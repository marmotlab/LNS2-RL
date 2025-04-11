#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <string>
#include <sstream>
#include <filesystem>
#include <chrono>

// Include our rerun configuration to get RERUN_ENABLED definition
#include "./inc/rerun_config.h"

// Conditionally include OpenCV
#if defined(OPENCV_ENABLED) || defined(CV_VERSION)
#include <opencv2/opencv.hpp>
#define OPENCV_AVAILABLE 1
#else
// Try to include OpenCV, but don't fail if it's not available
#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#define OPENCV_AVAILABLE 1
#else
#define OPENCV_AVAILABLE 0
#endif
#endif

#include "./inc/mylns2.h"
#include "./inc/SingleAgentSolver.h"

// Handle filesystem includes based on compiler support
#if defined(HAVE_STD_FILESYSTEM)
#include <filesystem>
namespace fs = std::filesystem;
#elif defined(HAVE_STD_EXPERIMENTAL_FILESYSTEM)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <sys/stat.h>
// Simple directory creation function if filesystem is not available
namespace fs {
    inline void create_directories(const std::string& path) {
        // Simple implementation to create a directory
        mkdir(path.c_str(), 0755);
    }
}
#endif

// Create a 64x64 map with random obstacles.
std::vector<std::vector<int>> createMap64x64(double obstacleProb = 0.1) {
    const int SIZE = 64;
    std::vector<std::vector<int>> map(SIZE, std::vector<int>(SIZE, 0));
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            double p = static_cast<double>(std::rand()) / RAND_MAX;
            if (p < obstacleProb) {
                map[r][c] = 1; // Obstacle
            }
        }
    }
    return map;
}

// For simplicity, this function returns a fixed set of 5 agent start and goal positions.
void createAgentPositions(std::vector<std::pair<int, int>> &starts,
                          std::vector<std::pair<int, int>> &goals) {
    starts = {
        {2, 2}, {3, 50}, {50, 10}, {30, 30}, {60, 5}
    };
    goals = {
        {10, 10}, {10, 40}, {55, 55}, {2, 60}, {5, 60}
    };
}

// Save a debug visualization image of the environment and paths.
void dumpGlobalVisualization(const std::vector<std::vector<int>> &map,
                              int gridSize,
                              const std::vector<std::pair<int,int>> &agentStarts,
                              const std::vector<std::pair<int,int>> &agentGoals,
                              const std::vector<std::vector<std::pair<int,int>>> &paths,
                              int iteration,
                              const std::string& scenarioName)
{
#if OPENCV_AVAILABLE
    const int scale = 10;
    cv::Mat vis(gridSize * scale, gridSize * scale, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw obstacles (black)
    for (int r = 0; r < gridSize; r++) {
        for (int c = 0; c < gridSize; c++) {
            if (map[r][c] == 1) {
                cv::rectangle(vis,
                              cv::Point(c * scale, r * scale),
                              cv::Point((c + 1) * scale - 1, (r + 1) * scale - 1),
                              cv::Scalar(0, 0, 0), cv::FILLED);
            }
        }
    }

    // Draw agent goals (blue)
    for (const auto &g : agentGoals) {
        cv::circle(vis,
                   cv::Point(g.second * scale + scale / 2, g.first * scale + scale / 2),
                   scale / 2,           // Increased radius
                   cv::Scalar(255, 0, 0),// Blue (BGR)
                   cv::FILLED);
    }

    // Draw agent starts (red)
    for (const auto &s : agentStarts) {
        cv::circle(vis,
                   cv::Point(s.second * scale + scale / 2, s.first * scale + scale / 2),
                   scale / 2,           // Increased radius
                   cv::Scalar(0, 0, 255),// Red (BGR)
                   cv::FILLED);
    }

    // Draw paths in green with thicker lines and intermediate markers
    for (const auto &path : paths) {
        for (size_t i = 1; i < path.size(); i++) {
            auto [r0, c0] = path[i - 1];
            auto [r1, c1] = path[i];
            cv::line(vis,
                     cv::Point(c0 * scale + scale / 2, r0 * scale + scale / 2),
                     cv::Point(c1 * scale + scale / 2, r1 * scale + scale / 2),
                     cv::Scalar(0, 255, 0), // Green
                     2,                   // Thicker line
                     cv::LINE_AA);
            cv::circle(vis,
                       cv::Point(c0 * scale + scale / 2, r0 * scale + scale / 2),
                       scale / 4,          // Intermediate marker size
                       cv::Scalar(0, 255, 0),
                       cv::FILLED);
        }
        if (!path.empty()) {
            auto [r_last, c_last] = path.back();
            cv::circle(vis,
                       cv::Point(c_last * scale + scale / 2, r_last * scale + scale / 2),
                       scale / 4,          // Final marker size
                       cv::Scalar(0, 255, 0),
                       cv::FILLED);
        }
    }

    fs::create_directories("global_visualizations");
    std::stringstream ss;
    ss << "global_visualizations/" << scenarioName << "_iter_" << (iteration + 2) << ".png";
    bool saved = cv::imwrite(ss.str(), vis);
    assert(saved && "Failed to save visualization image");
    std::cout << "Saved visualization: " << ss.str() << std::endl;
#else
    // OpenCV not available, just print a message
    std::cout << "OpenCV not available, skipping visualization for scenario: " << scenarioName
              << " iteration: " << iteration << std::endl;

    // Still create the directory for consistency
    fs::create_directories("global_visualizations");
#endif
}

// Scenario struct for testing
struct Scenario {
    int gridSize;
    int numAgents;
    double obstacleProb;
    std::string description;
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --enable-rerun    Enable Rerun visualization\n"
              << "  --disable-rerun   Disable Rerun visualization\n"
              << "  --help            Display this help message\n";
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    bool useRerun = true;  // Default value

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--enable-rerun") {
            useRerun = true;
        } else if (arg == "--disable-rerun") {
            useRerun = false;
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "Rerun visualization: " << (useRerun ? "enabled" : "disabled") << std::endl;

    std::srand(420);

    // Define a set of scenarios to test.
    std::vector<Scenario> scenarios = {
        {64, 5, 0.1, "5_agents_low_obstacles"},
        {64, 5, 0.2, "5_agents_medium_obstacles"},
        {64, 5, 0.3, "5_agents_high_obstacles"},
        {64, 8, 0.1, "8_agents_low_obstacles"},
        {64, 8, 0.2, "8_agents_medium_obstacles"},
        {64, 8, 0.3, "8_agents_high_obstacles"}
    };

    // Loop over scenarios.
    for (const auto& sc : scenarios) {
        std::cout << "Running scenario: " << sc.description << std::endl;
        // Create the map.
        auto map_data = createMap64x64(sc.obstacleProb);

        // For now, if sc.numAgents == 5, use fixed positions; otherwise, for 8 agents, we generate extra random positions.
        std::vector<std::pair<int, int>> starts, goals;
        if (sc.numAgents == 5) {
            createAgentPositions(starts, goals);
        } else {
            // Generate random start/goal positions on free cells.
            // (This is a simple placeholder; in practice you might need more robust generation.)
            for (int i = 0; i < sc.numAgents; i++) {
                // Find a random free cell
                int r, c;
                do {
                    r = std::rand() % sc.gridSize;
                    c = std::rand() % sc.gridSize;
                } while (map_data[r][c] == 1);
                starts.push_back({r, c});

                // For goal, try a different random free cell.
                do {
                    r = std::rand() % sc.gridSize;
                    c = std::rand() % sc.gridSize;
                } while (map_data[r][c] == 1);
                goals.push_back({r, c});
            }
        }

        // Create the planner.
        // (Note: MyLns2 currently uses fixed positions for 5 agents. For testing 8 agents,
        //  you may need to adjust your MyLns2 class accordingly.)
        MyLns2 lns2(123, map_data, starts, goals, sc.numAgents, sc.gridSize);

        // Set Rerun usage based on command-line argument
        lns2.setUseRerun(useRerun);

        // Record start time.
        auto start_time = std::chrono::steady_clock::now();

        // Run the initial path planning.
        lns2.init_pp();
        assert(!lns2.vector_path.empty());

        // Record end time.
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> planning_duration = end_time - start_time;
        std::cout << "Planning time for scenario " << sc.description << ": "
                  << planning_duration.count() << " seconds" << std::endl;

        // Dump visualization of the initial plan.
        dumpGlobalVisualization(map_data, sc.gridSize, starts, goals, lns2.vector_path, 0, sc.description);

        // Optionally perform a partial replan for a subset of agents.
        std::vector<int> subset = {1, 3};
        int collisions = lns2.calculate_sipps(subset);
        std::cout << "Partial replan for agents {1, 3}, collisions: " << collisions << std::endl;
        for (size_t i = 0; i < subset.size(); i++) {
            int ag_id = subset[i];
            if (i < lns2.sipps_path.size())
                lns2.vector_path[ag_id] = lns2.sipps_path[i];
        }
        dumpGlobalVisualization(map_data, sc.gridSize, starts, goals, lns2.vector_path, 1, sc.description);

        // Print final path plan.
        std::cout << "=== Final path plan for scenario " << sc.description << " ===" << std::endl;
        for (size_t ag = 0; ag < lns2.vector_path.size(); ag++) {
            std::cout << "Agent " << ag << ": ";
            for (const auto &pt : lns2.vector_path[ag])
                std::cout << "(" << pt.first << "," << pt.second << ") -> ";
            std::cout << "[GOAL]" << std::endl;
        }
        dumpGlobalVisualization(map_data, sc.gridSize, starts, goals, lns2.vector_path, 2, sc.description);

        std::cout << "----------------------------------------" << std::endl;
    }

    return 0;
}