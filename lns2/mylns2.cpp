#include "mylns2.h"
#include "config.h"  // For DEFAULT_RERUN_VIEWER_ADDRESS
#include <iostream>
#include <random>
#include "common.h"
#include <utility>
#include <cstdlib>  // for std::getenv
#include <unistd.h>
#include <chrono>
#include <thread>

// Init MyLns2 with dynamic map sequence
MyLns2::MyLns2(int seed,
               std::vector<std::vector<std::vector<float>>> dynamic_map_seq,
               std::vector<std::pair<int, int>> start_poss,
               std::vector<std::pair<int, int>> goal_poss,
               int all_ag_num,
               int map_size)
    : instance(dynamic_map_seq, start_poss, goal_poss, all_ag_num, map_size),
      path_table(map_size * map_size, all_ag_num)
{
#if RERUN_ENABLED
    if (use_rerun) {
        rec = std::make_shared<rerun::RecordingStream>("rerun_mylns2_" + std::to_string(seed));

        // Check for the viewer address from the environment; if not set, use the default.
        const char* viewer_addr = std::getenv("RERUN_VIEWER_ADDRESS");
        if (!viewer_addr) {
            viewer_addr = DEFAULT_RERUN_VIEWER_ADDRESS;
        }
        // In Rerun v0.22.1, there is no connect() method. Instead, we can spawn the viewer.
        auto spawn_result = rec->spawn();
        spawn_result.exit_on_failure(); // Abort if spawning fails

        log_prefix = "env_" + std::to_string(seed);
        if (!map_logged) {
            // Extract static map from the first frame of the dynamic map sequence
            // In MAPF: -1 is obstacle, 0 is free space
            // In GridBasedPathPlanning: 1 is obstacle, 0 is free space
            // For Rerun visualization: 1 is obstacle, 0 is free space
            std::vector<std::vector<int>> static_map(dynamic_map_seq[0].size(), std::vector<int>(dynamic_map_seq[0][0].size(), 0));

            // Extract static map based on GridMapEnv representation
            // In GridMapEnv, static obstacles have values > 0.9
            for (size_t i = 0; i < dynamic_map_seq[0].size(); i++) {
                for (size_t j = 0; j < dynamic_map_seq[0][0].size(); j++) {
                    if (dynamic_map_seq[0][i][j] > 0.9) {
                        static_map[i][j] = 1; // Set as obstacle for Rerun
                    }
                }
            }

            // Print some debug info about the map
            int obstacle_count = 0;
            for (const auto& row : static_map) {
                for (int cell : row) {
                    if (cell == 1) obstacle_count++;
                }
            }
            // std::cout << "[DEBUG] Static map size: " << static_map.size() << "x" << static_map[0].size() << std::endl;
            // std::cout << "[DEBUG] Static map obstacle count: " << obstacle_count << std::endl;
            publishMapRerun(static_map, instance.start_locations);
            map_logged = true;
        }
    }
#else
    // Rerun is disabled, set use_rerun to false
    use_rerun = false;
    log_prefix = "env_" + std::to_string(seed);
#endif

    srand(seed);
    agents.reserve(all_ag_num);
    for (int i = 0; i < all_ag_num; i++) {
        agents.emplace_back(instance, i, instance.start_locations, instance.goal_locations);

#if RERUN_ENABLED
        // Set the use_rerun flag for each agent's path planner
        agents[i].path_planner->setUseRerun(use_rerun);
#endif
    }
}

// Legacy init with static map only
MyLns2::MyLns2(int seed,
               std::vector<std::vector<int>> obs_map,
               std::vector<std::pair<int, int>> start_poss,
               std::vector<std::pair<int, int>> goal_poss,
               int all_ag_num,
               int map_size)
    : instance(obs_map, start_poss, goal_poss, all_ag_num, map_size),
      path_table(map_size * map_size, all_ag_num)
{
#if RERUN_ENABLED
    if (use_rerun) {
        rec = std::make_shared<rerun::RecordingStream>("rerun_mylns2_" + std::to_string(seed));

        // Check for the viewer address from the environment; if not set, use the default.
        const char* viewer_addr = std::getenv("RERUN_VIEWER_ADDRESS");
        if (!viewer_addr) {
            viewer_addr = DEFAULT_RERUN_VIEWER_ADDRESS;
        }
        // In Rerun v0.22.1, there is no connect() method. Instead, we can spawn the viewer.
        auto spawn_result = rec->spawn();
        spawn_result.exit_on_failure(); // Abort if spawning fails

        log_prefix = "env_" + std::to_string(seed);
        if (!map_logged) {
            // Check if this is a MAPF representation map (-1 for obstacles)
            bool is_mapf_format = false;
            for (const auto& row : obs_map) {
                for (int cell : row) {
                    if (cell == -1) {
                        is_mapf_format = true;
                        break;
                    }
                }
                if (is_mapf_format) break;
            }

            if (is_mapf_format) {
                // Convert from MAPF format to Rerun format
                std::vector<std::vector<int>> rerun_map(obs_map.size(), std::vector<int>(obs_map[0].size(), 0));

                // Count obstacles in original map
                int orig_obstacle_count = 0;
                for (const auto& row : obs_map) {
                    for (int cell : row) {
                        if (cell == -1) orig_obstacle_count++;
                    }
                }
                // std::cout << "[DEBUG] Original map has " << orig_obstacle_count << " obstacles (-1 values)" << std::endl;

                // Convert and count converted obstacles
                int converted_obstacle_count = 0;
                for (size_t i = 0; i < obs_map.size(); i++) {
                    for (size_t j = 0; j < obs_map[i].size(); j++) {
                        if (obs_map[i][j] == -1) {
                            rerun_map[i][j] = 1; // Convert -1 to 1 for obstacles
                            converted_obstacle_count++;
                        }
                    }
                }
                // std::cout << "[DEBUG] Converting MAPF format map to Rerun format" << std::endl;
                // std::cout << "[DEBUG] Converted map has " << converted_obstacle_count << " obstacles (1 values)" << std::endl;

                // Force a small delay to ensure visualization is initialized
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                publishMapRerun(rerun_map, instance.start_locations);
            } else {
                // Use the map as is
                // std::cout << "[DEBUG] Map is already in Rerun format (using 1 for obstacles)" << std::endl;
                publishMapRerun(obs_map, instance.start_locations);
            }
            map_logged = true;
        }
    }
#else
    // Rerun is disabled, set use_rerun to false
    use_rerun = false;
    log_prefix = "env_" + std::to_string(seed);
#endif

    srand(seed);
    agents.reserve(all_ag_num);
    for (int i = 0; i < all_ag_num; i++) {
        agents.emplace_back(instance, i, instance.start_locations, instance.goal_locations);

#if RERUN_ENABLED
        // Set the use_rerun flag for each agent's path planner
        agents[i].path_planner->setUseRerun(use_rerun);
#endif
    }
}

MyLns2::~MyLns2() = default;


#if RERUN_ENABLED
void MyLns2::publishMapRerun(const std::vector<std::vector<int>> &map,
    const std::vector<int> &start_locations)
{
    if (!use_rerun || !rec) return;

    int height = map.size();
    if (height == 0) {
        std::cout << "[ERROR] Empty map passed to publishMapRerun!" << std::endl;
        return;
    }

    int width = map[0].size();
    if (width == 0) {
        std::cout << "[ERROR] Empty map row passed to publishMapRerun!" << std::endl;
        return;
    }

    // Print debug info about the map
    int obstacle_count = 0;
    for (const auto& row : map) {
        for (int cell : row) {
            if (cell == 1) obstacle_count++;
        }
    }
    // std::cout << "[DEBUG] publishMapRerun - Map size: " << height << "x" << width << std::endl;
    // std::cout << "[DEBUG] publishMapRerun - Obstacle count: " << obstacle_count << std::endl;

    // Create a simple grid representation with points
    std::vector<rerun::components::Position3D> grid_positions;
    std::vector<rerun::components::Color> grid_colors;
    std::vector<float> grid_radii;

    // Create points for ALL cells (both free space and obstacles)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            grid_positions.emplace_back(x + 0.5f, y + 0.5f, -0.1f); // Center of cell, negative z to be below everything else

            if (map[y][x] == 0) { // Free space
                grid_colors.push_back(rerun::components::Color(150, 150, 150, 255)); // Grey for free space
            } else { // Obstacle
                grid_colors.push_back(rerun::components::Color(0, 0, 0, 255)); // Black for obstacles
            }

            grid_radii.push_back(0.49f); // Slightly smaller than full-sized to see grid lines
        }
    }

    // std::cout << "[DEBUG] Created grid with " << grid_positions.size() << " cells" << std::endl;

    // Log the entire grid with timeline set to 0
    rec->set_time_sequence("planning_time", 0);
    rec->log(log_prefix + "/map/static_grid",
            rerun::Points3D(rerun::Collection<rerun::components::Position3D>(grid_positions))
            .with_colors(grid_colors)
            .with_radii(grid_radii));

    // Also log a separate layer just for free space cells with white color
    std::vector<rerun::components::Position3D> free_positions;
    std::vector<rerun::components::Color> free_colors;
    std::vector<float> free_radii;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (map[y][x] == 0) { // Free space
                free_positions.emplace_back(x + 0.5f, y + 0.5f, -0.05f); // Slightly above the base grid
                free_colors.push_back(rerun::components::Color(150, 150, 150, 255)); // Grey for free space
                free_radii.push_back(0.48f); // Slightly smaller than the base grid
            }
        }
    }

    // std::cout << "[DEBUG] Created free space layer with " << free_positions.size() << " cells" << std::endl;

    // Log the free space layer with timeline set to 0
    if (!free_positions.empty()) {
        rec->set_time_sequence("planning_time", 0);
        rec->log(log_prefix + "/map/free_space_layer",
                rerun::Points3D(rerun::Collection<rerun::components::Position3D>(free_positions))
                .with_colors(free_colors)
                .with_radii(free_radii));
    }

    // Create grid lines to make the map structure more visible
    std::vector<rerun::components::LineStrip3D> horizontal_lines;
    std::vector<rerun::components::LineStrip3D> vertical_lines;

    // Create horizontal grid lines
    for (int y = 0; y <= height; y++) {
        std::vector<rerun::components::Position3D> line_points;
        line_points.push_back(rerun::components::Position3D(0, y, 0.01f));
        line_points.push_back(rerun::components::Position3D(width, y, 0.01f));
        horizontal_lines.push_back(rerun::components::LineStrip3D(
            rerun::Collection<rerun::components::Position3D>(line_points)
        ));
    }

    // Create vertical grid lines
    for (int x = 0; x <= width; x++) {
        std::vector<rerun::components::Position3D> line_points;
        line_points.push_back(rerun::components::Position3D(x, 0, 0.01f));
        line_points.push_back(rerun::components::Position3D(x, height, 0.01f));
        vertical_lines.push_back(rerun::components::LineStrip3D(
            rerun::Collection<rerun::components::Position3D>(line_points)
        ));
    }

    // Log horizontal grid lines with timeline set to 0
    rec->set_time_sequence("planning_time", 0);
    rec->log(log_prefix + "/map/grid_lines_horizontal",
            rerun::LineStrips3D(rerun::Collection<rerun::components::LineStrip3D>(horizontal_lines))
            .with_colors(std::vector<rerun::components::Color>(horizontal_lines.size(), rerun::components::Color(100, 100, 100, 150))) // Medium gray with transparency
            .with_radii(std::vector<float>(horizontal_lines.size(), 0.02f)));

    // Log vertical grid lines with timeline set to 0
    rec->set_time_sequence("planning_time", 0);
    rec->log(log_prefix + "/map/grid_lines_vertical",
            rerun::LineStrips3D(rerun::Collection<rerun::components::LineStrip3D>(vertical_lines))
            .with_colors(std::vector<rerun::components::Color>(vertical_lines.size(), rerun::components::Color(100, 100, 100, 150))) // Medium gray with transparency
            .with_radii(std::vector<float>(vertical_lines.size(), 0.02f)));

    // std::cout << "[DEBUG] Created grid lines: " << horizontal_lines.size() << " horizontal, " << vertical_lines.size() << " vertical" << std::endl;

    // Create points for obstacles with a raised z-coordinate for better visibility
    std::vector<rerun::components::Position3D> obstacle_positions;
    std::vector<rerun::components::Color> obstacle_colors;
    std::vector<float> obstacle_radii;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (map[y][x] == 1) { // Static obstacle
                obstacle_positions.emplace_back(x + 0.5f, y + 0.5f, 0.0f);  // At z=0 to be clearly visible
                obstacle_colors.push_back(rerun::components::Color(0, 0, 0, 255)); // Black with full opacity
                obstacle_radii.push_back(0.48f); // Almost full-sized
            }
        }
    }

    // std::cout << "[DEBUG] Found " << obstacle_positions.size() << " obstacles to visualize" << std::endl;

    if (!obstacle_positions.empty()) {
        // Force a small delay to ensure visualization is initialized
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Log obstacles in smaller batches to avoid overwhelming Rerun
        const size_t batch_size = 100;
        for (size_t i = 0; i < obstacle_positions.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, obstacle_positions.size());
            size_t count = end - i;

            std::vector<rerun::components::Position3D> batch_positions;
            std::vector<rerun::components::Color> batch_colors;
            std::vector<float> batch_radii;

            for (size_t j = i; j < end; j++) {
                batch_positions.push_back(obstacle_positions[j]);
                batch_colors.push_back(obstacle_colors[j]);
                batch_radii.push_back(obstacle_radii[j]);
            }

            rec->log(log_prefix + "/map/obstacles_batch_" + std::to_string(i/batch_size),
                    rerun::Points3D(rerun::Collection<rerun::components::Position3D>(batch_positions))
                    .with_colors(batch_colors)
                    .with_radii(batch_radii));

            // std::cout << "[DEBUG] Logged obstacle batch " << i/batch_size << " with " << count << " obstacles" << std::endl;

            // Small delay between batches
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // No additional visualizations needed - we already have the static grid and obstacle points
    } else {
        std::cout << "[WARNING] No obstacles to visualize!" << std::endl;

        // If no obstacles found, create test obstacles to verify visualization works
        // std::cout << "[DEBUG] Creating test obstacles to verify visualization" << std::endl;

        // Create a 3x3 grid of test obstacles
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                obstacle_positions.emplace_back(x + 0.5f, y + 0.5f, 0.2f);
                obstacle_colors.push_back(rerun::components::Color(0, 0, 0)); // Black test obstacles
                obstacle_radii.push_back(0.45f);
            }
        }

        rec->log(log_prefix + "/map/test_obstacle",
                rerun::Points3D(rerun::Collection<rerun::components::Position3D>(obstacle_positions))
                .with_colors(obstacle_colors)
                .with_radii(obstacle_radii));
    }

    // Log agent start positions (red)
    std::vector<rerun::components::Position3D> all_start_positions;
    std::vector<rerun::components::Color> all_start_colors;
    std::vector<float> all_start_radii;

    for (int loc : start_locations) {
        int x = loc % width;
        int y = loc / width;
        all_start_positions.push_back(rerun::components::Position3D(x + 0.5f, y + 0.5f, 0.05f)); // Above obstacles
        all_start_colors.push_back(rerun::components::Color(255, 105, 180)); // Pink for start positions
        all_start_radii.push_back(0.4f);
    }

    if (!all_start_positions.empty()) {
        rec->log(log_prefix + "/agent_starts",
                rerun::Points3D(rerun::Collection<rerun::components::Position3D>(all_start_positions))
                .with_colors(all_start_colors)
                .with_radii(all_start_radii));
    }

    // Log agent goal positions (green)
    std::vector<rerun::components::Position3D> all_goal_positions;
    std::vector<rerun::components::Color> all_goal_colors;
    std::vector<float> all_goal_radii;

    for (int goal_loc : instance.goal_locations) {
        int x = goal_loc % width;
        int y = goal_loc / width;
        all_goal_positions.push_back(rerun::components::Position3D(x + 0.5f, y + 0.5f, 0.05f)); // Above obstacles
        all_goal_colors.push_back(rerun::components::Color(0, 0, 255)); // Blue for agent goal positions
        all_goal_radii.push_back(0.4f);
    }

    if (!all_goal_positions.empty()) {
        rec->log(log_prefix + "/agent_goals",
                rerun::Points3D(rerun::Collection<rerun::components::Position3D>(all_goal_positions))
                .with_colors(all_goal_colors)
                .with_radii(all_goal_radii));
    }
}
#else
void MyLns2::publishMapRerun(const std::vector<std::vector<int>> &map,
    const std::vector<int> &start_locations)
{
    // Empty implementation when Rerun is disabled
}
#endif

void MyLns2::visualizeReferencePaths(int current_agent_id)
{
#if RERUN_ENABLED
    if (!use_rerun || !rec) return;

    // Visualize paths of previously planned agents as reference paths (light blue)
    for (int agent_id = 0; agent_id < (int)agents.size(); agent_id++) {
        // Skip the current agent and agents without paths
        if (agent_id == current_agent_id || agents[agent_id].path.empty()) {
            continue;
        }

        // Create path positions for this agent
        std::vector<rerun::components::Position3D> path_positions;
        for (const auto& step : agents[agent_id].path) {
            int px = instance.getColCoordinate(step.location);
            int py = instance.getRowCoordinate(step.location);
            path_positions.push_back(rerun::components::Position3D(px + 0.5f, py + 0.5f, 0.03f)); // Above obstacles but below agent markers
        }

        // Skip if path is empty
        if (path_positions.empty()) {
            continue;
        }

        // Create a line strip for the path
        if (path_positions.size() > 1) {
            std::vector<rerun::components::LineStrip3D> strips = {
                rerun::components::LineStrip3D(
                    rerun::Collection<rerun::components::Position3D>(path_positions)
                )
            };

            // Log the path as a line strip (light blue)
            rec->log(log_prefix + "/reference_paths/agent_" + std::to_string(agent_id),
                     rerun::LineStrips3D(rerun::Collection<rerun::components::LineStrip3D>(strips))
                         .with_colors({rerun::components::Color(173, 216, 230)}) // Light blue
                         .with_radii({0.1f}));
        }
    }
#endif
}

/*#########################
% MyLns2::init_pp:
%
% Inputs:
%   - agents: az összes ügynök objektuma, amelyekhez SIPP alapú útvonaltervezést rendelünk.
%   - instance: a térkép és ügynökök adatait tartalmazza.
%   - path_table: globális útvonal-tábla, amelybe az egyes ügynökök útvonalait beillesztjük.
%
% Function behaviour:
%   - Minden ügynök id-jét hozzáadja a neighbor.agents listához.
%   - Ha egy ügynök path_planner-je SIPP típusú, beállítja a log_prefix, rec és use_rerun értékeket.
%   - Újra feltölti a neighbor.agents listát és véletlenszerű sorrendbe rendezi.
%   - Létrehoz egy ConstraintTable-t a térkép méretei és a path_table alapján.
%   - Minden ügynöknél a SIPP-et hívja meg a findPath() függvénnyel, majd ellenőrzi, hogy az útvonal nem üres és a cél megfelelő.
%   - Beilleszti az útvonalat a path_table-ba, majd a lineáris koordinátákat eltárolja a vector_path-ben.
%
% Outputs:
%   - Az ügynökök inicializált útvonalai bekerülnek a path_table-ba és vector_path-be.
*/
void MyLns2::init_pp()
{
    for (int i = 0; i < (int)agents.size(); i++) {
        neighbor.agents.push_back(i);
        auto* sipp_solver = dynamic_cast<SIPP*>(agents[i].path_planner);
        if (sipp_solver) {
            sipp_solver->log_prefix = log_prefix;
#if RERUN_ENABLED
            sipp_solver->rec = rec;
#endif
            sipp_solver->use_rerun = use_rerun;
        }
    }
    neighbor.agents.reserve(agents.size());
    for (int i = 0; i < (int)agents.size(); i++)
        neighbor.agents.push_back(i);
    std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    for (auto id : neighbor.agents)
    {
#if RERUN_ENABLED
        // Collect other agents' information for path planning and visualization
        // This should always run, regardless of visualization settings
        std::vector<int> other_starts;
        std::vector<int> other_goals;
        std::vector<Path> other_paths;

        for (int other_id = 0; other_id < (int)agents.size(); other_id++) {
            if (other_id != id && !agents[other_id].path.empty()) {
                other_starts.push_back(agents[other_id].path_planner->start_location);
                other_goals.push_back(agents[other_id].path_planner->goal_location);
                other_paths.push_back(agents[other_id].path);
            }
        }

        // Set the information for path planning and visualization
        // The path planner will handle visualization conditionally
        agents[id].path_planner->setOtherAgentsInfo(other_starts, other_goals, other_paths);

        // Highlight the current agent's goal in yellow

        if (use_rerun && rec) {
            int goal_x = instance.getColCoordinate(agents[id].path_planner->goal_location);
            int goal_y = instance.getRowCoordinate(agents[id].path_planner->goal_location);

            // Create a yellow marker for the current agent's goal
            std::vector<rerun::components::Position3D> current_goal_pos;
            current_goal_pos.push_back(rerun::components::Position3D(goal_x + 0.5f, goal_y + 0.5f, 0.1f));
            std::vector<rerun::components::Color> current_goal_colors;
            current_goal_colors.push_back(rerun::components::Color(255, 255, 0)); // Yellow
            std::vector<float> current_goal_radii;
            current_goal_radii.push_back(0.45f);

            rec->log(log_prefix + "/current_agent_goal",
                    rerun::Points3D()
                        .with_positions(current_goal_pos)
                        .with_colors(current_goal_colors)
                        .with_radii(current_goal_radii));
        }
#endif

        // Visualize reference paths of previously planned agents
        visualizeReferencePaths(id);

        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        // Add this before the assertion
    if (agents[id].path.empty()) {
        std::cout << "ERROR: Agent " << id << " has empty path after planning" << std::endl;
        std::cout << "  Start location: " << agents[id].path_planner->start_location << std::endl;
        std::cout << "  Goal location: " << agents[id].path_planner->goal_location << std::endl;
    } else if (agents[id].path.back().location != agents[id].path_planner->goal_location) {
        std::cout << "ERROR: Agent " << id << " final location mismatch" << std::endl;
        std::cout << "  Path back location: " << agents[id].path.back().location << std::endl;
        std::cout << "  Goal location: " << agents[id].path_planner->goal_location << std::endl;
    }  // no hard obstacle, thus must find path
        path_table.insertPath(agents[id].id, agents[id].path);

        // Clear the current agent's goal highlight after planning is complete
#if RERUN_ENABLED
        if (use_rerun && rec) {
            rec->log(log_prefix + "/current_agent_goal", rerun::Clear());
        }
#endif
    }
    vector_path.reserve(instance.num_of_agents);
    for (const auto &agent : agents)
    {
        vector<pair<int,int>> single_path;
        for (const auto &state : agent.path)
            single_path.push_back(instance.getCoordinate(state.location));
        vector_path.push_back(single_path);
    }
}


/*#########################
% MyLns2::updateCollidingPairs:
%
% Inputs:
%   - colliding_pairs: egy üres vagy meglévő halmaz, amelybe a konfliktusos ügynök párokat fogja gyűjteni.
%   - agent_id: az aktuálisan ellenőrzendő ügynök azonosítója.
%   - path: az adott ügynök aktuális útvonala.
%
% Function behaviour:
%   - Iterál az útvonal időlépésein, és ellenőrzi:
%       a) Vertex konfliktusokat: ha más ügynök is eléri ugyanazt a cellát egy adott időpillanatban.
%       b) Edge konfliktusokat: ha két ügynök ellentétes irányban mozog ugyanazon él mentén.
%       c) Target konfliktusokat: ha az útvonal áthalad egy olyan cellán, ami már más ügynök célja.
%   - Ha konfliktust talál, hozzáadja a megfelelő ügynök-azonosító párokat a colliding_pairs halmazhoz.
%
% Outputs:
%   - Visszatér egy logikai értékkel, amely jelzi, hogy történt-e konfliktus (true), vagy nem (false).
*/
bool MyLns2::updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path)
{
    bool succ = false;
    if (path.size() < 2)
        return succ;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {
                succ = true;
                colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));// emplace: insert new element into set
            }
        }
        if (from != to && path_table.table[to].size() >= t && path_table.table[from].size() > t) // edge conflicts(swapping conflicts)
        {
            for (auto a1 : path_table.table[to][t - 1])
            {
                for (auto a2: path_table.table[from][t])
                {
                    if (a1 == a2)
                    {
                        succ = true;
                        colliding_pairs.emplace(min(agent_id, a2), max(agent_id, a2));
                        break;
                    }
                }
            }
        }
        if (!path_table.goals.empty() && path_table.goals[to] < t) // target conflicts, already has agent in its goal, so the new agent can not tarverse it
        { // this agent traverses the target of another agent
            for (auto id : path_table.table[to][path_table.goals[to]]) // look at all agents at the goal time
            {
                if (agents[id].path.back().location == to) // if agent id's goal is to, then this is the agent we want
                {
                    succ = true;
                    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
                    break;
                }
            }
        }
    }
    int goal = path.back().location; // target conflicts - some other agent traverses the target of this agent
    for (int t = (int)path.size(); t < path_table.table[goal].size(); t++)
    {
        for (auto id : path_table.table[goal][t])
        {
            succ = true;
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
    return succ;
}


/*#########################
% MyLns2::calculate_sipps:
%
% Inputs:
%   - new_agents: egy vektor, amely azoknak az ügynököknek az id-jeit tartalmazza, akik újratervezésre kerülnek.
%
% Function behaviour:
%   - Törli az újratervezésre kijelölt ügynökök útvonalait a path_table-ból.
%   - Létrehoz egy ConstraintTable-t a frissített path_table-ból.
%   - Iteratív módon, minden kijelölt ügynöknél meghívja a SIPP-et az új útvonal kiszámolására.
%   - Ha egy ügynök útvonala üres, fallback stratégiát alkalmaz (pl. a start pozíció mentése).
%   - Ellenőrzi, hogy az út végén az ügynök célpozíciója egyezik-e a várttal.
%   - Frissíti a colliding_pairs halmazt, ha az ügynök útvonalán konfliktusok találhatók.
%   - Az új útvonalakat beilleszti a path_table-ba, és a lineáris koordinátákat eltárolja a sipps_path-ben.
%
% Outputs:
%   - Visszatér a jelenlegi colliding_pairs számával, jelezve, hány konfliktus maradt.
*/

// Helper function to count waiting actions in a path
int countWaitingActions(const Path& path) {
    if (path.size() <= 1) return 0;

    int wait_count = 0;
    int consecutive_waits = 0;
    int max_consecutive_waits = 0;

    for (size_t i = 1; i < path.size(); i++) {
        if (path[i].location == path[i-1].location) {
            wait_count++;
            consecutive_waits++;

            // Track the maximum consecutive waits
            max_consecutive_waits = std::max(max_consecutive_waits, consecutive_waits);
        } else {
            consecutive_waits = 0;
        }
    }

    // Weight consecutive waits more heavily
    // This prioritizes agents with long waiting sequences
    return wait_count + max_consecutive_waits * 3;
}

int MyLns2::calculate_sipps(vector<int> new_agents)
{
    neighbor.colliding_pairs.clear();

    // Sort agents by waiting actions (if they have paths)
    // This prioritizes agents with more waiting actions for replanning
    std::vector<std::pair<int, int>> agent_wait_counts;
    for (int agent_id : new_agents) {
        int wait_count = 0;
        if (agent_id < agents.size() && !agents[agent_id].path.empty()) {
            wait_count = countWaitingActions(agents[agent_id].path);
        }
        agent_wait_counts.push_back(std::make_pair(agent_id, wait_count));
    }

    // Sort by wait count in descending order
    std::sort(agent_wait_counts.begin(), agent_wait_counts.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.second > b.second;
              });

    // Extract just the agent IDs in the new order
    neighbor.agents.clear();
    for (const auto& pair : agent_wait_counts) {
        neighbor.agents.push_back(pair.first);
        // Debug output to show prioritization
        if (pair.second > 0) {
            // Count actual waits and consecutive waits for detailed output
            int actual_waits = 0;
            int consecutive_waits = 0;
            int max_consecutive_waits = 0;

            for (size_t i = 1; i < agents[pair.first].path.size(); i++) {
                if (agents[pair.first].path[i].location == agents[pair.first].path[i-1].location) {
                    actual_waits++;
                    consecutive_waits++;
                    max_consecutive_waits = std::max(max_consecutive_waits, consecutive_waits);
                } else {
                    consecutive_waits = 0;
                }
            }

            std::cout << "Agent " << pair.first << " has " << actual_waits
                      << " waiting actions (max " << max_consecutive_waits
                      << " consecutive) - priority score: " << pair.second
                      << " - prioritizing for replanning" << std::endl;
        }
    }

    for (int i = 0; i < (int)neighbor.agents.size(); i++)
        path_table.deletePath(neighbor.agents[i]);
    makespan=path_table.makespan;
    auto p = neighbor.agents.begin();
    sipps_path.clear();
    sipps_path.reserve(new_agents.size());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    while (p != neighbor.agents.end())
    {
        int id = *p;

#if RERUN_ENABLED
        // Collect other agents' information for path planning and visualization
        // This should always run, regardless of visualization settings
        std::vector<int> other_starts;
        std::vector<int> other_goals;
        std::vector<Path> other_paths;

        for (int other_id = 0; other_id < (int)agents.size(); other_id++) {
            if (other_id != id && !agents[other_id].path.empty()) {
                other_starts.push_back(agents[other_id].path_planner->start_location);
                other_goals.push_back(agents[other_id].path_planner->goal_location);
                other_paths.push_back(agents[other_id].path);
            }
        }

        // Set the information for path planning and visualization
        // The path planner will handle visualization conditionally
        agents[id].path_planner->setOtherAgentsInfo(other_starts, other_goals, other_paths);

        // Highlight the current agent's goal in yellow

        if (use_rerun && rec) {
            int goal_x = instance.getColCoordinate(agents[id].path_planner->goal_location);
            int goal_y = instance.getRowCoordinate(agents[id].path_planner->goal_location);

            // Create a yellow marker for the current agent's goal
            std::vector<rerun::components::Position3D> current_goal_pos;
            current_goal_pos.push_back(rerun::components::Position3D(goal_x + 0.5f, goal_y + 0.5f, 0.1f));
            std::vector<rerun::components::Color> current_goal_colors;
            current_goal_colors.push_back(rerun::components::Color(255, 255, 0)); // Yellow
            std::vector<float> current_goal_radii;
            current_goal_radii.push_back(0.45f);

            rec->log(log_prefix + "/current_agent_goal",
                    rerun::Points3D()
                        .with_positions(current_goal_pos)
                        .with_colors(current_goal_colors)
                        .with_radii(current_goal_radii));
        }
#endif

        // Visualize reference paths of previously planned agents
        visualizeReferencePaths(id);

        agents[id].path = agents[id].path_planner->findPath(constraint_table);

        if (agents[id].path.empty()) {
            std::cout << "Warning: No path found for agent " << id << ". Using fallback strategy." << std::endl;

            // Fallback strategy options:
            // 1. Keep the agent at its start position
            Path fallback_path;
            fallback_path.push_back(PathEntry(agents[id].path_planner->start_location));
            agents[id].path = fallback_path;

            // 2. Or try alternative pathfinding method
            // agents[id].path = findAlternativePath(...);
        }

        // Now check that the path ends at the goal (or use an alternative goal if needed)
        if (agents[id].path.back().location != agents[id].path_planner->goal_location) {
            std::cout << "Warning: Agent " << id << " couldn't reach goal. Path ends at "
                      << agents[id].path.back().location << " instead of "
                      << agents[id].path_planner->goal_location << std::endl;
        }
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path);
        vector<pair<int,int>> single_path;
        for (const auto &state : agents[id].path)
        {   single_path.push_back(instance.getCoordinate(state.location));
        }
        sipps_path.push_back(single_path);
        path_table.insertPath(agents[id].id, agents[id].path);

        // Clear the current agent's goal highlight after planning is complete
#if RERUN_ENABLED
        if (use_rerun && rec) {
            rec->log(log_prefix + "/current_agent_goal", rerun::Clear());
        }
#endif

        ++p;
    }
    for (auto id : neighbor.agents)
    {
        if (agents[id].path.size()==1)
        {
            int to = agents[id].path[0].location;
            int t=0;
            for (auto & ag_list : path_table.table[to])
            {
                if (t!=0 && !ag_list.empty())
                {
                    for (int another_id: ag_list)
                        neighbor.colliding_pairs.emplace(min(another_id, id), max(another_id, id));// emplace: insert new element into set
                }
                t++;
            }
        }
    }

    return (int)neighbor.colliding_pairs.size();
}

/*#########################
% MyLns2::single_sipp:
%
% Inputs:
%   - dy_obs_path: Dinamikus akadályként kezelt útvonalak, amelyeket az ügynökök követnek.
%   - start_poss, goal_poss: Az ügynökök többi részének start és cél koordinátái.
%   - self_start_poss, self_goal_poss: Az adott ügynök saját start és cél koordinátái.
%   - global_num_agent: Az összes ügynök száma.
%
% Function behaviour:
%   - Létrehoz egy új ügynök listát (add_agents) és egy útvonal-táblát (add_path_table) a megadott globális ügynökszám alapján.
%   - Átalakítja a start és cél koordinátákat lineáris indexekké.
%   - Az első (global_num_agent - 1) ügynök útvonalait a dy_obs_path alapján rögzíti.
%   - Az utolsó ügynök esetében a SIPP segítségével kiszámolja az új útvonalat a ConstraintTable-t használva.
%   - Visszatér az utolsó ügynök útvonalának hosszával.
%
% Outputs:
%   - Az utolsó ügynök útvonalának hossza (lépések száma), mint int.
*/
int MyLns2::single_sipp(vector<vector<pair<int,int>>> dy_obs_path,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,
                                  pair<int,int> self_start_poss,pair<int,int> self_goal_poss,int global_num_agent)
{
    vector<Agent> add_agents;
    PathTableWC add_path_table(instance.map_size,global_num_agent);
    add_agents.reserve(global_num_agent); //vector.reserve: adjust capacity
    vector<int> start_locations;
    vector<int> goal_locations;
    start_locations.reserve(global_num_agent);
    goal_locations.reserve(global_num_agent);
    for(int id=0;id<global_num_agent-1;id++)
    {
        start_locations.push_back(instance.linearizeCoordinate(start_poss[id].first, start_poss[id].second));
        goal_locations.push_back(instance.linearizeCoordinate(goal_poss[id].first, goal_poss[id].second));}
    start_locations.push_back(instance.linearizeCoordinate(self_start_poss.first, self_start_poss.second));
    goal_locations.push_back(instance.linearizeCoordinate(self_goal_poss.first, self_goal_poss.second));
    for(int id=0;id<global_num_agent-1;id++)
    {
        add_agents.emplace_back(instance, id,start_locations,goal_locations);
        add_agents[id].path.resize(dy_obs_path[id].size());
        for (int t=0;t<(int)dy_obs_path[id].size();t++)
        {
            add_agents[id].path[t].location=instance.linearizeCoordinate(dy_obs_path[id][t].first, dy_obs_path[id][t].second);
        }
        add_path_table.insertPath(id, add_agents[id].path);
    }
    add_agents.emplace_back(instance, global_num_agent-1,start_locations,goal_locations);
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &add_path_table);
    add_agents[global_num_agent-1].path = add_agents[global_num_agent-1].path_planner->findPath(constraint_table);
    assert(!add_agents[global_num_agent-1].path.empty() && add_agents[global_num_agent-1].path.back().location == add_agents[global_num_agent-1].path_planner->goal_location);
    int path_ln=(int)add_agents[global_num_agent-1].path.size();
    return path_ln;

}







