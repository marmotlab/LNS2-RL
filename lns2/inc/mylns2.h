#pragma once

#include "config.h"
#include "common.h"
#include "Instance.h"
#include "BasicLNS.h"
#include "string"
#include <memory>
#include <chrono>
#include <thread>

// Include Rerun configuration
#include "rerun_config.h"

enum init_destroy_heuristic { TARGET_BASED, COLLISION_BASED, RANDOM_BASED, INIT_COUNT };

class MyLns2 {
public:
    // Constructor with dynamic map sequence
    MyLns2(int seed, vector<vector<vector<int>>> dynamic_map_seq, vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss, int all_ag_num, int map_size);

    // Legacy constructor with static map only
    MyLns2(int seed, vector<vector<int>> obs_map, vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss, int all_ag_num, int map_size);
    ~MyLns2();
    int makespan=0;
    void init_pp();
    vector<vector<pair<int,int>>> vector_path;
    vector<vector<pair<int,int>>> sipps_path;
    int calculate_sipps(vector<int> new_agents);
    int single_sipp(vector<vector<pair<int,int>>> dy_obs_path,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,
                    pair<int,int> self_start_poss,pair<int,int> self_goal_poss,int global_num_agent);

#if RERUN_ENABLED
    std::shared_ptr<rerun::RecordingStream> rec;
#endif
    bool use_rerun = true;  // Default value, can be overridden at runtime
    string log_prefix;
    bool map_logged = false;

    // Set whether to use Rerun visualization
    void setUseRerun(bool enable) {
        use_rerun = enable;
#if RERUN_ENABLED
        // Also update all agents' path planners
        for (auto& agent : agents) {
            if (agent.path_planner) {
                agent.path_planner->setUseRerun(enable);
            }
        }
#endif
    }

private:
    Neighbor neighbor;
    const Instance instance;
    vector<Agent> agents;
    PathTableWC path_table;
    bool updateCollidingPairs(set<pair<int, int>>& colliding_pairs,int agent_id, const Path& path);

    void publishMapRerun(const std::vector<std::vector<int>>& map, const std::vector<int>& start_locations);

    // Visualize reference paths of previously planned agents
    void visualizeReferencePaths(int current_agent_id);
};