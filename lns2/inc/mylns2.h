#pragma once
#include "common.h"
#include "Instance.h"
#include "BasicLNS.h"
#include "string"

enum init_destroy_heuristic { TARGET_BASED, COLLISION_BASED, RANDOM_BASED, INIT_COUNT };

class MyLns2 {
public:
    MyLns2(int seed, vector<vector<int>> obs_map,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,int all_ag_num,int map_size);
    int makespan=0;
    void init_pp();
    vector<vector<pair<int,int>>> vector_path;
    vector<vector<pair<int,int>>> sipps_path;
    int calculate_sipps(vector<int> new_agents);
    int single_sipp(vector<vector<pair<int,int>>> dy_obs_path,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,
                    pair<int,int> self_start_poss,pair<int,int> self_goal_poss,int global_num_agent);
private:
    Neighbor neighbor;
    const Instance instance;
    vector<Agent> agents;
    PathTableWC path_table;
    bool updateCollidingPairs(set<pair<int, int>>& colliding_pairs,int agent_id, const Path& path);
};
