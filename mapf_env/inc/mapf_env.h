#pragma once

#include "common.h"
#include "dynamic_state.h"
#include "static_state.h"


class MapfEnv {
public:
    MapfEnv(int seed, int global_num_agent, int fov_size, int row, int column, double episode_len, int k_steps,
            int num_time_slice,
            int window_size, pair<int, int> uti_window, pair<double, double> dis_time_weight,
            vector<vector<pair<int, int>>> paths,
            MatrixXi fix_state, std::map<pair<int, int>, vector<int>> fix_state_dict,
            MatrixXi obstacle_map, vector<pair<int, int>> start_list,
            vector<pair<int, int>> goal_list);

    ~MapfEnv();

    void joint_move(const vector<int> &actions);

    void observe(vector<int> actions);

    void predict_next();

    void update_ulti();

    bool joint_step(vector<int> actions);

    void local_reset(int makespan, int old_coll_pair_num, double sipp_coll_pair_num,
                     vector<vector<pair<int, int>>> sipps_path,
                     std::map<int, vector<pair<int, int>>> path_new_agent, vector<vector<pair<int, int>>> paths,
                     std::map<int, vector<pair<int, int>>> prev_path, vector<int> prev_agents,
                     vector<int> local_agents);

    void replan_part1();

    bool replan_part2(double rl_max_len, vector<vector<pair<int, int>>> new_path, set<pair<int, int>> new_coll_pair);

    void replan_part3(vector<vector<pair<int, int>>> new_path, set<pair<int, int>> new_coll_pair);

    void next_valid_actions();

    vector<vector<pair<int, int>>> paths;
    State world;
    Dystate dynamic_state;
    bool rupt = false;
    int timestep = 0;
    set<pair<int, int>> new_collision_pairs;
    vector<vector<pair<int, int>>> local_path;
    int makespan;
    int old_coll_pair_num;
    vector<vector<pair<int, int>>> sipps_path;
    double sipp_coll_pair_num;
    double sipps_max_len;
    int local_num_agents;
    deque<vector<MatrixXd>> agent_util_map_action;
    deque<MatrixXd> agent_util_map_vertex;
    vector<pair<int, int> *> local_agents_poss;
    vector<pair<int, int>> agents_poss;
    vector<int> local_agents;
    vector<vector<pair<int, int>>> all_next_poss;
    vector<MatrixXd> space_ulti_action;
    MatrixXd space_ulti_vertex;
    vector<vector<pair<int, int>>> old_path;
    vector<int> replan_ag;
    vector<vector<int>> valid_actions;
    vector<vector<MatrixXd>> all_obs;
    MatrixXd all_vector;
    MatrixXi fix_state;
    std::map<pair<int, int>, vector<int>> fix_state_dict;
    vector<pair<int, int>> start_list;
    vector<pair<int, int>> goal_list;
    MatrixXi obstacle_map;
    int num_on_goal=0;

private:
    std::unordered_map<int, pair<int, int>> dirDict = {{0, {0,  0}},
                                                       {1, {0,  1}},
                                                       {2, {1,  0}},
                                                       {3, {0,  -1}},
                                                       {4, {-1, 0}},
                                                       {5, {1,  1}},
                                                       {6, {1,  -1}},
                                                       {7, {-1, -1}},
                                                       {8, {-1, 1}}};
    const int global_num_agent;
    const int fov_size;
    const int row;
    const int column;
    const double episode_len;
    pair<double, double> dis_time_weight;
    const int k_steps;
    const int num_time_slice;
    const int window_size;
    pair<int, int> uti_window;
};
