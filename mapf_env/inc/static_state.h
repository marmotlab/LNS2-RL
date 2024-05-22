#pragma once

#include "common.h"
#include <Eigen/Dense>

class State {

public:
    State(int global_num_agent, MatrixXi fix_state, map<pair<int, int>, vector<int>> fix_state_dict);

    ~State();

    void reset(MatrixXi fix_state, map<pair<int, int>, vector<int>> fix_state_dict, const vector<int> &local_agents,
               const vector<pair<int, int>> &agents_goals);

    void get_heuri_map(const vector<int> &local_agents, const vector<pair<int, int>> &agents_goals);

    MatrixXi state;
    map<pair<int, int>, vector<int>> state_dict;
    vector<vector<MatrixXd>> heuri_map;
private:
    vector<pair<int, int>> directions = {{-1, 0},
                                         {1,  0},
                                         {0,  -1},
                                         {0,  1}};

};