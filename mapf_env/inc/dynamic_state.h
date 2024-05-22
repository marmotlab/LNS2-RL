#pragma once

#include "common.h"
#include <Eigen/Dense>

class Dystate {
public:
    Dystate(int global_num_agent, int row, int column, const vector<vector<pair<int, int>>> &paths);

    ~Dystate();

    void reset(int new_max_len, vector<int> new_agents, std::map<int, vector<pair<int, int>>> &path_new_agent,
               vector<int> &prev_agents,
               std::map<int, vector<pair<int, int>>> &old_path);

    int max_lens;
    int row;
    int column;
    vector<vector<MatrixXd>> util_map_action;
    vector<MatrixXi> state;

private:
    unordered_map<pair<int, int>, int> actionDict = {{{0,  0},  0},
                                                     {{0,  1},  1},
                                                     {{1,  0},  2},
                                                     {{0,  -1}, 3},
                                                     {{-1, 0},  4},
                                                     {{1,  1},  5},
                                                     {{1,  -1}, 6},
                                                     {{-1, -1}, 7},
                                                     {{-1, 1},  8}};
};




