#include "dynamic_state.h"

Dystate::Dystate(int global_num_agent, int row, int column, const vector<vector<pair<int, int>>> &paths) :
        row(row), column(column) {
    max_lens = 0;
    for (const auto &path: paths)
        max_lens = std::max(max_lens, static_cast<int>(path.size()));

    state.resize(max_lens, MatrixXi::Zero(row, column));
    util_map_action.resize(max_lens, vector<MatrixXd>(5, MatrixXd::Zero(row, column)));

    for (int a = 0; a < global_num_agent; ++a) {
        int max_len = (int) paths[a].size();
        for (int t = 0; t < max_len; ++t) {
            auto &poss = paths[a][t];
            state[t](poss.first, poss.second) += 1;

            if (t > 0) {
                auto &prev_poss = paths[a][t - 1];
                int action = actionDict[{poss.first - prev_poss.first, poss.second - prev_poss.second}];
                util_map_action[t][action](poss.first, poss.second) += 1.0;
            }
        }

        if (max_len < max_lens) {
            auto &poss = paths[a].back();
            for (int t = max_len; t < max_lens; ++t) {
                state[t](poss.first, poss.second) += 1;
                util_map_action[t][0](poss.first, poss.second) += 1.0;
            }
        }
    }
}

Dystate::~Dystate() {}

void Dystate::reset(int new_max_len, vector<int> new_agents, std::map<int, vector<pair<int, int>>> &path_new_agent,
                    vector<int> &prev_agents,
                    std::map<int, vector<pair<int, int>>> &old_path) {

    int diff = max_lens - new_max_len;
    max_lens = new_max_len;

    if (diff > 0) {
        state.resize(max_lens);
        util_map_action.resize(max_lens);
    } else if (diff < 0) {
        for (int i = 0; i < -diff; ++i) {
            // Adding new layers to util_map_action
            util_map_action.emplace_back(5, MatrixXd::Zero(row, column));
            util_map_action.back()[0] += state.back().cast<double>();

            MatrixXi lastState = state.back();
            state.push_back(lastState);
        }
    }
    std::set<int> delete_agent, add_agent;

    if (prev_agents[0]!=-1) {
        std::sort(new_agents.begin(), new_agents.end());
        std::sort(prev_agents.begin(), prev_agents.end());
        std::set_difference(new_agents.begin(), new_agents.end(), prev_agents.begin(), prev_agents.end(),
                            std::inserter(delete_agent, delete_agent.end()));
        std::set_difference(prev_agents.begin(), prev_agents.end(), new_agents.begin(), new_agents.end(),
                            std::inserter(add_agent, add_agent.end()));
    } else {
        delete_agent = std::set<int>(new_agents.begin(), new_agents.end());
    }

    pair<int,int> poss;
    pair<int,int> prev_poss;
    // Delete agents
    for (int i: delete_agent) {
        int path_len = (int) path_new_agent[i].size();
        for (int t = 0; t < max_lens; ++t) {
            if (path_len <= t){
                poss = path_new_agent[i].back();
            }
            else {
                poss = path_new_agent[i][t];
            }
            state[t](poss.first, poss.second) -= 1;
            if (t > 0) {
                if (path_len <= t-1){
                    prev_poss = path_new_agent[i].back();
                }
                else{
                    prev_poss = path_new_agent[i][t-1];
                }
                int action = actionDict[{poss.first - prev_poss.first, poss.second - prev_poss.second}];
                util_map_action[t][action](poss.first, poss.second) -= 1.0;  //poss(2,9),action:0, should be the problem of t=27
            }
        }
    }
    // Add agents
    for (int i: add_agent) {
        int path_len = (int) old_path[i].size();
        for (int t = 0; t < max_lens; ++t) {
            if (path_len <= t){
                poss = old_path[i].back();
            }
            else {
                poss = old_path[i][t];
            }
            state[t](poss.first, poss.second) += 1;

            if (t > 0) {
                if (path_len <= t-1){
                    prev_poss = old_path[i].back();
                }
                else{
                    prev_poss = old_path[i][t-1];
                }
                int action = actionDict[{poss.first - prev_poss.first, poss.second - prev_poss.second}];
                util_map_action[t][action](poss.first, poss.second) += 1.0;
            }
        }
    }
}


