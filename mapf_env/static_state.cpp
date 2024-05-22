#include "static_state.h"

#include <utility>

State::State(int global_num_agent, MatrixXi fix_state, map<pair<int, int>, vector<int>> fix_state_dict) :
        state(std::move(fix_state)), state_dict(std::move(fix_state_dict)), heuri_map(global_num_agent) {}

State::~State() = default;

void State::reset(MatrixXi fix_state, map<pair<int, int>, vector<int>> fix_state_dict, const vector<int> &local_agents,
                  const vector<pair<int, int>> &agents_goals) {
    state = std::move(fix_state);
    state_dict = std::move(fix_state_dict);
    get_heuri_map(local_agents, agents_goals);
}


void State::get_heuri_map(const vector<int> &local_agents, const vector<pair<int, int>> &agents_goals) {

    for (int a: local_agents) {
        if (heuri_map[a].empty()) {
            MatrixXi dist_map = MatrixXi::Constant(state.rows(), state.cols(), std::numeric_limits<int>::max());
            deque<pair<int, int>> open_list;
            vector<vector<bool>> in_open_list(state.rows(), vector<bool>(state.cols(), false));

            auto goal = agents_goals[a];
            open_list.emplace_back(goal);
            in_open_list[goal.first][goal.second] = true;
            dist_map(goal.first, goal.second) = 0;

            while (!open_list.empty()) {
                auto [x, y] = open_list.front();
                open_list.pop_front();
                in_open_list[x][y] = false;
                int dist = dist_map(x, y);

                for (auto &[dx, dy]: directions) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < state.rows() && ny >= 0 && ny < state.cols() && state(nx, ny) != -1 &&
                        dist_map(nx, ny) > dist + 1) {
                        dist_map(nx, ny) = dist + 1;
                        if (!in_open_list[nx][ny]) {
                            open_list.emplace_back(nx, ny);
                            in_open_list[nx][ny] = true;
                        }
                    }
                }
            }

            vector<MatrixXd> heuristic_maps(4, MatrixXd::Zero(state.rows(), state.cols()));

            for (int x = 0; x < state.rows(); ++x) {
                for (int y = 0; y < state.cols(); ++y) {
                    if (state(x, y) != -1) {
                        if (x > 0 && dist_map(x - 1, y) < dist_map(x, y))
                            heuristic_maps[0](x, y) = 1.0;
                        if (x < state.rows() - 1 && dist_map(x + 1, y) < dist_map(x, y))
                            heuristic_maps[1](x, y) = 1.0;
                        if (y > 0 && dist_map(x, y - 1) < dist_map(x, y))
                            heuristic_maps[2](x, y) = 1.0;
                        if (y < state.cols() - 1 && dist_map(x, y + 1) < dist_map(x, y))
                            heuristic_maps[3](x, y) = 1.0;
                    }
                }
            }
            heuri_map[a] = heuristic_maps;
        }
    }
}