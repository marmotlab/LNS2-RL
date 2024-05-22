#include "mapf_env.h"

#include <utility>

MapfEnv::MapfEnv(int seed, int global_num_agent, int fov_size, int row, int column, double episode_len, int k_steps,
                 int num_time_slice,
                 int window_size, pair<int, int> uti_window, pair<double, double> dis_time_weight,
                 vector<vector<pair<int, int>>> paths,
                 MatrixXi fix_state, std::map<pair<int, int>, vector<int>> fix_state_dict,
                 MatrixXi obstacle_map, vector<pair<int, int>> start_list,
                 vector<pair<int, int>> goal_list) :
        paths(std::move(paths)), global_num_agent(global_num_agent), fov_size(fov_size), row(row), column(column),
        episode_len(episode_len),
        fix_state(fix_state), fix_state_dict(fix_state_dict), start_list(std::move(start_list)), dis_time_weight(std::move(dis_time_weight)),
        uti_window(std::move(uti_window)),
        k_steps(k_steps), num_time_slice(num_time_slice), window_size(window_size), obstacle_map(obstacle_map),
        goal_list(std::move(goal_list)), world(global_num_agent, fix_state, fix_state_dict),
        dynamic_state(global_num_agent, row, column, this->paths) {
    std::srand(seed);
    old_path.reserve(global_num_agent);
    space_ulti_action.reserve(5);
    space_ulti_vertex = MatrixXd::Zero(row, column);
    for (int i = 0; i < 5; ++i)
        space_ulti_action.emplace_back(MatrixXd::Zero(row, column));
    for (int i = 0; i < 2; ++i) {
        agent_util_map_action.emplace_back(5, MatrixXd::Zero(row, column));
        agent_util_map_vertex.emplace_back(MatrixXd::Zero(row, column));
    }
}

MapfEnv::~MapfEnv() = default;


void MapfEnv::local_reset(int makespan, int old_coll_pair_num, double sipp_coll_pair_num,
                          vector<vector<pair<int, int>>> sipps_path,
                          std::map<int, vector<pair<int, int>>> path_new_agent, vector<vector<pair<int, int>>> paths,
                          std::map<int, vector<pair<int, int>>> prev_path, vector<int> prev_agents,
                          vector<int> local_agents) {
    timestep = 0;
    num_on_goal = 0;
    rupt = false;
    this->makespan = makespan;
    this->old_coll_pair_num = old_coll_pair_num;
    this->sipps_path = std::move(sipps_path);
    this->sipp_coll_pair_num = sipp_coll_pair_num;
    this->paths = std::move(paths);
    this->local_agents = std::move(local_agents);
    local_num_agents = (int) this->local_agents.size();
    sipps_max_len = (double) std::max_element(this->sipps_path.begin(), this->sipps_path.end(),
                                              [](const vector<pair<int, int>> &a, const vector<pair<int, int>> &b) {
                                                  return a.size() < b.size();
                                              })->size();
    local_agents_poss.clear();
    local_agents_poss.reserve(local_num_agents);
    agents_poss = start_list;
    for (int i: this->local_agents)
        local_agents_poss.push_back(&agents_poss[i]);
    all_next_poss.clear();
    all_next_poss.reserve(local_num_agents);
    for (auto &actionMaps: agent_util_map_action) {
        for (auto &mat: actionMaps) {
            mat.setZero();
        }
    }
    for (auto &mat: agent_util_map_vertex)
        mat.setZero();
    for (int i = 0; i < local_num_agents; ++i)
        agent_util_map_vertex.back()(local_agents_poss[i]->first, local_agents_poss[i]->second) += 1.0;
    world.reset(fix_state, fix_state_dict, this->local_agents, goal_list);
    dynamic_state.reset(makespan + 1, this->local_agents, path_new_agent, prev_agents, prev_path);
    update_ulti();
    predict_next();
    new_collision_pairs.clear();
    valid_actions.reserve(local_num_agents);
    all_obs.clear();
    all_obs.reserve(local_num_agents);
    for (int i = 0; i < local_num_agents; ++i) {
        vector<MatrixXd> obs(31, MatrixXd::Zero(fov_size, fov_size));
        all_obs.push_back(obs);
    }
    all_vector = MatrixXd::Zero(local_num_agents, 8);
    local_path.clear();
    local_path.reserve(local_num_agents);
    for (int i = 0; i < local_num_agents; ++i)
        local_path.push_back({*local_agents_poss[i]});
}

void MapfEnv::update_ulti() {
    for (auto &mat: space_ulti_action)
        mat.setZero();
    space_ulti_vertex.setZero();
    for (int t = uti_window.first; t < uti_window.second; ++t) {
        if (timestep + t + 1 < 0)  // Skip if time_step + t + 1 is negative
            continue;

        if (t < 0) {
            // Assuming agent_util_map_action and agent_util_map_vertex are accessible here
            for (int i = 0; i < 5; ++i)
                space_ulti_action[i] += agent_util_map_action[t + 2][i];
            space_ulti_vertex += agent_util_map_vertex[t + 2];
        }

        if (timestep + t + 1 >= dynamic_state.max_lens) {
            space_ulti_action[0] += dynamic_state.state.back().cast<double>(); // Assuming state is a deque or vector of Eigen matrices
            space_ulti_vertex += dynamic_state.state.back().cast<double>();
        } else {
            for (int i = 0; i < 5; ++i)
                space_ulti_action[i] += dynamic_state.util_map_action[timestep + t + 1][i];
            space_ulti_vertex += dynamic_state.state[timestep + t + 1].cast<double>();
        }
    }
    double scale_factor = 10.0 / static_cast<double>(global_num_agent);
    space_ulti_vertex *= scale_factor;
    for (auto &mat: space_ulti_action)
        mat *= scale_factor;
}

void MapfEnv::predict_next() {
    all_next_poss.clear();
    vector<double> final_distances;
    if (timestep != 0) {
        for (int local_agent_index = 0; local_agent_index < local_num_agents; ++local_agent_index) {
            vector<pair<int, int>> next_poss_list;
            next_poss_list.reserve(k_steps);
            final_distances.clear();
            int t = 0;
            for (const auto &pos: sipps_path[local_agent_index]) {
                double dis_x = local_agents_poss[local_agent_index]->first - pos.first;
                double dis_y = local_agents_poss[local_agent_index]->second - pos.second;
                double dis = std::sqrt(dis_x * dis_x + dis_y * dis_y);
                int time_dis = std::abs(timestep - t);
                double final_dis = dis * dis_time_weight.first + static_cast<double>(time_dis) * dis_time_weight.second;
                final_distances.push_back(final_dis);
                t++;
            }
            auto min_it = std::min_element(final_distances.begin(), final_distances.end());
            int poss_index = std::distance(final_distances.begin(), min_it);
            for (int k = 0; k < k_steps; ++k) {
                std::pair<int, int> next_poss;
                if (k == 0) {
                    next_poss = (poss_index + 1 < sipps_path[local_agent_index].size()) ?
                                sipps_path[local_agent_index][poss_index + 1] :
                                sipps_path[local_agent_index].back();
                    double pre_dis_x = next_poss.first - local_agents_poss[local_agent_index]->first;
                    double pre_dis_y = next_poss.second - local_agents_poss[local_agent_index]->second;
                    double pre_dis = pre_dis_x * pre_dis_x + pre_dis_y * pre_dis_y;
                    if (pre_dis > 1)
                        next_poss = *local_agents_poss[local_agent_index];
                } else {
                    next_poss = (poss_index + k + 1 < sipps_path[local_agent_index].size()) ?
                                sipps_path[local_agent_index][poss_index + k + 1] :
                                sipps_path[local_agent_index].back();
                }
                next_poss_list.push_back(next_poss);
            }
            all_next_poss.push_back(next_poss_list);
        }
    } else {
        for (int local_agent_index = 0; local_agent_index < local_num_agents; ++local_agent_index) {
            vector<pair<int, int>> next_poss_list;
            next_poss_list.reserve(k_steps);
            for (int k = 0; k < k_steps; ++k) {
                pair<int, int> next_poss = (k + 1 < sipps_path[local_agent_index].size()) ?
                                           sipps_path[local_agent_index][k + 1] :
                                           sipps_path[local_agent_index].back();
                next_poss_list.push_back(next_poss);
            }

            all_next_poss.push_back(next_poss_list);
        }
    }
}


void MapfEnv::observe( vector<int> actions) {
    for (auto &Maps: all_obs) {
        for (auto &mat: Maps)
            mat.setZero();
    }
    all_vector.setZero();
    vector<int> visible_agents;
    vector<pair<int, int>> window_path;
    window_path.reserve(2 * window_size);
    auto d_timestep = static_cast<double>(timestep);

    for (int local_agent_index = 0; local_agent_index < local_num_agents; ++local_agent_index) {
        visible_agents.clear();
        window_path.clear();
        int agent_index = local_agents[local_agent_index];
        int agent_x = local_agents_poss[local_agent_index]->first;
        int agent_y = local_agents_poss[local_agent_index]->second;

        int top_poss = std::max(agent_x - fov_size / 2, 0);
        int bottom_poss = std::min(agent_x + fov_size / 2 + 1, row);
        int left_poss = std::max(agent_y - fov_size / 2, 0);
        int right_poss = std::min(agent_y + fov_size / 2 + 1, column);
        pair<int, int> top_left = make_pair(agent_x - fov_size / 2, agent_y - fov_size / 2);
        int FOV_top = std::max(fov_size / 2 - agent_x, 0);
        int FOV_left = std::max(fov_size / 2 - agent_y, 0);
        int hight = bottom_poss - top_poss;
        int width = right_poss - left_poss;

        int min_time = 0;
        int max_time = 0;
        auto path_length = (int) sipps_path[local_agent_index].size();

        if (timestep - window_size < 0) {
            min_time = 0;
        } else if (timestep >= path_length) {
            min_time = std::max(0, path_length - window_size);
        } else {
            min_time = timestep - window_size;
        }

        max_time = std::min(timestep + window_size, path_length);
        window_path.insert(window_path.end(), sipps_path[local_agent_index].begin() + min_time,
                           sipps_path[local_agent_index].begin() + max_time);

        auto &[goal_x, goal_y] = goal_list[agent_index];
        if (goal_x >= top_poss && goal_x < bottom_poss && goal_y >= left_poss && goal_y < right_poss)
            all_obs[local_agent_index][10](goal_x - top_left.first, goal_y - top_left.second) = 1.0;  // goal map
        all_obs[local_agent_index][9](agent_x - top_left.first, agent_y - top_left.second) = 1.0;  // local poss map

        for (int i = 0; i < 4; ++i) // guide map
            all_obs[local_agent_index][13 + i].block(FOV_top, FOV_left, hight,
                                                     width) = world.heuri_map[agent_index][i].block(top_poss, left_poss,
                                                                                                    hight, width);
        all_obs[local_agent_index][20].block(FOV_top, FOV_left, hight, width) = space_ulti_vertex.block(top_poss,
                                                                                                        left_poss,
                                                                                                        hight,
                                                                                                        width); // util
        for (int i = 0; i < 5; ++i) // util action
            all_obs[local_agent_index][21 + i].block(FOV_top, FOV_left, hight, width) = space_ulti_action[i].block(
                    top_poss, left_poss, hight, width);

        for (int i = top_left.first; i < top_left.first + fov_size; ++i) {
            for (int j = top_left.second; j < top_left.second + fov_size; ++j) {
                int map_i = i - top_left.first;
                int map_j = j - top_left.second;
                if (i >= row || i < 0 || j >= column || j < 0) {
                    all_obs[local_agent_index][19](map_i, map_j) = 1.0 - d_timestep/episode_len;  //occupy map 19
                    all_obs[local_agent_index][12](map_i, map_j) = 1.0;  //obs map
                    continue;
                }
                if (world.state(i, j) == -1) {
                    all_obs[local_agent_index][19](map_i, map_j) = 1.0 - d_timestep/episode_len;   //occupy map 19
                    all_obs[local_agent_index][12](map_i, map_j) = 1.0;  //obs map
                    continue;
                }
                if (std::find(window_path.begin(), window_path.end(), make_pair(i, j)) != window_path.end())
                    all_obs[local_agent_index][17](map_i, map_j) = 1.0; //sipps map

                for (int k = 0; k < k_steps; ++k) {
                    for (int iter_a = 0; iter_a < local_num_agents; ++iter_a) {
                        if (iter_a != local_agent_index && i == all_next_poss[iter_a][k].first &&
                            j == all_next_poss[iter_a][k].second)
                            all_obs[local_agent_index][26 + k](map_i, map_j) += 1.0;  //next step map
                    }
                    if (all_obs[local_agent_index][26 + k](map_i, map_j) != 0.0)
                        all_obs[local_agent_index][26 + k](map_i, map_j) =
                                0.5 + 0.5 * std::tanh((all_obs[local_agent_index][26 + k](map_i, map_j) - 1) / 3);
                }

                for (auto item: world.state_dict[{i, j}]) {
                    if (item != agent_index &&
                        std::find(local_agents.begin(), local_agents.end(), item) != local_agents.end()) {
                        visible_agents.push_back(item);
                        all_obs[local_agent_index][9](map_i, map_j) += 1.0; //local poss map
                    }
                }
                if (all_obs[local_agent_index][9](map_i, map_j) != 0.0)
                    all_obs[local_agent_index][9](map_i, map_j) = 0.5 + 0.5 * std::tanh(
                            (all_obs[local_agent_index][9](map_i, map_j) - 1) / 3); //local poss map

                for (int t = 0; t < num_time_slice; ++t) {
                    if (timestep + t < dynamic_state.max_lens) {
                        all_obs[local_agent_index][t](map_i, map_j) = (double)dynamic_state.state[timestep + t](i, j) ; //dynamic poss map
                    } else {
                        all_obs[local_agent_index][t](map_i, map_j) =(double)dynamic_state.state.back()(i, j);
                    }
                    if (all_obs[local_agent_index][t](map_i, map_j) != 0.0)
                        all_obs[local_agent_index][t](map_i, map_j) = 0.5 + 0.5 * std::tanh(
                                (all_obs[local_agent_index][t](map_i, map_j) - 1) / 3); //local poss map
                }

                if (timestep >= makespan) {
                    if (dynamic_state.state.back()(i, j) > 0) {
                        all_obs[local_agent_index][19](map_i, map_j) = 1.0 - d_timestep/episode_len; //occupyed
                    } else {
                        all_obs[local_agent_index][18](map_i, map_j) = 1.0 - (d_timestep + 1.0)/episode_len; //blank
                    }
                } else {
                    double occupy_t = 0.0;
                    if (dynamic_state.state[timestep](i, j) > 0) {
                        for (int t = timestep; t <= episode_len; ++t) {
                            if (t >= makespan) {
                                if (dynamic_state.state.back()(i, j) > 0)
                                    occupy_t = episode_len - d_timestep;
                                break;
                            }
                            if (dynamic_state.state[t](i, j) > 0) {
                                occupy_t += 1.0;
                            } else {
                                break;
                            }
                        }
                    }
                    all_obs[local_agent_index][19](map_i, map_j) = occupy_t/episode_len; //occupy

                    double blank_t = 0.0;
                    for (int t = timestep + 1; t <= episode_len; ++t) {
                        if (t >= makespan) {
                            if (dynamic_state.state.back()(i, j) == 0)
                                blank_t = episode_len - d_timestep - 1.0;
                            break;
                        }
                        if (dynamic_state.state[t](i, j) == 0) {
                            blank_t += 1.0;
                        } else {
                            break;
                        }
                    }
                    all_obs[local_agent_index][18](map_i, map_j) = blank_t/episode_len;  //blank
                }
            }
        }

        for (const auto &visible_index: visible_agents) {
            int min_x = std::max(top_left.first,
                                 std::min(top_left.first + fov_size - 1, goal_list[visible_index].first));
            int min_y = std::max(top_left.second,
                                 std::min(top_left.second + fov_size - 1, goal_list[visible_index].second));
            all_obs[local_agent_index][11](min_x - top_left.first, min_y - top_left.second) = 1.0; //local goal map
        }

        double dx = goal_list[agent_index].first - agent_x;
        double dy = goal_list[agent_index].second - agent_y;
        double mag = std::sqrt(dx * dx + dy * dy);

        if (mag != 0) {
            dx /= mag;
            dy /= mag;
        }
        mag = std::min(mag, 50.0);

        all_vector(local_agent_index, 0) = dx;
        all_vector(local_agent_index, 1) = dy;
        all_vector(local_agent_index, 2) = mag;
        all_vector(local_agent_index, 7) = (double) actions[local_agent_index];
    }
    double coll_ratio =
            (sipp_coll_pair_num - static_cast<double>(new_collision_pairs.size())) / (sipp_coll_pair_num + 1.0);
    double time_step_episode_ratio = d_timestep / episode_len;
    double time_step_sipps_ratio = d_timestep / sipps_max_len;
    double goal_ratio = static_cast<double>(num_on_goal) / static_cast<double>(local_num_agents);

    all_vector.col(3) = VectorXd::Constant(local_num_agents, coll_ratio);
    all_vector.col(4) = VectorXd::Constant(local_num_agents, time_step_episode_ratio);
    all_vector.col(5) = VectorXd::Constant(local_num_agents, time_step_sipps_ratio);
    all_vector.col(6) = VectorXd::Constant(local_num_agents, goal_ratio);
}

void MapfEnv::joint_move(const vector<int> &actions) {
    num_on_goal = 0;
    world.state = (timestep < dynamic_state.max_lens) ? dynamic_state.state[timestep] + obstacle_map :
                  dynamic_state.state.back() + obstacle_map;

    for (int i = 0; i < global_num_agent; ++i) {
        if (std::find(local_agents.begin(), local_agents.end(), i) == local_agents.end()) {
            int max_len = (int) paths[i].size();
            if (max_len <= timestep) continue;
            agents_poss[i] = paths[i][timestep];
            auto it = std::find(world.state_dict[paths[i][timestep - 1]].begin(),
                                world.state_dict[paths[i][timestep - 1]].end(), i);
            world.state_dict[paths[i][timestep - 1]].erase(it);
            world.state_dict[paths[i][timestep]].push_back(i);
        }
    }
    vector<pair<int, int>> local_past_position(local_agents_poss.size());
    std::transform(local_agents_poss.begin(), local_agents_poss.end(), local_past_position.begin(),
                   [](const auto &ptr) { return *ptr; });
    agent_util_map_action.pop_front();
    agent_util_map_vertex.pop_front();
    agent_util_map_action.emplace_back(5, MatrixXd::Zero(row, column));
    agent_util_map_vertex.emplace_back(MatrixXd::Zero(row, column));

    for (int local_i = 0; local_i < local_num_agents; ++local_i) {
        int i = local_agents[local_i];
        auto direction = dirDict[actions[local_i]];
        auto [ax, ay] = agents_poss[i];
        int new_ax = ax + direction.first;
        int new_ay = ay + direction.second;

        agents_poss[i] = make_pair(new_ax, new_ay);
        world.state(new_ax, new_ay) += 1;

        auto &vec = world.state_dict[make_pair(ax, ay)];
        auto it = std::find(vec.begin(), vec.end(), i);
        vec.erase(it);
        world.state_dict[make_pair(new_ax, new_ay)].push_back(i);
        agent_util_map_action.back()[actions[local_i]](new_ax, new_ay) += 1.0;
        agent_util_map_vertex.back()(new_ax, new_ay) += 1.0;
    }

    for (int local_i = 0; local_i < local_num_agents; ++local_i) {
        int i = local_agents[local_i];
        // Checking for vertex collision with dynamic obstacles
        if (world.state(agents_poss[i].first, agents_poss[i].second) > 1) {
            auto &collide_agents_id = world.state_dict[agents_poss[i]];
            for (int j: collide_agents_id) {
                if (j != i)
                    new_collision_pairs.insert({std::min(j, i), std::max(j, i)});
            }
        }

        auto &collide_agent_id = world.state_dict[local_past_position[local_i]];
        for (int j: collide_agent_id) {
            if (j != i) {
                auto it = std::find(local_agents.begin(), local_agents.end(), j);
                if (it != local_agents.end()) {
                    size_t local_j = std::distance(local_agents.begin(), it);
                    auto &past_poss = local_past_position[local_j];
                    if (past_poss == agents_poss[i] && agents_poss[j] != past_poss)
                        new_collision_pairs.insert({std::min(j, i), std::max(j, i)});
                } else {
                    size_t max_len = paths[j].size();
                    if (max_len <= timestep) continue;
                    auto &past_poss = paths[j][timestep - 1];
                    if (past_poss == agents_poss[i] && past_poss != paths[j][timestep])
                        new_collision_pairs.insert({std::min(j, i), std::max(j, i)});
                }
            }
        }

        if (agents_poss[i] == goal_list[i])
            num_on_goal++;
    }
}

bool MapfEnv::joint_step(vector<int> actions) {
    timestep++;
    joint_move(actions);
    if (new_collision_pairs.size() > old_coll_pair_num) {
        rupt = true;
        return true;
    }
    for (int i = 0; i < local_num_agents; ++i)
        local_path[i].push_back(*local_agents_poss[i]);

    bool all_reach_goal = (num_on_goal == local_num_agents);
    if (!all_reach_goal) return false;

    if (timestep < makespan) {
        for (int i: local_agents) {
            for (size_t j = timestep + 1; j < dynamic_state.state.size(); ++j) {
                if (dynamic_state.state[j](goal_list[i].first, goal_list[i].second) > 0)
                    return false;
            }
        }
    }
    return true;
}

void MapfEnv::replan_part1() {
    old_path.clear();
    for (const auto &path: paths) {
        if (timestep < path.size()) {
            old_path.emplace_back(path.begin() + timestep, path.end());
        } else {
            old_path.emplace_back(vector<pair<int, int>>{path.back()});
        }
    }
}

bool
MapfEnv::replan_part2(double rl_max_len, vector<vector<pair<int, int>>> new_path, set<pair<int, int>> new_coll_pair) {
    replan_ag.clear();
    for (size_t local_i = 0; local_i < new_path.size(); ++local_i) {
        local_path[local_i].insert(local_path[local_i].end(), new_path[local_i].begin() + 1, new_path[local_i].end());
        if (local_path[local_i].size() >= rl_max_len)
            replan_ag.push_back(local_agents[local_i]);
    }
    new_collision_pairs.insert(new_coll_pair.begin(), new_coll_pair.end());
    if (replan_ag.empty())
        return false;
    for (auto it = new_collision_pairs.begin(); it != new_collision_pairs.end();) {
        if (std::find(replan_ag.begin(), replan_ag.end(), it->first) != replan_ag.end() ||
            std::find(replan_ag.begin(), replan_ag.end(), it->second) != replan_ag.end()) {
            it = new_collision_pairs.erase(it); // 直接在遍历时移除，避免收集后再删除
        } else {
            ++it;
        }
    }
    return true;
}

void MapfEnv::replan_part3(vector<vector<pair<int, int>>> new_path, set<pair<int, int>> new_coll_pair) {
    new_collision_pairs.insert(new_coll_pair.begin(), new_coll_pair.end());

    for (size_t local_i = 0; local_i < replan_ag.size(); ++local_i) {
        int i = replan_ag[local_i];
        auto it = std::find(local_agents.begin(), local_agents.end(), i);
        size_t local_i_2 = std::distance(local_agents.begin(), it);
        local_path[local_i_2] = new_path[local_i];
    }
}

void MapfEnv::next_valid_actions() {
    valid_actions.clear();
    for (int local_agent_index = 0; local_agent_index < local_num_agents; ++local_agent_index) {
        vector<int> available_actions;
        available_actions.reserve(5);
        available_actions.push_back(0);

        auto &[ax, ay] = *local_agents_poss[local_agent_index];

        for (int action = 1; action <= 4; ++action) { // Actions except 0
            auto &[dx, dy] = dirDict[action];
            int new_ax = ax + dx;
            int new_ay = ay + dy;
            if (new_ax >= row || new_ax < 0 || new_ay >= column || new_ay < 0 ||
                world.state(new_ax, new_ay) < 0)
                continue;
            available_actions.push_back(action);
        }
        valid_actions.push_back(available_actions);
    }
}
