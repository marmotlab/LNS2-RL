#include<boost/tokenizer.hpp>
#include <algorithm>    // std::shuffle
#include"Instance.h"


// Init map instance with static map only
Instance::Instance(const vector<vector<int>>& obs_map, const vector<pair<int,int>>& start_poss, const vector<pair<int,int>>& goal_poss, int num_of_agents, int num_of_rows):
        num_of_agents(num_of_agents), num_of_rows(num_of_rows), num_of_cols(num_of_rows)
{
    std::cout << "[INSTANCE] Initializing with static map and " << num_of_agents << " agents\n";

    loadMap(obs_map);
    loadAgents(start_poss, goal_poss);
}

// Init map instance with dynamic map sequence
Instance::Instance(const vector<vector<vector<int>>>& grid_sequence, const vector<pair<int,int>>& start_poss, const vector<pair<int,int>>& goal_poss, int num_of_agents, int num_of_rows):
        num_of_agents(num_of_agents), num_of_rows(num_of_rows), num_of_cols(num_of_rows), dynamic_map_seq(grid_sequence)
{
    std::cout << "[INSTANCE] Initializing with dynamic map sequence and " << num_of_agents << " agents\n";
    std::cout << "[INSTANCE] Grid sequence size: " << grid_sequence.size()
              << ", frame shape: (" << grid_sequence[0].size() << ", " << grid_sequence[0][0].size() << ")\n";

    // Extract static map from the first frame of the dynamic map sequence
    // In our convention, static obstacles have value 1 in the dynamic map
    vector<vector<int>> static_map(grid_sequence[0].size(), vector<int>(grid_sequence[0][0].size(), 0));

    for (size_t i = 0; i < grid_sequence[0].size(); i++) {
        for (size_t j = 0; j < grid_sequence[0][0].size(); j++) {
            // Static obstacles have value 1 in our convention
            if (grid_sequence[0][i][j] == 1) {
                static_map[i][j] = 1;
            }
        }
    }

    loadMap(static_map);
    loadAgents(start_poss, goal_poss);
}

/* loadMap:
% Inputs:
%   - obs_map: 2D vektor, amelyben a 0 érték szabad cellát, a nem 0 érték akadályt jelöl.
*/
bool Instance::loadMap(const vector<vector<int>>& obs_map)
{
    map_size = num_of_cols * num_of_rows;
    my_map.resize(map_size, false);

    std::cout << "[LOADMAP] Loading static obstacle map...\n";
    int obs_count = 0;
    for (int i = 0; i < num_of_rows; i++) {
        for (int j = 0; j < num_of_cols; j++) {
            bool is_obs = (obs_map[i][j] != 0);
            my_map[linearizeCoordinate(i, j)] = is_obs;
            if (is_obs) obs_count++;
        }
    }
    std::cout << "[LOADMAP] Obstacle count: " << obs_count << "\n";
    return true;
}

/*
% loadAgents:
% Inputs:
%   - start_poss és goal_poss: ügynökök start és cél koordinátái.
%
% Behavior:
%   - Átalakítja a koordinátákat lineáris indexekké és eltárolja őket a start_locations és goal_locations vektorokban.
%
% Outputs:
%   - start_locations, goal_locations: a lineáris indexek listája.
*/
bool Instance::loadAgents(const vector<pair<int,int>>& start_poss,
    const vector<pair<int,int>>& goal_poss)
{
    start_locations.resize(num_of_agents);
    goal_locations.resize(num_of_agents);

    std::cout << "[LOADAGENTS] Loading agent start and goal positions...\n";
    for (int i = 0; i < num_of_agents; i++) {
        int sx = start_poss[i].first;
        int sy = start_poss[i].second;
        int gx = goal_poss[i].first;
        int gy = goal_poss[i].second;
        start_locations[i] = linearizeCoordinate(sx, sy);
        goal_locations[i] = linearizeCoordinate(gx, gy);

        /*
        std::cout << "  → Agent " << i
        << " start=(" << sx << "," << sy << ") val=" << dynamic_map_seq[0][sx][sy]
        << " | goal=(" << gx << "," << gy << ") val=" << dynamic_map_seq[0][gx][gy]
        << std::endl;
        */
    }
    return true;
}

/*
% getNeighbors:
% Inputs:
%   - curr: az aktuális cella indexe.
%
% Behavior:
%   - Kiszámolja a 4 szomszédos cella indexét (jobbra, balra, fel, le).
%   - A validMove függvény alapján kiválasztja az érvényes (mozgatható) szomszédokat.
%
% Outputs:
%   - Visszatér egy listával a megfelelő szomszédos cellák indexeiről.
*/
list<int> Instance::getNeighbors(int curr) const  // get truely moveable agent
{
	list<int> neighbors;
	int candidates[4] = {curr + 1, curr - 1, curr + num_of_cols, curr - num_of_cols};  // right, left, up,down
	for (int next : candidates)  // for next in candidates
	{
		if (validMove(curr, next))
			neighbors.emplace_back(next);
	}
	return neighbors;
}

/*
% hasDynamicObstacle:
% Inputs:
%   - timestep, row, col: the current time step and cell coordinates.
%
% Behavior:
%   - Checks if there is a dynamic obstacle at the specified cell and time step (value == 2).
%
% Outputs:
%   - True if the cell has a dynamic obstacle, false otherwise.
*/
bool Instance::hasDynamicObstacle(int timestep, int row, int col) const {
    if (dynamic_map_seq.empty() || timestep >= dynamic_map_seq.size()) return false;
    if (row < 0 || row >= num_of_rows || col < 0 || col >= num_of_cols) return false;
    return dynamic_map_seq[timestep][row][col] == 2;
    // Note: This method is kept for compatibility, but we now use getDynamicObstacleCost
    // to handle dynamic obstacles as soft constraints with increased cost
}
