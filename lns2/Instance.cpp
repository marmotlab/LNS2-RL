#include<boost/tokenizer.hpp>
#include <algorithm>    // std::shuffle
#include"Instance.h"


Instance::Instance(const vector<vector<int>>& obs_map, const vector<pair<int,int>>& start_poss, const vector<pair<int,int>>& goal_poss,int num_of_agents,int num_of_rows,int num_of_cols):
        num_of_agents(num_of_agents),num_of_rows(num_of_rows),num_of_cols(num_of_cols)
{
    loadMap(obs_map);
    loadAgents(start_poss,goal_poss);
}


bool Instance::loadMap(const vector<vector<int>>& obs_map)
{
    map_size = num_of_cols * num_of_rows;  // linearized
    my_map.resize(map_size, false);  // release rest space
    // read map (and start/goal locations)
    for (int i = 0; i < num_of_rows; i++) {
        for (int j = 0; j < num_of_cols; j++) {
            my_map[linearizeCoordinate(i, j)] = (obs_map[i][j] != 0); // @=1 obstacle, .=0 empty, trasfer form 2 D to 1 D
        }
    }
    return true;
}

bool Instance::loadAgents(const vector<pair<int,int>>& start_poss, const vector<pair<int,int>>& goal_poss)
{
    start_locations.resize(num_of_agents);
    goal_locations.resize(num_of_agents);
    for (int i = 0; i < num_of_agents; i++)
    {
        start_locations[i] = linearizeCoordinate(start_poss[i].first, start_poss[i].second);
        goal_locations[i] = linearizeCoordinate(goal_poss[i].first, goal_poss[i].second);
    }
    return true;
}


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
