#pragma once  // only include this file once
#include"common.h"


// Currently only works for undirected unweighted 4-nighbor grids
class Instance 
{
public:
	int num_of_cols;
	int num_of_rows;
	int map_size;
    vector<int> start_locations;
    vector<int> goal_locations;
    int num_of_agents;

	Instance()=default;
    Instance(const vector<vector<int>>& obs_map, const vector<pair<int,int>>& start_poss, const vector<pair<int,int>>& goal_poss,int num_of_agents,int num_of_rows);

    inline bool validMove(int curr, int next) const
    {
        if (next < 0 || next >= map_size)
            return false;
        if (my_map[next])  // =1 obstacle
            return false;
        return getManhattanDistance(curr, next) < 2;  // if one step
    }
    list<int> getNeighbors(int curr) const;

    inline int linearizeCoordinate(int row, int col) const { return ( this->num_of_cols * row + col); }  // x-y map to one dimension
    inline int getRowCoordinate(int id) const { return id / this->num_of_cols; }
    inline int getColCoordinate(int id) const { return id % this->num_of_cols; }
    inline pair<int, int> getCoordinate(int id) const { return make_pair(id / this->num_of_cols, id % this->num_of_cols); }

    inline int getManhattanDistance(int loc1, int loc2) const
    {
        int loc1_x = getRowCoordinate(loc1);
        int loc1_y = getColCoordinate(loc1);
        int loc2_x = getRowCoordinate(loc2);
        int loc2_y = getColCoordinate(loc2);
        return abs(loc1_x - loc2_x) + abs(loc1_y - loc2_y);
    }



private:
	  // int moves_offset[MOVE_COUNT];
	  vector<bool> my_map;
      string map_fname;
      string agent_fname;

      bool loadMap(const vector<vector<int>>& obs_map);
      bool loadAgents(const vector<pair<int,int>>& start_poss, const vector<pair<int,int>>& goal_poss);

	  // Class  SingleAgentSolver can access private members of Node 
	  friend class SingleAgentSolver;
};

