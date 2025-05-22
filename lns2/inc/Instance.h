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

    // Store full dynamic map sequence [t][row][col]
    // Using float to handle GridMapEnv representation directly
    std::vector<std::vector<std::vector<float>>> dynamic_map_seq;

	Instance()=default;

    // Constructor with dynamic map sequence
    Instance(const std::vector<std::vector<std::vector<float>>>& grid_sequence,
        const std::vector<std::pair<int, int>>& start_poss,
        const std::vector<std::pair<int, int>>& goal_poss,
        int num_of_agents, int num_of_rows);

    // Legacy constructor with static map only
    Instance(const vector<vector<int>>& obs_map, const vector<pair<int,int>>& start_poss, const vector<pair<int,int>>& goal_poss,int num_of_agents,int num_of_rows);

    // Check if a move is valid at a specific time step
    inline bool validMove(int curr, int next, int timestep = 0) const
    {
        if (next < 0 || next >= map_size)
            return false;

        // Check static obstacles
        if (my_map[next])  // =1 obstacle
            return false;

        // Dynamic obstacles (radar zones) are now soft constraints, so we don't block the path here
        // We'll handle them with a cost factor in getDynamicObstacleCost

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

    // Returns a constant reference to the map data.
    const vector<bool>& getMap() const { return my_map; }

    // Check if a cell has a dynamic obstacle at a specific time
    // In GridMapEnv, dynamic obstacles have values between 0.1 and 0.9
    bool hasDynamicObstacle(int timestep, int row, int col) const;

    // Get the cost of moving through a dynamic obstacle (radar zone)
    // Returns 3 if there's a dynamic obstacle, 1 otherwise
    // In GridMapEnv, dynamic obstacles have values between 0.1 and 0.9
    inline int getDynamicObstacleCost(int timestep, int row, int col) const {
        if (!dynamic_map_seq.empty() && timestep < dynamic_map_seq.size()) {
            if (row >= 0 && row < num_of_rows && col >= 0 && col < num_of_cols) {
                float value = dynamic_map_seq[timestep][row][col];
                if (value > 0.1 && value <= 0.9) // Dynamic obstacle (radar zone)
                    return 3; // Higher cost for radar zones
            }
        }
        return 1; // Normal cost
    }

    // Get the cost of moving to a location at a specific time
    inline int getMoveCost(int curr, int next, int timestep) const {
        if (next < 0 || next >= map_size)
            return INT_MAX; // Invalid move

        if (my_map[next]) // Static obstacle
            return INT_MAX; // Invalid move

        // Check if this is a waiting action (staying in the same location)
        if (curr == next) {
            // Waiting is 8x more expensive than moving
            return 8;
        }

        int row = getRowCoordinate(next);
        int col = getColCoordinate(next);
        return getDynamicObstacleCost(timestep, row, col);
    }

    // Check if we have dynamic map data
    bool hasDynamicMapData() const { return !dynamic_map_seq.empty(); }

    // Get the number of time steps in the dynamic map sequence
    size_t getNumTimeSteps() const { return dynamic_map_seq.size(); }

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

