#pragma once
#include "common.h"
#include "SIPP.h"

struct Agent
{
    int id;
    SingleAgentSolver* path_planner = nullptr; // start, goal, and heuristics are stored in the path planner
    Path path;

    Agent(const Instance& instance, int id,const vector<int>& start_locations,const vector<int>& goal_locations) : id(id)
    {
        path_planner = new SIPP(instance, id,start_locations,goal_locations);
    }
    ~Agent(){ delete path_planner; }
};

struct Neighbor
{
    vector<int> agents;
    set<pair<int, int>> colliding_pairs;  // id1 < id2
};
