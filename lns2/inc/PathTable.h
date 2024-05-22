#pragma once
#include "common.h"

#define NO_AGENT -1

class PathTableWC // with collisions
{
public:
    int makespan = 0;
    vector< vector< list<int> > > table; // this stores the paths, the value is the id of the agent
    vector<int> goals; // this stores the goal locatons of the paths: key is the location, while value is the timestep when the agent reaches the goal
    void insertPath(int agent_id, const Path& path);
    void insertPath(int agent_id);
    void deletePath(int agent_id);
    int getFutureNumOfCollisions(int loc, int time) const; // return #collisions when the agent waiting at loc starting from time forever
    bool hasEdgeCollisions(int from, int to, int to_time) const;
    int getLastCollisionTimestep(int location) const;
    // return the agent who reaches its target target_location before timestep earliest_timestep
    explicit PathTableWC(int map_size = 0, int num_of_agents = 0) : table(map_size), goals(map_size, MAX_COST),
        paths(num_of_agents, nullptr) {}
private:
    vector<const Path*> paths;
};