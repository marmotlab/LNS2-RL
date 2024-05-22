#include "PathTable.h"


void PathTableWC::insertPath(int agent_id, const Path& path)
{
    paths[agent_id] = &path;
    if (path.empty())
        return;
    for (int t = 0; t < (int)path.size(); t++)
    {
        if (table[path[t].location].size() <= t)
            table[path[t].location].resize(t + 1);
        table[path[t].location][t].push_back(agent_id);
    }
    assert(goals[path.back().location] == MAX_TIMESTEP);
    goals[path.back().location] = (int) path.size() - 1;
    makespan = max(makespan, (int) path.size() - 1);
}

void PathTableWC::insertPath(int agent_id)
{
    assert(paths[agent_id] != nullptr);
    insertPath(agent_id, *paths[agent_id]);
}

void PathTableWC::deletePath(int agent_id)
{
    const Path & path = *paths[agent_id];
    if (path.empty())
        return;
    for (int t = 0; t < (int)path.size(); t++)
    {
        assert(table[path[t].location].size() > t &&
               std::find (table[path[t].location][t].begin(), table[path[t].location][t].end(), agent_id)
               != table[path[t].location][t].end());
        table[path[t].location][t].remove(agent_id);
    }
    goals[path.back().location] = MAX_TIMESTEP;
    if (makespan == (int) path.size() - 1) // re-compute makespan
    {
        makespan = 0;
        for (int time : goals)
        {
            if (time < MAX_TIMESTEP && time > makespan)
                makespan = time;
        }

    }
}

int PathTableWC::getFutureNumOfCollisions(int loc, int time) const
{
    assert(goals[loc] == MAX_TIMESTEP);
    int rst = 0;
    if (!table.empty() && (int)table[loc].size() > time)
    {
        for (int t = time + 1; t < (int)table[loc].size(); t++)
            rst += (int)table[loc][t].size();  // vertex conflict, HAS OTHER AGENTS REACH THIS POSITION AT TIME T
    }
    return rst;
}

bool PathTableWC::hasEdgeCollisions(int from, int to, int to_time) const
{
    if (!table.empty() && from != to && table[to].size() >= to_time && table[from].size() > to_time)
    {
        for (auto a1 : table[to][to_time - 1])
        {
            for (auto a2: table[from][to_time])
            {
                if (a1 == a2)
                    return true; // edge conflict
            }
        }
    }
    return false;
}

int PathTableWC::getLastCollisionTimestep(int location) const
{
    if (table.empty())
        return -1;
    for (int t = (int)table[location].size() - 1; t >= 0; t--)
    {
        if (!table[location][t].empty())
            return t;
    }
    return -1;
}
