#include "mylns2.h"
#include <iostream>
#include <random>
#include "common.h"
#include <utility>

MyLns2::MyLns2(int seed, vector<vector<int>> obs_map,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,int all_ag_num,int map_size):
        instance(obs_map,start_poss,goal_poss,all_ag_num,map_size),path_table(map_size*map_size, all_ag_num)
{
    srand(seed);
    agents.reserve(all_ag_num); //vector.reserve: adjust capacity
    for (int i = 0; i < all_ag_num; i++)
        agents.emplace_back(instance, i,instance.start_locations,instance.goal_locations);  //  add element to the last place
}

void MyLns2::init_pp()
{
    neighbor.agents.reserve(agents.size());
    for (int i = 0; i < (int)agents.size(); i++)
        neighbor.agents.push_back(i);
    std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    for (auto id : neighbor.agents)
    {
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);  // no hard obstacle, thus must find path
        path_table.insertPath(agents[id].id, agents[id].path);
    }
    vector_path.reserve(instance.num_of_agents);
    for (const auto &agent : agents)
    {
        vector<pair<int,int>> single_path;
        for (const auto &state : agent.path)
            single_path.push_back(instance.getCoordinate(state.location));
        vector_path.push_back(single_path);
    }
}


bool MyLns2::updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path)
{
    bool succ = false;
    if (path.size() < 2)
        return succ;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {
                succ = true;
                colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));// emplace: insert new element into set
            }
        }
        if (from != to && path_table.table[to].size() >= t && path_table.table[from].size() > t) // edge conflicts(swapping conflicts)
        {
            for (auto a1 : path_table.table[to][t - 1])
            {
                for (auto a2: path_table.table[from][t])
                {
                    if (a1 == a2)
                    {
                        succ = true;
                        colliding_pairs.emplace(min(agent_id, a1), max(agent_id, a1));
                        break;
                    }
                }
            }
        }
        if (!path_table.goals.empty() && path_table.goals[to] < t) // target conflicts, already has agent in its goal, so the new agent can not tarverse it
        { // this agent traverses the target of another agent
            for (auto id : path_table.table[to][path_table.goals[to]]) // look at all agents at the goal time
            {
                if (agents[id].path.back().location == to) // if agent id's goal is to, then this is the agent we want
                {
                    succ = true;
                    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
                    break;
                }
            }
        }
    }
    int goal = path.back().location; // target conflicts - some other agent traverses the target of this agent
    for (int t = (int)path.size(); t < path_table.table[goal].size(); t++)
    {
        for (auto id : path_table.table[goal][t])
        {
            succ = true;
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
    return succ;
}

int MyLns2::calculate_sipps(vector<int> new_agents)
{
    neighbor.colliding_pairs.clear();
    neighbor.agents=new_agents;
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
        path_table.deletePath(neighbor.agents[i]);
    makespan=path_table.makespan;
    auto p = neighbor.agents.begin();
    sipps_path.clear();
    sipps_path.reserve(new_agents.size());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    while (p != neighbor.agents.end())
    {
        int id = *p;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path);
        vector<pair<int,int>> single_path;
        for (const auto &state : agents[id].path)
        {   single_path.push_back(instance.getCoordinate(state.location));
        }
        sipps_path.push_back(single_path);
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
    }
    for (auto id : neighbor.agents)
    {
        if (agents[id].path.size()==1)
        {
            int to = agents[id].path[0].location;
            int t=0;
            for (auto & ag_list : path_table.table[to])
            {
                if (t!=0 && !ag_list.empty())
                {
                    for (int another_id: ag_list)
                        neighbor.colliding_pairs.emplace(min(another_id, id), max(another_id, id));// emplace: insert new element into set
                }
                t++;
            }
        }
    }

    return (int)neighbor.colliding_pairs.size();
}

int MyLns2::single_sipp(vector<vector<pair<int,int>>> dy_obs_path,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,
                                  pair<int,int> self_start_poss,pair<int,int> self_goal_poss,int global_num_agent)
{
    vector<Agent> add_agents;
    PathTableWC add_path_table(instance.map_size,global_num_agent);
    add_agents.reserve(global_num_agent); //vector.reserve: adjust capacity
    vector<int> start_locations;
    vector<int> goal_locations;
    start_locations.reserve(global_num_agent);
    goal_locations.reserve(global_num_agent);
    for(int id=0;id<global_num_agent-1;id++)
    {
        start_locations.push_back(instance.linearizeCoordinate(start_poss[id].first, start_poss[id].second));
        goal_locations.push_back(instance.linearizeCoordinate(goal_poss[id].first, goal_poss[id].second));}
    start_locations.push_back(instance.linearizeCoordinate(self_start_poss.first, self_start_poss.second));
    goal_locations.push_back(instance.linearizeCoordinate(self_goal_poss.first, self_goal_poss.second));
    for(int id=0;id<global_num_agent-1;id++)
    {
        add_agents.emplace_back(instance, id,start_locations,goal_locations);
        add_agents[id].path.resize(dy_obs_path[id].size());
        for (int t=0;t<(int)dy_obs_path[id].size();t++)
        {
            add_agents[id].path[t].location=instance.linearizeCoordinate(dy_obs_path[id][t].first, dy_obs_path[id][t].second);
        }
        add_path_table.insertPath(id, add_agents[id].path);
    }
    add_agents.emplace_back(instance, global_num_agent-1,start_locations,goal_locations);
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &add_path_table);
    add_agents[global_num_agent-1].path = add_agents[global_num_agent-1].path_planner->findPath(constraint_table);
    assert(!add_agents[global_num_agent-1].path.empty() && add_agents[global_num_agent-1].path.back().location == add_agents[global_num_agent-1].path_planner->goal_location);
    int path_ln=(int)add_agents[global_num_agent-1].path.size();
    return path_ln;

}







