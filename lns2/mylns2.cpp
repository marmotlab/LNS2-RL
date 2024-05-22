#include "mylns2.h"
#include <iostream>
#include <random>
#include "common.h"
#include <utility>

MyLns2::MyLns2(int seed,vector<vector<int>> obs_map,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,int all_ag_num,int map_size):
        instance(obs_map,start_poss,goal_poss,all_ag_num,map_size),path_table(map_size*map_size, all_ag_num), collision_graph(all_ag_num),goal_table(map_size*map_size, -1)
{
    srand(seed);
    agents.reserve(all_ag_num); //vector.reserve: adjust capacity
    for (int i = 0; i < all_ag_num; i++)
        agents.emplace_back(instance, i,instance.start_locations,instance.goal_locations);  //  add element to the last place
    for (auto& i:agents) {
        goal_table[i.path_planner->goal_location] = i.id;
    }
    destroy_weights.assign(INIT_COUNT, 1);
    decay_factor = 0.05;
    reaction_factor = 0.05;
}

void MyLns2::init_pp()
{
    neighbor.agents.clear();
    neighbor.agents.reserve(agents.size());
    for (int i = 0; i < (int)agents.size(); i++)
        neighbor.agents.push_back(i);
    std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
    set<pair<int, int>> colliding_pairs;
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    for (auto id : neighbor.agents)
    {
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);  // no hard obstacle, thus must find path
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(colliding_pairs,agents[id].id, agents[id].path);
        path_table.insertPath(agents[id].id, agents[id].path);
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
                        colliding_pairs.emplace(min(another_id, id), max(another_id, id));// emplace: insert new element into set
                }
                t++;
            }
        }
    }
    num_of_colliding_pairs = colliding_pairs.size();
    for(const auto& agent_pair : colliding_pairs)
    {
        collision_graph[agent_pair.first].emplace(agent_pair.second);
        collision_graph[agent_pair.second].emplace(agent_pair.first);
    }
    extract_path();
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

bool MyLns2::select_and_sipps(bool if_update,bool first_flag,vector<vector<pair<int,int>>> new_path,set<pair<int, int>> new_collsion_pair,int num_local_ag)
{
    neighbor_size = num_local_ag;
    neighbor.agents.clear();
    if (if_update and !first_flag)
    {
        for (int i = 0; i < (int)used_shuffled_agents.size(); i++)
        {
            int id = used_shuffled_agents[i];
            path_table.deletePath(id);
            int clip_index=-1;
            pair<int,int> goal_poss=instance.getCoordinate(instance.goal_locations[id]);
            for (int t=(int)new_path[i].size()-1;t>-1;t--)
            {
                if (new_path[i][t]!=goal_poss)
                {   clip_index=t;
                    break;}
            }
            new_path[i].resize(clip_index+2);
            agents[id].path.resize(new_path[i].size());
            for (int t=0;t<(int)new_path[i].size();t++)
            {
                agents[id].path[t].location=instance.linearizeCoordinate(new_path[i][t].first, new_path[i][t].second);
            }
            path_table.insertPath(agents[id].id, agents[id].path);
        }
        num_of_colliding_pairs += (int)new_collsion_pair.size() - (int)neighbor.old_colliding_pairs.size();
        for(const auto& agent_pair : neighbor.old_colliding_pairs)
        {
            collision_graph[agent_pair.first].erase(agent_pair.second);
            collision_graph[agent_pair.second].erase(agent_pair.first);
        }
        for(const auto& agent_pair : new_collsion_pair)
        {
            collision_graph[agent_pair.first].emplace(agent_pair.second);
            collision_graph[agent_pair.second].emplace(agent_pair.first);
        }
        if (new_collsion_pair.size() < neighbor.old_colliding_pairs.size())
            destroy_weights[selected_neighbor] =
                    reaction_factor * (double)(neighbor.old_colliding_pairs.size() -
                                               new_collsion_pair.size()) // / neighbor.agents.size()
                    + (1 - reaction_factor) * destroy_weights[selected_neighbor];
        else
            destroy_weights[selected_neighbor] =
                    (1 - decay_factor) * destroy_weights[selected_neighbor];
    }
    if (!if_update and !first_flag)
    {
        for (int i = 0; i < (int)neighbor.agents.size(); i++)
        {
            int a = neighbor.agents[i];
            path_table.deletePath(a);
            agents[a].path = neighbor.old_paths[i];
            path_table.insertPath(agents[a].id, agents[a].path);
        }
        destroy_weights[selected_neighbor] =
                (1 - decay_factor) * destroy_weights[selected_neighbor];
    }

    if (num_of_colliding_pairs==0)
        return true;
    bool succ = false;
    while (true)
    {
        chooseDestroyHeuristicbyALNS();
        switch (init_destroy_strategy)
        {
            case TARGET_BASED:
                succ = generateNeighborByTarget();
                break;
            case COLLISION_BASED:
                succ = generateNeighborByCollisionGraph();
                break;
            case RANDOM_BASED:
                succ = generateNeighborRandomly();
                break;
        }
        if (!succ || neighbor.agents.size()!=num_local_ag)
            continue;
        neighbor.old_colliding_pairs.clear();
        for (int a : neighbor.agents)
        {
            for (auto j: collision_graph[a])
            {
                neighbor.old_colliding_pairs.emplace(min(a, j), max(a, j));
            }
        }
        old_coll_pair_num=(int)neighbor.old_colliding_pairs.size();
        if (neighbor.old_colliding_pairs.empty()) // no need to replan
        {
            assert(init_destroy_strategy == RANDOM_BASED);
            destroy_weights[selected_neighbor] = (1 - decay_factor) * destroy_weights[selected_neighbor];
            continue;
        }
        break;
    }

    neighbor.old_paths.resize(neighbor.agents.size());
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
    {
        int a = neighbor.agents[i];
        neighbor.old_paths[i] = agents[a].path;
        path_table.deletePath(a);
    }
    makespan=path_table.makespan;
    runPP();
    return false;
}

void MyLns2::runPP()
{
    used_shuffled_agents.clear();
    used_shuffled_agents = neighbor.agents;
    std::random_shuffle(used_shuffled_agents.begin(), used_shuffled_agents.end());
    auto p = used_shuffled_agents.begin();
    neighbor.colliding_pairs.clear();
    sipps_path.clear();
    sipps_path.reserve(neighbor.agents.size());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);

    while (p != used_shuffled_agents.end())
    {
        int id = *p;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path);
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
        vector<pair<int,int>> single_path;
        for (const auto &state : agents[id].path)
        {   single_path.push_back(instance.getCoordinate(state.location));
        }
        sipps_path.push_back(single_path);
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
}


void MyLns2::rouletteWheel()
{
    double sum = 0;
    for (const auto& h : destroy_weights)
        sum += h;
    double r = (double) rand() / RAND_MAX;
    double threshold = destroy_weights[0];
    selected_neighbor = 0;
    while (threshold < r * sum)
    {
        selected_neighbor++;
        threshold += destroy_weights[selected_neighbor];
    }
}


void MyLns2::chooseDestroyHeuristicbyALNS()
{
    rouletteWheel();
    switch (selected_neighbor)
    {
        case 0 : init_destroy_strategy = TARGET_BASED; break;
        case 1 : init_destroy_strategy = COLLISION_BASED; break;
        case 2 : init_destroy_strategy = RANDOM_BASED; break;
        default : cerr << "ERROR" << endl; exit(-1);
    }
}

bool MyLns2::generateNeighborByCollisionGraph()
{
    vector<int> all_vertices;
    all_vertices.reserve(collision_graph.size());
    for (int i = 0; i < (int)collision_graph.size(); i++)
    {
        if (!collision_graph[i].empty())
            all_vertices.push_back(i);
    }
    unordered_map<int, set<int>> G;
    auto v = all_vertices[rand() % all_vertices.size()];
    findConnectedComponent(collision_graph, v, G);
    assert(G.size() > 1);

    assert(neighbor_size <= (int)agents.size());
    set<int> neighbors_set;
    if ((int)G.size() <= neighbor_size)
    {
        for (const auto& node : G)
            neighbors_set.insert(node.first);
        int count = 0;
        while ((int)neighbors_set.size() < neighbor_size && count < 10)
        {
            int a1 = *std::next(neighbors_set.begin(), rand() % neighbors_set.size());
            int a2 = randomWalk(a1);
            if (a2 != NO_AGENT)
                neighbors_set.insert(a2);
            else
                count++;
        }
    }
    else
    {
        int a = std::next(G.begin(), rand() % G.size())->first;
        neighbors_set.insert(a);
        while ((int)neighbors_set.size() < neighbor_size)
        {
            a = *std::next(G[a].begin(), rand() % G[a].size());
            neighbors_set.insert(a);
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    return true;

}
bool MyLns2::generateNeighborByTarget()
{
    int a = -1;
    auto r = rand() % (num_of_colliding_pairs * 2);
    int sum = 0;
    for (int i = 0 ; i < (int)collision_graph.size(); i++)
    {
        sum += (int)collision_graph[i].size();
        if (r <= sum and !collision_graph[i].empty())
        {
            a = i;
            break;
        }
    }
    assert(a != -1 and !collision_graph[a].empty());
    set<pair<int,int>> A_start; // an ordered set of (time, id) pair.
    set<int> A_target;


    for(int t = 0 ;t< path_table.table[agents[a].path_planner->start_location].size();t++){
        for(auto id : path_table.table[agents[a].path_planner->start_location][t]){
            if (id!=a)
                A_start.insert(make_pair(t,id));
        }
    }

    agents[a].path_planner->findMinimumSetofColldingTargets(goal_table,A_target);// generate non-wait path and collect A_target
    set<int> neighbors_set;

    neighbors_set.insert(a);

    if(A_start.size() + A_target.size() >= neighbor_size-1){
        if (A_start.empty()){
            vector<int> shuffled_agents;
            shuffled_agents.assign(A_target.begin(),A_target.end());
            std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
            neighbors_set.insert(shuffled_agents.begin(), shuffled_agents.begin() + neighbor_size-1);
        }
        else if (A_target.size() >= neighbor_size){
            vector<int> shuffled_agents;
            shuffled_agents.assign(A_target.begin(),A_target.end());
            std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
            neighbors_set.insert(shuffled_agents.begin(), shuffled_agents.begin() + neighbor_size-2);

            neighbors_set.insert(A_start.begin()->second);
        }
        else{
            neighbors_set.insert(A_target.begin(), A_target.end());
            for(auto e : A_start){
                //A_start is ordered by time.
                if (neighbors_set.size()>= neighbor_size)
                    break;
                neighbors_set.insert(e.second);

            }
        }
    }
    else if (!A_start.empty() || !A_target.empty()){
        neighbors_set.insert(A_target.begin(), A_target.end());
        for(auto e : A_start){
            neighbors_set.insert(e.second);
        }

        set<int> tabu_set;
        while(neighbors_set.size()<neighbor_size){
            int rand_int = rand() % neighbors_set.size();
            auto it = neighbors_set.begin();
            std::advance(it, rand_int);
            a = *it;
            tabu_set.insert(a);

            if(tabu_set.size() == neighbors_set.size())
                break;

            vector<int> targets;
            for(auto p: agents[a].path){
                if(goal_table[p.location]>-1){
                    targets.push_back(goal_table[p.location]);
                }
            }

            if(targets.empty())
                continue;
            rand_int = rand() %targets.size();
            neighbors_set.insert(*(targets.begin()+rand_int));
        }
    }

    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    return true;
}
bool MyLns2::generateNeighborRandomly()
{
    if (neighbor_size >= agents.size())
    {
        neighbor.agents.resize(agents.size());
        for (int i = 0; i < (int)agents.size(); i++)
            neighbor.agents[i] = i;
        return true;
    }
    set<int> neighbors_set;
    auto total = num_of_colliding_pairs * 2 + agents.size();
    while(neighbors_set.size() < neighbor_size)
    {
        vector<int> r(neighbor_size - neighbors_set.size());
        for (auto i = 0; i < neighbor_size - neighbors_set.size(); i++)
            r[i] = rand() % total;
        std::sort(r.begin(), r.end());
        int sum = 0;
        for (int i = 0, j = 0; i < agents.size() and j < r.size(); i++)
        {
            sum += (int)collision_graph[i].size() + 1;
            if (sum >= r[j])
            {
                neighbors_set.insert(i);
                while (j < r.size() and sum >= r[j])
                    j++;
            }
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    return true;
}


unordered_map<int, set<int>>& MyLns2::findConnectedComponent(const vector<set<int>>& graph, int vertex,
                                                             unordered_map<int, set<int>>& sub_graph)
{
    std::queue<int> Q;
    Q.push(vertex);
    sub_graph.emplace(vertex, graph[vertex]);
    while (!Q.empty())
    {
        auto v = Q.front(); Q.pop();
        for (const auto & u : graph[v])
        {
            auto ret = sub_graph.emplace(u, graph[u]);
            if (ret.second) // insert successfully
                Q.push(u);
        }
    }
    return sub_graph;
}

int MyLns2::randomWalk(int agent_id)
{
    int t = rand() % agents[agent_id].path.size();
    int loc = agents[agent_id].path[t].location;
    while (t <= path_table.makespan and
           (path_table.table[loc].size() <= t or
            path_table.table[loc][t].empty() or
            (path_table.table[loc][t].size() == 1 and path_table.table[loc][t].front() == agent_id)))
    {
        auto next_locs = instance.getNeighbors(loc);
        next_locs.push_back(loc);
        int step = rand() % next_locs.size();
        auto it = next_locs.begin();
        loc = *std::next(next_locs.begin(), rand() % next_locs.size());
        t = t + 1;
    }
    if (t > path_table.makespan)
        return NO_AGENT;
    else
        return *std::next(path_table.table[loc][t].begin(), rand() % path_table.table[loc][t].size());
}

void MyLns2::extract_path()
{
    vector_path.clear();
    for (const auto &agent : agents)
    {
        vector<pair<int,int>> single_path;
        for (const auto &state : agent.path)
            single_path.push_back(instance.getCoordinate(state.location));
        vector_path.push_back(single_path);}
}


int MyLns2::add_sipps(vector<vector<pair<int,int>>> temp_path,vector<int> selected_ag,const vector<pair<int,int>>& vector_start_locations)
{
    add_neighbor.agents=selected_ag;
    add_neighbor.colliding_pairs.clear();
    vector<Agent> add_agents;
    PathTableWC add_path_table(instance.map_size,instance.num_of_agents);
    add_agents.reserve(agents.size()); //vector.reserve: adjust capacity
    vector<int> start_locations;
    start_locations.reserve((int)agents.size());
    for(int id=0;id<(int)agents.size();id++)
        start_locations.push_back(instance.linearizeCoordinate(vector_start_locations[id].first, vector_start_locations[id].second));
    for(int id=0;id<(int)agents.size();id++)
    {
        add_agents.emplace_back(instance, id,start_locations,instance.goal_locations);
        add_agents[id].path.resize(temp_path[id].size());
        for (int t=0;t<(int)temp_path[id].size();t++)
        {
            add_agents[id].path[t].location=instance.linearizeCoordinate(temp_path[id][t].first, temp_path[id][t].second);
        }
        add_path_table.insertPath(id, add_agents[id].path);
    }
    for (int id : add_neighbor.agents)
    {
        add_path_table.deletePath(id);
    }
    auto add_shuffled_agents = add_neighbor.agents;
    std::random_shuffle(add_shuffled_agents.begin(), add_shuffled_agents.end());
    auto p = add_shuffled_agents.begin();
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &add_path_table);
    while (p != add_shuffled_agents.end())
    {
        int id = *p;
        add_agents[id].path = add_agents[id].path_planner->findPath(constraint_table);
        assert(!add_agents[id].path.empty() && add_agents[id].path.back().location == add_agents[id].path_planner->goal_location);
        if (add_agents[id].path_planner->num_collisions > 0)
            add_updateCollidingPairs( add_neighbor.colliding_pairs,add_agents[id].id, add_agents[id].path,add_path_table,add_agents);
        add_path_table.insertPath(add_agents[id].id, add_agents[id].path);
        ++p;
    }
    add_sipps_path.clear();
    p = add_neighbor.agents.begin();
    while (p != add_neighbor.agents.end())
    {
        int id=*p;
        if (add_agents[id].path.size()==1)
        {
            int to = add_agents[id].path[0].location;
            int t=0;
            for (auto & ag_list : add_path_table.table[to])
            {
                if (t!=0 && !ag_list.empty())
                {
                    for (int another_id: ag_list)
                        add_neighbor.colliding_pairs.emplace(min(another_id, id), max(another_id, id));// emplace: insert new element into set
                }
                t++;
            }
        }
        vector<pair<int,int>> single_path;
        for (const auto &state : add_agents[id].path)
            single_path.push_back(instance.getCoordinate(state.location));
        add_sipps_path.push_back(single_path);
        ++p;
    }
    return (int)add_neighbor.colliding_pairs.size();
}

bool MyLns2::add_updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path,const PathTableWC& add_path_table,const vector<Agent>& add_agents)
{
    bool succ = false;
    if (path.size() < 2)
        return succ;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)add_path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : add_path_table.table[to][t])
            {
                succ = true;
                colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));// emplace: insert new element into set
            }
        }
        if (from != to && add_path_table.table[to].size() >= t && add_path_table.table[from].size() > t) // edge conflicts(swapping conflicts)
        {
            for (auto a1 : add_path_table.table[to][t - 1])
            {
                for (auto a2: add_path_table.table[from][t])
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
        if (!add_path_table.goals.empty() && add_path_table.goals[to] < t) // target conflicts, already has agent in its goal, so the new agent can not tarverse it
        { // this agent traverses the target of another agent
            for (auto id : add_path_table.table[to][add_path_table.goals[to]]) // look at all agents at the goal time
            {
                if (add_agents[id].path.back().location == to) // if agent id's goal is to, then this is the agent we want
                {
                    succ = true;
                    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
                    break;
                }
            }
        }
    }
    int goal = path.back().location; // target conflicts - some other agent traverses the target of this agent
    for (int t = (int)path.size(); t < add_path_table.table[goal].size(); t++)
    {
        for (auto id : add_path_table.table[goal][t])
        {
            succ = true;
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
    return succ;
}

void MyLns2::replan_sipps(vector<vector<pair<int,int>>> temp_path,vector<int> selected_ag)
{
    for (int i = 0; i < (int)used_shuffled_agents.size(); i++)
    {
        int id = used_shuffled_agents[i];
        path_table.deletePath(id);
        if (std::find(selected_ag.begin(), selected_ag.end(), id) == selected_ag.end())
        {
            int clip_index=-1;
            pair<int,int> goal_poss=instance.getCoordinate(instance.goal_locations[id]);
            for (int t=(int)temp_path[i].size()-1;t>-1;t--)
            {
                if (temp_path[i][t]!=goal_poss)
                {   clip_index=t;
                    break;}
            }
            temp_path[i].resize(clip_index+2);
            agents[id].path.resize(temp_path[i].size());
            for (int t=0;t<(int)temp_path[i].size();t++)
            {
                agents[id].path[t].location=instance.linearizeCoordinate(temp_path[i][t].first, temp_path[i][t].second);
            }
            path_table.insertPath(agents[id].id, agents[id].path);
        }
    }
    replan_neighbor.agents=selected_ag;
    replan_neighbor.colliding_pairs.clear();
    auto replan_shuffled_agents = replan_neighbor.agents;
    std::random_shuffle(replan_shuffled_agents.begin(), replan_shuffled_agents.end());
    auto p = replan_shuffled_agents.begin();
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    while (p != replan_shuffled_agents.end())
    {
        int id = *p;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(replan_neighbor.colliding_pairs, agents[id].id, agents[id].path);
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
    }
    replan_sipps_path.clear();
    p = replan_neighbor.agents.begin();
    while (p != replan_neighbor.agents.end())
    {
        int id=*p;
        vector<pair<int,int>> single_path;
        for (const auto &state : agents[id].path)
            single_path.push_back(instance.getCoordinate(state.location));
        replan_sipps_path.push_back(single_path);
        ++p;
    }
}



