#include "PathTable.h"

/*#########################
% insertPath (int agent_id, const Path& path):
%
% Inputs:
%   - agent_id: az ügynök azonosítója.
%   - path: Az ügynök által megtervezett időindexelt útvonal.
%
% Behavior:
%   - Az adott ügynök útvonalát rögzíti a 'paths' vektorban.
%   - Minden időpillanatban (t) a cella (path[t].location) számára 
%     hozzáadja az ügynök azonosítóját a 'table' megfelelő pozíciójához.
%   - A cél cellánál a 'goals' értéket beállítja az út hossza - 1-re.
%   - Frissíti a makespan értékét.
%
% Outputs:
%   - Frissített path table és goals, makespan.
*/
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

/*#########################
% insertPath (int agent_id):
%
% Inputs:
%   - agent_id: az ügynök azonosítója, melynek útvonalát már elmentettük a 'paths'-ban.
% Behavior:
%   - Ellenőrzi, hogy paths[agent_id] nem üres,
%     majd meghívja a fenti insertPath(int, const Path&) függvényt.
% Outputs:
%   - A paths beillesztése megtörténik.
*/
void PathTableWC::insertPath(int agent_id)
{
    assert(paths[agent_id] != nullptr);
    insertPath(agent_id, *paths[agent_id]);
}

/*#########################
% deletePath (int agent_id):
%
% Inputs:
%   - agent_id: az ügynök azonosítója, amelynek útvonalát törölni kell.
% Behavior:
%   - Iterál az ügynök útvonalán (Path), és minden időpillanatból
%     eltávolítja az ügynök azonosítóját a 'table'-ből.
%   - A cél cella esetében reseteli a goals értékét (MAX_TIMESTEP).
%   - Ha az aktuális makespan megegyezik az ügynök útvonalhosszával, újraszámolja a makespan értékét.
% Outputs:
%   - Eltávolítja az ügynök útvonalát a path table-ból, frissítve a goals és makespan értékeket.
*/
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

/*#########################
% getFutureNumOfCollisions:
%
% Inputs:
%   - loc: A cella indexe.
%   - time: Az időpillanat, után mely ütközéseket számoljuk.
% Behavior:
%   - Iterál a 'table[loc]' safe interval listán az adott időpont után,
%     és összeadja az egyes időlépésekben szereplő ügynökök számát.
% Outputs:
%   - Az ütközések száma (az érkező ügynökök mennyisége) a jövőben az adott cellában.
*/
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

/*#########################
% hasEdgeCollisions:
%
% Inputs:
%   - from, to: Két szomszédos cella indexe.
%   - to_time: Az időpillanat, amikor a mozgás befejeződik.
%
% Behavior:
%   - Ellenőrzi, hogy az 'from' cellából a 'to' cellába való mozgás során
%     van-e olyan ügynök, amelyik időben konfliktusba kerül (edge conflict).
%
% Outputs:
%   - True, ha van élütközés, egyébként false.
*/
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

/*#########################
% getLastCollisionTimestep:
%
% Inputs:
%   - location: A cella indexe.
% Behavior:
%   - Visszafelé iterál az adott cella 'table' bejegyzésein, és az első időpillanatot adja vissza,
%     amikor legalább egy ügynök elérte a cellát.
% Outputs:
%   - Az utolsó ütközési (érkezési) időpillanat, vagy -1, ha nincs ütközés.
*/
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
