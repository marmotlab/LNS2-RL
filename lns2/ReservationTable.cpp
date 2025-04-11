#include "ReservationTable.h"

/*#########################
% insertSoftConstraint2SIT:
% Inputs:
%   - location: a cella indexe, amelyre a constraint vonatkozik.
%   - t_min, t_max: az új constraint időintervallumának kezdete és vége.
%
% Function behaviour:
%   - Iterál a 'sit[location]' (Safe Interval Table a cellára) intervallumain.
%   - Ha az új intervallum (t_min, t_max) átfedi a meglévő intervallumot (és az aktuális intervallumban nincs ütközés),
%     akkor frissíti vagy összevonja az intervallumokat úgy, hogy a megadott időablakban a cella ütközést jelöl.
%
% Outputs:
%   - Frissített safe interval lista a 'sit[location]' számára.
*/
void ReservationTable::insertSoftConstraint2SIT(int location, int t_min, int t_max)
{
    assert(t_min >= 0 && t_min < t_max and !sit[location].empty());
    for (auto it = sit[location].begin(); it != sit[location].end(); ++it)
    {
        if (t_min >= get<1>(*it) || get<2>(*it))  // no intersection  or has collision
            continue;
        else if (t_max <= get<0>(*it))  // no intersection
            break;

        auto i_min = get<0>(*it);
        auto i_max = get<1>(*it);
        if (i_min < t_min && i_max <= t_max)  // early stage intersection
        {
            if (it != sit[location].end() and std::next(it) != sit[location].end() and
                    (location != goal_location || i_max != constraint_table.length_min) and
                    i_max == get<0>(*std::next(it)) and get<2>(*std::next(it))) // we can merge the current interval with the next one
            {
                (*it) = make_tuple(i_min, t_min, false);
                ++it;
                (*it) = make_tuple(t_min, get<1>(*it), true);
            }
            else
            {
                sit[location].insert(it, make_tuple(i_min, t_min, false));
                (*it) = make_tuple(t_min, i_max, true);  // insert to safe interval
            }

        }
        else if (t_min <= i_min && t_max < i_max)  // later stage intersection
        {
            if (it != sit[location].begin() and (location != goal_location || i_min != constraint_table.length_min) and
                    i_min == get<1>(*std::prev(it)) and get<2>(*std::prev(it))) // we can merge the current interval with the previous one
            {
                (*std::prev(it)) = make_tuple(get<0>(*std::prev(it)), t_max, true);
            }
            else
            {
                sit[location].insert(it, make_tuple(i_min, t_max, true));
            }
            (*it) = make_tuple(t_max, i_max, false);
        }
        else if (i_min < t_min && t_max < i_max)  // contain the while t_min-t_max
        {
            sit[location].insert(it, make_tuple(i_min, t_min, false));
            sit[location].insert(it, make_tuple(t_min, t_max, true));
            (*it) = make_tuple(t_max, i_max, false);
        }
        else // constraint_min <= get<0>(*it) && get<1> <= constraint_max
        {
            if (it != sit[location].begin() and (location != goal_location || i_min != constraint_table.length_min) and
                i_min == get<1>(*std::prev(it)) and get<2>(*std::prev(it))) // we can merge the current interval with the previous one
            {
                if (it != sit[location].end() and std::next(it) != sit[location].end() and
                        (location != goal_location || i_max != constraint_table.length_min) and
                        i_max == get<0>(*std::next(it)) and get<2>(*std::next(it))) // we can merge the current interval with the next one
                {
                    (*std::prev(it)) = make_tuple(get<0>(*std::prev(it)), get<1>(*std::next(it)), true);
                    sit[location].erase(std::next(it));
                    it = sit[location].erase(it);
                }
                else
                {
                    (*std::prev(it)) = make_tuple(get<0>(*std::prev(it)), i_max, true);
                    it = sit[location].erase(it);
                }
                --it;
            }
            else
            {
                if (it != sit[location].end() and std::next(it) != sit[location].end() and
                        (location != goal_location || i_max != constraint_table.length_min) and
                        i_max == get<0>(*std::next(it)) and get<2>(*std::next(it))) // we can merge the current interval with the next one
                {
                    (*it) = make_tuple(i_min, get<1>(*std::next(it)), true);
                    sit[location].erase(std::next(it));
                }
                else
                {
                    (*it) = make_tuple(i_min, i_max, true);
                }
            }
        }
    }
}

// update SIT at the given location
/*#########################
% updateSIT:
% Inputs:
%   - location: Az adott cella indexe, amelyre a safe interval táblát frissítjük.
%
% Function behaviour:
%   - Ha a cella a célpozíció, két intervallumra osztja: [0, length_min) és [length_min, min(length_max+1, MAX_TIMESTEP)).
%   - Egyéb cellák esetén létrehoz egy intervallumot [0, min(length_max, MAX_TIMESTEP-1)+1), jelezve, hogy az adott cella kezdetben szabad.
%   - Ha létezik soft path table (CAT), akkor a cél- vagy élütközések (vertex/edge conflicts) alapján meghívja az insertSoftConstraint2SIT 
%     függvényt, hogy az adott cella safe interval-jeit frissítse.
%
% Outputs:
%   - A 'sit[location]' listában frissített intervalumokat tartalmaz, melyek figyelembe veszik a statikus akadályokat és
%     a soft ütközési korlátozásokat (ha van CAT).
*/
void ReservationTable::updateSIT(int location)
{
    assert(sit[location].empty());  //[t_min, t_max), num_of_collisions
    // length constraints for the goal location
    if (location == goal_location) // we need to divide the same intervals into 2 parts [0, length_min) and [length_min, length_max + 1) for location on goal
    {
        if (constraint_table.length_min > constraint_table.length_max) // the location is blocked for the entire time horizon
        {
            sit[location].emplace_back(0, 0, false);
            return;
        }
        if (0 < constraint_table.length_min)
        {
            sit[location].emplace_back(0, constraint_table.length_min, false);
        }
        assert(constraint_table.length_min >= 0);
        sit[location].emplace_back(constraint_table.length_min, min(constraint_table.length_max + 1, MAX_TIMESTEP), false);
    }
    else
    {
        sit[location].emplace_back(0, min(constraint_table.length_max, MAX_TIMESTEP - 1) + 1, false);
    }


    // soft path table
    if (constraint_table.path_table_for_CAT != nullptr and
        !constraint_table.path_table_for_CAT->table.empty())  // insert soft constraint, that may contain collision with the soft path
    {
        if (location < constraint_table.map_size) // vertex conflict
        {
            for (int t = 0; t < (int)constraint_table.path_table_for_CAT->table[location].size(); t++)
            {
                if (!constraint_table.path_table_for_CAT->table[location][t].empty())
                {
                    insertSoftConstraint2SIT(location, t, t+1);
                }
            }
            if (constraint_table.path_table_for_CAT->goals[location] < MAX_TIMESTEP) // target conflict
                insertSoftConstraint2SIT(location, constraint_table.path_table_for_CAT->goals[location], MAX_TIMESTEP + 1);
        }
        else // edge conflict
        {
            auto from = location / constraint_table.map_size - 1;
            auto to = location % constraint_table.map_size;
            if (from != to)
            {
                int t_max = (int) min(constraint_table.path_table_for_CAT->table[from].size(),
                                      constraint_table.path_table_for_CAT->table[to].size() + 1);
                for (int t = 1; t < t_max; t++)
                {
                    bool found = false;
                    for (auto a1 : constraint_table.path_table_for_CAT->table[to][t - 1])
                    {
                        for (auto a2: constraint_table.path_table_for_CAT->table[from][t])
                        {
                            if (a1 == a2)
                            {
                                insertSoftConstraint2SIT(location, t, t+1);
                                found = true;
                                break;
                            }
                            if (found)
                                break;
                        }
                    }
                }
            }
        }
    }

}

// return <upper_bound, low, high,  vertex collision, edge collision>--algorithm 2
/*#########################
% get_safe_intervals:
% Inputs:
%   - from, to: az induló és cél cella indexe.
%   - lower_bound, upper_bound: az intervallum, melyben érkezni kell (pl. curr->timestep+1-tól high_expansion+1-ig).
%
% Function behaviour:
%   - Végigiterál a 'sit[to]' listán, azaz a célcellához tartozó biztonságos intervallumokon.
%   - Csak azokat az intervallumokat veszi figyelembe, amelyek átfedést mutatnak a [lower_bound, upper_bound) időablakkal.
%   - Lekéri az adott intervallumon belüli legkorábbi érkezési időt (t1) a get_earliest_arrival_time függvény segítségével.
%   - Ha az intervallum soft vertex ütközést tartalmaz (get<2>(interval) == true), beállítja az edge collision flag-et is.
%   - Amennyiben az él mentén nincs ütközés (t1 == t2), az egész intervallumot átfogóan jelöli ütközésmentesnek.
%   - Ha részeredményként edge collision érzékelhető, két részre osztja az intervallumot.
%
% Outputs:
%   - Lista, mely minden eleme egy tuple:
%     <upper_bound, low (legkorábbi érkezés), high (legkésőbbi érkezés), vertex_collision flag, edge_collision flag>.
*/
list<tuple<int, int, int, bool, bool>> ReservationTable::get_safe_intervals(int from, int to, int lower_bound, int upper_bound)
{
    list<tuple<int, int, int, bool, bool>> rst;
    if (lower_bound >= upper_bound)  //lower_bound=curr_t+1
        return rst;

    if (sit[to].empty())
        updateSIT(to);
    // return all movable safe interval of next position (line 2-3 of algorithm 2)
    for(auto interval : sit[to])  // all safe interval at the location
    {
        if (lower_bound >= get<1>(interval))  // no intersection
            continue;
        else if (upper_bound <= get<0>(interval))
            break;
        // the interval overlaps with [lower_bound, upper_bound)
        auto t1 = get_earliest_arrival_time(from, to,
                max(lower_bound, get<0>(interval)), min(upper_bound, get<1>(interval)));
        if (t1 < 0) // the interval is not reachable
            continue;  // line9
        else if (get<2>(interval)) // the interval has soft vertix collisions
        {
            rst.emplace_back(get<1>(interval), t1, get<1>(interval), true, false);  //<upper_bound, low, high,  vertex collision, edge collision>
        }
        else // the interval does not have soft vertix collisions
        { // so we need to check the move action has collisions or not
            auto t2 = get_earliest_no_collision_arrival_time(from, to, interval, t1, upper_bound);
            if (t1 == t2)  // no edge collision and vertics collison during the whole interval
                rst.emplace_back(get<1>(interval), t1, get<1>(interval), false, false);  // line 17
            else if (t2 < 0)  // line 17   has edge collision
                rst.emplace_back(get<1>(interval), t1, get<1>(interval), false, true);
            else// 0<t2<t1, can reach early with collision t1 and reach later but without collision t2
            {
                rst.emplace_back(get<1>(interval), t1, t2, false, true);  //line12
                rst.emplace_back(get<1>(interval), t2, get<1>(interval), false, false); //line14
            }
        }
    }
    return rst;
}

/*#########################
% get_first_safe_interval:
% Inputs:
%   - location: A cella indexe.
%
% Function behaviour:
%   - Ha a cellához nincs még safe interval, hívja az updateSIT-t.
%   - Visszaadja a safe interval lista első elemét.
%
% Outputs:
%   - Az első (legkorábbi) safe interval az adott cellához.
*/
Interval ReservationTable::get_first_safe_interval(size_t location)
{
    if (sit[location].empty())  // if this location does not has safe interval
	    updateSIT(location);
    return sit[location].front();  // return the first element of list
}


// find a new safe interval with t_min as given from exit interval
/*#########################
% find_safe_interval:
% Inputs:
%   - interval (referenciaként): A megtalálandó safe interval kimeneti értéke.
%   - location: Az adott cella indexe.
%   - t_min: Az alsó időhatár, amelytől keresünk.
%
% Function behaviour:
%   - Végigiterál a safe interval listán a megadott cellához, és ha t_min benne van egy intervallumban,
%     beállítja az interval kimenetet, majd visszatér igaz értékkel.
%   - Ha nem talál megfelelő intervallumot, hamissal tér vissza.
%
% Outputs:
%   - Boolean érték, amely jelzi, hogy sikerült-e megtalálni a megfelelő safe intervallumot.
*/
bool ReservationTable::find_safe_interval(Interval& interval, size_t location, int t_min)
{
	if (t_min >= min(constraint_table.length_max, MAX_TIMESTEP - 1) + 1)
		return false;
    if (sit[location].empty())
	    updateSIT(location);
    for( auto & i : sit[location])
    {
        if ((int)get<0>(i) <= t_min && t_min < (int)get<1>(i))
        {
            interval = Interval(t_min, get<1>(i), get<2>(i));
            return true;
        }
        else if (t_min < (int)get<0>(i))
            break;
    }
    return false;
}

/*#########################
% get_earliest_arrival_time:
% Inputs:
%   - from, to: Kezdő és cél cellák.
%   - lower_bound, upper_bound: Az intervallum, melyben az érkezésnek meg kell történnie.
%
% Function behaviour:
%   - Iterál az időablakban és visszaadja a legkisebb olyan időpontot, amikor
%     a constraint_table koncepciója szerint nincs tilalom.
%
% Outputs:
%   - A legkorábbi érkezési idő (ha nincs, -1-et ad vissza).
*/
int ReservationTable::get_earliest_arrival_time(int from, int to, int lower_bound, int upper_bound) const
{
    for (auto t = lower_bound; t < upper_bound; t++)  // lower bound:the earlist step can reach the location;upper_bound: the maximum length can stay on the location
    {
        if (!constraint_table.constrained(from, to, t)) // no colision
            return t;
    }
    return -1;
}

/*#########################
% get_earliest_no_collision_arrival_time:
% Inputs:
%   - from, to: Kezdő és cél cellák.
%   - interval: Az aktuális safe interval.
%   - lower_bound, upper_bound: Az érkezési időablak.
%
% Function behaviour:
%   - Megkeresi azt a legkorábbi időpontot az adott intervallumban, amikor
%     az él mentén nincs ütközés (hasEdgeConflict).
%
% Outputs:
%   - A legkorábbi érkezési idő, ahol az él ütközésmentes; ha nem találja, -1-et ad vissza.
*/
int ReservationTable::get_earliest_no_collision_arrival_time(int from, int to, const Interval& interval,
                                                             int lower_bound, int upper_bound) const
{
    for (auto t = max(lower_bound, get<0>(interval)); t < min(upper_bound, get<1>(interval)); t++)
    {
        if (!constraint_table.hasEdgeConflict(from, to, t))
            return t;
    }
    return -1;
}