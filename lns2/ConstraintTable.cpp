#include "ConstraintTable.h"

/*
% ConstraintTable::getLastCollisionTimestep:
%
% Inputs:
%   - location: A cella indexe.
%
% Behavior:
%   - Ha elérhető (path_table_for_CAT nem nullptr), visszaadja a
%     path_table_for_CAT->getLastCollisionTimestep(location) értéket.
%
% Outputs:
%   - Az utolsó ütközési (érkezési) időpont, vagy -1, ha nem található.
*/
int ConstraintTable::getLastCollisionTimestep(int location) const
{
    int rst = -1;
    if (path_table_for_CAT != nullptr)
        rst = path_table_for_CAT->getLastCollisionTimestep(location);
    return rst;
}

// - Jelenleg mindig false értékkel tér vissza  (nincs explicit constraint ellenőrzés).
bool ConstraintTable::constrained(size_t curr_loc, size_t next_loc, int next_t) const
{
    return false;
}

//  Másolja a constraint paramétereke
void ConstraintTable::copy(const ConstraintTable& other)
{
	length_min = other.length_min;
	length_max = other.length_max;
	num_col = other.num_col;
	map_size = other.map_size;
    path_table_for_CAT = other.path_table_for_CAT;
}

//  Ellenőrzi, hogy a path_table_for_CAT alapján élütközés van-e a két cella között.
bool ConstraintTable::hasEdgeConflict(size_t curr_id, size_t next_id, int next_timestep) const
{
    assert(curr_id != next_id);
    if (path_table_for_CAT != nullptr and path_table_for_CAT->hasEdgeCollisions(curr_id, next_id, next_timestep))
        return true;
    return false;
}

//  Visszaadja a path_table_for_CAT alapján a jövőbeni ütközések számát.
int ConstraintTable::getFutureNumOfCollisions(int loc, int t) const
{
    int rst = 0;
    if (path_table_for_CAT != nullptr)
        rst = path_table_for_CAT->getFutureNumOfCollisions(loc, t);
    return rst;
}

// return the earliest timestep that the agent can hold the location
// Jelenleg visszaadja az earliest_timestep értéket.
int ConstraintTable::getHoldingTime( int earliest_timestep) const
{
    // path table
    int rst = earliest_timestep;
	return rst;
}
