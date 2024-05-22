#include "ConstraintTable.h"

int ConstraintTable::getLastCollisionTimestep(int location) const
{
    int rst = -1;
    if (path_table_for_CAT != nullptr)
        rst = path_table_for_CAT->getLastCollisionTimestep(location);
    return rst;
}


bool ConstraintTable::constrained(size_t curr_loc, size_t next_loc, int next_t) const
{
    return false;
}

void ConstraintTable::copy(const ConstraintTable& other)
{
	length_min = other.length_min;
	length_max = other.length_max;
	num_col = other.num_col;
	map_size = other.map_size;
    path_table_for_CAT = other.path_table_for_CAT;
}


bool ConstraintTable::hasEdgeConflict(size_t curr_id, size_t next_id, int next_timestep) const
{
    assert(curr_id != next_id);
    if (path_table_for_CAT != nullptr and path_table_for_CAT->hasEdgeCollisions(curr_id, next_id, next_timestep))
        return true;
    return false;
}
int ConstraintTable::getFutureNumOfCollisions(int loc, int t) const
{
    int rst = 0;
    if (path_table_for_CAT != nullptr)
        rst = path_table_for_CAT->getFutureNumOfCollisions(loc, t);
    return rst;
}

// return the earliest timestep that the agent can hold the location
int ConstraintTable::getHoldingTime( int earliest_timestep) const
{
    // path table
    int rst = earliest_timestep;
	return rst;
}
