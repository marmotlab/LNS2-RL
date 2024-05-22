#pragma once
#include "common.h"
#include "PathTable.h"

class ConstraintTable
{
public:
	int length_min = 0;
	int length_max = MAX_TIMESTEP;  // changed by conflict, but did not be changed in LNS2
	size_t num_col;
	size_t map_size;
    const PathTableWC * path_table_for_CAT;  //  paths of all agents with collisions   // soft path table

	int getHoldingTime(int earliest_timestep) const; // the earliest timestep that the agent can hold the location after earliest_timestep
    int getLastCollisionTimestep(int location) const;
    // void clear(){ct.clear(); cat_small.clear(); cat_large.clear(); landmarks.clear(); length_min = 0, length_max = INT_MAX; latest_timestep = 0;}

    bool constrained(size_t curr_loc, size_t next_loc, int next_t) const;
    bool hasEdgeConflict(size_t curr_id, size_t next_id, int next_timestep) const;
    int getFutureNumOfCollisions(int loc, int t) const;

	ConstraintTable(size_t num_col, size_t map_size,
	        const PathTableWC * path_table_for_CAT = nullptr) :
            num_col(num_col), map_size(map_size),
            path_table_for_CAT(path_table_for_CAT) {}
	ConstraintTable(const ConstraintTable& other) { copy(other); }
    ~ConstraintTable() = default;

	void copy(const ConstraintTable& other);

protected:
    friend class ReservationTable;
};

