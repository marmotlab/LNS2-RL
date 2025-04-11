#include "common.h"

std::ostream& operator<<(std::ostream& os, const Path& path)
{
	for (const auto& state : path)
	{
		os << state.location << "\t"; // << "(" << state.is_single() << "),";
	}
	return os;
}
