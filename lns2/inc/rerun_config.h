#pragma once

// This header provides configuration for conditional Rerun usage

#ifndef RERUN_ENABLED
// Default to disabled if not defined by CMake
#define RERUN_ENABLED 0
#endif

// Conditional includes for Rerun
#if RERUN_ENABLED
#include <rerun/archetypes.hpp>
#include <rerun/components.hpp>
#include <rerun.hpp>
#endif

// Helper macros for conditional code
#if RERUN_ENABLED
#define RERUN_IF(x) x
#define RERUN_ELSE(x)
#else
#define RERUN_IF(x)
#define RERUN_ELSE(x) x
#endif
