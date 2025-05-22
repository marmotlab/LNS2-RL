#!/usr/bin/env python3
"""
Drone Solution Benchmark - Compare performance of different path planning algorithms

This script benchmarks three algorithms:
1. Original LNS2 with static maps
2. TJPS algorithm with dynamic environment
3. Our dynamic drone solution with LNS2 extension and kinematic conversion

For each algorithm, we measure:
- Runtime (excluding visualization)
- Path lengths
- Success rate in resolving collisions
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import gc

# continuous
from continuous_visualization import visualize_continuous_space
visualize_continuous = visualize_continuous_space

# low-level discrete helpers
from mapf_env_test import visualize_paths, create_video_from_images

# ——— adapter: full‐map “visualize_solution” wrapper ———
def visualize_solution(paths, grid_seq, start_list, goal_list, output_dir, fps=10):
    """
    Dump one PNG per timestep via visualize_paths, then
    call create_video_from_images(...) to make agent_movements.gif.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # figure out how many frames we need
    max_t = max(len(p) for p in paths)
    for t in range(max_t):
        img_path = os.path.join(output_dir, f"agent_movements_t{t:03d}.png")
        visualize_paths(
            grid_seq,
            start_list,
            goal_list,
            paths,
            t,
            img_path
        )
    # now build the GIF
    create_video_from_images(
        output_dir,
        'agent_movements_t',
        os.path.join(output_dir, 'agent_movements.gif'),
        fps=fps 
    )
    print(f"✔ Discrete GIF saved to {output_dir}/agent_movements.gif")

# Add necessary paths
sys.path.append("GridBasedPathPlanning")
sys.path.append("GridBasedPathPlanning/PathPlanning_main")
sys.path.append("multi_agent_path_planning/centralized")

# Import required modules
from GridBasedPathPlanning.Environment.GridMapEnv import GridMapEnv
from GridBasedPathPlanning.Plan.TJPS import TJPS
from mapf_env_test import create_test_env, test_mapf_env
from dynamic_drone_solution import (
    step1_generate_initial_paths,
    step2_convert_to_schedule,
    step3_build_temporal_plan_graph,
    step4_construct_simple_temporal_network,
    step5_optimize_temporal_plan,
    step6_generate_final_trajectories,
    UAV_PARAMS,
    GRID_DT,
    RADAR_ZONE_SPEED_FACTOR
)

#—––– Our “local” LNS2 implementation —–––––––––––––––––––––––––––
# we assume it lives in lns2rl/lns2/build/my_lns2.py
sys.path.insert(0, "/lns2rl")
try:
    from lns2.build import my_lns2 as OurMyLns2
except ImportError as e:
    print(f"Warning: could not load local LNS2 at lns2rl/lns2: {e}")

#—––– The “original” upstream LNS2 implementation —––––––––––––––
# lives under lns2rl/original_lns2rl/LNS2-RL/build/my_lns2.py
sys.path.insert(0, "/lns2rl/original_lns2rl/LNS2-RL")
try:
   from lns2.build import my_lns2 as OriginalMyLns2
except ImportError as e:
    print(f"Warning: could not load original LNS2 at original_lns2rl/LNS2-RL: {e}")

# Create symbolic link for CBS module if needed
try:
    # Check if we're in Docker or local environment
    if os.path.exists('/lns2rl'):
        # Docker environment
        base_path = '/lns2rl'
    else:
        # Local environment
        base_path = '/home/ubuntu/LNS2-RL'

    # Create the CBS symbolic link
    cbs_path = os.path.join(base_path, 'multi_agent_path_planning/centralized/cbs')
    cbs_symlink_path = os.path.join(base_path, 'multi_agent_path_planning/centralized/scheduling/cbs')

    # Also try to create a direct link to the cbs.py file
    cbs_file_path = os.path.join(cbs_path, 'cbs.py')
    cbs_file_symlink_path = os.path.join(cbs_symlink_path, 'cbs.py')

    # Create directory if it doesn't exist
    if not os.path.exists(cbs_symlink_path):
        os.makedirs(cbs_symlink_path, exist_ok=True)
        print(f"Created directory: {cbs_symlink_path}")

    # Create symbolic links
    if os.path.exists(cbs_path) and not os.path.exists(os.path.join(cbs_symlink_path, 'cbs.py')):
        print(f"Creating symbolic link from {cbs_file_path} to {cbs_file_symlink_path}")
        if os.path.exists(cbs_file_path):
            os.symlink(cbs_file_path, cbs_file_symlink_path)
            print(f"Created symbolic link for cbs.py")
        else:
            print(f"Warning: cbs.py not found at {cbs_file_path}")
except Exception as e:
    print(f"Warning: Could not create CBS symbolic link: {e}")

# Import start_end_points function for consistent agent start/goal generation
from GridBasedPathPlanning.RL_Environment.environment.map_generator import start_end_points
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import GridToRGB
import re

# Constants
MAX_ITERATIONS = 10  # Maximum iterations for replanning
BENCHMARK_OUTPUT_DIR = "benchmark_results"

# Make sure UAV_PARAMS is properly defined
if 'UAV_PARAMS' not in globals():
    # Define UAV parameters if not already defined
    UAV_PARAMS = {
        'v_max': 5.0,         # Maximum velocity (grid cells per second)
        'v_min': 0.5,         # Minimum velocity (grid cells per second)
        'omega_max': 0.5,     # Maximum turn rate (rad/s)
        'a_max': 2.0,         # Maximum acceleration (grid cells per second^2)
        'mass': 1.0,          # Mass (kg)
        'F_max': 10.0,        # Maximum force (N)
        'safety_distance': 2.0,  # Minimum distance between UAVs (grid cells)
        'radar_zone_cost': 3.0   # Cost multiplier for passing through radar zones
    }

def calculate_path_length(path):
    """Calculate the length of a path in grid cells"""
    # Handle TJPS path format (dictionary with 'path_extracted' key)
    if isinstance(path, dict) and 'path_extracted' in path:
        path = path['path_extracted']
        if isinstance(path, np.ndarray):
            path = path.tolist()

    # Handle empty or too short paths
    if not path or not hasattr(path, '__len__') or len(path) < 2:
        return 0

    # Special case for numpy arrays
    if isinstance(path, np.ndarray):
        if path.ndim == 2 and path.shape[1] >= 2:
            # Format: numpy array of shape (n, 2) or more
            length = 0
            for i in range(1, len(path)):
                length += abs(path[i, 0] - path[i-1, 0]) + abs(path[i, 1] - path[i-1, 1])
            return length

    # Try to get the length of the path
    try:
        path_length = len(path)
    except (TypeError, AttributeError):
        print(f"Warning: Path has no length attribute: {type(path)}")
        return 0

    length = 0
    for i in range(1, path_length):
        try:
            # Try different path formats
            if isinstance(path[i], tuple) and isinstance(path[i-1], tuple):
                # Format: [(x1, y1), (x2, y2), ...]
                x1, y1 = path[i-1]
                x2, y2 = path[i]
            elif isinstance(path[i], dict) and isinstance(path[i-1], dict):
                # Format: [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}, ...]
                x1, y1 = path[i-1]['x'], path[i-1]['y']
                x2, y2 = path[i]['x'], path[i]['y']
            elif isinstance(path[i], list) and isinstance(path[i-1], list):
                # Format: [[x1, y1], [x2, y2], ...] or [[x1, y1, t1], [x2, y2, t2], ...]
                if len(path[i-1]) == 2:
                    x1, y1 = path[i-1]
                    x2, y2 = path[i]
                elif len(path[i-1]) >= 3:  # Handle case with time or other values
                    x1, y1 = path[i-1][0], path[i-1][1]
                    x2, y2 = path[i][0], path[i][1]
                else:
                    raise ValueError(f"List format not recognized: {path[i-1]}")
            elif hasattr(path[i], 'x') and hasattr(path[i], 'y'):
                # Format: [obj1, obj2, ...] where obj has x,y attributes
                x1, y1 = path[i-1].x, path[i-1].y
                x2, y2 = path[i].x, path[i].y
            else:
                # Unknown format, try to access by index
                x1, y1 = path[i-1][0], path[i-1][1]
                x2, y2 = path[i][0], path[i][1]

            # Calculate Manhattan distance between consecutive points
            length += abs(x2 - x1) + abs(y2 - y1)
        except (IndexError, KeyError, TypeError, AttributeError) as e:
            print(f"Warning: Error calculating path length at index {i}: {e}")
            try:
                print(f"Path format: {type(path)}, First element type: {type(path[0]) if len(path) > 0 else 'empty'}")
            except:
                print(f"Path format: {type(path)}, Cannot access elements")
            # Continue with next point
            continue

    return length

def calculate_trajectory_length(trajectory):
    """Calculate the length of a continuous trajectory"""
    if not trajectory or len(trajectory.waypoints) < 2:
        return 0

    length = 0
    for i in range(1, len(trajectory.waypoints)):
        # Calculate Euclidean distance between consecutive points
        p1 = trajectory.waypoints[i-1]
        p2 = trajectory.waypoints[i]
        length += np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

    return length

def benchmark_original_lns2(static_map_path, num_agents, output_dir=None):
    """Benchmark the original LNS2 algorithm with static maps"""
    print("\n===== Benchmarking Original LNS2 =====")

    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load the static map (either .map text or image) via our loader
    try:
        grid_array = load_map(static_map_path)
        if grid_array is None:
            raise ValueError(f"Could not load map from {static_map_path}")

        height, width = grid_array.shape
        grid = GridMapEnv(grid_size=(height, width))
        # convert_nonzero_to_inf will treat any non-zero cell as obstacle
        grid.SetStaticGrid(grid_static=grid_array, convert_nonzero_to_inf=True)

        # Generate random start and goal positions
        start_list = []
        goal_list = []

        # Calculate minimum Manhattan distance
        min_manhattan_distance = height // 2
        print(f"Using minimum Manhattan distance of {min_manhattan_distance} between start and goal positions")

        # Helper function to calculate Manhattan distance
        def manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        # Helper function to check if a position is valid
        def is_valid_position(pos, positions_list):
            x, y = pos
            # Check if position is within bounds
            if x < 0 or x >= height or y < 0 or y >= width:
                return False
            # Check if position is not an obstacle in static map
            if img[x, y] > 0:
                return False
            # Check if position is not already occupied by another agent
            if pos in positions_list:
                return False
            return True

        # Generate start positions
        for _ in range(num_agents):
            attempts = 0
            while attempts < 1000:
                pos = (np.random.randint(0, height), np.random.randint(0, width))
                if is_valid_position(pos, start_list):
                    start_list.append(pos)
                    break
                attempts += 1
            if attempts >= 1000:
                print(f"Warning: Could not find valid start position after 1000 attempts")
                return None, None, None

        # Generate goal positions with minimum Manhattan distance
        for i in range(num_agents):
            start_pos = start_list[i]
            attempts = 0
            found_valid_goal = False

            while attempts < 2000 and not found_valid_goal:
                pos = (np.random.randint(0, height), np.random.randint(0, width))

                if is_valid_position(pos, goal_list) and manhattan_distance(start_pos, pos) >= min_manhattan_distance:
                    goal_list.append(pos)
                    found_valid_goal = True
                    print(f"Agent {i}: Start {start_pos}, Goal {pos}, Manhattan distance: {manhattan_distance(start_pos, pos)}")
                    break

                attempts += 1

            if not found_valid_goal:
                print(f"Warning: Could not find valid goal position with minimum Manhattan distance after 2000 attempts")
                return None, None, None

        # Create static grid for LNS2
        static_grid = np.zeros((height, width), dtype=np.float32)
        for x in range(height):
            for y in range(width):
                if img[x, y] > 0:
                    static_grid[x, y] = 1  # Obstacle

        # Initialize LNS2
        print("Initializing original LNS2...")
        seed = 42
        start_time = time.time()

        # Create a 3D grid sequence with just the static map repeated
        grid_seq = np.zeros((100, height, width), dtype=np.float32)
        for t in range(100):
            grid_seq[t] = static_grid

        # Initialize LNS2 model
        lns2_model = MyLns2(seed, grid_seq, start_list, goal_list, num_agents, height)
        lns2_model.setUseRerun(False)  # Disable visualization
        lns2_model.init_pp()

        # Get initial paths
        paths = lns2_model.vector_path

        # Iterative conflict resolution
        iteration = 0
        num_collisions = lns2_model.calculate_sipps(list(range(num_agents)))
        initial_collisions = num_collisions

        print(f"Initial planning complete with {num_collisions} collisions")

        while num_collisions > 0 and iteration < MAX_ITERATIONS:
            iteration += 1
            print(f"Iteration {iteration}/{MAX_ITERATIONS}")

            # Replan for all agents
            agents_to_replan = list(range(num_agents))
            num_collisions = lns2_model.calculate_sipps(agents_to_replan)

            # Get updated paths
            paths = lns2_model.vector_path

            print(f"Replanning complete with {num_collisions} collisions")

            if num_collisions == 0:
                print("All collisions resolved!")
                break

        # Calculate total runtime
        total_runtime = time.time() - start_time

        # Calculate path lengths
        path_lengths = [calculate_path_length(path) for path in paths]

        # Calculate success rate
        success_rate = 1.0 if num_collisions == 0 else 1.0 - (num_collisions / initial_collisions)

        print(f"Original LNS2 benchmark completed in {total_runtime:.2f} seconds")
        print(f"Path lengths: {path_lengths}")
        print(f"Success rate: {success_rate:.2f}")

        # Return results
        return {
            "algorithm": "Original LNS2",
            "runtime": total_runtime,
            "path_lengths": path_lengths,
            "avg_path_length": np.mean(path_lengths),
            "success_rate": success_rate,
            "iterations": iteration,
            "final_collisions": num_collisions
        }

    except Exception as e:
        print(f"Error in original LNS2 benchmark: {e}")
        return None

def benchmark_tjps(static_map_path, num_agents, num_obstacles, output_dir=None):
    """Benchmark the TJPS algorithm with dynamic environment"""
    print("\n===== Benchmarking TJPS Algorithm =====")

    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Create test environment with dynamic obstacles
        print("Creating test environment...")
        grid, grid_seq, start_list, goal_list = create_test_env(
            static_map_path=static_map_path,
            num_agents=num_agents,
            num_dynamic_obstacles=num_obstacles,
            frames=1000,
            output_dir=output_dir,
            visualize=False  # Disable visualization
        )

        # Benchmark TJPS for each agent
        start_time = time.time()
        tjps_paths = []
        tjps_runtimes = []
        tjps_success = []

        for agent_id in range(num_agents):
            print(f"Planning for agent {agent_id}...")
            agent_start_time = time.time()

            # Initialize TJPS planner
            planner = TJPS(
                grid_seq,
                start_list[agent_id],
                goal_list[agent_id],
                obs_threshold=0.1,  # Threshold for considering a cell as an obstacle
                goal_threshold=1,   # Distance threshold to goal
                max_wait=10,        # Maximum wait time
                max_execution_time=60  # Maximum execution time in seconds
            )

            # Run TJPS
            path, success = planner.plan()

            # Record results
            agent_runtime = time.time() - agent_start_time
            tjps_runtimes.append(agent_runtime)
            tjps_success.append(success)

            if success:
                tjps_paths.append(path)
                print(f"Agent {agent_id} planning successful in {agent_runtime:.2f} seconds")
            else:
                tjps_paths.append([])
                print(f"Agent {agent_id} planning failed after {agent_runtime:.2f} seconds")

        # Calculate total runtime
        total_runtime = time.time() - start_time

        # Calculate path lengths for successful paths
        path_lengths = [calculate_path_length(path) for path in tjps_paths if path]

        # Calculate success rate
        success_rate = sum(tjps_success) / num_agents

        print(f"TJPS benchmark completed in {total_runtime:.2f} seconds")
        print(f"Path lengths: {path_lengths}")
        print(f"Success rate: {success_rate:.2f}")

        # Return results
        return {
            "algorithm": "TJPS",
            "runtime": total_runtime,
            "agent_runtimes": tjps_runtimes,
            "path_lengths": path_lengths,
            "avg_path_length": np.mean(path_lengths) if path_lengths else 0,
            "success_rate": success_rate,
            "iterations": 1,  # TJPS doesn't do iterations
            "final_collisions": num_agents - sum(tjps_success)
        }

    except Exception as e:
        print(f"Error in TJPS benchmark: {e}")
        return None

def benchmark_dynamic_drone_solution(static_map_path, num_agents, num_obstacles, output_dir=None):
    """Benchmark our dynamic drone solution with LNS2 extension and kinematic conversion"""
    print("\n===== Benchmarking Dynamic Drone Solution =====")

    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Generate initial paths
        print("Step 1: Generating initial paths...")
        step1_start_time = time.time()
        paths, grid_seq, start_list, goal_list = step1_generate_initial_paths(
            static_map_path,
            num_agents,
            num_obstacles,
            output_dir
        )
        step1_time = time.time() - step1_start_time
        print(f"Step 1 completed in {step1_time:.2f} seconds")

        # Calculate initial path lengths
        initial_path_lengths = [calculate_path_length(path) for path in paths]
        print(f"Initial path lengths: {initial_path_lengths}")

        # Step 2: Convert to schedule
        print("Step 2: Converting to schedule...")
        step2_start_time = time.time()
        schedule = step2_convert_to_schedule(paths)
        step2_time = time.time() - step2_start_time
        print(f"Step 2 completed in {step2_time:.2f} seconds")

        # Step 3: Build Temporal Plan Graph
        print("Step 3: Building Temporal Plan Graph...")
        step3_start_time = time.time()
        tpg = step3_build_temporal_plan_graph(schedule, UAV_PARAMS)
        step3_time = time.time() - step3_start_time
        print(f"Step 3 completed in {step3_time:.2f} seconds")

        # Step 4: Construct Simple Temporal Network
        print("Step 4: Constructing Simple Temporal Network...")
        step4_start_time = time.time()

        # Extract dynamic obstacles from the grid sequence
        dynamic_obstacles = {}
        if grid_seq is not None and len(grid_seq) > 1:
            for t in range(len(grid_seq)):
                grid = grid_seq[t]
                dynamic_mask = np.logical_and(grid > 0.05, grid < 0.95)
                obstacle_positions = np.where(dynamic_mask)

                for i in range(len(obstacle_positions[0])):
                    x, y = obstacle_positions[0][i], obstacle_positions[1][i]
                    obstacle_id = f"obstacle_{x}_{y}"

                    if obstacle_id not in dynamic_obstacles:
                        dynamic_obstacles[obstacle_id] = []

                    dynamic_obstacles[obstacle_id].append({
                        'x': float(x),
                        'y': float(y),
                        't': float(t) * GRID_DT * RADAR_ZONE_SPEED_FACTOR
                    })

        stn = step4_construct_simple_temporal_network(tpg, UAV_PARAMS, dynamic_obstacles)
        step4_time = time.time() - step4_start_time
        print(f"Step 4 completed in {step4_time:.2f} seconds")

        # Step 5: Optimize Temporal Plan
        print("Step 5: Optimizing Temporal Plan...")
        step5_start_time = time.time()
        solution = step5_optimize_temporal_plan(stn)
        step5_time = time.time() - step5_start_time
        print(f"Step 5 completed in {step5_time:.2f} seconds")

        # Step 6: Generate Final Trajectories
        print("Step 6: Generating Final Trajectories...")
        step6_start_time = time.time()
        trajectories = step6_generate_final_trajectories(
                solution,              # schedule
                UAV_PARAMS,            # uav_params
                dynamic_obstacles,     # dynamic_obstacles
                grid_seq               # grid_seq
            )
        step6_time = time.time() - step6_start_time
        print(f"Step 6 completed in {step6_time:.2f} seconds")

        # Calculate trajectory lengths
        trajectory_lengths = [calculate_trajectory_length(traj) for traj in trajectories]
        print(f"Final trajectory lengths: {trajectory_lengths}")

        # Calculate total runtime (excluding visualization)
        total_runtime = step1_time + step2_time + step3_time + step4_time + step5_time + step6_time

        # Calculate success rate (based on whether all trajectories were generated successfully)
        success_rate = len(trajectories) / num_agents

        print(f"Dynamic Drone Solution benchmark completed in {total_runtime:.2f} seconds")
        print(f"Trajectory lengths: {trajectory_lengths}")
        print(f"Success rate: {success_rate:.2f}")

        # Return results
        return {
            "algorithm": "Dynamic Drone Solution",
            "runtime": total_runtime,
            "step_runtimes": {
                "step1": step1_time,
                "step2": step2_time,
                "step3": step3_time,
                "step4": step4_time,
                "step5": step5_time,
                "step6": step6_time
            },
            "initial_path_lengths": initial_path_lengths,
            "avg_initial_path_length": np.mean(initial_path_lengths),
            "trajectory_lengths": trajectory_lengths,
            "avg_trajectory_length": np.mean(trajectory_lengths),
            "success_rate": success_rate
        }

    except Exception as e:
        print(f"Error in Dynamic Drone Solution benchmark: {e}")
        return None

def load_map(map_path):
    """Load a map from file and return as a grid"""
    try:
        # Check if the map is a .map file (text format) or an image
        if map_path.endswith('.map'):
            # Load .map file (text format)
            with open(map_path, 'r') as f:
                lines = f.readlines()

            # Parse header
            height = None
            width = None
            for line in lines:
                if line.startswith('height'):
                    height = int(line.split()[1])
                elif line.startswith('width'):
                    width = int(line.split()[1])
                if height is not None and width is not None:
                    break

            if height is None or width is None:
                raise ValueError("Could not parse map dimensions from .map file")

            # Parse map data
            map_data = []
            for line in lines:
                if line.startswith('.') or line.startswith('@') or line.startswith('T') or line.startswith('S'):
                    row = []
                    for char in line.strip():
                        if char == '.':  # Traversable
                            row.append(0)
                        else:  # Obstacle
                            row.append(1)
                    map_data.append(row)

            if len(map_data) != height:
                print(f"Warning: Map height ({len(map_data)}) doesn't match header ({height})")

            # Convert to numpy array
            grid = np.array(map_data, dtype=np.float32)
        else:
            # Load image file
            img = np.array(Image.open(map_path).convert('L'))
            grid = (img > 0).astype(np.float32)  # Convert to binary grid

        return grid
    except Exception as e:
        print(f"Error loading map: {e}")
        return None

def create_shared_environment(map_path, num_agents, num_obstacles, frames=1000):
    """Create a shared environment for all benchmarks to use"""
    print("\n===== Creating Shared Environment =====")

    try:
        # Load static grid: either .map text, image, or .npy
        if map_path.endswith(".npy"):
            static_grid = np.load(map_path)
            print(f"Loaded static grid from .npy with shape {static_grid.shape}")
        else:
            # existing loader for .map or image
            static_grid = load_map(map_path)
            if static_grid is None:
                raise ValueError(f"Could not load static map from {map_path}")
            print(f"Loaded static grid from {map_path} with shape {static_grid.shape}")

        height, width = static_grid.shape
        print(f"Loaded map with dimensions {height}x{width}")

        # Create GridMapEnv
        grid = GridMapEnv(grid_size=(height, width))
        grid.SetStaticGrid(grid_static=static_grid, convert_nonzero_to_inf=True)

        # Generate start and goal positions using start_end_points
        # This ensures the same positions are used for all benchmarks
        min_manhattan_dist = height // 2  # Use half the map height as minimum distance
        print(f"Using minimum Manhattan distance of {min_manhattan_dist} between start and goal positions")

        # Generate start positions first
        start_coords = []
        for _ in range(num_agents):
            start_pos = grid.GetRandomFreeCell()
            start_coords.append(start_pos)

        # Use start_end_points to generate end positions with minimum distance
        agent_positions = start_end_points(start_coords, grid, width=width, min_manhattan_dist=min_manhattan_dist)

        if agent_positions is None:
            raise ValueError("Failed to generate valid start/goal positions")

        # Extract start and goal positions
        start_list = []
        goal_list = []

        for i in range(0, len(agent_positions), 2):
            start_list.append(tuple(agent_positions[i]))
            goal_list.append(tuple(agent_positions[i+1]))

        print(f"Generated {len(start_list)} agent start/goal pairs")
        for i in range(len(start_list)):
            print(f"Agent {i}: Start {start_list[i]}, Goal {goal_list[i]}")

        # Add dynamic obstacles
        grid.addRandomDynamicObstacles(no_of_obs=num_obstacles, d_range=(46, 70), alpha=0.2)

        # Generate grid sequence with dynamic obstacles
        grid_seq = grid.GenerateGridSequence(frames, reset_after=True)

        # Create a static grid sequence (no dynamic obstacles) for original LNS2
        static_grid_seq = np.zeros((frames, height, width), dtype=np.float32)
        for t in range(frames):
            # Copy only the static obstacles (value 1.0)
            static_grid_seq[t] = (static_grid > 0).astype(np.float32)

        return grid, grid_seq, static_grid_seq, start_list, goal_list

    except Exception as e:
        print(f"Error creating shared environment: {e}")
        return None, None, None, None, None

def save_benchmark_results(results, output_dir, start_list=None, goal_list=None):
    """Save benchmark results to JSON and generate plots"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Calculate Manhattan distance ground truth for each agent
    manhattan_distances = []
    if start_list and goal_list:
        for i in range(len(start_list)):
            start = start_list[i]
            goal = goal_list[i]
            manhattan_dist = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
            manhattan_distances.append(manhattan_dist)
        avg_manhattan_dist = np.mean(manhattan_distances)
        print(f"Average Manhattan distance (ground truth): {avg_manhattan_dist:.2f}")

    # Create enhanced results with additional metrics
    enhanced_results = []
    for result in results:
        enhanced_result = {k: v for k, v in result.items() if k not in ['paths', 'trajectories']}

        # Add algorithm type
        if "Original LNS2" in result["algorithm"]:
            enhanced_result["algorithm_type"] = "Original LNS2 mean"
        elif "TJPS" in result["algorithm"]:
            enhanced_result["algorithm_type"] = "TJPS mean"
        elif "Dynamic Drone Solution" in result["algorithm"] or "Dynamic Drone (Before Kinematic)" in result["algorithm"]:
            enhanced_result["algorithm_type"] = "kinematic_dynamic mean"

        # Calculate path length ratio compared to Manhattan distance
        if manhattan_distances:
            if "Original LNS2" in result["algorithm"] or "TJPS" in result["algorithm"]:
                if "path_lengths" in result and result["path_lengths"]:
                    path_lengths = result["path_lengths"]
                    path_length_ratios = []
                    for i, length in enumerate(path_lengths):
                        if i < len(manhattan_distances) and manhattan_distances[i] > 0:
                            ratio = length / manhattan_distances[i]
                            path_length_ratios.append(ratio)

                    if path_length_ratios:
                        enhanced_result["avg_path_length_ratio"] = np.mean(path_length_ratios)
                        enhanced_result["path_length_ratios"] = path_length_ratios

            elif "Dynamic Drone Solution" in result["algorithm"] or "Dynamic Drone (Before Kinematic)" in result["algorithm"]:
                # For initial paths
                if "initial_path_lengths" in result and result["initial_path_lengths"]:
                    path_lengths = result["initial_path_lengths"]
                    path_length_ratios = []
                    for i, length in enumerate(path_lengths):
                        if i < len(manhattan_distances) and manhattan_distances[i] > 0:
                            ratio = length / manhattan_distances[i]
                            path_length_ratios.append(ratio)

                    if path_length_ratios:
                        enhanced_result["avg_initial_path_length_ratio"] = np.mean(path_length_ratios)
                        enhanced_result["initial_path_length_ratios"] = path_length_ratios

                # For trajectories
                if "trajectory_lengths" in result and result["trajectory_lengths"]:
                    traj_lengths = result["trajectory_lengths"]
                    traj_length_ratios = []
                    for i, length in enumerate(traj_lengths):
                        if i < len(manhattan_distances) and manhattan_distances[i] > 0:
                            ratio = length / manhattan_distances[i]
                            traj_length_ratios.append(ratio)

                    if traj_length_ratios:
                        enhanced_result["avg_trajectory_length_ratio"] = np.mean(traj_length_ratios)
                        enhanced_result["trajectory_length_ratios"] = traj_length_ratios

        # Add per-agent metrics
        if "agent_runtimes" in result:
            enhanced_result["min_agent_runtime"] = min(result["agent_runtimes"])
            enhanced_result["max_agent_runtime"] = max(result["agent_runtimes"])

        # Add step-by-step metrics for Dynamic Drone Solution
        if "step_runtimes" in result:
            enhanced_result["step_runtimes"] = result["step_runtimes"]
            enhanced_result["planning_time"] = result["step_runtimes"].get("step1", 0)
            enhanced_result["conversion_time"] = sum([
                result["step_runtimes"].get("step2", 0),
                result["step_runtimes"].get("step3", 0),
                result["step_runtimes"].get("step4", 0),
                result["step_runtimes"].get("step5", 0),
                result["step_runtimes"].get("step6", 0)
            ])

        enhanced_results.append(enhanced_result)

    # Add ground truth metrics
    benchmark_metadata = {
        "manhattan_distances": manhattan_distances,
        "avg_manhattan_distance": avg_manhattan_dist if 'avg_manhattan_dist' in locals() else None,
        "num_agents": len(start_list) if start_list else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Create the final results dictionary
    final_results = {
        "metadata": benchmark_metadata,
        "results": enhanced_results
    }

    # Save to JSON
    import json
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f"Benchmark results saved to {json_path}")

    # Create a DataFrame for plotting
    df = pd.DataFrame(enhanced_results)

    # Generate runtime comparison plot
    plt.figure(figsize=(10, 6))
    plt.bar(df["algorithm"], df["runtime"])
    plt.title("Algorithm Runtime Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_comparison.png"))

    # Generate path length comparison plot
    plt.figure(figsize=(10, 6))
    labels      = [r["algorithm"] for r in enhanced_results]
    avg_lengths = []
    for r in enhanced_results:
        if "path_lengths" in r:
            avg_lengths.append(r["avg_path_length"])
        elif "initial_path_lengths" in r:
            # before kinematics  
            avg_lengths.append(
                r.get("avg_initial_path_length",
                      r.get("avg_path_length", 0))
            )
        else:
            # after kinematics  
            avg_lengths.append(r["avg_trajectory_length"])

    plt.bar(labels, avg_lengths)
    plt.title("Average Path Length Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Path Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "path_length_comparison.png"))
 
    # Generate path length ratio comparison plot (if available)
    if any("path_length_ratios" in r or
            "initial_path_length_ratios" in r or
            "trajectory_length_ratios" in r
            for r in enhanced_results):
        plt.figure(figsize=(10, 6))
        labels     = [r["algorithm"] for r in enhanced_results]
        avg_ratios = []
        for r in enhanced_results:
            if "path_length_ratios" in r:
                avg_ratios.append(r["avg_path_length_ratio"])
            elif "initial_path_length_ratios" in r:
                avg_ratios.append(
                    r.get("avg_initial_path_length_ratio",
                          r.get("avg_path_length_ratio", 0))
                )
            else:
                avg_ratios.append(r["avg_trajectory_length_ratio"])

        plt.bar(labels, avg_ratios)
        plt.title("Path Length Ratio Comparison (vs Manhattan Distance)")
        plt.xlabel("Algorithm")
        plt.ylabel("Average Path Length Ratio")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "path_length_ratio_comparison.png"))

    # Generate success rate comparison plot
    plt.figure(figsize=(10, 6))
    plt.bar(df["algorithm"], df["success_rate"])
    plt.title("Success Rate Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate_comparison.png"))

    print(f"Benchmark plots saved to {output_dir}")


def main():

    parser = argparse.ArgumentParser(description="Benchmark different path planning algorithms")
    parser.add_argument("--map", type=str, default="GridBasedPathPlanning/Data/BenchmarkMaps/Paris_1_256.map", help="Path to the static map")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--obstacles", type=int, default=15, help="Number of dynamic obstacles")
    parser.add_argument("--output", type=str, default="drone_solution_test", help="Output directory")
    parser.add_argument("--algorithms", type=str, default="all", help="Algorithms to benchmark (comma-separated: original,tjps,dynamic,all)")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of results")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for visualization GIFs (1 for slow, detailed viewing)")

    args = parser.parse_args()

    print(f"Benchmarking with map: {args.map}")
    print(f"Number of agents: {args.agents}")
    print(f"Number of dynamic obstacles: {args.obstacles}")
    print(f"Output directory: {args.output}")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create visualization directories if needed
    if args.visualize:
        os.makedirs(os.path.join(args.output, "original_lns2"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "tjps"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "dynamic_drone"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "kinematic_plan"), exist_ok=True)

    # Determine which algorithms to benchmark
    algorithms = args.algorithms.lower().split(",")
    run_all = "all" in algorithms

    # Store benchmark results
    benchmark_results = []

    # Create a shared environment for all benchmarks
    print("\n===== Creating shared environment for all benchmarks =====")
    grid, grid_seq, static_grid_seq, start_list, goal_list = create_shared_environment(
        args.map, args.agents, args.obstacles, frames=1000
    )

    if grid is None or grid_seq is None or start_list is None or goal_list is None:
        print("Failed to create shared environment. Exiting.")
        return

    # Run benchmarks
    if run_all or "original" in algorithms:
        # For original LNS2, use the same static map and start/goal positions
        # but without dynamic obstacles
        print("\n===== Benchmarking Original LNS2 =====")
        try:
            # Initialize original LNS2
            seed = 42
            start_time = time.time()

            # Initialize LNS2 model with static grid sequence
            lns2_model = OriginalMyLns2.MyLns2(seed, static_grid_seq, start_list, goal_list, args.agents, grid_seq.shape[1])
            lns2_model.setUseRerun(False)  # Disable visualization
            lns2_model.init_pp()

            # Get initial paths
            paths = lns2_model.vector_path

            # Iterative conflict resolution
            iteration = 0
            num_collisions = lns2_model.calculate_sipps(list(range(args.agents)))
            initial_collisions = num_collisions

            print(f"Initial planning complete with {num_collisions} collisions")

            while num_collisions > 0 and iteration < MAX_ITERATIONS:
                iteration += 1
                print(f"Iteration {iteration}/{MAX_ITERATIONS}")

                # Replan for all agents
                agents_to_replan = list(range(args.agents))
                num_collisions = lns2_model.calculate_sipps(agents_to_replan)

                # Get updated paths
                paths = lns2_model.vector_path

                print(f"Replanning complete with {num_collisions} collisions")

                if num_collisions == 0:
                    print("All collisions resolved!")
                    break

            # Calculate total runtime
            total_runtime = time.time() - start_time

            # Calculate path lengths
            path_lengths = [calculate_path_length(path) for path in paths]

            # Calculate success rate
            success_rate = 1.0 if num_collisions == 0 else 1.0 - (num_collisions / initial_collisions)

            print(f"Original LNS2 benchmark completed in {total_runtime:.2f} seconds")
            print(f"Path lengths: {path_lengths}")
            print(f"Success rate: {success_rate:.2f}")

            # Visualize discrete solution with our new helper
            if args.visualize:
                vis_dir = os.path.join(args.output, "original_lns2")
                os.makedirs(vis_dir, exist_ok=True)
                visualize_solution(
                    paths,
                    grid_seq,
                    start_list,
                    goal_list,
                    output_dir=vis_dir
                )
                print(f"✔ Original LNS2 discrete GIF in {vis_dir}/agent_movements.gif")

            # Add results
            benchmark_results.append({
                "algorithm": "Original LNS2",
                "runtime": total_runtime,
                "path_lengths": path_lengths,
                "avg_path_length": np.mean(path_lengths),
                "success_rate": success_rate,
                "iterations": iteration,
                "final_collisions": num_collisions,
                "paths": paths  # Store paths for later comparison
            })

        except Exception as e:
            print(f"Error in original LNS2 benchmark: {e}")

        finally:
        # Clean up EVERYTHING that came from static_grid_seq, lns2_model, paths, etc.
            del lns2_model, paths
            # static_grid_seq and grid_seq are reused by other benchmarks, do NOT delete those here
            gc.collect()

    if run_all or "tjps" in algorithms:
        # Use the shared environment for TJPS
        print("\n===== Benchmarking TJPS Algorithm =====")
        try:
            # Benchmark TJPS for each agent
            start_time = time.time()
            tjps_paths = []
            tjps_runtimes = []
            tjps_success = []

            for agent_id in range(args.agents):
                print(f"Planning for agent {agent_id}...")
                agent_start_time = time.time()

                # Initialize TJPS planner
                planner = TJPS(
                    grid_seq,
                    start_list[agent_id],
                    goal_list[agent_id],
                    obs_threshold=0.1,  # Threshold for considering a cell as an obstacle
                    goal_threshold=1,   # Distance threshold to goal
                    max_wait=10,        # Maximum wait time
                    max_execution_time=60  # Maximum execution time in seconds
                )

                # Run TJPS
                try:
                    result = planner.plan()

                    # Check the format of the result
                    if isinstance(result, tuple) and len(result) == 2:
                        # Format: (path, success)
                        path, success = result
                    elif isinstance(result, dict) and 'path_extracted' in result:
                        # Format: {'path_extracted': path, 'cost': cost, ...}
                        path = result
                        success = result['path_extracted'] is not None and len(result['path_extracted']) > 0
                    else:
                        # If not a recognized format, assume it's just the path
                        path = result
                        success = path is not None and hasattr(path, '__len__') and len(path) > 0
                except Exception as e:
                    print(f"Error running TJPS for agent {agent_id}: {e}")
                    path = []
                    success = False

                # Record results
                agent_runtime = time.time() - agent_start_time
                tjps_runtimes.append(agent_runtime)
                tjps_success.append(success)

                if success:
                    tjps_paths.append(path)
                    print(f"Agent {agent_id} planning successful in {agent_runtime:.2f} seconds")
                else:
                    tjps_paths.append([])
                    print(f"Agent {agent_id} planning failed after {agent_runtime:.2f} seconds")

            # Calculate total runtime
            total_runtime = time.time() - start_time

            # Calculate path lengths for successful paths
            path_lengths = [calculate_path_length(path) for path in tjps_paths if path]

            # Calculate success rate
            success_rate = sum(tjps_success) / args.agents

            print(f"TJPS benchmark completed in {total_runtime:.2f} seconds")
            print(f"Path lengths: {path_lengths}")
            print(f"Success rate: {success_rate:.2f}")

            # Add results
            benchmark_results.append({
                "algorithm": "TJPS",
                "runtime": total_runtime,
                "agent_runtimes": tjps_runtimes,
                "path_lengths": path_lengths,
                "avg_path_length": np.mean(path_lengths) if path_lengths else 0,
                "success_rate": success_rate,
                "iterations": 1,  # TJPS doesn't do iterations
                "final_collisions": args.agents - sum(tjps_success),
                "paths": tjps_paths  # Store paths for later comparison
            })

        except Exception as e:
            print(f"Error in TJPS benchmark: {e}")

            # Add partial results if available
            if 'tjps_paths' in locals() and any(tjps_paths):
                path_lengths = [calculate_path_length(path) for path in tjps_paths if path]
                success_rate = sum(1 for path in tjps_paths if path) / args.agents

                benchmark_results.append({
                    "algorithm": "TJPS (Partial)",
                    "runtime": time.time() - start_time if 'start_time' in locals() else 0,
                    "agent_runtimes": tjps_runtimes if 'tjps_runtimes' in locals() else [],
                    "path_lengths": path_lengths,
                    "avg_path_length": np.mean(path_lengths) if path_lengths else 0,
                    "success_rate": success_rate,
                    "iterations": 1,
                    "final_collisions": args.agents - sum(1 for path in tjps_paths if path),
                    "paths": tjps_paths
                })

                # Visualize discrete TJPS solution
                if args.visualize:
                    vis_dir = os.path.join(args.output, "tjps")
                    os.makedirs(vis_dir, exist_ok=True)
                    visualize_solution(
                        tjps_paths,
                        grid_seq,
                        start_list,
                        goal_list,
                        output_dir=vis_dir
                        )
                    print(f"✔ TJPS discrete GIF in {vis_dir}/agent_movements.gif")

        finally:
            # TJPS uses only grid_seq & planner; free paths & runtimes
            del tjps_paths, tjps_runtimes, tjps_success, planner
            gc.collect()

    if run_all or "dynamic" in algorithms:
        # Use the shared environment for dynamic drone solution
        print("\n===== Benchmarking Dynamic Drone Solution =====")
        try:
            # Step 1: Generate initial paths using our LNS2 with dynamic obstacles
            print("Step 1: Generating initial paths...")
            step1_start_time = time.time()

            # Initialize LNS2 model with dynamic grid sequence
            seed = 42
            lns2_model = OurMyLns2.MyLns2(seed, grid_seq, start_list, goal_list, args.agents, grid_seq.shape[1])
            lns2_model.setUseRerun(False)  # Disable visualization
            lns2_model.init_pp()

            # Get initial paths
            paths = lns2_model.vector_path

            # Iterative conflict resolution
            iteration = 0
            num_collisions = lns2_model.calculate_sipps(list(range(args.agents)))

            print(f"Initial planning complete with {num_collisions} collisions")

            while num_collisions > 0 and iteration < MAX_ITERATIONS:
                iteration += 1
                print(f"Iteration {iteration}/{MAX_ITERATIONS}")

                # Replan for all agents
                agents_to_replan = list(range(args.agents))
                num_collisions = lns2_model.calculate_sipps(agents_to_replan)

                # Get updated paths
                paths = lns2_model.vector_path

                print(f"Replanning complete with {num_collisions} collisions")

                if num_collisions == 0:
                    print("All collisions resolved!")
                    break

            step1_time = time.time() - step1_start_time
            print(f"Step 1 completed in {step1_time:.2f} seconds")

            # Calculate initial path lengths
            initial_path_lengths = [calculate_path_length(path) for path in paths]
            print(f"Initial path lengths: {initial_path_lengths}")

            # Step 2: Convert to schedule
            print("Step 2: Converting to schedule...")
            step2_start_time = time.time()
            schedule = step2_convert_to_schedule(paths)
            step2_time = time.time() - step2_start_time
            print(f"Step 2 completed in {step2_time:.2f} seconds")

            # Step 3: Build Temporal Plan Graph
            print("Step 3: Building Temporal Plan Graph...")
            step3_start_time = time.time()
            tpg = step3_build_temporal_plan_graph(schedule, UAV_PARAMS)
            step3_time = time.time() - step3_start_time
            print(f"Step 3 completed in {step3_time:.2f} seconds")

            # Step 4: Construct Simple Temporal Network
            print("Step 4: Constructing Simple Temporal Network...")
            step4_start_time = time.time()

            # Extract dynamic obstacles from the grid sequence
            dynamic_obstacles = {}
            if grid_seq is not None and len(grid_seq) > 1:
                for t in range(len(grid_seq)):
                    grid_frame = grid_seq[t]
                    dynamic_mask = np.logical_and(grid_frame > 0.05, grid_frame < 0.95)
                    obstacle_positions = np.where(dynamic_mask)

                    for i in range(len(obstacle_positions[0])):
                        x, y = obstacle_positions[0][i], obstacle_positions[1][i]
                        obstacle_id = f"obstacle_{x}_{y}"

                        if obstacle_id not in dynamic_obstacles:
                            dynamic_obstacles[obstacle_id] = []

                        dynamic_obstacles[obstacle_id].append({
                            'x': float(x),
                            'y': float(y),
                            't': float(t) * GRID_DT * RADAR_ZONE_SPEED_FACTOR
                        })

            import traceback
            try:
                print("Attempting to construct STN with original implementation…")
                stn = step4_construct_simple_temporal_network(
                    tpg, UAV_PARAMS, dynamic_obstacles
                )
            except Exception:
                print("❌ Original STN construction failed, here’s the traceback:")
                traceback.print_exc()
                print("ℹ️ Falling back to the SimpleTemporalNetwork class…")
                from multi_agent_path_planning.centralized.scheduling.stn import SimpleTemporalNetwork
                stn = SimpleTemporalNetwork(tpg, dynamic_obstacles)
                stn.generate_bounds()
                stn.generate_end_edges()

            step4_time = time.time() - step4_start_time
            print(f"Step 4 completed in {step4_time:.2f} seconds")

            # Step 5: Optimize Temporal Plan
            print("Step 5: Optimizing Temporal Plan...")
            step5_start_time = time.time()

            try:
                print("Attempting to optimize temporal plan with original implementation...")
                solution = step5_optimize_temporal_plan(stn)
            except Exception as e:
                print(f"Error optimizing temporal plan with original implementation: {e}")
                print("Using fallback optimization implementation...")

                # Use the OptimizationClass from dynamic_drone_solution.py
                from dynamic_drone_solution import OptimizationClass

                # Create an instance of OptimizationClass
                optimizer = OptimizationClass(stn)
                solution = optimizer.generate_schedule()

            step5_time = time.time() - step5_start_time
            print(f"Step 5 completed in {step5_time:.2f} seconds")

            # Step 6: Generate Final Trajectories
            print("Step 6: Generating Final Trajectories...")
            step6_start_time = time.time()
            trajectories = step6_generate_final_trajectories(
                solution,              # schedule
                UAV_PARAMS,            # uav_params
                dynamic_obstacles,     # dynamic_obstacles
                grid_seq               # grid_seq
            )
            step6_time = time.time() - step6_start_time
            print(f"Step 6 completed in {step6_time:.2f} seconds")

            # Calculate trajectory lengths
            trajectory_lengths = [calculate_trajectory_length(traj) for traj in trajectories]
            print(f"Final trajectory lengths: {trajectory_lengths}")

            # Calculate total runtime (excluding visualization)
            total_runtime = step1_time + step2_time + step3_time + step4_time + step5_time + step6_time

            # Calculate success rate (based on whether all trajectories were generated successfully)
            success_rate = len(trajectories) / args.agents
            
            # Before kinematic
            benchmark_results.append({
            "algorithm": "Dynamic Drone (Before Kinematic)",
            "runtime": step1_time,
            "initial_path_lengths": initial_path_lengths,
            "avg_path_length": np.mean(initial_path_lengths),
            "success_rate": success_rate,       # or based on initial collisions
            "iterations": iteration,
            "final_collisions": num_collisions,
            "paths": paths,
            })

            # After kinematic
            benchmark_results.append({
            "algorithm": "Dynamic Drone Solution",
            "runtime": total_runtime,
            "step_runtimes": {
                "step1": step1_time,
                "step2": step2_time,
                "step3": step3_time,
                "step4": step4_time,
                "step5": step5_time,
                "step6": step6_time,
            },
            "initial_path_lengths": initial_path_lengths,
            "avg_initial_path_length": np.mean(initial_path_lengths),
            "trajectory_lengths": trajectory_lengths,
            "avg_trajectory_length": np.mean(trajectory_lengths),
            "success_rate": success_rate,
            "paths": paths,
            })

            print(f"Dynamic Drone Solution benchmark completed in {total_runtime:.2f} seconds")
            print(f"Trajectory lengths: {trajectory_lengths}")
            print(f"Success rate: {success_rate:.2f}")

            # Visualize discrete paths
            if args.visualize:
                vis_dir = os.path.join(args.output, "dynamic_drone")
                os.makedirs(vis_dir, exist_ok=True)
                visualize_solution(
                    paths,
                    grid_seq,
                    start_list,
                    goal_list,
                    output_dir=vis_dir
                )
                print(f"✔ Dynamic-drone discrete GIF in {vis_dir}/agent_movements.gif")

                # Visualize continuous trajectories
                cont_dir = os.path.join(args.output, "dynamic_drone_continuous")
                os.makedirs(cont_dir, exist_ok=True)
                # new signature: (trajectories, grid_seq, output_dir, uav_params)
                visualize_continuous_space(
                    trajectories,
                    grid_seq,
                    cont_dir,
                    UAV_PARAMS
                )
                print(f"✔ Dynamic-drone continuous visualization in {cont_dir}")
        except Exception as e:
             print(f"Error in dynamic drone solution benchmark: {e}")
        # Force garbage collection (always runs, even on error)
        finally:
            # Delete every local large buffer you created
            del paths, grid_seq, dynamic_obstacles, solution, trajectories
            gc.collect()

    # Save benchmark results
    if benchmark_results:
        save_benchmark_results(benchmark_results, args.output, start_list, goal_list)

    print("Benchmark completed!")

if __name__ == "__main__":
    main()
