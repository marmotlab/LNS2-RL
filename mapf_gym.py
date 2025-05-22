import copy
import random
import sys

import gym
import numpy as np
from lns2.build import my_lns2
from world_property import State
from alg_parameters import *
from dynamic_state import DyState
# Dictionary mapping actions to their opposites (used to penalize reversing actions)
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

# "CIMBIBOY includes" – additional imports from WarehouseEnvironment code
import os
import re
from PIL import Image, ImageDraw
import matplotlib as plt
from GridBasedPathPlanning.RL_Environment.environment.map_generator import map_to_value
from GridBasedPathPlanning.Environment.GridMapEnv import GridMapEnv
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import GridToRGB
# "2025 mar 3." – date stamp or version


class CL_MAPFEnv(gym.Env):
    """
    CL_MAPFEnv is a Gym environment that maps a Multi-Agent Path Finding (MAPF) problem.
    It integrates core components from WarehouseEnvironment (such as static map generation
    and dynamic obstacle updates) with the MAPF planning and RL logic.
    """

    def __init__(self, env_id,
                 global_num_agents_range=EnvParameters.GLOBAL_N_AGENT_LIST,
                 fov_size=EnvParameters.FOV_SIZE,
                 size=EnvParameters.WORLD_SIZE_LIST,
                 prob=EnvParameters.OBSTACLE_PROB_LIST,
                 im_flag=False,
                 enable_visualization=True):
        """Initialization of the environment.

        Parameters:
          env_id: Environment identifier.
          global_num_agents_range: List of possible agent counts.
          fov_size: Field-of-view size for partial observations.
          size: List of possible grid sizes.
          prob: Obstacle density parameters.
          im_flag: Flag indicating whether using imitation mode.
        """
        # Set basic environment parameters
        self.global_num_agents_range = global_num_agents_range
        self.fov_size = fov_size
        self.SIZE = size      # Available sizes for the grid
        self.PROB = prob      # Obstacle density options
        self.env_id = env_id
        self.im_flag = im_flag
        self.enable_visualization = enable_visualization  # Flag to control visualization
# ______________________________________________________________________________________________________________________
        # WarehouseEnvironment logic for dynamic obstacles and grid creation is integrated later.
        self.amr_count = 16
        self.max_amr = 16
        self.FRAMES = 420
        self.curriculum = 100

    def set_grid(self, static_map, map_idx, add_num_dyn_obj, FRAMES):
        self.map_idx = map_idx
        self.map_a = np.asarray(static_map)

        # Convert from MAPF representation (-1 for obstacles, 0 for free)
        # to GridBasedPathPlanning representation (1 for obstacles, 0 for free)
        # Simply convert -1 to 1 for obstacles

        # Create a copy of the map and convert -1 to 1 for obstacles
        self.map_v = self.map_a.copy()
        self.map_v[self.map_v == -1] = 1

        # Save this conversion for debugging
        debug_dir = "debug_maps"
        os.makedirs(debug_dir, exist_ok=True)

        # Initialize the grid with the converted map
        self.grid.SetStaticGrid(self.map_v)
        self.grid.ClearGrid(gridtype="grid_dynamic")
        # Add more dynamic obstacles (radar zones) with larger size for drone environment
        self.grid.addRandomDynamicObstacles(no_of_obs=add_num_dyn_obj, d_range=(2, 7), alpha=0.4)
        self.frames = FRAMES
        self.grid.ResetDynamicObstacles()
        self.grid_seq = self.grid.GenerateGridSequence(self.frames, reset_after=True)
        self.grid.RenderAllObstacles(typ="static")
        self.grid.RenderAllObstacles(typ="dynamic")
        self.grid.RenderGrid()

        # Generate the RGB representation using GridToRGB
        self.init_arr = GridToRGB(self.grid.grid)

        # Save the visualization using init_arr (already in RGB format)
        debug_dir = "debug_maps"
        os.makedirs(debug_dir, exist_ok=True)
        vis_path = f"{debug_dir}/grid_vis_{map_idx}.png"
        Image.fromarray(self.init_arr.astype(np.uint8)).save(vis_path)


        self.initial_random_steps = False
        self.was_reached = False
        self.arrived = 0
        self.iter = 0
        self.first_reset = False

        self.info = {
            'R_max_step': False,
            'no_global_guidance': False,
            'goal_reached': False,
            'collision': False,
            'blocked': False,
            'steps': -1,
            # 'path': [],
            'reward': 0,
            'total_arrived': 0
        }

    def reGenerateAndAdd(self):
        self.grid.addRandomDynamicObstacles(no_of_obs=1, d_range=(1,4), alpha = 0.4)
        self.grid_seq = self.grid.GenerateGridSequence(self.frames, reset_after=True)

    def update_dyn_map(self, step = 1):
        """
        Update dynamic obstacles and merge them with the static map to produce an effective map.
        This should be called at every step.
        """
        self.grid.ClearGrid(gridtype="grid_dynamic")
        if step == 1:
            self.grid.StepDynamicObstacles()
        self.grid.RenderAllObstacles(typ="dynamic")
        self.grid.RenderGrid()
        self.init_arr = GridToRGB(self.grid.grid)


    def debug_dynamic(self, num_steps=10, did='d1'):
        """
        Debug function to visualize dynamic obstacle movement over multiple steps.
        Generates a sequence of images showing the dynamic obstacles moving.

        Args:
            num_steps: Number of steps to visualize
            did: Debug ID to differentiate between multiple debug runs
        """
        print(f"[DEBUG] Visualizing dynamic obstacle movement for {num_steps} steps...")

        # Create debug directory
        debug_dir = "debug_maps/debug_dynamic"
        os.makedirs(debug_dir, exist_ok=True)

        # Reset dynamic obstacles to initial state
        self.grid.ResetDynamicObstacles()

        # Generate sequence of images
        for step in range(num_steps):
            # Clear dynamic grid and render obstacles
            self.grid.ClearGrid(gridtype="grid_dynamic")
            self.grid.RenderAllObstacles(typ="dynamic")
            self.grid.RenderGrid()

            # Generate RGB representation
            rgb_grid = GridToRGB(self.grid.grid)

            # Add agent start positions (red)
            for row, col in self.start_list:
                if 0 <= row < rgb_grid.shape[0] and 0 <= col < rgb_grid.shape[1]:
                    rgb_grid[row, col] = [255, 0, 0]  # Red

            # Add agent goal positions (blue)
            for row, col in self.goal_list:
                if 0 <= row < rgb_grid.shape[0] and 0 <= col < rgb_grid.shape[1]:
                    rgb_grid[row, col] = [0, 0, 255]  # Blue

            # Save image
            img_path = f"{debug_dir}/dynamic_step_{did}_{step}.png"
            Image.fromarray(rgb_grid.astype(np.uint8)).save(img_path)
            print(f"[DEBUG] Saved dynamic visualization for step {step} to: {img_path}")

            # Step dynamic obstacles
            self.grid.StepDynamicObstacles()

        # Reset dynamic obstacles to initial state
        self.grid.ResetDynamicObstacles()
        self.grid.ClearGrid(gridtype="grid_dynamic")
        self.grid.RenderAllObstacles(typ="dynamic")
        self.grid.RenderGrid()
        print(f"[DEBUG] Dynamic obstacle visualization complete. Images saved to {debug_dir}/")

    def debug_gridmap(self, converted_grid_seq, marker = "base"):
        """
        Debug function to visualize the first 3 timesteps (0, 1, 2) of the dynamic map array.

        Args:
            converted_grid_seq: 3D numpy array of shape (420, 96, 96) containing dynamic obstacle data
        """
        print("################################################################")
        print("Visualizing timesteps 0, 1, 2 of dynamic map array...")
        print(f"Dynamic map shape: {converted_grid_seq.shape}")
        print("################################################################")

        # Verify the shape of the dynamic map
        if len(converted_grid_seq.shape) != 3:
            print(f"[ERROR] Expected 3D array, got shape {converted_grid_seq.shape}")
            return

        # Create debug directory
        debug_dir = "debug_maps"
        os.makedirs(debug_dir, exist_ok=True)

        # Visualize the first 3 timesteps (0, 1, 2)
        for t in range(min(10, converted_grid_seq.shape[0])):
            # Create a temporary grid for this timestep
            temp_grid = np.zeros_like(converted_grid_seq[t])
            # Copy the data for this timestep
            temp_grid[:] = converted_grid_seq[t]

            # Use the GridToRGB function to convert to RGB
            rgb_map = GridToRGB(temp_grid)

            # Save image
            img_path = f"{debug_dir}/gridmapenv_slices_{marker}_{t}.png"
            Image.fromarray(rgb_map).save(img_path)
            print(f"Saved timestep {t} visualization to: {img_path}")

        print("################################################################")
        print("Dynamic map visualization complete. Images saved to debug_maps/")
        print("################################################################")

    # The convert_grid_seq_for_mylns2 function has been removed
    # We now pass grid_seq directly to mylns2

# ______________________________________________________________________________________________________________________

    def global_set_world(self, cl_num_task):
        """Randomly generate a new task (i.e. a MAPF instance).

        This method generates:
          - A random static grid (map) with obstacles.
          - Random start positions for all agents.
          - Random goal positions, ensuring they are in the same connected region as the agent.
        The parameter cl_num_task is used to choose among curriculum task levels.
        """
        # Helper function to get a connected region (used to ensure goal is reachable)
        def get_connected_region(world0, regions_dict, x0, y0):
            # Increase recursion limit to allow deep region searches.
            sys.setrecursionlimit(1000000)
            if (x0, y0) in regions_dict:  # region already computed for this cell
                return regions_dict[(x0, y0)]
            visited = set()
            sx, sy = world0.shape[0], world0.shape[1]
            work_list = [(x0, y0)]
            while len(work_list) > 0:
                (i, j) = work_list.pop()
                # Skip out-of-bounds or obstacle cells.
                if i < 0 or i >= sx or j < 0 or j >= sy:
                    continue
                if world0[i, j] == -1:
                    continue  # cell is a static obstacle
                # If cell is marked as occupied, store region info.
                if world0[i, j] > 0:
                    regions_dict[(i, j)] = visited
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                # Add neighboring cells (up, right, down, left) for BFS
                work_list.append((i + 1, j))
                work_list.append((i, j + 1))
                work_list.append((i - 1, j))
                work_list.append((i, j - 1))
            regions_dict[(x0, y0)] = visited
            return visited

        # Choose a random obstacle probability from the available set (based on task level)
        prob = random.choice(self.PROB[cl_num_task])
        # Choose a random grid size index
        task_set = random.choice(range(len(self.SIZE)))
        self.size = self.SIZE[task_set]
        self.height = self.width = self.size
        self.episode_len = EnvParameters.EPISODE_LEN[task_set]
        # Compute the total number of agents as a function of grid size and a random multiplier
        self.global_num_agent = int(round(random.choice(self.global_num_agents_range[cl_num_task]) * self.size * self.size))

        # Create a random static grid: cells are 0 (free) or -1 (obstacle)
        self.map = -(np.random.rand(int(self.size), int(self.size)) < prob).astype(int)

        # Create the Grid environment first
        self.grid = GridMapEnv(grid_size=(self.height, self.width))

        # Initialize the grid with static map and create dynamic obstacles
        self.set_grid(self.map, self.env_id, self.amr_count, self.FRAMES)

        # Copy the map to fix state (used for later visualization or reference)
        self.fix_state = copy.copy(self.map)
        # Create a dictionary for tracking dynamic state per cell.
        self.fix_state_dict = {}
        for i in range(int(self.size)):
            for j in range(int(self.size)):
                self.fix_state_dict[i, j] = []

        # Randomize agent start positions - now checking both static map and dynamic obstacles at time 0
        agent_counter = 0
        self.start_list = []
        while agent_counter < self.global_num_agent:
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            # Check if cell is free in static map and not occupied by dynamic obstacle at time 0
            if self.fix_state[x, y] == 0:  # Free in static map
                if self.grid_seq[0, x, y] < 0.1:  # No dynamic obstacle
                    self.fix_state[x, y] += 1
                    self.fix_state_dict[x, y].append(agent_counter)
                    self.start_list.append((x, y))
                    agent_counter += 1
                else:
                    # Skip this position due to dynamic obstacle
                    print(f"[DEBUG] Skipping start position ({x},{y}) due to dynamic obstacle (value: {self.grid_seq[0, x, y]})")
        # Verify that total occupancy in fix_state equals total agents plus obstacles.
        assert(sum(sum(self.fix_state)) == self.global_num_agent + sum(sum(self.map)))

        # Randomize goal positions for each agent ensuring goals lie in reachable connected regions.
        goals = np.zeros((int(self.size), int(self.size))).astype(int)
        goal_counter = 0
        agent_regions = dict()
        self.goal_list = []
        while goal_counter < self.global_num_agent:
            agent_pos = self.start_list[goal_counter]
            valid_tiles = get_connected_region(self.fix_state, agent_regions, agent_pos[0], agent_pos[1])
            # Filter valid tiles to exclude those with dynamic obstacles at time 0
            valid_tiles_filtered = []
            skipped_count = 0
            for tile in valid_tiles:
                x, y = tile
                if self.grid_seq[0, x, y] < 0.1:  # No dynamic obstacle at time 0
                    valid_tiles_filtered.append(tile)
                else:
                    skipped_count += 1

            # If no valid tiles remain after filtering, use original set (better than getting stuck)
            if not valid_tiles_filtered:
                print(f"[WARNING] No valid tiles without dynamic obstacles for agent {goal_counter}. Using original set.")
                valid_tiles_filtered = list(valid_tiles)
            elif skipped_count > 0:
                print(f"[DEBUG] Skipped {skipped_count} potential goal positions for agent {goal_counter} due to dynamic obstacles")

            x, y = random.choice(valid_tiles_filtered)
            if goals[x, y] == 0 and self.fix_state[x, y] != -1:
                goals[x, y] = goal_counter + 1
                self.goal_list.append((x, y))
                goal_counter += 1

        # Initialize the world state with the fix_state and agent positions.
        self.world = State(self.fix_state, self.fix_state_dict, self.global_num_agent, self.start_list, self.goal_list)

    def joint_move(self, actions):
        """
        Execute simultaneous moves for all agents and update the world state.

        The method:
          - Updates the world state based on the current time step and dynamic obstacles.
          - For agents that are not in local control, updates their position using the pre-computed paths.
          - For local agents, calculates the new position based on the provided actions.
          - Updates collision counts by checking overlapping occupancy.
        """
        # Update world state based on dynamic state.
        if self.time_step < self.dynamic_state.max_lens:
            self.world.state = self.dynamic_state.state[self.time_step] + self.map
        else:
            self.world.state = self.dynamic_state.state[-1] + self.map

        # For non-local agents, use their planned path.
        for i in range(self.global_num_agent):
            if i not in self.world.local_agents:
                max_len = len(self.paths[i])
                if max_len <= self.time_step:
                    continue
                else:
                    self.world.agents_poss[i] = self.paths[i][self.time_step]
                    # Update state_dict: remove from previous cell and add to new cell.
                    self.world.state_dict[self.paths[i][self.time_step - 1]].remove(i)
                    self.world.state_dict[self.paths[i][self.time_step]].append(i)

        # Copy the current local agents' positions before updating.
        local_past_position = copy.copy(self.world.local_agents_poss)
        dynamic_collision_status = np.zeros(self.local_num_agents)
        agent_collision_status = np.zeros(self.local_num_agents)
        reach_goal_status = np.zeros(self.local_num_agents)

        # Update action and vertex maps for utility calculation.
        self.agent_util_map_action.pop(0)
        self.agent_util_map_vertex.pop(0)
        self.agent_util_map_action.append(np.zeros((5, self.map.shape[0], self.map.shape[1])))
        self.agent_util_map_vertex.append(np.zeros((self.map.shape[0], self.map.shape[1])))

        # For each local (RL-controlled) agent:
        for local_i, i in enumerate(self.world.local_agents):
            # Determine movement direction from the action dictionary.
            direction = self.world.get_dir(actions[local_i])
            ax, ay = self.world.local_agents_poss[local_i]
            dx, dy = direction[0], direction[1]
            # Check boundaries.
            if ax + dx >= self.world.state.shape[0] or ax + dx < 0 or ay + dy >= self.world.state.shape[1] or ay + dy < 0:
                raise ValueError("out of boundaries")
            # Check if moving into an obstacle.
            if self.map[ax + dx, ay + dy] < 0:
                raise ValueError("collide with static obstacles")
            # Update positions for local agents.
            self.world.agents_poss[i] = (ax + dx, ay + dy)
            self.world.local_agents_poss[local_i] = (ax + dx, ay + dy)
            self.world.state[ax + dx, ay + dy] += 1
            self.world.state_dict[ax, ay].remove(i)
            self.world.state_dict[ax + dx, ay + dy].append(i)
            self.agent_util_map_action[-1][int(actions[local_i]), ax + dx, ay + dy] += 1
            self.agent_util_map_vertex[-1][ax + dx, ay + dy] += 1

        # Check for collisions:
        for local_i, i in enumerate(self.world.local_agents):
            if self.world.state[self.world.local_agents_poss[local_i]] > 1:
                collide_agents_index = self.world.state_dict[self.world.local_agents_poss[local_i]]
                # Ensure consistency: number of agents in state_dict equals occupancy count.
                assert (len(collide_agents_index) == self.world.state[self.world.local_agents_poss[local_i]])
                for j in collide_agents_index:
                    if j != i:
                        # Count collisions differently based on whether colliding agent is local or not.
                        if j in self.world.local_agents:
                            agent_collision_status[local_i] += 1
                        else:
                            dynamic_collision_status[local_i] += 1
                        self.new_collision_pairs.add((min(j, i), max(j, i)))
            # Check past positions for additional collision events.
            if self.world.state[local_past_position[local_i]] > 0:
                collide_agents_index = self.world.state_dict[local_past_position[local_i]]
                assert (len(collide_agents_index) == self.world.state[local_past_position[local_i]])
                for j in collide_agents_index:
                    if j != i:
                        if j in self.world.local_agents:
                            local_j = self.world.local_agents.index(j)
                            past_poss = local_past_position[local_j]
                            if past_poss == self.world.local_agents_poss[local_i] and self.world.agents_poss[j] != past_poss:
                                agent_collision_status[local_i] += 1
                                self.new_collision_pairs.add((min(j, i), max(j, i)))
                        else:
                            max_len = len(self.paths[j])
                            if max_len <= self.time_step:
                                continue
                            else:
                                past_poss = self.paths[j][self.time_step - 1]
                                if past_poss == self.world.local_agents_poss[local_i] and past_poss != self.paths[j][self.time_step]:
                                    dynamic_collision_status[local_i] += 1
                                    self.new_collision_pairs.add((min(j, i), max(j, i)))
            # Check if agent has reached its goal.
            if self.world.local_agents_poss[local_i] == self.goal_list[i]:
                reach_goal_status[local_i] = 1

        return dynamic_collision_status, agent_collision_status, reach_goal_status

    def observe(self, local_agent_index):
        """
        Return a single agent's observation.

        This method extracts a local field-of-view (FOV) around the agent's current position,
        and constructs various maps (goal, obstacles, agent positions, dynamic obstacles, etc.)
        that together form the observation input for the RL agent.
        """
        agent_index = self.world.local_agents[local_agent_index]

        # Determine boundaries for the FOV window around the agent.
        top_poss = max(self.world.agents_poss[agent_index][0] - self.fov_size // 2, 0)
        bottom_poss = min(self.world.agents_poss[agent_index][0] + self.fov_size // 2 + 1, self.size)
        left_poss = max(self.world.agents_poss[agent_index][1] - self.fov_size // 2, 0)
        right_poss = min(self.world.agents_poss[agent_index][1] + self.fov_size // 2 + 1, self.size)
        top_left = (self.world.agents_poss[agent_index][0] - self.fov_size // 2,
                    self.world.agents_poss[agent_index][1] - self.fov_size // 2)
        # Calculate offsets for the FOV slice.
        FOV_top = max(self.fov_size // 2 - self.world.agents_poss[agent_index][0], 0)
        FOV_left = max(self.fov_size // 2 - self.world.agents_poss[agent_index][1], 0)
        FOV_bottom = FOV_top + (bottom_poss - top_poss)
        FOV_right = FOV_left + (right_poss - left_poss)

        obs_shape = (self.fov_size, self.fov_size)
        # Initialize various maps for the observation.
        goal_map = np.zeros(obs_shape)           # Map for agent's own goal.
        local_poss_map = np.zeros(obs_shape)       # Map for local agent positions.
        local_goals_map = np.zeros(obs_shape)      # Map for goals of visible other agents.
        obs_map = np.ones(obs_shape)               # Map for obstacles.
        guide_map = np.zeros((4, obs_shape[0], obs_shape[1]))  # Guidance map (e.g., heuristic directions).
        visible_agents = set()
        dynamic_poss_maps = np.zeros((EnvParameters.NUM_TIME_SLICE, self.fov_size, self.fov_size))
        sipps_map = np.zeros(obs_shape)            # Map based on planned paths.
        util_map_action = np.zeros((5, self.fov_size, self.fov_size))
        util_map = np.zeros(obs_shape)
        blank_map = np.zeros(obs_shape)
        occupy_map = np.zeros(obs_shape)
        next_step_map = np.zeros((EnvParameters.K_STEPS, self.fov_size, self.fov_size))

        # Determine time window for planned path (SIPPS path) extraction.
        if self.time_step - EnvParameters.WINDOWS < 0:
            min_time = 0
        elif self.time_step >= len(self.sipps_path[local_agent_index]):
            min_time = max(0, len(self.sipps_path[local_agent_index]) - EnvParameters.WINDOWS)
        else:
            min_time = self.time_step - EnvParameters.WINDOWS
        max_time = min(self.time_step + EnvParameters.WINDOWS, len(self.sipps_path[local_agent_index]))
        window_path = self.sipps_path[local_agent_index][min_time:max_time]

        # Mark the agent's own goal if it falls within the FOV.
        if self.goal_list[agent_index][0] in range(top_poss, bottom_poss) and self.goal_list[agent_index][1] in range(left_poss, right_poss):
            goal_map[self.goal_list[agent_index][0] - top_left[0],
                     self.goal_list[agent_index][1] - top_left[1]] = 1
        # Mark the current position of the agent.
        local_poss_map[self.world.agents_poss[agent_index][0] - top_left[0],
                       self.world.agents_poss[agent_index][1] - top_left[1]] = 1
        # Set obstacle values from the static map.
        obs_map[FOV_top:FOV_bottom, FOV_left:FOV_right] = -self.map[top_poss:bottom_poss, left_poss:right_poss]
        # Set guidance map using pre-computed heuristic directions.
        guide_map[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = self.world.heuri_map[agent_index][:, top_poss:bottom_poss, left_poss:right_poss]
        # Set utility maps based on current space utilization.
        util_map[FOV_top:FOV_bottom, FOV_left:FOV_right] = self.space_ulti_vertex[top_poss:bottom_poss, left_poss:right_poss]
        util_map_action[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = self.space_ulti_action[:, top_poss:bottom_poss, left_poss:right_poss]

        # Loop over each pixel in the FOV to adjust occupancy and other maps.
        for i in range(top_left[0], top_left[0] + self.fov_size):
            for j in range(top_left[1], top_left[1] + self.fov_size):
                # If out-of-bound, mark occupancy accordingly.
                if i >= self.size or i < 0 or j >= self.size or j < 0:
                    occupy_map[i - top_left[0], j - top_left[1]] = 1 - self.time_step / self.episode_len
                    continue
                # If cell is an obstacle in the static map, mark occupancy.
                if self.world.state[i, j] == -1:
                    occupy_map[i - top_left[0], j - top_left[1]] = 1 - self.time_step / self.episode_len
                    continue
                # Mark SIPPS path presence.
                if (i, j) in window_path:
                    sipps_map[i - top_left[0], j - top_left[1]] = 1
                # For other agents in the local region, update local agent map.
                for iter_a in range(self.local_num_agents):
                    if iter_a != local_agent_index:
                        for k in range(EnvParameters.K_STEPS):
                            if (i, j) == self.all_next_poss[iter_a][k]:
                                next_step_map[k, i - top_left[0], j - top_left[1]] += 1
                if self.world.state[i, j] > 0:
                    for item in self.world.state_dict[i, j]:
                        if item in self.world.local_agents and item != agent_index:
                            visible_agents.add(item)
                            local_poss_map[i - top_left[0], j - top_left[1]] += 1

                # Update dynamic occupancy maps over a time slice.
                for t in range(EnvParameters.NUM_TIME_SLICE):
                    if self.time_step + t < self.dynamic_state.max_lens:
                        dynamic_poss_maps[t, i - top_left[0], j - top_left[1]] = self.dynamic_state.state[self.time_step + t, i, j]
                    else:
                        dynamic_poss_maps[t, i - top_left[0], j - top_left[1]] = self.dynamic_state.state[-1, i, j]

                # Update occupancy and blank maps based on future dynamic state.
                if self.time_step >= self.makespan:
                    if self.dynamic_state.state[-1, i, j] > 0:
                        occupy_map[i - top_left[0], j - top_left[1]] = 1 - self.time_step / self.episode_len
                    else:
                        blank_map[i - top_left[0], j - top_left[1]] = 1 - (self.time_step + 1) / self.episode_len
                else:
                    occupy_t = 0
                    if self.dynamic_state.state[self.time_step, i, j] > 0:
                        for t in range(self.time_step, self.episode_len + 1):
                            if t >= self.makespan:
                                if self.dynamic_state.state[-1, i, j] > 0:
                                    occupy_t = self.episode_len - self.time_step
                                break
                            if self.dynamic_state.state[t, i, j] > 0:
                                occupy_t += 1
                            else:
                                break
                    occupy_map[i - top_left[0], j - top_left[1]] = occupy_t / self.episode_len
                    blank_t = 0
                    for t in range(self.time_step + 1, self.episode_len + 1):
                        if t >= self.makespan:
                            if self.dynamic_state.state[-1, i, j] == 0:
                                blank_t = self.episode_len - self.time_step - 1
                            break
                        if self.dynamic_state.state[t, i, j] == 0:
                            blank_t += 1
                        else:
                            break
                    blank_map[i - top_left[0], j - top_left[1]] = blank_t / self.episode_len

        # Normalize maps using tanh function for smoothing.
        zero_mask = local_poss_map == 0
        local_poss_map = 0.5 + 0.5 * np.tanh((local_poss_map - 1) / 3)
        local_poss_map[zero_mask] = 0
        zero_mask = next_step_map == 0
        next_step_map = 0.5 + 0.5 * np.tanh((next_step_map - 1) / 3)
        next_step_map[zero_mask] = 0
        zero_mask = dynamic_poss_maps == 0
        dynamic_poss_maps = 0.5 + 0.5 * np.tanh((dynamic_poss_maps - 1) / 3)
        dynamic_poss_maps[zero_mask] = 0

        # For each visible other agent, project its goal onto the FOV boundary.
        for vis_agent_index in visible_agents:
            x, y = self.world.agents_goals[vis_agent_index]
            min_node = (max(top_left[0], min(top_left[0] + self.fov_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.fov_size - 1, y)))
            local_goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        # Compute normalized direction vector from agent position to goal.
        dx = self.world.agents_goals[agent_index][0] - self.world.agents_poss[agent_index][0]
        dy = self.world.agents_goals[agent_index][1] - self.world.agents_poss[agent_index][1]
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag

        # Compute off-route penalty based on the difference between agent position and planned path.
        window_path = np.array(window_path)
        diff = window_path - self.world.agents_poss[agent_index]
        x = diff[:, 0]
        y = diff[:, 1]
        distance = np.sqrt(x ** 2 + y ** 2)
        off_rout_penalty = -np.min(distance) * self.off_route_factor

        # Return observation layers, vector of state features, and off-route penalty.
        return ([dynamic_poss_maps[0], dynamic_poss_maps[1], dynamic_poss_maps[2], dynamic_poss_maps[3],
                 dynamic_poss_maps[4], dynamic_poss_maps[5], dynamic_poss_maps[6], dynamic_poss_maps[7], dynamic_poss_maps[8],
                 local_poss_map, goal_map, local_goals_map,
                 obs_map, guide_map[0], guide_map[1], guide_map[2], guide_map[3], sipps_map, blank_map, occupy_map, util_map,
                 util_map_action[0], util_map_action[1], util_map_action[2], util_map_action[3], util_map_action[4],
                 next_step_map[0], next_step_map[1], next_step_map[2], next_step_map[3], next_step_map[4]],
                [dx, dy, mag],
                off_rout_penalty)

    def predict_next(self):
        """
        Predict the next positions along the SIPPS path for each local agent.

        For each local agent, this method calculates future positions (up to K_STEPS)
        based on the current time step and the SIPPS planned path. If the current time step is 0,
        it defaults to the next positions in the SIPPS path.
        """
        self.all_next_poss = []
        if self.time_step != 0:
            for local_agent_index in range(self.local_num_agents):
                next_poss_list = []
                for k in range(EnvParameters.K_STEPS):
                    if k == 0:
                        # Calculate distance from current position to all positions in the SIPPS path.
                        dis_x = self.world.local_agents_poss[local_agent_index][0] - np.array(self.sipps_path[local_agent_index])[:, 0]
                        dis_y = self.world.local_agents_poss[local_agent_index][1] - np.array(self.sipps_path[local_agent_index])[:, 1]
                        dis = np.sqrt(dis_x ** 2 + dis_y ** 2)
                        time_dis = np.abs(self.time_step - np.array(range(len(self.sipps_path[local_agent_index]))))
                        # Combine spatial and temporal distance.
                        final_dis = dis * EnvParameters.DIS_TIME_WEIGHT[0] + time_dis * EnvParameters.DIS_TIME_WEIGHT[1]
                        poss_index = np.argmin(final_dis)
                        # Choose the next position along the SIPPS path.
                        if poss_index + 1 < len(self.sipps_path[local_agent_index]):
                            next_poss = self.sipps_path[local_agent_index][poss_index + 1]
                        else:
                            next_poss = self.sipps_path[local_agent_index][-1]
                        pre_dis_x = next_poss[0] - self.world.local_agents_poss[local_agent_index][0]
                        pre_dis_y = next_poss[1] - self.world.local_agents_poss[local_agent_index][1]
                        pre_dis = pre_dis_x ** 2 + pre_dis_y ** 2
                        if pre_dis > 1:
                            next_poss = self.world.local_agents_poss[local_agent_index]
                    else:
                        if poss_index + k + 1 < len(self.sipps_path[local_agent_index]):
                            next_poss = self.sipps_path[local_agent_index][poss_index + k + 1]
                        else:
                            next_poss = self.sipps_path[local_agent_index][-1]
                    next_poss_list.append(next_poss)
                self.all_next_poss.append(next_poss_list)
        else:
            for local_agent_index in range(self.local_num_agents):
                next_poss_list = []
                for k in range(EnvParameters.K_STEPS):
                    if k + 1 < len(self.sipps_path[local_agent_index]):
                        next_poss = self.sipps_path[local_agent_index][k + 1]
                    else:
                        next_poss = self.sipps_path[local_agent_index][-1]
                    next_poss_list.append(next_poss)
                self.all_next_poss.append(next_poss_list)

    def update_ulti(self):
        """
        Update space utilization maps based on future dynamic states.

        This method computes two maps:
          - space_ulti_action: a weighted map of actions based on future dynamic obstacle usage.
          - space_ulti_vertex: a weighted map of occupancy.
        The maps are normalized by the global number of agents.
        """
        self.space_ulti_action = np.zeros((5, self.map.shape[0], self.map.shape[1]))
        self.space_ulti_vertex = np.zeros(self.map.shape)
        for t in EnvParameters.UTI_WINDOWS:
            if self.time_step + t + 1 < 0:
                continue
            if t < 0:
                self.space_ulti_action += self.agent_util_map_action[t + 2]
                self.space_ulti_vertex += self.agent_util_map_vertex[t + 2]
            if self.time_step + t + 1 >= self.dynamic_state.max_lens:
                self.space_ulti_action[0, :, :] += self.dynamic_state.state[-1]
                self.space_ulti_vertex += self.dynamic_state.state[-1]
            else:
                self.space_ulti_action += self.dynamic_state.util_map_action[self.time_step + t + 1]
                self.space_ulti_vertex += self.dynamic_state.state[self.time_step + t + 1]
        self.space_ulti_vertex = 10 * self.space_ulti_vertex / self.global_num_agent
        self.space_ulti_action = 10 * self.space_ulti_action / self.global_num_agent

    def joint_step(self, actions):
        """
        Execute a joint action for all local agents and compute the resulting reward.

        This method:
          1. Increments the time step and performs joint_move to update agent positions.
          2. Updates space utilization maps.
          3. Computes reward based on collisions, reaching goals, and action costs.
          4. Aggregates observation data from each agent.
          5. Returns the observation, state vector, reward, done flag, next valid actions,
             and some performance metrics.
        """
        # Step the dynamic obstacles before processing agent actions:
        self.update_dyn_map(1)

        self.time_step += 1
        dynamic_collision_status, agent_collision_status, reach_goal_status = self.joint_move(actions)

        # Initialize reward and observation arrays.
        rewards = np.zeros((1, self.local_num_agents), dtype=np.float32)
        obs = np.zeros((1, self.local_num_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE), dtype=np.float32)
        vector = np.zeros((1, self.local_num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        next_valid_actions = []

        # Compute rewards for each local agent based on space utilization maps.
        for i in range(self.local_num_agents):
            rewards[:, i] += EnvParameters.OVERALL_WEIGHT * EnvParameters.UTI_WEIGHT[1] * self.space_ulti_vertex[self.world.local_agents_poss[i]]
            rewards[:, i] += EnvParameters.OVERALL_WEIGHT * EnvParameters.UTI_WEIGHT[0] * self.space_ulti_action[int(actions[i]),
                                                                                                            self.world.local_agents_poss[i][0],
                                                                                                            self.world.local_agents_poss[i][1]]
        # Update utility maps and predict next positions.
        self.update_ulti()
        self.predict_next()

        # For each local agent, add collision and goal-based rewards.
        for i in range(self.local_num_agents):
            rewards[:, i] += EnvParameters.DY_COLLISION_COST * dynamic_collision_status[i]
            rewards[:, i] += EnvParameters.AG_COLLISION_COST * agent_collision_status[i]
            if reach_goal_status[i] == 1:
                rewards[:, i] += EnvParameters.GOAL_REWARD
            else:
                if actions[i] == opposite_actions[self.previous_action[i]]:
                    rewards[:, i] += EnvParameters.MOVE_BACK_COST
                if actions[i] == 0:
                    rewards[:, i] += self.idle_cost
                else:
                    rewards[:, i] += self.action_cost
                if self.time_step > self.sipps_max_len:
                    rewards[:, i] += EnvParameters.ADD_COST

            # Compute distance-based penalty.
            dis = np.sqrt((self.world.local_agents_poss[i][0] - self.world.local_agents_goal[i][0]) ** 2 +
                          (self.world.local_agents_poss[i][1] - self.world.local_agents_goal[i][1]) ** 2)
            rewards[:, i] -= EnvParameters.DIS_FACTOR * (TrainingParameters.GAMMA * dis - self.world.old_dis[i])
            self.world.old_dis[i] = dis

            # Get observation and update observation array.
            state = self.observe(i)
            rewards[:, i] += state[-1]
            obs[:, i, :, :, :] = state[0]
            vector[:, i, :3] = state[1]
            next_valid_actions.append(self.world.list_next_valid_actions(i))

        num_dynamic_collide = sum(dynamic_collision_status)
        num_agent_collide = sum(agent_collision_status)
        num_on_goal = sum(reach_goal_status)
        real_r = (EnvParameters.DY_COLLISION_COST * num_dynamic_collide +
                  EnvParameters.AG_COLLISION_COST * num_agent_collide +
                  EnvParameters.GOAL_REWARD * num_on_goal +
                  (self.local_num_agents - num_on_goal) * self.action_cost)
        self.previous_action = actions
        all_reach_goal = (num_on_goal == self.local_num_agents)
        vector[:, :, 3] = (self.sipp_coll_pair_num - len(self.new_collision_pairs)) / (self.sipp_coll_pair_num + 1)
        vector[:, :, 4] = self.time_step / self.episode_len
        vector[:, :, 5] = self.time_step / self.sipps_max_len
        vector[:, :, 6] = num_on_goal / self.local_num_agents
        vector[:, :, 7] = actions
        done = False
        success = False
        if all_reach_goal and self.time_step >= self.makespan:
            done = True
            if len(self.new_collision_pairs) <= self.sipp_coll_pair_num:
                success = True
        if self.time_step >= self.episode_len:
            done = True
        return obs, vector, rewards, done, next_valid_actions, num_on_goal, num_dynamic_collide, num_agent_collide, success, real_r

    def visualize_paths(self, paths, timestep, output_path):
        """
        Visualize the map with agents and their paths at a specific timestep.
        """
        # Get the map at the current timestep
        current_map = self.grid_seq[min(timestep, self.grid_seq.shape[0]-1)].copy()

        # Convert to RGB
        rgb_map = GridToRGB(current_map)

        # Add agent positions at current timestep
        for i, path in enumerate(paths):
            if timestep < len(path):
                pos = path[timestep]
                # Mark current position in bright red
                rgb_map[pos[0], pos[1]] = [255, 0, 0]  # Bright red for current position
            else:
                # If agent has reached goal, use last position
                pos = path[-1]
                # Mark final position in purple
                rgb_map[pos[0], pos[1]] = [255, 0, 255]  # Purple for agents at goal

        # Mark goal positions in blue
        for pos in self.goal_list:
            rgb_map[pos[0], pos[1]] = [0, 0, 255]  # Blue for goals

        # Save the image
        Image.fromarray(rgb_map.astype(np.uint8)).save(output_path)

    def create_video_from_images(self, image_folder, pattern, output_path, fps=5, scale=1):
        """
        Create an animated GIF from a sequence of images.
        """
        try:
            # Get list of all image filenames in the folder that match the pattern
            images = [img for img in os.listdir(image_folder) if img.endswith(".png") and pattern in img]

            # Sort images numerically by extracting the number from the filename
            images.sort(key=lambda x: int(re.search(r'\d+', x).group()))

            if not images:
                print(f"No images found matching pattern: {pattern}")
                return False

            print(f"Found {len(images)} images for GIF creation")

            # Ensure output path has .gif extension
            if not output_path.endswith('.gif'):
                output_path = output_path.replace('.mp4', '.gif')
                output_path = output_path.replace('.avi', '.gif')
                if not output_path.endswith('.gif'):
                    output_path += '.gif'

            # Load all images
            frames = []
            for image in images:
                image_path = os.path.join(image_folder, image)
                img = Image.open(image_path)

                # Resize if needed
                if scale != 1:
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.NEAREST)

                frames.append(img)

            if not frames:
                print("No valid images found for GIF creation")
                return False

            # Calculate duration in milliseconds (1000 / fps)
            duration = int(1000 / fps)

            # Save as animated GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=duration,
                loop=0  # Loop forever
            )

            print(f"Animated GIF created successfully: {output_path}")
            return True

        except Exception as e:
            print(f"Error creating GIF: {e}")
            return False

    def visualize_solution(self, output_dir="sipps_visualizations", stage="final", is_final=False):
        """
        Visualize the solution found by the LNS2 algorithm.

        Args:
            output_dir: Base directory to save the visualization images
            stage: Stage of the solution process ("initial", "iteration_X", or "final")
            is_final: Whether this is the final solution (no collisions)
        """
        # Create a stage-specific directory
        stage_dir = os.path.join(output_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)

        # Get the paths from the LNS2 model
        paths = self.lns2_model.vector_path

        # Find the longest path (maximum number of timesteps needed)
        max_path_length = max([len(path) for path in paths])
        print(f"Visualizing {stage} solution with maximum path length: {max_path_length} timesteps")

        # Count collisions - safely handle the case when neighbor isn't available
        try:
            # Try to access neighbor.colliding_pairs
            if hasattr(self.lns2_model, 'neighbor') and hasattr(self.lns2_model.neighbor, 'colliding_pairs'):
                num_collisions = len(self.lns2_model.neighbor.colliding_pairs)
            else:
                # If not available, use sipp_coll_pair_num if it exists
                num_collisions = getattr(self, 'sipp_coll_pair_num', 0)
        except Exception as e:
            # If any error occurs, default to 0
            print(f"Warning: Could not determine collision count: {e}")
            num_collisions = 0

        # Visualize every timestep until all agents reach their goals
        for t in range(0, max_path_length):
            self.visualize_paths(paths, t, os.path.join(stage_dir, f"t{t:03d}.png"))

        # Create an animated GIF from the generated images
        print(f"Creating animated GIF for {stage} solution...")
        self.create_video_from_images(stage_dir, "t", os.path.join(stage_dir, "animation.gif"), fps=3)

        # Create a larger-scale GIF for better visibility
        print(f"Creating larger-scale animation for {stage} solution...")
        self.create_video_from_images(stage_dir, "t", os.path.join(stage_dir, "animation_large.gif"), fps=3, scale=2)

        # If this is the final solution (no collisions), also save it to the final directory
        if is_final or stage == "final":
            final_dir = os.path.join(output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)

            # Copy the images to the final directory
            for t in range(0, max_path_length):
                self.visualize_paths(paths, t, os.path.join(final_dir, f"t{t:03d}.png"))

            # Create animated GIFs in the final directory
            self.create_video_from_images(final_dir, "t", os.path.join(final_dir, "animation.gif"), fps=3)
            self.create_video_from_images(final_dir, "t", os.path.join(final_dir, "animation_large.gif"), fps=3, scale=2)

            print(f"Final solution with {num_collisions} collisions saved to {final_dir}/")

        print(f"Visualization for {stage} solution with {num_collisions} collisions complete. Saved to {stage_dir}/")

    def _global_reset(self, cl_num_task):
        # Reinitialize the MAPF task (this call resets start/goal positions, etc.)
        self.global_set_world(cl_num_task)

        # Dynamic obstacles already created in global_set_world
        # Just reset them to initial state
        self.grid.ResetDynamicObstacles()
        self.grid.ClearGrid(gridtype="grid_dynamic")
        self.grid.RenderAllObstacles(typ="dynamic")
        self.grid.RenderGrid()
        # Generate RGB representation
        rgb_grid = GridToRGB(self.grid.grid)

        # Debug visualization by debug_dynamic
        # self.debug_dynamic(did='d1')
        # self.debug_dynamic(did='d2')
        # self.debug_gridmap(self.grid_seq, marker = "base")

        # Initialize the LNS2 planner with the dynamic map sequence directly
        # No conversion needed - passing grid_seq directly to mylns2
        self.lns2_model = my_lns2.MyLns2(self.env_id * 123, self.grid_seq, self.start_list, self.goal_list, self.global_num_agent, self.map.shape[0])

        # Set visualization delay to make the simulation more real-time
        # Adjust this value to control the speed of the visualization (higher = slower)
        self.lns2_model.setVisualizationDelay(200)  # 200ms delay between steps
        self.lns2_model.init_pp()
        self.paths = self.lns2_model.vector_path
        # Initialize dynamic state for the agents based on the planned paths.
        self.dynamic_state = DyState(self.paths, self.global_num_agent, self.map.shape)
        self.idle_cost = EnvParameters.IDLE_COST[cl_num_task]
        self.action_cost = EnvParameters.ACTION_COST[cl_num_task]
        self.off_route_factor = EnvParameters.OFF_ROUTE_FACTOR[cl_num_task]

        # Visualize the initial solution after global reset, but only if visualization is enabled
        # This is not the final solution, so is_final=False
        if self.enable_visualization:
            self.visualize_solution(stage="initial", is_final=False)

        return

    def _local_reset(self, local_num_agents, first_time):
        """
        Reset the environment locally for a subset of agents (local agents).

        This method:
          1. Selects a random subset of agents from the global set.
          2. Resets the time step and previous action information.
          3. If not the first reset, stores previous paths for continuity.
          4. Updates agent paths and re-computes dynamic state for local tasks.
          5. Reinitializes utility maps and resets collision pairs.
        """
        # Step the dynamic obstacles before processing agent actions:
        self.update_dyn_map(0)

        self.local_num_agents = local_num_agents
        new_agents = random.sample(range(self.global_num_agent), local_num_agents)
        self.time_step = 0
        self.previous_action = np.zeros(local_num_agents)
        if not first_time:
            prev_path = {}
            for local_index in range(self.local_num_agents):
                prev_path[self.world.local_agents[local_index]] = self.sipps_path[local_index]
            prev_agents = self.local_agents
        else:
            prev_agents = None
            prev_path = None  # No previous agents, new task.
        path_new_agent = {}
        for global_index in new_agents:
            path_new_agent[global_index] = self.paths[global_index]  # Use the precomputed path.
        if not first_time:
            for local_index in range(self.local_num_agents):
                self.paths[self.world.local_agents[local_index]] = self.sipps_path[local_index]
        self.local_agents = new_agents
        self.sipp_coll_pair_num = self.lns2_model.calculate_sipps(self.local_agents)
        self.makespan = self.lns2_model.makespan
        self.sipps_path = self.lns2_model.sipps_path
        self.sipps_max_len = max([len(path) for path in self.sipps_path])
        # Reset dynamic state for local agents.
        self.dynamic_state.reset_local_tasks(self.local_agents, path_new_agent, prev_agents, prev_path, self.makespan + 1)
        # Reset world state for local agents.
        self.world.reset_local_tasks(self.fix_state, self.fix_state_dict, self.start_list, self.local_agents)
        # Initialize utility maps for space utilization.
        self.agent_util_map_action = [np.zeros((5, self.map.shape[0], self.map.shape[1])) for _ in range(2)]
        self.agent_util_map_vertex = [np.zeros((self.map.shape[0], self.map.shape[1])) for _ in range(2)]
        for local_i in range(self.local_num_agents):
            self.agent_util_map_vertex[-1][self.world.local_agents_poss[local_i]] += 1
        self.new_collision_pairs = set()
        self.update_ulti()
        self.predict_next()

        # Visualize the solution after local reset
        # Use a counter to track iterations
        if not hasattr(self, 'local_reset_counter'):
            self.local_reset_counter = 1
        else:
            self.local_reset_counter += 1

        # Determine if this is the final solution (no collisions)
        # Safely handle the case when sipp_coll_pair_num might not be available
        try:
            is_final = (self.sipp_coll_pair_num == 0)
        except AttributeError:
            # If sipp_coll_pair_num is not available, assume it's not final
            is_final = False

        # Visualize the solution after this iteration, but only if visualization is enabled
        # Only visualize if this is not the first time (to avoid duplicate with initial)
        if self.enable_visualization and not first_time and self.local_reset_counter % 5 == 0:
            self.visualize_solution(
                stage=f"iteration_{self.local_reset_counter}",
                is_final=is_final
            )

        # If this is the final iteration of the entire process (e.g., in driver.py),
        # we'll save a separate final visualization at the end

        return

    def list_next_valid_actions(self, local_agent_index):
        """Return the list of valid actions for the agent at local_agent_index."""
        return self.world.list_next_valid_actions(local_agent_index)
