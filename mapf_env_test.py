import os
import random
import sys
import time
import re
from PIL import Image
import numpy as np
import cv2
from lns2.build import my_lns2
from GridBasedPathPlanning.Environment.GridMapEnv import GridMapEnv, GridToRGB
from GridBasedPathPlanning.RL_Environment.environment.map_generator import map_to_value
from pathlib import Path

# Define action directions
dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}  # 0: stay, 1:right, 2: down, 3:left, 4:up
actionDict = {v: k for k, v in dirDict.items()}

FREE  = {'.', 'G', 'S'}        # amit szabadnak tekintünk
BLOCK = {'@', 'T', 'O'}        # akadály jelek

def read_movingai_map(path):
    """
    Moving-AI .map  → uint8 mátrix   (0 = szabad, 1 = akadály)
    Robusztus fejléc-kezeléssel.
    """
    with open(path, 'r') as f:
        header = {}
        # tetszőleges sorrendben, üres sorokat átugorva
        for line in f:
            line = line.strip()
            if not line:
                continue                       # üres sor
            if line.lower().startswith('map'):
                break                          # vége a fejlécnek
            key, val = line.split()[:2]        # csak az első 2 token kell
            header[key.lower()] = val

        h = int(header['height'])
        w = int(header['width'])

        grid = np.zeros((h, w), dtype=np.uint8)
        for r in range(h):
            line = f.readline().rstrip('\n')
            for c, ch in enumerate(line):
                grid[r, c] = 0 if ch in FREE else 1
    return grid

def load_static_map(path):
    ext = Path(path).suffix.lower()
    if ext == '.png':
        return map_to_value(np.asarray(Image.open(path)))
    elif ext == '.map':                 # MovingAI
        return read_movingai_map(path)
    else:
        raise ValueError(f'Ismeretlen formátum: {ext}')

# Define a more complete version of the DyState class
class SimpleDyState:
    def __init__(self, all_paths, num_agents, grid_shape):
        """Initialize the dynamic state with all agent paths"""
        self.max_len = max([len(path) for path in all_paths])
        self.grid_shape = grid_shape
        self.num_agents = num_agents
        self.global_num_agents = num_agents
        self.local_num_agents = None

        # Check if the map is too large for full 3D arrays
        map_size = grid_shape[0] * grid_shape[1]

        # For very large maps, use a sparse representation
        if map_size > 10000 or self.max_len > 1000:  # For maps larger than 100x100 or very long paths
            print(f"Large map detected ({grid_shape[0]}x{grid_shape[1]}) or long paths ({self.max_len} steps). Using memory-efficient sparse representation.")

            # Use a sparse representation: dictionary mapping (t, x, y) to count
            self.use_sparse = True
            self.state_sparse = {}
            self.util_map_action_sparse = {}

            # Fill the sparse state with agent positions at each timestep
            for agent_id in range(num_agents):
                path = all_paths[agent_id]
                path_len = len(path)

                for t in range(self.max_len):
                    if t < path_len:
                        pos = path[t]
                    else:
                        pos = path[-1]  # Stay at the goal after reaching it

                    # Increment the count of agents at this position and timestep
                    key = (t, pos[0], pos[1])
                    self.state_sparse[key] = self.state_sparse.get(key, 0) + 1

                    # Update action utility maps
                    if t > 0:
                        if t < path_len:
                            prev_pos = path[t-1]
                        else:
                            prev_pos = path[-1]

                        # Calculate the action taken to move from prev_pos to pos
                        action = actionDict.get((pos[0] - prev_pos[0], pos[1] - prev_pos[1]), 0)  # Default to 'stay' if not found
                        action_key = (t, action, pos[0], pos[1])
                        self.util_map_action_sparse[action_key] = self.util_map_action_sparse.get(action_key, 0) + 1
        else:
            # For smaller maps, use the original dense representation
            self.use_sparse = False

            # Create a 3D array to store the dynamic state
            # Each cell contains the number of agents occupying it at each timestep
            self.state = np.zeros((self.max_len, grid_shape[0], grid_shape[1]), dtype=np.float32)  # Use float32 instead of float64

            # Create action utility maps (similar to mapf_gym's DyState)
            self.util_map_action = [np.zeros((5, grid_shape[0], grid_shape[1]), dtype=np.float32) for _ in range(self.max_len)]

            # Fill the state with agent positions at each timestep
            for agent_id in range(num_agents):
                path = all_paths[agent_id]
                path_len = len(path)

                for t in range(self.max_len):
                    if t < path_len:
                        pos = path[t]
                    else:
                        pos = path[-1]  # Stay at the goal after reaching it

                    # Increment the count of agents at this position and timestep
                    self.state[t][pos[0]][pos[1]] += 1

                    # Update action utility maps
                    if t > 0:
                        if t < path_len:
                            prev_pos = path[t-1]
                        else:
                            prev_pos = path[-1]

                        # Calculate the action taken to move from prev_pos to pos
                        action = actionDict.get((pos[0] - prev_pos[0], pos[1] - prev_pos[1]), 0)  # Default to 'stay' if not found
                        self.util_map_action[t][action][pos[0]][pos[1]] += 1

    def reset_local_tasks(self, new_agents, path_new_agent, prev_agents=None, old_path=None, new_max_len=None):
        """Update the dynamic state after replanning

        Args:
            new_agents: List of agent IDs that are being replanned
            path_new_agent: Dictionary mapping agent IDs to their new paths
            prev_agents: List of agent IDs that were previously replanned (can be None)
            old_path: Dictionary mapping agent IDs to their old paths (can be None)
            new_max_len: New maximum path length (can be None)
        """
        self.local_num_agents = len(new_agents)

        # Update max_len if provided
        if new_max_len is not None:
            old_max_len = self.max_len
            self.max_len = new_max_len

            # Handle different representations
            if self.use_sparse:
                # For sparse representation, we need to filter out keys with t >= new_max_len
                if new_max_len < old_max_len:
                    # Remove entries beyond the new max_len
                    self.state_sparse = {k: v for k, v in self.state_sparse.items() if k[0] < new_max_len}
                    self.util_map_action_sparse = {k: v for k, v in self.util_map_action_sparse.items() if k[0] < new_max_len}
            else:
                # For dense representation, adjust arrays
                diff = old_max_len - new_max_len
                if diff > 0:
                    # Truncate arrays if new_max_len is smaller
                    self.state = self.state[:self.max_len, :, :]
                    self.util_map_action = self.util_map_action[:self.max_len]
                elif diff < 0:
                    # Extend arrays if new_max_len is larger
                    # Create new arrays with the new size
                    new_state = np.zeros((self.max_len, self.grid_shape[0], self.grid_shape[1]), dtype=np.float32)
                    new_util_map_action = [np.zeros((5, self.grid_shape[0], self.grid_shape[1]), dtype=np.float32) for _ in range(self.max_len)]

                    # Copy existing data
                    new_state[:old_max_len] = self.state
                    for t in range(old_max_len):
                        new_util_map_action[t] = self.util_map_action[t]

                    # For the new timesteps, copy the last timestep
                    for t in range(old_max_len, self.max_len):
                        new_state[t] = self.state[-1]
                        new_util_map_action[t] = self.util_map_action[-1]

                    # Update state and util_map_action
                    self.state = new_state
                    self.util_map_action = new_util_map_action

        # Determine which agents to delete from the dynamic state
        delete_agents = set()
        if prev_agents is not None:
            # Delete agents that were in prev_agents but not in new_agents
            delete_agents = set(prev_agents) - set(new_agents)
        else:
            # If prev_agents is None, delete all new_agents
            delete_agents = set(new_agents)

        # First, remove the old paths of the agents being deleted
        for agent_id in delete_agents:
            if old_path is not None and agent_id in old_path:
                agent_path = old_path[agent_id]
            elif agent_id in path_new_agent:
                agent_path = path_new_agent[agent_id]
            else:
                continue  # Skip if we don't have a path for this agent

            path_len = len(agent_path)

            for t in range(self.max_len):
                if t < path_len:
                    pos = agent_path[t]
                else:
                    pos = agent_path[-1]  # Stay at the goal after reaching it

                # Handle different representations
                if self.use_sparse:
                    # Decrement the count in the sparse representation
                    key = (t, pos[0], pos[1])
                    if key in self.state_sparse:
                        self.state_sparse[key] -= 1
                        if self.state_sparse[key] <= 0:
                            del self.state_sparse[key]  # Remove if count reaches 0
                else:
                    # Decrement the count in the dense representation
                    self.state[t][pos[0]][pos[1]] -= 1

                # Update action utility maps
                if t > 0:
                    if t < path_len:
                        prev_pos = agent_path[t-1]
                    else:
                        prev_pos = agent_path[-1]

                    # Calculate the action taken to move from prev_pos to pos
                    action = actionDict.get((pos[0] - prev_pos[0], pos[1] - prev_pos[1]), 0)  # Default to 'stay' if not found

                    # Handle different representations
                    if self.use_sparse:
                        # Decrement the count in the sparse representation
                        action_key = (t, action, pos[0], pos[1])
                        if action_key in self.util_map_action_sparse:
                            self.util_map_action_sparse[action_key] -= 1
                            if self.util_map_action_sparse[action_key] <= 0:
                                del self.util_map_action_sparse[action_key]  # Remove if count reaches 0
                    else:
                        # Decrement the count in the dense representation
                        self.util_map_action[t][action][pos[0]][pos[1]] -= 1

        # Then, add the new paths of the agents being replanned
        for agent_id in new_agents:
            if agent_id not in path_new_agent:
                continue  # Skip if we don't have a new path for this agent

            new_path = path_new_agent[agent_id]
            path_len = len(new_path)

            for t in range(self.max_len):
                if t < path_len:
                    pos = new_path[t]
                else:
                    pos = new_path[-1]  # Stay at the goal after reaching it

                # Handle different representations
                if self.use_sparse:
                    # Increment the count in the sparse representation
                    key = (t, pos[0], pos[1])
                    self.state_sparse[key] = self.state_sparse.get(key, 0) + 1
                else:
                    # Increment the count in the dense representation
                    self.state[t][pos[0]][pos[1]] += 1

                # Update action utility maps
                if t > 0:
                    if t < path_len:
                        prev_pos = new_path[t-1]
                    else:
                        prev_pos = new_path[-1]

                    # Calculate the action taken to move from prev_pos to pos
                    action = actionDict.get((pos[0] - prev_pos[0], pos[1] - prev_pos[1]), 0)  # Default to 'stay' if not found

                    # Handle different representations
                    if self.use_sparse:
                        # Increment the count in the sparse representation
                        action_key = (t, action, pos[0], pos[1])
                        self.util_map_action_sparse[action_key] = self.util_map_action_sparse.get(action_key, 0) + 1
                    else:
                        # Increment the count in the dense representation
                        self.util_map_action[t][action][pos[0]][pos[1]] += 1

    def get_dynamic_grid_seq(self):
        """Convert the dynamic state to a grid sequence for visualization"""
        # Create a grid sequence with the same shape as the dynamic state
        grid_seq = np.zeros((self.max_len, self.grid_shape[0], self.grid_shape[1]), dtype=np.float32)

        # Handle different representations
        if self.use_sparse:
            # For sparse representation, only fill in the non-zero entries
            for (t, x, y), value in self.state_sparse.items():
                if t < self.max_len and 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1]:
                    grid_seq[t, x, y] = value
        else:
            # For dense representation, copy the state directly
            # Cells with agents are marked as occupied (value > 0)
            for t in range(self.max_len):
                for i in range(self.grid_shape[0]):
                    for j in range(self.grid_shape[1]):
                        if self.state[t][i][j] > 0:
                            grid_seq[t][i][j] = self.state[t][i][j]

        return grid_seq


def create_test_env(static_map_path="GridBasedPathPlanning/RL_Environment/data/cleaned_empty/empty-48-48-random-10_60_agents.png", num_agents=5, num_dynamic_obstacles=10, frames=2420, output_dir="test_images", visualize=True):
    """
    Create a test environment with a static map, dynamic obstacles, and random agent start/goal positions.

    Args:
        static_map_path: Path to the static map image file
        num_agents: Number of agents to place (default: 5)
        num_dynamic_obstacles: Number of dynamic obstacles to add (default: 10)
        frames: Number of frames for the dynamic map sequence (default: 420)
        output_dir: Directory to save output images

    Returns:
        grid: The GridMapEnv instance
        grid_seq: The dynamic map sequence
        start_list: List of agent start positions
        goal_list: List of agent goal positions
    """
    if visualize:
        os.makedirs(output_dir, exist_ok=True)

    # Load the static map from file
    print(f"Loading static map from {static_map_path}")
    # Convert the image to a value map (0 for free space, 1 for obstacles)
    map_v = load_static_map(static_map_path)

    # Get map dimensions
    height, width = map_v.shape
    print(f"Map dimensions: {height}x{width}")

    # Create the grid environment with the map dimensions
    grid = GridMapEnv(grid_size=(height, width))

    # Set the static grid
    grid.SetStaticGrid(map_v)

    # For large maps, limit the number of frames to reduce memory usage
    if height * width > 10000:  # For maps larger than 100x100
        # Limit frames based on map size
        max_frames = 1000
        if frames > max_frames:
            print(f"Large map detected. Limiting frames from {frames} to {max_frames} to save memory")
            frames = max_frames

    # Add dynamic obstacles - use smaller range for large maps
    if height > 512:
        # Use larger range for large maps to make obstacles bigger
        d_range = (64, 256)
    if height > 256 and height <= 512:
        # Use larger range for large maps to make obstacles bigger
        d_range = (32, 128)
    if height > 128 and height <= 256:
        # Use larger range for large maps to make obstacles bigger
        d_range = (16, 64)
    else:
        d_range = (4, 8)

    grid.addRandomDynamicObstacles(no_of_obs=num_dynamic_obstacles, d_range=d_range, alpha=0.3)

    # Generate the grid sequence
    grid.ResetDynamicObstacles()
    grid_seq = grid.GenerateGridSequence(frames, reset_after=True)

    # Force garbage collection to free memory
    import gc
    gc.collect()

    # Visualize the initial map if visualization is enabled
    if visualize:
        grid.RenderAllObstacles(typ="static")
        grid.RenderAllObstacles(typ="dynamic")
        grid.RenderGrid()
        init_arr = GridToRGB(grid.grid)
        Image.fromarray(init_arr.astype(np.uint8)).save(os.path.join(output_dir, "initial_map.png"))
        print(f"Saved initial map to {os.path.join(output_dir, 'initial_map.png')}")

    # Generate random start and goal positions for agents
    start_list = []
    goal_list = []

    # Calculate minimum Manhattan distance between start and goal
    min_manhattan_distance = height // 2  # At least half the map height
    print(f"Using minimum Manhattan distance of {min_manhattan_distance} between start and goal positions")

    # Helper function to calculate Manhattan distance
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Helper function to check if a position is valid (not an obstacle and not already occupied)
    def is_valid_position(pos, positions_list):
        x, y = pos
        # Check if position is within bounds
        if x < 0 or x >= height or y < 0 or y >= width:
            return False
        # Check if position is not an obstacle in static map
        if map_v[x, y] > 0:
            return False
        # Check if position is not a dynamic obstacle at time 0
        if grid_seq[0, x, y] > 0.1:
            return False
        # Check if position is not already occupied by another agent
        if pos in positions_list:
            return False
        return True

    # Generate start positions
    for _ in range(num_agents):
        attempts = 0
        while attempts < 1000:  # Limit attempts to avoid infinite loops
            pos = (np.random.randint(0, height), np.random.randint(0, width))
            if is_valid_position(pos, start_list):
                start_list.append(pos)
                break
            attempts += 1
        if attempts >= 1000:
            print(f"Warning: Could not find valid start position after 1000 attempts")
            # Use a fallback position - scan the map systematically
            for x in range(height):
                for y in range(width):
                    if is_valid_position((x, y), start_list):
                        start_list.append((x, y))
                        break
                if len(start_list) == _+1:
                    break

    # Generate goal positions with minimum Manhattan distance from corresponding start positions
    for i in range(num_agents):
        start_pos = start_list[i]
        attempts = 0
        found_valid_goal = False

        while attempts < 2000 and not found_valid_goal:  # Increased attempts for finding distant goals
            pos = (np.random.randint(0, height), np.random.randint(0, width))

            # Check if position is valid and has minimum Manhattan distance from start
            if is_valid_position(pos, goal_list) and manhattan_distance(start_pos, pos) >= min_manhattan_distance:
                goal_list.append(pos)
                found_valid_goal = True
                print(f"Agent {i}: Start {start_pos}, Goal {pos}, Manhattan distance: {manhattan_distance(start_pos, pos)}")
                break

            attempts += 1

        if not found_valid_goal:
            print(f"Warning: Could not find valid goal position with minimum Manhattan distance after 2000 attempts")
            print(f"Falling back to any valid goal position for agent {i}")

            # First try to find any valid goal with at least some distance
            reduced_distance = min_manhattan_distance // 2
            while reduced_distance > 0 and not found_valid_goal:
                print(f"Trying with reduced distance: {reduced_distance}")
                attempts = 0
                while attempts < 500 and not found_valid_goal:
                    pos = (np.random.randint(0, height), np.random.randint(0, width))
                    if is_valid_position(pos, goal_list) and manhattan_distance(start_pos, pos) >= reduced_distance:
                        goal_list.append(pos)
                        found_valid_goal = True
                        print(f"Agent {i}: Found goal with reduced distance {reduced_distance}: {pos}")
                        break
                    attempts += 1
                reduced_distance = reduced_distance // 2

            # If still not found, use any valid position
            if not found_valid_goal:
                attempts = 0
                while attempts < 1000 and not found_valid_goal:
                    pos = (np.random.randint(0, height), np.random.randint(0, width))
                    if is_valid_position(pos, goal_list):
                        goal_list.append(pos)
                        found_valid_goal = True
                        print(f"Agent {i}: Found any valid goal: {pos}")
                        break
                    attempts += 1

            # Last resort: scan the map systematically
            if not found_valid_goal:
                print(f"Last resort: scanning map systematically for agent {i}")
                for x in range(height):
                    for y in range(width):
                        if is_valid_position((x, y), goal_list):
                            goal_list.append((x, y))
                            found_valid_goal = True
                            print(f"Agent {i}: Found goal by scanning: {(x, y)}")
                            break
                    if found_valid_goal:
                        break

    print(f"Created environment with {num_agents} agents")
    print(f"Start positions: {start_list}")
    print(f"Goal positions: {goal_list}")

    return grid, grid_seq, start_list, goal_list


def visualize_paths(grid_seq, start_list, goal_list, paths, timestep, output_path, dynamic_state=None):
    """
    Visualize the map with agents and their paths at a specific timestep.

    Args:
        grid_seq: The dynamic map sequence
        start_list: List of agent start positions
        goal_list: List of agent goal positions
        paths: List of agent paths
        timestep: The timestep to visualize
        output_path: Path to save the output image
        dynamic_state: Optional dynamic state object to use for visualization
    """
    # Get the map at the current timestep
    if dynamic_state is not None:
        # Use the dynamic state to get the current map
        # This ensures we're using the updated dynamic state after replanning
        current_map = np.zeros_like(grid_seq[0])

        # Add dynamic obstacles from grid_seq
        if timestep < grid_seq.shape[0]:
            current_map = grid_seq[timestep].copy()
        else:
            current_map = grid_seq[-1].copy()
    else:
        # Use the original grid_seq
        current_map = grid_seq[min(timestep, grid_seq.shape[0]-1)].copy()

    # Convert to RGB
    rgb_map = GridToRGB(current_map)

    # draw each agent’s trail up to this timestep
    for i, path in enumerate(paths):
        # build a list of points up to t
        pts = path[:timestep+1] if timestep < len(path) else path
        if len(pts) >= 2:
            # convert to numpy array of shape (n,1,2) in (col,row) order
            poly = np.array([[p[1], p[0]] for p in pts], dtype=np.int32).reshape(-1,1,2)
            # draw a thin semi-transparent grey line
            cv2.polylines(rgb_map, [poly], isClosed=False, color=(200,200,200), thickness=2, lineType=cv2.LINE_AA)

    # now draw the agents themselves
    for i, path in enumerate(paths):
        # pick position: current or last
        pos = path[timestep] if timestep < len(path) else path[-1]
        # choose color: red if still moving, purple if at goal
        color = (255,0,0) if timestep < len(path) else (255,0,255)
        # draw filled circle at (col,row)
        cv2.circle(rgb_map, (pos[1], pos[0]), radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # draw goals
    for gx, gy in goal_list:
        cv2.circle(rgb_map, (gy, gx), radius=3, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)

    # save
    Image.fromarray(rgb_map).save(output_path)
    

def create_video_from_images(image_folder, pattern, output_path, fps=5, scale=1):
    """
    Create an animated GIF from a sequence of images.

    Args:
        image_folder: Folder containing the images
        pattern: Pattern to match image files (e.g., 'final_solution_t')
        output_path: Path to save the output GIF
        fps: Frames per second for the animation
        scale: Scale factor for the images (default: 1)

    Returns:
        True if GIF was created successfully, False otherwise
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

        # Memory management: Limit the number of frames to process
        max_frames = 500  # Limit to 500 frames to prevent memory issues
        if len(images) > max_frames:
            print(f"Warning: Limiting GIF to {max_frames} frames to prevent memory issues")
            # Take every nth frame to maintain the full timeline but reduce memory usage
            step = len(images) // max_frames + 1
            images = images[::step]
            if len(images) > max_frames:
                images = images[:max_frames]
            print(f"Reduced to {len(images)} frames")

        # Ensure output path has .gif extension
        if not output_path.endswith('.gif'):
            output_path = output_path.replace('.mp4', '.gif')
            output_path = output_path.replace('.avi', '.gif')
            if not output_path.endswith('.gif'):
                output_path += '.gif'

        # Process images in batches to reduce memory usage
        batch_size = 50  # Process 50 images at a time
        frames = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_frames = []

            for image in batch:
                image_path = os.path.join(image_folder, image)
                img = Image.open(image_path)

                # Resize if needed
                if scale != 1:
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.NEAREST)

                batch_frames.append(img)

            frames.extend(batch_frames)

            # Force garbage collection after each batch
            import gc
            gc.collect()

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

        # Clear frames to free memory
        frames.clear()
        import gc
        gc.collect()

        return True

    except Exception as e:
        print(f"Error creating GIF: {e}")
        return False


def test_mapf_env(static_map_path="GridBasedPathPlanning/RL_Environment/data/cleaned_empty/empty-48-48-random-10_60_agents.png", num_agents=5, num_dynamic_obstacles=10, frames=420, output_dir="test_images", visualize=True):
    """
    Test the MAPF environment with dynamic obstacles and iterative SIPPS planning.

    Args:
        map_size: Size of the square grid (default: 64x64)
        num_agents: Number of agents to place (default: 5)
        num_dynamic_obstacles: Number of dynamic obstacles to add (default: 10)
        frames: Number of frames for the dynamic map sequence (default: 420)
        output_dir: Directory to save output images
    """
    # Import time for performance measurement
    import time

    # Create output directory if visualization is enabled
    if visualize:
        os.makedirs(output_dir, exist_ok=True)

    # Create a performance log file
    perf_log_path = os.path.join(output_dir, "mapf_performance_log.txt")
    with open(perf_log_path, 'w') as f:
        f.write("MAPF Performance Analysis Log\n")
        f.write("===========================\n\n")

    # Create the test environment
    print("\nCreating test environment...")
    start_time = time.time()
    grid, grid_seq, start_list, goal_list = create_test_env(
        static_map_path=static_map_path,
        num_agents=num_agents,
        num_dynamic_obstacles=num_dynamic_obstacles,
        frames=frames,
        output_dir=output_dir,
        visualize=visualize
    )
    env_creation_time = time.time() - start_time
    print(f"Environment creation completed in {env_creation_time:.2f} seconds")

    # Initialize mylns2 with the dynamic map sequence
    seed = 42
    print("\nInitializing mylns2 with dynamic map sequence...")
    init_start_time = time.time()

    # Optimize grid_seq to reduce memory usage and improve performance
    # For large maps, we can downsample the grid sequence in time
    # This reduces the memory footprint and speeds up initialization
    original_frames = len(grid_seq)

    # If the grid sequence is very large, downsample it
    if original_frames > 500:
        # Take every nth frame to reduce the sequence size
        step = original_frames // 500 + 1
        optimized_grid_seq = grid_seq[::step]
        print(f"Optimized grid sequence: reduced from {original_frames} to {len(optimized_grid_seq)} frames")
    else:
        optimized_grid_seq = grid_seq

    # Force garbage collection before initializing LNS2
    import gc
    gc.collect()

    # Detailed performance logging
    with open(os.path.join(output_dir, "detailed_performance_log.txt"), 'w') as f:
        f.write("Detailed Performance Analysis Log\n")
        f.write("===============================\n\n")

        # Log grid sequence optimization
        f.write(f"Grid sequence optimization: {time.time() - init_start_time:.2f} seconds\n")
        grid_opt_time = time.time()

        # Initialize LNS2 with the optimized grid sequence
        print("Creating LNS2 model...")
        lns2_model = my_lns2.MyLns2(seed, optimized_grid_seq, start_list, goal_list, num_agents, grid.grid_size[0])
        constructor_time = time.time() - grid_opt_time
        print(f"LNS2 constructor completed in {constructor_time:.2f} seconds")
        f.write(f"LNS2 constructor: {constructor_time:.2f} seconds\n")

        # Disable Rerun visualization to save memory
        lns2_model.setUseRerun(False)
        lns2_model.setVisualizationDelay(0)  # No delay to speed up processing

        # Force garbage collection again
        gc.collect()

        # Initialize path planning
        print("Initializing path planning...")
        init_pp_start = time.time()
        lns2_model.init_pp()
        init_pp_time = time.time() - init_pp_start
        print(f"init_pp() completed in {init_pp_time:.2f} seconds")
        f.write(f"init_pp(): {init_pp_time:.2f} seconds\n")

        # Total initialization time
        init_time = time.time() - init_start_time
        print(f"LNS2 initialization completed in {init_time:.2f} seconds")
        f.write(f"Total LNS2 initialization: {init_time:.2f} seconds\n\n")

        # Percentage breakdown
        f.write("Percentage breakdown of LNS2 initialization:\n")
        f.write(f"Grid sequence optimization: {(grid_opt_time - init_start_time) / init_time * 100:.1f}%\n")
        f.write(f"LNS2 constructor: {constructor_time / init_time * 100:.1f}%\n")
        f.write(f"init_pp(): {init_pp_time / init_time * 100:.1f}%\n")

        # Most expensive part
        if constructor_time > init_pp_time:
            f.write("\nMost expensive part: LNS2 constructor\n")
            f.write("This is likely due to processing the dynamic map sequence.\n")
        else:
            f.write("\nMost expensive part: init_pp()\n")
            f.write("This is likely due to the initialization of the path planning algorithm.\n")

    # Return the total initialization time for the main performance log
    init_time = time.time() - init_start_time

    # Get the initial paths
    paths = lns2_model.vector_path
    print(f"Initial paths created for {len(paths)} agents")

    # Find the longest path (maximum number of timesteps needed)
    max_path_length = max([len(path) for path in paths])
    print(f"Initial maximum path length: {max_path_length} timesteps")

    # Skip initial path visualization to save memory
    # We'll only visualize the final solution

    # Implement iterative conflict resolution similar to mapf_gym
    max_iterations = 10
    iteration = 0

    # First, plan paths for all agents
    all_agents = list(range(num_agents))
    print(f"\nInitial planning for all agents {all_agents}...")
    initial_planning_start_time = time.time()

    # Plan paths one by one to avoid unnecessary waiting
    # This is similar to how mapf_gym handles planning
    for agent_id in all_agents:
        # Plan path for this agent
        agent_start_time = time.time()
        num_collisions = lns2_model.calculate_sipps([agent_id])
        agent_time = time.time() - agent_start_time
        print(f"Planning for agent {agent_id} complete with {num_collisions} collisions in {agent_time:.2f} seconds")

        # Get the updated path
        sipps_paths = lns2_model.sipps_path

        # Update the vector_path with the sipps_path result
        paths[agent_id] = sipps_paths[0]

        # Update the lns2_model's vector_path with our updated paths
        lns2_model.vector_path = paths

        # After each agent is planned, update the path table
        # This ensures that the next agent sees the current agent's path as a soft constraint
        lns2_model.calculate_sipps([agent_id])

    # Final planning to get the total number of collisions
    final_planning_start_time = time.time()
    num_collisions = lns2_model.calculate_sipps(all_agents)
    final_planning_time = time.time() - final_planning_start_time

    initial_planning_time = time.time() - initial_planning_start_time
    print(f"Initial planning complete with {num_collisions} collisions in {initial_planning_time:.2f} seconds")

    # Store the initial number of collisions
    initial_collisions = num_collisions
    previous_collisions = num_collisions

    # Create a dynamic state object to track agent positions over time
    # This is similar to the DyState class in mapf_gym
    dynamic_state = SimpleDyState(paths, num_agents, (grid_seq.shape[1], grid_seq.shape[2]))

    # Find the longest path (maximum number of timesteps needed)
    max_path_length = max([len(path) for path in paths])
    print(f"Maximum path length: {max_path_length} timesteps")

    # Visualize the initial paths if visualization is enabled
    if visualize:
        # Visualize every timestep until all agents reach their goals
        for t in range(0, max_path_length):
            visualize_paths(grid_seq, start_list, goal_list, paths, t,
                           os.path.join(output_dir, f"initial_paths_t{t:03d}.png"), dynamic_state)

    # If no collisions, we're done
    if num_collisions == 0:
        print("No collisions detected. Planning complete!")
        iterative_resolution_time = 0
    else:
        # Implement iterative conflict resolution
        print("\nStarting iterative conflict resolution...")
        iterative_resolution_start_time = time.time()

        # Store timing for each iteration
        iteration_times = []

        # Custom function to detect collisions and identify agents to replan
        def get_agents_to_replan():
            # Try to get colliding pairs from lns2_model.neighbor if it exists
            try:
                if hasattr(lns2_model, 'neighbor') and hasattr(lns2_model.neighbor, 'colliding_pairs'):
                    colliding_pairs = lns2_model.neighbor.colliding_pairs
                    print(f"Colliding pairs from lns2_model.neighbor: {colliding_pairs}")

                    # Extract the agents involved in collisions
                    agents_to_replan = set()
                    for pair in colliding_pairs:
                        agents_to_replan.add(pair[0])
                        agents_to_replan.add(pair[1])

                    return list(agents_to_replan), len(colliding_pairs)
            except Exception as e:
                print(f"Warning: Could not get colliding pairs from lns2_model.neighbor: {e}")

            # Fallback: Use the number of collisions from the last call to calculate_sipps
            # and select agents with the most conflicts
            print("Using fallback collision detection method")

            # If we have collisions but don't know which agents are involved,
            # replan for all agents (or a subset of them)
            if num_collisions > 0:
                # For simplicity, replan for all agents
                # In a more sophisticated implementation, we could use a heuristic
                # to select agents with the most conflicts
                agents_to_replan = all_agents
                print(f"Fallback: Replanning for all {len(agents_to_replan)} agents")
                return agents_to_replan, num_collisions
            else:
                # No collisions detected
                return [], 0

        # Iterative conflict resolution
        iteration = 1
        while iteration <= max_iterations:
            print(f"\nIteration {iteration}/{max_iterations}")
            iteration_start_time = time.time()

            # Identify agents involved in collisions using the lns2_model's colliding_pairs
            agents_to_replan, num_collisions = get_agents_to_replan()

            # If no agents to replan, we're done
            if len(agents_to_replan) == 0:
                print("No agents involved in collisions. Planning complete!")
                break

            print(f"Detected {num_collisions} collisions involving {len(agents_to_replan)} agents")
            print(f"Replanning for agents: {agents_to_replan}")

            # Store previous paths for continuity (similar to mapf_gym's _local_reset)
            prev_path = {}
            for agent_id in agents_to_replan:
                prev_path[agent_id] = paths[agent_id]

            # Call calculate_sipps with the agents involved in collisions
            replan_start_time = time.time()
            num_collisions = lns2_model.calculate_sipps(agents_to_replan)
            replan_time = time.time() - replan_start_time
            print(f"Replanning complete with {num_collisions} collisions in {replan_time:.2f} seconds")

            # Get the updated paths
            sipps_paths = lns2_model.sipps_path

            # Update the vector_path with the sipps_path results
            for i, agent_id in enumerate(agents_to_replan):
                paths[agent_id] = sipps_paths[i]

            # The key insight from mapf_gym is that after replanning, it updates the dynamic state
            # with the new paths, which ensures that other agents' paths are treated as soft constraints

            # Update the dynamic state with the new paths
            # This is similar to how mapf_gym calls dynamic_state.reset_local_tasks()
            # Convert sipps_paths to a dictionary mapping agent IDs to paths
            path_new_agent = {agent_id: sipps_paths[i] for i, agent_id in enumerate(agents_to_replan)}

            # Get the makespan (maximum path length)
            makespan = lns2_model.makespan

            # Call reset_local_tasks with the same parameters as mapf_gym
            dynamic_state.reset_local_tasks(agents_to_replan, path_new_agent, agents_to_replan, prev_path, makespan + 1)

            # Update the lns2_model's vector_path with our updated paths
            lns2_model.vector_path = paths

            # Now call calculate_sipps again with all agents to update the path table
            # This ensures that the path table is properly updated with the new paths
            num_collisions = lns2_model.calculate_sipps(all_agents)

            # Find the longest path (maximum number of timesteps needed)
            max_path_length = max([len(path) for path in paths])
            print(f"Maximum path length: {max_path_length} timesteps")

            # Skip iteration visualization to save memory
            # We'll only visualize the final solution

            # If the number of collisions is 0, we're done
            if num_collisions == 0:
                print("No more collisions detected. Planning complete!")
                break

            # In mapf_gym, we don't need to handle the "not making progress" case separately
            # The algorithm will naturally converge to a solution or reach the maximum iterations

            # Store the current number of collisions for the next iteration
            previous_collisions = num_collisions

            # Record the time for this iteration
            iteration_time = time.time() - iteration_start_time
            iteration_times.append(iteration_time)
            print(f"Iteration {iteration} completed in {iteration_time:.2f} seconds")

            iteration += 1

        # Calculate total iterative resolution time
        if iteration > 1:
            iterative_resolution_time = time.time() - iterative_resolution_start_time
            print(f"Iterative conflict resolution completed in {iterative_resolution_time:.2f} seconds")
        else:
            iterative_resolution_time = 0

    if iteration == max_iterations and num_collisions > 0:
        print(f"Warning: Maximum iterations reached with {num_collisions} collisions remaining")

    # Find the longest path (maximum number of timesteps needed)
    max_path_length = max([len(path) for path in paths])
    print(f"Final maximum path length: {max_path_length} timesteps")

    # Visualize the final solution if visualization is enabled
    if visualize:
        print("\nCreating final solution visualization...")
        # Memory management: Limit the number of frames to visualize
        max_frames = 500  # Limit to 500 frames to prevent memory issues
        if max_path_length > max_frames:
            print(f"Warning: Limiting visualization to {max_frames} frames to prevent memory issues")
            # Take every nth frame to maintain the full timeline but reduce memory usage
            step = max_path_length // max_frames + 1
            frames_to_visualize = range(0, max_path_length, step)
            if len(frames_to_visualize) > max_frames:
                frames_to_visualize = range(0, max_path_length, step + 1)
        else:
            frames_to_visualize = range(0, max_path_length)

        # Visualize selected timesteps until all agents reach their goals
        for t in frames_to_visualize:
            visualize_paths(grid_seq, start_list, goal_list, paths, t,
                           os.path.join(output_dir, f"final_solution_t{t:03d}.png"), dynamic_state)

            # Force garbage collection periodically
            if t % 100 == 0:
                import gc
                gc.collect()

    # Calculate total planning time
    total_planning_time = init_time + initial_planning_time + iterative_resolution_time

    # Write performance log
    with open(perf_log_path, 'a') as f:
        f.write(f"Environment creation: {env_creation_time:.2f} seconds\n")
        f.write(f"LNS2 initialization: {init_time:.2f} seconds\n")
        f.write(f"Initial planning: {initial_planning_time:.2f} seconds\n")
        f.write(f"Iterative conflict resolution: {iterative_resolution_time:.2f} seconds\n")
        f.write(f"Total planning time: {total_planning_time:.2f} seconds\n\n")

        f.write("Percentage breakdown:\n")
        f.write(f"Environment creation: {(env_creation_time/total_planning_time)*100:.1f}%\n")
        f.write(f"LNS2 initialization: {(init_time/total_planning_time)*100:.1f}%\n")
        f.write(f"Initial planning: {(initial_planning_time/total_planning_time)*100:.1f}%\n")
        f.write(f"Iterative conflict resolution: {(iterative_resolution_time/total_planning_time)*100:.1f}%\n\n")

        # Identify the most computationally expensive part
        max_time = max(env_creation_time, init_time, initial_planning_time, iterative_resolution_time)
        if max_time == env_creation_time:
            f.write("Most computationally expensive part: Environment creation\n")
            f.write("This is likely due to the generation of dynamic obstacles and grid sequence.\n")
        elif max_time == init_time:
            f.write("Most computationally expensive part: LNS2 initialization\n")
            f.write("This is likely due to the initialization of the path planning algorithm.\n")
        elif max_time == initial_planning_time:
            f.write("Most computationally expensive part: Initial planning\n")
            f.write("This is likely due to the SIPPS algorithm which has to search through the state space.\n")
        elif max_time == iterative_resolution_time:
            f.write("Most computationally expensive part: Iterative conflict resolution\n")
            f.write("This is likely due to the repeated calls to the SIPPS algorithm to resolve conflicts.\n")

        # If we have iteration times, analyze them
        if 'iteration_times' in locals() and iteration_times:
            f.write(f"\nIteration times analysis:\n")
            f.write(f"Number of iterations: {len(iteration_times)}\n")
            f.write(f"Average iteration time: {sum(iteration_times)/len(iteration_times):.2f} seconds\n")
            f.write(f"Minimum iteration time: {min(iteration_times):.2f} seconds\n")
            f.write(f"Maximum iteration time: {max(iteration_times):.2f} seconds\n")

            # Identify the most expensive iteration
            max_iter_time = max(iteration_times)
            max_iter_idx = iteration_times.index(max_iter_time) + 1
            f.write(f"Most expensive iteration: Iteration {max_iter_idx} ({max_iter_time:.2f} seconds)\n")

    if visualize:
        print(f"\nTest complete. Results saved to {output_dir}/")
        print(f"Final solution has {num_collisions} collisions")
        print(f"Performance analysis saved to {perf_log_path}")

        # Create GIF only for the final solution to save memory
        print("\nCreating animated GIF for final solution only...")
        gif_start_time = time.time()

        # Create GIF for final solution (after all replanning iterations) - slower animation (3 FPS)
        create_video_from_images(output_dir, "final_solution_t",
                               f"{output_dir}/final_solution.gif", fps=3)

        gif_time = time.time() - gif_start_time
        print(f"Animation creation completed in {gif_time:.2f} seconds")

        # Add visualization time to performance log
        with open(perf_log_path, 'a') as f:
            f.write(f"\nVisualization time: {gif_time:.2f} seconds\n")
    else:
        print(f"\nTest complete. Visualization disabled.")
        print(f"Final solution has {num_collisions} collisions")
        print(f"Performance analysis saved to {perf_log_path}")

    # Return the final paths and collision count
    return paths, num_collisions


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test MAPF environment with dynamic obstacles')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false', help='Disable visualization')
    parser.add_argument('--agents', type=int, default=5, help='Number of agents')
    parser.add_argument('--obstacles', type=int, default=15, help='Number of dynamic obstacles')
    parser.add_argument('--output', type=str, default='test_images', help='Output directory')
    parser.add_argument('--map', type=str, default='GridBasedPathPlanning/RL_Environment/data/cleaned_empty/empty-48-48-random-10_60_agents.png', help='Path to the static map')
    parser.set_defaults(visualize=True)  # Visualization is enabled by default

    args = parser.parse_args()

    test_mapf_env(
        static_map_path=args.map,
        num_agents=args.agents,
        num_dynamic_obstacles=args.obstacles,
        output_dir=args.output,
        visualize=args.visualize
    )