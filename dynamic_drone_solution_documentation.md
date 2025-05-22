# Dynamic Drone Solution Documentation

## Overview

The `dynamic_drone_solution.py` script provides a solution for planning continuous, kinematically feasible paths for fixed-wing drones in environments with both static and dynamic obstacles (radar zones). The solution integrates with the existing MAPF (Multi-Agent Path Finding) framework and extends it to handle the specific constraints of fixed-wing UAVs, which cannot stop mid-air and have limited turning capabilities.

## Key Features

- **Continuous Motion Planning**: Ensures drones maintain continuous motion without stopping, which is critical for fixed-wing UAVs
- **Dynamic Obstacle Avoidance**: Handles moving radar zones with temporal constraints
- **Kinematic Feasibility**: Validates trajectories against UAV dynamics constraints (velocity, acceleration, turn rate)
- **Temporal Planning**: Uses a Simple Temporal Network (STN) to coordinate multiple drones and avoid collisions
- **Visualization**: Provides animated visualizations of drone trajectories and dynamic obstacles

## UAV Dynamics Model

The solution models fixed-wing UAV dynamics with the following constraints:

```python
# UAV dynamics parameters
UAV_PARAMS = {
    'v_max': 10.0,        # Maximum velocity (m/s)
    'v_min': 0.5,         # Minimum velocity (m/s) - critical for fixed-wing UAVs
    'omega_max': 1.0,     # Maximum turn rate (rad/s)
    'a_max': 5.0,         # Maximum acceleration (m/s^2)
    'mass': 1.0,          # Mass (kg)
    'F_max': 20.0,        # Maximum force (N)
    'safety_distance': 2.0  # Minimum distance between UAVs (m)
}
```

These parameters ensure that:
1. Drones maintain a minimum velocity (cannot stop mid-air)
2. Drones cannot turn too sharply (limited by `omega_max`)
3. Drones have realistic acceleration/deceleration capabilities

## Solution Pipeline

The solution follows a multi-step pipeline:

1. **Initial Path Generation**: Generate discrete paths using the existing MAPF solver
2. **Schedule Conversion**: Convert discrete paths to a schedule format
3. **Temporal Plan Graph**: Build a graph representing temporal constraints
4. **Simple Temporal Network**: Construct a network to enforce temporal constraints
5. **Temporal Plan Optimization**: Optimize the plan to ensure continuous motion
6. **Trajectory Generation**: Generate final trajectories with control inputs
7. **Visualization**: Visualize the trajectories with dynamic obstacles

## Integration with Existing Code

The solution integrates with several components from the centralized library:

### Temporal Planning Components

- **Temporal Plan Graph (TPG)**: Used from `centralized/scheduling/tpg.py`
- **Simple Temporal Network (STN)**: Used from `centralized/scheduling/stn.py`
- **Optimization**: Used from `centralized/scheduling/minimize.py`

### SIPP Components

- **Safe Interval Path Planning**: Used from `centralized/graph_generation.py`
- **State and SippGrid classes**: Used for handling dynamic obstacles

## Ensuring Kinematically Feasible Plans

Several techniques are used to ensure kinematically feasible plans:

### 1. Continuous Motion Enforcement

The `OptimizationClass.generate_schedule()` method enforces continuous motion by:

```python
# Use a constant velocity approach for timing
constant_velocity = 3.0  # Units per time step
total_time = total_distance / constant_velocity

# Distribute time based on segment lengths
for i, wp in enumerate(sorted_waypoints):
    if i == 0:
        adjusted_time = 0.0
    else:
        # Calculate distance from start to this waypoint
        # Time is proportional to distance traveled
        adjusted_time = (cumulative_distance / total_distance) * total_time
```

This ensures that:
- Time is proportional to distance traveled
- Velocity remains relatively constant
- No stopping occurs between waypoints

### 2. Path Smoothing

The `_apply_linear_interpolation()` function creates smoother paths by:

```python
# Create a densely interpolated trajectory
interpolated = Trajectory(agent_id)

# Number of interpolation points between each pair of waypoints
num_interp_points = 5  # Increased for smoother motion

# Add interpolated points
for j in range(1, num_interp_points):
    alpha = j / float(num_interp_points)
    interp_x = wp1.x * (1 - alpha) + wp2.x * alpha
    interp_y = wp1.y * (1 - alpha) + wp2.y * alpha
    interp_t = t1 * (1 - alpha) + t2 * alpha
```

### 3. Dynamic Feasibility Validation

The `is_dynamically_feasible()` method validates trajectories against UAV constraints:

```python
def is_dynamically_feasible(self, uav_params):
    # Check velocity constraints
    if v_mag < uav_params['v_min'] or v_mag > uav_params['v_max']:
        return False

    # Check turn rate constraints
    if turn_rate > uav_params['omega_max']:
        return False

    # Check speed-curvature coupling
    if v_mag > 0:
        max_turn_rate = uav_params['F_max'] / (uav_params['mass'] * v_mag)
        if turn_rate > max_turn_rate:
            return False
```

### 4. Trajectory Adjustment

If a trajectory is not feasible, the solution attempts to adjust it:

```python
# Determine if we need to scale the trajectory
if max_velocity > uav_params['v_max']:
    # Need to slow down (stretch time)
    scale_factor = max_velocity / uav_params['v_max'] * 1.05

# Apply scaling to the entire trajectory
for i in range(1, len(trajectory.waypoints)):
    original_dt = trajectory.times[i] - trajectory.times[0]
    adjusted_time = trajectory.times[0] + original_dt * scale_factor
```

## Handling Dynamic Obstacles (Radar Zones)

Dynamic obstacles (radar zones) are handled as soft constraints with increased path cost (3 instead of 1). The solution:

1. Identifies dynamic obstacles in the grid sequence
2. Tracks their movement over time
3. Adds temporal constraints to the STN to avoid collisions
4. Visualizes the obstacles moving during planning

```python
# Check if this edge crosses any dynamic obstacle
if self._paths_intersect(loc_a, loc_b, obs_loc_1, obs_loc_2):
    # Option 1: Pass before the obstacle arrives
    self.edges.append({
        'vertex_a': edge.vertex_b,
        'vertex_b': None,  # Special marker for absolute time
        'bound': [-float('inf'), obs_t_1 - 0.1]
    })

    # Option 2: Pass after the obstacle has left
    self.edges.append({
        'vertex_a': None,
        'vertex_b': edge.vertex_a,
        'bound': [obs_t_2 + 0.1, float('inf')]
    })
```

## Visualization

The solution provides several visualization options:

1. **Trajectory Animation**: Shows drones moving along their paths
2. **Dynamic Obstacle Animation**: Shows radar zones moving through the environment
3. **Orientation Indicators**: Shows drone heading with arrows
4. **GIF/MP4 Output**: Creates animated GIFs or MP4 videos of the solution

## Notes on Implementation

- **Fallback Implementations**: The solution provides fallback implementations for STN and optimization when the centralized library components are not available
- **Time Scaling**: Dynamic obstacles are slowed down by a factor of 5 to make their movement more visible
- **Interpolation**: Linear interpolation is used for smoother visualization
- **Error Handling**: Robust error handling ensures the solution works even with challenging inputs

## Usage

```bash
python dynamic_drone_solution.py --map <map_path> --agents <num_agents> --obstacles <num_obstacles> --output <output_dir>
```

## Implementation Details

### Classes and Data Structures

#### Waypoint Class
```python
class Waypoint:
    def __init__(self, x, y, time=0.0):
        self.x = x
        self.y = y
        self.time = time
```
Represents a single point in a drone's trajectory with position and time.

#### Trajectory Class
```python
class Trajectory:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.waypoints = []
        self.times = []
        self.velocities = []
        self.accelerations = []
        self.turn_rates = []
```
Stores a complete trajectory for a drone, including waypoints, times, and computed dynamics.

#### SimpleTemporalNetwork Class
A fallback implementation of the STN from the centralized library, with added support for dynamic obstacles:
```python
class SimpleTemporalNetwork:
    def __init__(self, tpg=None, dynamic_obstacles=None):
        self.tpg = tpg
        self.vertices = []
        self.edges = []
        self.vmax = []
        self.dynamic_obstacles = dynamic_obstacles
```

### Key Functions

#### step1_generate_initial_paths
Generates initial discrete paths using the existing MAPF solver:
```python
def step1_generate_initial_paths(map_path, num_agents, num_obstacles, output_dir):
    # Create test environment
    env = create_test_env(map_path, num_agents, num_obstacles)
    # Run MAPF solver
    paths, grid_seq, start_list, goal_list = test_mapf_env(env)
    return paths, grid_seq, start_list, goal_list
```

#### step2_convert_to_schedule
Converts discrete paths to a schedule format:
```python
def step2_convert_to_schedule(paths):
    schedule = {}
    for agent_id, path in enumerate(paths):
        schedule[f"agent{agent_id}"] = []
        for t, (x, y) in enumerate(path):
            schedule[f"agent{agent_id}"].append({
                'x': x,
                'y': y,
                't': float(t)
            })
    return schedule
```

#### step3_build_temporal_plan_graph
Builds a graph representing temporal constraints:
```python
def step3_build_temporal_plan_graph(schedule, uav_params):
    tpg = TemporalPlanGraph()
    # Add vertices and edges
    for agent, waypoints in schedule.items():
        for i, wp in enumerate(waypoints):
            vertex = Vertex(agent, Location(wp['x'], wp['y']), wp['t'])
            tpg.add_vertex(vertex)
    return tpg
```

#### _apply_linear_interpolation
Creates smoother paths with continuous motion:
```python
def _apply_linear_interpolation(trajectory, agent_id, agent_name):
    # Calculate total path length
    total_distance = 0
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i+1]
        dx = wp2.x - wp1.x
        dy = wp2.y - wp1.y
        segment_length = math.sqrt(dx*dx + dy*dy)
        total_distance += segment_length

    # Use a constant velocity approach for timing
    constant_velocity = 4.0  # Units per time step
    total_time = total_distance / constant_velocity
```

### Handling Dynamic Obstacles

The solution identifies dynamic obstacles from the grid sequence:
```python
# Find cells with values between 0.1 and 0.9 (dynamic obstacles)
dynamic_mask = np.logical_and(grid > 0.05, grid < 0.95)
obstacle_positions = np.where(dynamic_mask)

for i in range(len(obstacle_positions[0])):
    x, y = obstacle_positions[0][i], obstacle_positions[1][i]
    obstacle_id = f"dyn_obs_{x}_{y}"

    # Add position with timestamp - scaled to make obstacles move slower
    dynamic_obstacles[obstacle_id].append({
        'x': x,
        'y': y,
        't': float(t) * time_scaling_factor  # Scale time to slow down movement
    })
```

### Visualization Techniques

The solution uses several techniques for effective visualization:

1. **Interpolation for Smooth Motion**:
```python
# Find the current position with interpolation for smoother movement
if len(all_t) >= 2:
    # Find the two waypoints that bracket the current time
    next_idx = None
    for i, t in enumerate(all_t):
        if t > current_time:
            next_idx = i
            break

    if next_idx is not None and next_idx > 0:
        # We have a previous and next waypoint - interpolate
        prev_idx = next_idx - 1
        prev_t = all_t[prev_idx]
        next_t = all_t[next_idx]

        # Calculate interpolation factor
        alpha = (current_time - prev_t) / (next_t - prev_t)

        # Interpolate position
        current_x = all_x[prev_idx] * (1 - alpha) + all_x[next_idx] * alpha
        current_y = all_y[prev_idx] * (1 - alpha) + all_y[next_idx] * alpha
```

2. **Orientation Indicators**:
```python
# Add orientation indicator (heading arrow)
if next_idx < len(all_x):
    # Calculate velocity vector
    vel_x = all_x[next_idx] - all_x[prev_idx]
    vel_y = all_y[next_idx] - all_y[prev_idx]

    # Normalize to get direction
    vel_mag = math.sqrt(vel_x**2 + vel_y**2)
    if vel_mag > 0.001:  # Avoid division by zero
        vel_x /= vel_mag
        vel_y /= vel_mag

        # Plot heading arrow
        ax.arrow(current_y, current_x, vel_y, vel_x,
                 head_width=0.8, head_length=1.2,
                 fc='blue', ec='blue', alpha=0.8)
```

## Future Improvements

1. **Better Turn Dynamics**: Implement more sophisticated turn dynamics for fixed-wing UAVs
2. **3D Visualization**: Add altitude as a dimension for more realistic visualization
3. **Wind Effects**: Model wind effects on drone trajectories
4. **Formation Flying**: Enable coordinated formation flying for multiple drones
5. **Obstacle Prediction**: Implement predictive avoidance for dynamic obstacles
6. **Adaptive Velocity**: Dynamically adjust velocity based on obstacle density
7. **Energy Optimization**: Minimize energy consumption while maintaining feasibility

## Challenges and Solutions

### Challenge 1: Ensuring Continuous Motion
**Problem**: Fixed-wing drones cannot stop mid-air, but the original MAPF solution allowed agents to wait.

**Solution**:
- Implemented a constant velocity approach in the schedule generation
- Distributed time proportionally to distance traveled
- Added temporal constraints to ensure minimum velocity

### Challenge 2: Dynamic Obstacle Integration
**Problem**: Dynamic obstacles (radar zones) needed to be integrated with the temporal planning.

**Solution**:
- Extracted dynamic obstacles from the grid sequence
- Created temporal constraints to avoid collisions
- Slowed down obstacle movement for better visualization
- Added soft constraints with increased path cost

### Challenge 3: Turn Rate Limitations
**Problem**: Fixed-wing drones have limited turning capabilities.

**Solution**:
- Added turn rate constraints to the trajectory validation
- Implemented a speed-curvature coupling model
- Adjusted trajectories to respect turn rate limits

### Challenge 4: Visualization
**Problem**: Needed to visualize continuous motion and dynamic obstacles effectively.

**Solution**:
- Implemented interpolation for smooth motion
- Added orientation indicators to show heading
- Created animated GIFs with adjustable frame rates
- Synchronized drone and obstacle movement

### Challenge 5: Integration with Existing Code
**Problem**: Needed to integrate with the existing MAPF framework.

**Solution**:
- Created fallback implementations for centralized components
- Maintained compatibility with the existing data structures
- Extended the existing classes to handle UAV-specific constraints
