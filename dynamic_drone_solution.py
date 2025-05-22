#!/usr/bin/env python3
"""
Dynamic Drone Solution - Post-processing for MAPF paths to create dynamically feasible trajectories for fixed-wing UAVs

This script takes the output from mapf_env_test.py and transforms the discrete paths into
continuous trajectories that respect fixed-wing UAV dynamics constraints.

The process follows these steps:
1. Generate initial discrete paths using mapf_env_test.py
2. Convert discrete paths to a schedule format compatible with the centralized library
3. Build a Temporal Plan Graph (TPG) using the existing implementation
4. Construct a Simple Temporal Network (STN) from the TPG
5. Optimize the temporal plan to find feasible timing
6. Generate final trajectories with control inputs and validate against UAV dynamics

Usage:
    python dynamic_drone_solution_revised.py --input <path_to_mapf_solution> --output <output_directory>
"""

import os
import sys
import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline, splprep, splev
import math
from copy import deepcopy

# ------------------------------------------------------------------
# UNIFIED SIMULATION PARAMETERS
# ------------------------------------------------------------------
# Physical duration of ONE discrete MAPF tick (seconds)
# This is the fundamental time unit for the simulation
GRID_DT = 4.0          # Increased to make dynamic obstacles move much slower

# Radar zone speed factor - how much slower radar zones move compared to agents
# Higher values = slower radar zones
RADAR_ZONE_SPEED_FACTOR = 3.0  # Radar zones move 3x slower than agents

# UAV dynamics parameters - adjusted for consistent simulation
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

# Add centralized library to path
# Detect if we're in Docker or local environment
if os.path.exists('/lns2rl'):
    # Docker environment
    base_dir = '/lns2rl'
    # Add additional paths for Docker environment
    sys.path.append('/lns2rl')
    sys.path.append('/lns2rl/multi_agent_path_planning')
    sys.path.append('/lns2rl/multi_agent_path_planning/centralized')
    sys.path.append('/lns2rl/multi_agent_path_planning/centralized/scheduling')
elif os.path.exists('/app'):
    # Alternative Docker environment
    base_dir = '/app'
    # Add additional paths for Docker environment
    sys.path.append('/app')
    sys.path.append('/app/multi_agent_path_planning')
    sys.path.append('/app/multi_agent_path_planning/centralized')
    sys.path.append('/app/multi_agent_path_planning/centralized/scheduling')
else:
    # Local environment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Add additional paths for local environment
    sys.path.append(base_dir)
    centralized_path = os.path.join(base_dir, 'multi_agent_path_planning', 'centralized')
    if os.path.exists(centralized_path):
        sys.path.append(os.path.join(base_dir, 'multi_agent_path_planning'))
        sys.path.append(centralized_path)
        sys.path.append(os.path.join(centralized_path, 'scheduling'))

print(f"Using base directory: {base_dir}")
print(f"Python path: {sys.path}")

# Create a symbolic link to make cbs module accessible from scheduling
centralized_path = os.path.join(base_dir, 'multi_agent_path_planning', 'centralized')
scheduling_path = os.path.join(centralized_path, 'scheduling')
cbs_path = os.path.join(centralized_path, 'cbs')
cbs_symlink_path = os.path.join(scheduling_path, 'cbs')

# Create the symbolic link if it doesn't exist
try:
    if os.path.exists(scheduling_path) and os.path.exists(cbs_path) and not os.path.exists(cbs_symlink_path):
        os.symlink(cbs_path, cbs_symlink_path)
        print(f"Created symbolic link from {cbs_path} to {cbs_symlink_path}")
except Exception as e:
    print(f"Warning: Could not create symbolic link: {e}")

# Import the modules
try:
    # First try direct imports
    from tpg import TemporalPlanGraph, Vertex, Edge
    try:
        from cbs.cbs import Location
    except ImportError:
        # Try importing directly if cbs is not a package
        import sys
        sys.path.append(os.path.join(base_dir, 'multi_agent_path_planning', 'centralized', 'cbs'))
        try:
            from cbs import Location
        except ImportError:
            # Define our own Location class as a fallback
            class Location:
                def __init__(self, x=-1, y=-1):
                    self.x = x
                    self.y = y
                def __eq__(self, other):
                    return self.x == other.x and self.y == other.y
                def __str__(self):
                    return str((self.x, self.y))

    from stn import SimpleTemporalNetwork
    from minimize import OptimizationClass
    print("Successfully imported modules using direct path")
except ImportError as e:
    print(f"Direct import failed: {e}")
    print("Trying absolute imports...")

    try:
        # Try absolute imports
        from multi_agent_path_planning.centralized.scheduling.tpg import TemporalPlanGraph, Vertex, Edge
        try:
            from multi_agent_path_planning.centralized.cbs.cbs import Location
        except ImportError:
            # Define our own Location class as a fallback
            class Location:
                def __init__(self, x=-1, y=-1):
                    self.x = x
                    self.y = y
                def __eq__(self, other):
                    return self.x == other.x and self.y == other.y
                def __str__(self):
                    return str((self.x, self.y))

        from multi_agent_path_planning.centralized.scheduling.stn import SimpleTemporalNetwork
        from multi_agent_path_planning.centralized.scheduling.minimize import OptimizationClass
        print("Successfully imported modules using absolute path")
    except ImportError as e2:
        print(f"Absolute import failed: {e2}")

        # Try to find the modules in the current directory
        print("Checking if modules exist in the current directory...")
        import glob
        print(f"Files in current directory: {glob.glob('*.py')}")
        print(f"Directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")

        # Define our own Location class as a fallback
        class Location:
            def __init__(self, x=-1, y=-1):
                self.x = x
                self.y = y
            def __eq__(self, other):
                return self.x == other.x and self.y == other.y
            def __str__(self):
                return str((self.x, self.y))

        # Define our own SimpleTemporalNetwork class as a fallback
        class SimpleTemporalNetwork:
            def __init__(self, tpg, dynamic_obstacles=None):
                self.tpg = tpg
                self.dynamic_obstacles = dynamic_obstacles
                self.edges = []                      # ← make sure this exists
                # build per‐cell time‐windows
                self.cell_windows = {}
                if dynamic_obstacles:
                    for sched in dynamic_obstacles.values():
                        for pt in sched:
                            key = (pt['x'], pt['y'])
                            self.cell_windows.setdefault(key, []).append(pt['t'])
                    for times in self.cell_windows.values():
                        times.sort()

                if tpg:
                    self.vertices = list(
                        tpg.augmented_vertices
                        if hasattr(tpg, 'augmented_vertices')
                        else tpg.vertices
                    )
                    self.initial_states = (
                        tpg.initial_states
                        if hasattr(tpg, 'initial_states')
                        else []
                    )
                    self.vmax = [5.0] * len(self.initial_states)

            def generate_bounds(self):
                from bisect import bisect_left
                print("Using patched SimpleTemporalNetwork.generate_bounds()")

                for edge in self.tpg.augmented_edges:
                    # 1) compute lb exactly as before:
                    if hasattr(edge, 'edge_length') and hasattr(edge, 'heading_change'):
                        if edge.vertex_a.agent == edge.vertex_b.agent:
                            idx = 0 if edge.vertex_a.agent == 'agent0' else 1
                            time_dist = edge.edge_length / self.vmax[idx]
                            time_turn = abs(edge.heading_change) / UAV_PARAMS['omega_max']
                            lb = max(time_dist, time_turn)
                        else:
                            lb = 0.1
                    else:
                        lb = 0.1           # ← properly indented under the else

                    ub = lb * 1.5

                    # append the normal travel‐time edge
                    self.edges.append({
                        'vertex_a': edge.vertex_a,
                        'vertex_b': edge.vertex_b,
                        'bound': [lb, ub]
                    })

                    # 2) now constant‐time obstacle check at the **arrival** vertex
                    b = edge.vertex_b
                    key = (b.location.x, b.location.y)
                    times = self.cell_windows.get(key)
                    if times:
                        i = bisect_left(times, b.time)
                        if i < len(times):
                            t_obs = times[i]
                            # forbid you arriving at or after t_obs
                            self.edges.append({
                                'vertex_a': b,
                                'vertex_b': None,
                                'bound': [-float('inf'), t_obs - 1e-3],
                                'absolute_time': True
                            })
                    
            # ─── in class SimpleTemporalNetwork ──────────────────────────────
# REPLACE the whole _paths_intersect() definition with:

            def _paths_intersect(self, p1, p2, p3, p4, radius=UAV_PARAMS['safety_distance']):
                """
                Minkowski-sum segment/segment test: true if the capsule
                defined by segment p1–p2 (inflated by 'radius') intersects
                the capsule of segment p3–p4.
                Works in grid units (x,y are integers).
                """
                def seg_dist(a, b, c, d):
                    """closest distance between 2 segments in 2-D"""
                    # parametric representations
                    u = b - a
                    v = d - c
                    w = a - c
                    denom = u.dot(u)*v.dot(v) - (u.dot(v))**2
                    if denom < 1e-9:            # almost parallel
                        # distance between the two infinite lines
                        return np.linalg.norm(np.cross(u, c-a))/np.linalg.norm(u)
                    s = (u.dot(v)*w.dot(v) - v.dot(v)*w.dot(u)) / denom
                    t = (u.dot(u)*w.dot(v) - u.dot(v)*w.dot(u)) / denom
                    s = np.clip(s, 0.0, 1.0)
                    t = np.clip(t, 0.0, 1.0)
                    closest = a + s*u - (c + t*v)
                    return np.linalg.norm(closest)

                a, b = np.array(p1, float), np.array(p2, float)
                c, d = np.array(p3, float), np.array(p4, float)
                return seg_dist(a, b, c, d) <= 2.0*radius   # two inflated tubes

            def generate_end_edges(self):
                print("Using fallback SimpleTemporalNetwork.generate_end_edges()")

                # Define a simple Vertex class for start and end vertices
                class SimpleVertex:
                    def __init__(self, agent, location, time):
                        self.agent = agent
                        self.location = location
                        self.time = time

                # Create start and end vertices using the class we just defined
                start = SimpleVertex('start', Location(-1, -1), -1)
                end = SimpleVertex('end', Location(-2, -2), -2)

                # Add start and end vertices to the list of vertices
                self.vertices.extend([start, end])

                # Define bounds
                bound_start = [0.0, 0.0]  # [min_time, max_time]
                bound_end = [0.0, float('inf')]  # [min_time, max_time]

                # Add edges from start to initial states
                if hasattr(self.tpg, 'initial_states') and self.tpg.initial_states:
                    for state in self.tpg.initial_states:
                        self.edges.append({
                            'vertex_a': start,
                            'vertex_b': state,
                            'bound': bound_start
                        })

                # Add edges from final states to end
                if hasattr(self.tpg, 'final_states') and self.tpg.final_states:
                    for state in self.tpg.final_states:
                        self.edges.append({
                            'vertex_a': state,
                            'vertex_b': end,
                            'bound': bound_end
                        })

        # Define our own OptimizationClass as a fallback
        class OptimizationClass:
            def __init__(self, stn):
                self.stn = stn

            def generate_schedule(self):
                print("Using fallback OptimizationClass.generate_schedule()")
                # Implementation that enforces continuous motion for fixed-wing drones
                # while preserving the original timing relationships as much as possible
                schedule = {}

                # Get the original schedule from the TPG if available
                if hasattr(self.stn, 'tpg') and hasattr(self.stn.tpg, 'schedule'):
                    # Start with the original schedule
                    original_schedule = self.stn.tpg.schedule

                    # Process each agent's path
                    for agent, waypoints in original_schedule.items():
                        schedule[agent] = []

                        if len(waypoints) < 2:
                            continue

                        # Sort waypoints by original time
                        sorted_waypoints = sorted(waypoints, key=lambda wp: wp['t'])

                        # Calculate segment lengths and original velocities
                        segment_lengths = []
                        original_velocities = []
                        waiting_segments = []

                        for i in range(len(sorted_waypoints) - 1):
                            wp1 = sorted_waypoints[i]
                            wp2 = sorted_waypoints[i+1]

                            # Calculate distance
                            dx = wp2['x'] - wp1['x']
                            dy = wp2['y'] - wp1['y']
                            segment_length = math.sqrt(dx*dx + dy*dy)
                            segment_lengths.append(segment_length)

                            # Calculate original velocity
                            dt = wp2['t'] - wp1['t']

                            # Check if this is a waiting segment (very small movement)
                            is_waiting = segment_length < 0.1
                            waiting_segments.append(is_waiting)

                            if dt > 0 and segment_length > 0:
                                velocity = segment_length / dt
                                original_velocities.append(velocity)
                            else:
                                # Handle zero time difference or zero distance
                                original_velocities.append(UAV_PARAMS['v_min'])

                        # Analyze velocity distribution
                        if original_velocities:
                            avg_velocity = sum(original_velocities) / len(original_velocities)
                            max_velocity = max(original_velocities)
                            min_velocity = min(v for v in original_velocities if v > 0)

                            print(f"Original velocity stats for {agent}:")
                            print(f"  Average: {avg_velocity:.2f}, Min: {min_velocity:.2f}, Max: {max_velocity:.2f}")

                            # Count waiting segments
                            wait_count = sum(1 for w in waiting_segments if w)
                            if wait_count > 0:
                                print(f"  Found {wait_count} waiting segments that need to be handled")

                            # Adjust velocities to be within UAV constraints
                            adjusted_velocities = []
                            for i, v in enumerate(original_velocities):
                                if waiting_segments[i]:
                                    # For waiting segments, use a higher velocity to reduce waiting time
                                    adjusted_v = UAV_PARAMS['v_max'] * 0.8  # 80% of max velocity
                                else:
                                    # For normal segments, preserve original velocity if possible
                                    adjusted_v = max(UAV_PARAMS['v_min'], min(v, UAV_PARAMS['v_max']))
                                adjusted_velocities.append(adjusted_v)
                        else:
                            # Default to a reasonable velocity if we couldn't calculate original velocities
                            adjusted_velocities = [3.0] * len(segment_lengths)
                            waiting_segments = [False] * len(segment_lengths)

                        # First waypoint starts at time 0
                        current_time = 0.0
                        schedule[agent].append({
                            'x': sorted_waypoints[0]['x'],
                            'y': sorted_waypoints[0]['y'],
                            't': current_time
                        })

                        # Calculate times based on adjusted velocities
                        for i in range(len(segment_lengths)):
                            # Time = distance / velocity
                            segment_time = segment_lengths[i] / adjusted_velocities[i] if adjusted_velocities[i] > 0 else 0.1

                            # For waiting segments, ensure minimum time (can't be too fast)
                            if waiting_segments[i]:
                                segment_time = max(segment_time, 0.1)

                            current_time += segment_time

                            schedule[agent].append({
                                'x': sorted_waypoints[i+1]['x'],
                                'y': sorted_waypoints[i+1]['y'],
                                't': current_time
                            })
                else:
                    # Fallback if no original schedule is available
                    # Group vertices by agent
                    agent_vertices = {}
                    for vertex in self.stn.vertices:
                        if hasattr(vertex, 'agent'):
                            agent = vertex.agent
                            if agent not in ['start', 'end']:
                                if agent not in agent_vertices:
                                    agent_vertices[agent] = []
                                agent_vertices[agent].append(vertex)

                    # Create a schedule for each agent
                    for agent, vertices in agent_vertices.items():
                        schedule[agent] = []

                        # Sort vertices by their original time
                        vertices.sort(key=lambda v: v.time if hasattr(v, 'time') else 0)

                        if len(vertices) < 2:
                            continue

                        # Calculate total path length
                        total_distance = 0
                        for i in range(len(vertices) - 1):
                            v1 = vertices[i]
                            v2 = vertices[i+1]

                            if hasattr(v1, 'location') and hasattr(v2, 'location'):
                                dx = v2.location.x - v1.location.x
                                dy = v2.location.y - v1.location.y
                                segment_length = math.sqrt(dx*dx + dy*dy)
                                total_distance += segment_length

                        # Use a constant velocity approach for timing
                        # This ensures continuous motion without stops
                        constant_velocity = 3.0  # Units per time step
                        total_time = total_distance / constant_velocity if total_distance > 0 else 1.0

                        # Distribute time based on segment lengths
                        cumulative_distance = 0
                        for i, vertex in enumerate(vertices):
                            if hasattr(vertex, 'location'):
                                if i == 0:
                                    # First waypoint starts at time 0
                                    adjusted_time = 0.0
                                else:
                                    # Calculate distance from previous vertex
                                    prev_vertex = vertices[i-1]
                                    if hasattr(prev_vertex, 'location'):
                                        dx = vertex.location.x - prev_vertex.location.x
                                        dy = vertex.location.y - prev_vertex.location.y
                                        segment_length = math.sqrt(dx*dx + dy*dy)
                                        cumulative_distance += segment_length

                                        # Time is proportional to distance traveled
                                        adjusted_time = (cumulative_distance / total_distance) * total_time if total_distance > 0 else float(i)
                                    else:
                                        adjusted_time = float(i)

                                schedule[agent].append({
                                    'x': vertex.location.x if hasattr(vertex.location, 'x') else 0,
                                    'y': vertex.location.y if hasattr(vertex.location, 'y') else 0,
                                    't': adjusted_time
                                })

                return schedule

        print("Using fallback classes for SimpleTemporalNetwork and OptimizationClass")

# Import from mapf_env_test.py to reuse functionality
from mapf_env_test import create_test_env, test_mapf_env

class Waypoint:
    """Represents a waypoint in a UAV's trajectory"""
    def __init__(self, x, y, time=0.0):
        self.x = x
        self.y = y
        self.time = time

    def __str__(self):
        return f"({self.x}, {self.y}, t={self.time})"

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Trajectory:
    """Continuous trajectory for a UAV"""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.waypoints = []
        self.times = []
        self.velocities = []
        self.accelerations = []
        self.turn_rates = []

    def add_waypoint(self, waypoint, time):
        self.waypoints.append(waypoint)
        self.times.append(time)

    def compute_dynamics(self):
        """Compute velocities, accelerations, and turn rates along the trajectory"""
        if len(self.waypoints) < 2:
            return

        # Compute velocities
        self.velocities = []
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i+1]
            dt = self.times[i+1] - self.times[i]

            if dt > 0:
                vx = (wp2.x - wp1.x) / dt
                vy = (wp2.y - wp1.y) / dt
                self.velocities.append((vx, vy))
            else:
                self.velocities.append((0, 0))

        # Add final velocity (same as previous)
        self.velocities.append(self.velocities[-1])

        # Compute accelerations
        self.accelerations = []
        for i in range(len(self.velocities) - 1):
            vx1, vy1 = self.velocities[i]
            vx2, vy2 = self.velocities[i+1]
            dt = self.times[i+1] - self.times[i]

            if dt > 0:
                ax = (vx2 - vx1) / dt
                ay = (vy2 - vy1) / dt
                self.accelerations.append((ax, ay))
            else:
                self.accelerations.append((0, 0))

        # Add final acceleration (same as previous)
        self.accelerations.append(self.accelerations[-1])

        # Compute turn rates
        self.turn_rates = []
        for i in range(len(self.velocities) - 1):
            vx1, vy1 = self.velocities[i]
            vx2, vy2 = self.velocities[i+1]

            v1_mag = math.sqrt(vx1**2 + vy1**2)
            v2_mag = math.sqrt(vx2**2 + vy2**2)

            if v1_mag > 0 and v2_mag > 0:
                # Compute angle between velocity vectors
                dot_product = vx1*vx2 + vy1*vy2
                cos_angle = dot_product / (v1_mag * v2_mag)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
                angle = math.acos(cos_angle)

                # Determine sign of turn rate (positive = counterclockwise)
                cross_product = vx1*vy2 - vy1*vx2
                sign = 1 if cross_product > 0 else -1

                dt = self.times[i+1] - self.times[i]
                if dt > 0:
                    turn_rate = sign * angle / dt
                    self.turn_rates.append(turn_rate)
                else:
                    self.turn_rates.append(0)
            else:
                self.turn_rates.append(0)

        # Add final turn rate (same as previous)
        self.turn_rates.append(self.turn_rates[-1])

    def is_dynamically_feasible(self, uav_params):
        """Check if the trajectory respects UAV dynamics constraints"""
        if len(self.waypoints) < 2:
            return False

        self.compute_dynamics()

        for i in range(len(self.waypoints)):
            # Check velocity constraints
            if i < len(self.velocities):
                vx, vy = self.velocities[i]
                v_mag = math.sqrt(vx**2 + vy**2)

                if v_mag < uav_params['v_min'] or v_mag > uav_params['v_max']:
                    return False

            # Check turn rate constraints
            if i < len(self.turn_rates):
                turn_rate = abs(self.turn_rates[i])

                if turn_rate > uav_params['omega_max']:
                    return False

                # Check speed-curvature coupling
                if i < len(self.velocities):
                    vx, vy = self.velocities[i]
                    v_mag = math.sqrt(vx**2 + vy**2)

                    if v_mag > 0:
                        max_turn_rate = uav_params['F_max'] / (uav_params['mass'] * v_mag)
                        if turn_rate > max_turn_rate:
                            return False

            # Check acceleration constraints
            if i < len(self.accelerations):
                ax, ay = self.accelerations[i]
                a_mag = math.sqrt(ax**2 + ay**2)

                if a_mag > uav_params['a_max']:
                    return False

        return True


def make_global_spline(points_xy, v_max, omega_max, nsamples=250, original_times=None, grid_seq=None):
    """
    points_xy : list of (x,y) tuples
    original_times : list of original timestamps (optional)
    returns    interp_x, interp_y, t_points   (np.ndarray, same length)
    One cubic B-spline for the whole path + curvature–based time law.

    If original_times is provided, tries to follow the original timing
    while ensuring kinematic feasibility.
    """

    Px, Py = zip(*points_xy)

    # ---------------------------------------------------------------- B-spline
    u_ctrl   = np.linspace(0.0, 1.0, len(Px))
    tck, _   = splprep([Px, Py], u=u_ctrl, s=0.0, k=min(3, len(Px)-1))
    u_dense  = np.linspace(0.0, 1.0, nsamples)
    interp_x, interp_y = splev(u_dense, tck)

    # ---------------------------------------------------------------- time law
    dx = np.diff(interp_x);   dy = np.diff(interp_y)
    ds = np.hypot(dx, dy) + 1e-12

    d1x = dx / ds;            d1y = dy / ds
    d2x = np.diff(d1x, prepend=d1x[0])
    d2y = np.diff(d1y, prepend=d1y[0])
    kappa = np.abs(d1x*d2y - d1y*d2x) / np.maximum((d1x**2 + d1y**2)**1.5, 1e-9)

    v_turn = np.where(kappa > 1e-6, omega_max / kappa, v_max)
    v_allowed = np.minimum(v_turn, v_max)
    dt = np.maximum(ds / v_allowed, 0.05)   # never shorter than 50 ms

    # Calculate kinematically feasible times
    kinematic_times = np.concatenate([[0.0], np.cumsum(dt)])

    # If original times are provided, try to follow them while respecting kinematic constraints
    if original_times is not None and len(original_times) >= 2:
        # Scale the original time range to match the kinematic time range
        orig_duration = original_times[-1] - original_times[0]
        kin_duration = kinematic_times[-1] - kinematic_times[0]

        # If the original duration is longer than the kinematic duration,
        # we can safely follow the original timing (slower is always safe)
        if orig_duration > kin_duration:
            # Interpolate original times to match the number of points
            orig_u = np.linspace(0, 1, len(original_times))
            interp_times = np.interp(u_dense, orig_u, original_times)
            return interp_x, interp_y, interp_times
        else:
            # Original timing is faster than kinematically feasible
            # Scale the original times to ensure kinematic feasibility
            scale_factor = kin_duration / orig_duration
            orig_u = np.linspace(0, 1, len(original_times))
            scaled_orig_times = original_times[0] + (original_times - original_times[0]) * scale_factor
            interp_times = np.interp(u_dense, orig_u, scaled_orig_times)
            return interp_x, interp_y, interp_times

    # If no original times or not enough points, use kinematic times
    return interp_x, interp_y, kinematic_times

# ─── utilities ────────────────────────────────────────────────────
def lock_axes(ax, img):
    """Keep square cells and stop autoscale after every imshow()."""
    ax.set_xlim(-0.5, img.shape[1]-0.5)
    ax.set_ylim(-0.5, img.shape[0]-0.5)
    ax.set_aspect('equal', adjustable='box')
# ──────────────────────────────────────────────────────────────────

# Implementation of the 6 steps

def step1_generate_initial_paths(map_path, num_agents, num_obstacles, output_dir="mapf_solution"):
    """
    Step 1: Generate initial discrete paths using mapf_env_test.py

    Args:
        map_path: Path to the static map
        num_agents: Number of agents
        num_obstacles: Number of dynamic obstacles
        output_dir: Directory to save the output

    Returns:
        paths: List of discrete paths for each agent
        grid_seq: The dynamic map sequence
        start_list: List of agent start positions
        goal_list: List of agent goal positions
    """
    print("Step 1: Generating initial discrete paths...")

    # Create test environment - disable visualization to save memory
    grid, grid_seq, start_list, goal_list = create_test_env(
        static_map_path=map_path,
        num_agents=num_agents,
        num_dynamic_obstacles=num_obstacles,
        output_dir=output_dir,
        visualize=False  # Disable visualization to save memory
    )

    # Run MAPF to get initial paths - disable visualization to save memory
    paths, num_collisions = test_mapf_env(
        static_map_path=map_path,
        num_agents=num_agents,
        num_dynamic_obstacles=num_obstacles,
        output_dir=output_dir,
        visualize=False  # Disable visualization to save memory
    )

    # Force garbage collection to free memory
    import gc
    gc.collect()

    print(f"Initial paths generated with {num_collisions} collisions")
    return paths, grid_seq, start_list, goal_list

def step2_convert_to_schedule(paths):
    """
    Step 2: Convert discrete paths to a schedule format compatible with the centralized library

    Args:
        paths: List of discrete paths for each agent

    Returns:
        schedule: Dictionary mapping agent names to lists of waypoints with timing
    """
    print("Step 2: Converting discrete paths to schedule format...")

    schedule = {}

    for agent_id, path in enumerate(paths):
        agent_name = f"agent{agent_id}"
        schedule[agent_name] = []

        last = None
        for t, point in enumerate(path):
            if last is not None and point == last:
                continue
            last = point
            schedule[agent_name].append({
                'x': float(point[0]),
                'y': float(point[1]),
                't': float(t) * GRID_DT      # stamp in real seconds
            })

    print(f"Converted {len(paths)} paths to schedule format")
    return schedule

def step3_build_temporal_plan_graph(schedule, uav_params):
    """
    Step 3: Build a Temporal Plan Graph (TPG) using a highly optimized implementation

    Args:
        schedule: Dictionary mapping agent names to lists of waypoints with timing
        uav_params: UAV dynamics parameters

    Returns:
        tpg: Temporal Plan Graph
    """
    print("Step 3: Building Temporal Plan Graph...")

    # Optimize the schedule by removing unnecessary waypoints
    # This reduces the number of vertices in the TPG, making it faster to build
    optimized_schedule = {}
    for agent, waypoints in schedule.items():
        # Skip if there are fewer than 2 waypoints
        if len(waypoints) < 2:
            optimized_schedule[agent] = waypoints
            continue

        # Keep only waypoints that represent significant changes in position
        # This reduces the number of vertices in the TPG
        optimized_waypoints = [waypoints[0]]  # Always keep the first waypoint
        last_pos = (waypoints[0]['x'], waypoints[0]['y'])

        for wp in waypoints[1:-1]:  # Process middle waypoints
            current_pos = (wp['x'], wp['y'])
            # Only keep waypoints that represent a change in position
            if current_pos != last_pos:
                optimized_waypoints.append(wp)
                last_pos = current_pos

        # Always keep the last waypoint
        if len(waypoints) > 1:
            optimized_waypoints.append(waypoints[-1])

        optimized_schedule[agent] = optimized_waypoints

    print(f"Optimized schedule: reduced from {sum(len(wps) for wps in schedule.values())} to {sum(len(wps) for wps in optimized_schedule.values())} waypoints")

    # COMPLETELY CUSTOM IMPLEMENTATION FOR LARGE MAPS
    # Instead of using the original TemporalPlanGraph class, we'll build our own optimized version

    # Create a custom TPG class that doesn't depend on the original implementation
    class CustomTPG:
        def __init__(self, schedule, safety_distance):
            self.schedule = schedule
            self.delta = safety_distance
            self.vertices = []
            self.edges_type_1 = []
            self.edges_type_2 = []
            self.augmented_edges = []
            self.augmented_vertices = []
            self.initial_states = []
            self.final_states = []

    # Create a new TPG object
    tpg = CustomTPG(optimized_schedule, uav_params['safety_distance'])

    print("Building optimized TPG...")

    # Dictionary to store vertices by agent and location for quick lookup
    vertex_dict = {}

    # Dictionary to map locations to vertices (for type-2 edges)
    location_to_vertices = {}

    # Step 1: Create vertices and type-1 edges
    print("Creating vertices and type-1 edges...")
    for agent, plan in optimized_schedule.items():
        if len(plan) < 1:
            continue

        # Create first vertex
        v0 = Vertex(agent, Location(plan[0]['x'], plan[0]['y']), plan[0]['t'])
        tpg.vertices.append(v0)

        # Store in dictionaries for quick lookup
        vertex_key = (agent, plan[0]['x'], plan[0]['y'], plan[0]['t'])
        vertex_dict[vertex_key] = v0

        loc_key = (plan[0]['x'], plan[0]['y'])
        if loc_key not in location_to_vertices:
            location_to_vertices[loc_key] = []
        location_to_vertices[loc_key].append(v0)

        # Store initial state
        tpg.initial_states.append(v0)

        # Process remaining waypoints
        last_dir = None
        for i in range(len(plan) - 1):
            # Get locations
            loc_a = Location(plan[i]['x'], plan[i]['y'])
            loc_b = Location(plan[i+1]['x'], plan[i+1]['y'])

            # Skip if same location
            if loc_a.x == loc_b.x and loc_a.y == loc_b.y:
                continue

            # Get or create vertices
            vertex_a_key = (agent, loc_a.x, loc_a.y, plan[i]['t'])
            vertex_a = vertex_dict.get(vertex_a_key)
            if not vertex_a:
                vertex_a = Vertex(agent, loc_a, plan[i]['t'])
                tpg.vertices.append(vertex_a)
                vertex_dict[vertex_a_key] = vertex_a

            vertex_b = Vertex(agent, loc_b, plan[i+1]['t'])
            tpg.vertices.append(vertex_b)
            vertex_b_key = (agent, loc_b.x, loc_b.y, plan[i+1]['t'])
            vertex_dict[vertex_b_key] = vertex_b

            # Store in location dictionary for type-2 edges
            loc_b_key = (loc_b.x, loc_b.y)
            if loc_b_key not in location_to_vertices:
                location_to_vertices[loc_b_key] = []
            location_to_vertices[loc_b_key].append(vertex_b)

            # Calculate direction and heading change
            dx = loc_b.x - loc_a.x
            dy = loc_b.y - loc_a.y
            seg_len = (dx*dx + dy*dy) ** 0.5
            dir_curr = (dx/seg_len, dy/seg_len)

            # Calculate heading change
            if last_dir is None:
                dpsi = 0.0
            else:
                dot = last_dir[0]*dir_curr[0] + last_dir[1]*dir_curr[1]
                dot = max(-1.0, min(1.0, dot))
                cross = last_dir[0]*dir_curr[1] - last_dir[1]*dir_curr[0]
                dpsi = abs(math.atan2(cross, dot))

            # Create edge
            edge_ab = Edge(vertex_a, vertex_b, heading_change=dpsi)
            tpg.edges_type_1.append(edge_ab)

            # Update last direction
            last_dir = dir_curr

        # Store final state
        final_state = Vertex(agent, Location(plan[-1]['x'], plan[-1]['y']), plan[-1]['t'])
        tpg.final_states.append(final_state)

    # Step 2: Create type-2 edges (much more efficiently)
    print("Creating type-2 edges...")
    type2_count = 0
    max_type2_edges = 10000  # Limit the number of type-2 edges to prevent excessive memory usage

    # Process locations with multiple vertices
    for loc, vertices in location_to_vertices.items():
        if len(vertices) > 1:
            # Sort vertices by time for more efficient processing
            vertices.sort(key=lambda v: v.time)

            # Create edges between vertices at the same location
            for i in range(len(vertices)):
                # Limit the number of comparisons to avoid excessive edges
                max_j = min(i + 10, len(vertices))  # Only check up to 10 vertices ahead

                for j in range(i+1, max_j):
                    # Only create edges between different agents
                    if vertices[i].agent != vertices[j].agent:
                        edge = Edge(vertices[i], vertices[j])
                        tpg.edges_type_2.append(edge)
                        type2_count += 1

                        # Check if we've reached the limit
                        if type2_count >= max_type2_edges:
                            print(f"Warning: Reached limit of {max_type2_edges} type-2 edges. Stopping to prevent excessive memory usage.")
                            break

                # Break outer loop if limit reached
                if type2_count >= max_type2_edges:
                    break

    print(f"Created {len(tpg.vertices)} vertices, {len(tpg.edges_type_1)} type-1 edges, and {len(tpg.edges_type_2)} type-2 edges")

    # Force garbage collection before augmenting the graph
    import gc
    gc.collect()

    # Step 3: Augment the graph (simplified version)
    print("Augmenting graph...")

    # Copy vertices to augmented vertices
    tpg.augmented_vertices = tpg.vertices.copy()

    # Process type-1 edges
    edge_count = 0
    for edge in tpg.edges_type_1:
        # Create safety vertices
        v1 = create_safety_vertex(edge.vertex_a, edge, tpg.delta, 1)
        v2 = create_safety_vertex(edge.vertex_b, edge, tpg.delta, -1)

        if v1 and v2:
            # Add to augmented vertices
            tpg.augmented_vertices.append(v1)
            tpg.augmented_vertices.append(v2)

            # Create augmented edges
            edge1 = Edge(edge.vertex_a, v1, heading_change=edge.heading_change)
            edge2 = Edge(v1, v2, heading_change=0.0)
            edge3 = Edge(v2, edge.vertex_b, heading_change=edge.heading_change)

            # Add to augmented edges
            tpg.augmented_edges.extend([edge1, edge2, edge3])
            edge_count += 3

            # Print progress every 1000 edges
            if edge_count % 1000 == 0:
                print(f"Processed {edge_count} augmented edges...")

    # Process type-2 edges (limit to prevent excessive processing)
    max_type2_augmented = 5000  # Limit to prevent excessive processing
    type2_processed = 0

    for edge_t2 in tpg.edges_type_2:
        # Create safety vertices
        v1 = create_safety_vertex(edge_t2.vertex_a, edge_t2, tpg.delta, 1)
        v2 = create_safety_vertex(edge_t2.vertex_b, edge_t2, tpg.delta, -1)

        if v1 and v2:
            # Create augmented edge
            edge4 = Edge(v1, v2)
            edge4.edge_length = 0
            tpg.augmented_edges.append(edge4)

            type2_processed += 1
            if type2_processed >= max_type2_augmented:
                print(f"Warning: Reached limit of {max_type2_augmented} augmented type-2 edges. Stopping to prevent excessive processing.")
                break

    print(f"Created {len(tpg.augmented_edges)} augmented edges")

    # Force garbage collection again
    gc.collect()

    print(f"Built TPG with {len(tpg.vertices)} vertices and {len(tpg.augmented_edges)} augmented edges")
    return tpg

# Helper function to create safety vertices
def create_safety_vertex(vertex, edge, delta, side):
    """Create a safety vertex for the augmented graph"""
    try:
        if side == -1:
            # Get direction from edge
            dir_x = edge.vertex_b.location.x - edge.vertex_a.location.x
            dir_y = edge.vertex_b.location.y - edge.vertex_a.location.y

            # Normalize
            mag = (dir_x**2 + dir_y**2) ** 0.5
            if mag < 1e-6:  # Avoid division by zero
                return None

            dir_x /= mag
            dir_y /= mag

            # Create new location
            new_loc_x = vertex.location.x - delta * dir_x
            new_loc_y = vertex.location.y - delta * dir_y
            new_loc = Location(new_loc_x, new_loc_y)

            # Create new vertex
            new_vertex = Vertex(vertex.agent, new_loc, vertex.time - 0.1)
            return new_vertex

        elif side == 1:
            # Get direction from edge
            dir_x = edge.vertex_b.location.x - edge.vertex_a.location.x
            dir_y = edge.vertex_b.location.y - edge.vertex_a.location.y

            # Normalize
            mag = (dir_x**2 + dir_y**2) ** 0.5
            if mag < 1e-6:  # Avoid division by zero
                return None

            dir_x /= mag
            dir_y /= mag

            # Create new location
            new_loc_x = vertex.location.x + delta * dir_x
            new_loc_y = vertex.location.y + delta * dir_y
            new_loc = Location(new_loc_x, new_loc_y)

            # Create new vertex
            new_vertex = Vertex(vertex.agent, new_loc, vertex.time + 0.1)
            return new_vertex
    except Exception as e:
        print(f"Error creating safety vertex: {e}")
        return None

    return None

def step4_construct_simple_temporal_network(tpg, uav_params, dynamic_obstacles=None):
    """
    Step 4: Construct a Simple Temporal Network (STN) from the TPG

    Args:
        tpg: Temporal Plan Graph (either original or custom implementation)
        uav_params: UAV dynamics parameters
        dynamic_obstacles: Dictionary of dynamic obstacles with their schedules

    Returns:
        stn: Simple Temporal Network
    """
    print("Step 4: Constructing Simple Temporal Network...")

    try:
        # Try to create STN from TPG with dynamic obstacles using the original implementation
        stn = SimpleTemporalNetwork(tpg, dynamic_obstacles)

        # Modify the STN to account for UAV dynamics
        # Set maximum velocity based on UAV parameters
        stn.vmax = [uav_params['v_max']] * len(tpg.initial_states)

        # Regenerate bounds with the new velocity constraints
        stn.generate_bounds()
        stn.generate_end_edges()

        print(f"Built STN with {len(stn.edges)} edges")
        return stn
    except Exception as e:
        print(f"Error creating STN with original implementation: {e}")
        print("Using fallback STN implementation...")

        # Create a custom STN implementation that works with our custom TPG
        class CustomSTN:
            def __init__(self, tpg, dynamic_obstacles=None):
                self.tpg = tpg
                self.dynamic_obstacles = dynamic_obstacles
                self.vertices = tpg.augmented_vertices
                self.edges = []
                self.vmax = [uav_params['v_max']] * len(tpg.initial_states)
                self.initial_states = tpg.initial_states
                self.final_states = tpg.final_states

                # Generate bounds based on TPG edges
                self.generate_bounds()
                self.generate_end_edges()

            def generate_bounds(self):
                print("Generating STN bounds...")

                # Process all augmented edges from TPG
                for edge in self.tpg.augmented_edges:
                    # Calculate lower bound based on edge length and velocity
                    if hasattr(edge, 'edge_length') and hasattr(edge, 'heading_change'):
                        # Use agent-specific velocity if available
                        agent_idx = 0  # Default to first agent
                        if hasattr(edge.vertex_a, 'agent') and edge.vertex_a.agent != 'agent0' and len(self.vmax) > 1:
                            agent_idx = 1

                        # Distance-based time
                        time_dist = edge.edge_length / self.vmax[agent_idx]
                        # Turn-based time
                        time_turn = abs(edge.heading_change) / uav_params['omega_max']
                        lb = max(time_dist, time_turn)
                    else:
                        lb = 0.1

                    # Upper bound is 1.5x lower bound
                    ub = lb * 1.5

                    # Create edge with bounds
                    stn_edge = {
                        'vertex_a': edge.vertex_a,
                        'vertex_b': edge.vertex_b,
                        'bound': [lb, ub]
                    }
                    self.edges.append(stn_edge)

                # Process dynamic obstacles if available
                if self.dynamic_obstacles:
                    self._add_dynamic_obstacle_constraints()

            def _add_dynamic_obstacle_constraints(self):
                # Add temporal constraints for dynamic obstacles
                # This is a simplified implementation
                for obstacle_id, schedule in self.dynamic_obstacles.items():
                    for i in range(len(schedule) - 1):
                        obs_loc_1 = (schedule[i]['x'], schedule[i]['y'])
                        obs_loc_2 = (schedule[i+1]['x'], schedule[i+1]['y'])
                        obs_t_1 = schedule[i]['t']
                        obs_t_2 = schedule[i+1]['t']

                        # Check for potential conflicts with agent paths
                        for edge in self.tpg.augmented_edges:
                            if hasattr(edge.vertex_a, 'location') and hasattr(edge.vertex_b, 'location'):
                                loc_a = (edge.vertex_a.location.x, edge.vertex_a.location.y)
                                loc_b = (edge.vertex_b.location.x, edge.vertex_b.location.y)

                                # Simple check for potential intersection
                                if self._paths_might_intersect(loc_a, loc_b, obs_loc_1, obs_loc_2):
                                    # Add constraints to either pass before or after the obstacle
                                    # Option 1: Pass before the obstacle arrives
                                    self.edges.append({
                                        'vertex_a': edge.vertex_b,
                                        'vertex_b': None,  # Special marker for absolute time
                                        'bound': [-float('inf'), obs_t_1 - 0.1],  # Must finish before obstacle arrives
                                        'absolute_time': True
                                    })

                                    # Option 2: Pass after the obstacle has left
                                    self.edges.append({
                                        'vertex_a': None,  # Special marker for absolute time
                                        'vertex_b': edge.vertex_a,
                                        'bound': [obs_t_2 + 0.1, float('inf')],  # Must start after obstacle leaves
                                        'absolute_time': True
                                    })

            def _paths_might_intersect(self, p1, p2, p3, p4, radius=2.0):
                """Simple check if two line segments might intersect"""
                # Convert to numpy arrays for easier calculation
                import numpy as np
                a = np.array(p1)
                b = np.array(p2)
                c = np.array(p3)
                d = np.array(p4)

                # Calculate bounding boxes
                min_ab = np.minimum(a, b)
                max_ab = np.maximum(a, b)
                min_cd = np.minimum(c, d)
                max_cd = np.maximum(c, d)

                # Check if bounding boxes overlap
                if np.any(max_ab + radius < min_cd - radius) or np.any(max_cd + radius < min_ab - radius):
                    return False

                # For simplicity, assume they might intersect if bounding boxes overlap
                return True

            def generate_end_edges(self):
                print("Generating STN end edges...")

                # Create start and end vertices
                start = type('SimpleVertex', (), {'agent': 'start', 'location': None, 'time': -1})
                end = type('SimpleVertex', (), {'agent': 'end', 'location': None, 'time': -2})

                # Add to vertices list
                self.vertices.extend([start, end])

                # Add edges from start to initial states
                for state in self.initial_states:
                    self.edges.append({
                        'vertex_a': start,
                        'vertex_b': state,
                        'bound': [0.0, 0.0]  # Must start at time 0
                    })

                # Add edges from final states to end
                for state in self.final_states:
                    self.edges.append({
                        'vertex_a': state,
                        'vertex_b': end,
                        'bound': [0.0, float('inf')]  # Can finish any time after reaching goal
                    })

        # Create custom STN
        stn = CustomSTN(tpg, dynamic_obstacles)
        print(f"Built custom STN with {len(stn.edges)} edges")
        return stn

def step5_optimize_temporal_plan(stn):
    """
    Step 5: Optimize the temporal plan to find feasible timing

    Args:
        stn: Simple Temporal Network (either original or custom implementation)

    Returns:
        solution: Optimized timing for each waypoint
    """
    print("Step 5: Optimizing temporal plan...")

    try:
        # Try to use the original optimizer
        optimizer = OptimizationClass(stn)

        # Generate optimized schedule
        solution = optimizer.generate_schedule()
        print("Temporal plan optimization successful")
        return solution
    except Exception as e:
        print(f"Original optimization failed: {e}")
        print("Using fallback optimization method...")

        # Custom optimization for our custom STN
        try:
            # Simple implementation of temporal plan optimization
            # This is a fallback when the original optimizer fails

            # Create a custom optimizer
            class CustomOptimizer:
                def __init__(self, stn):
                    self.stn = stn

                def generate_schedule(self):
                    print("Generating schedule with custom optimizer...")

                    # Create a dictionary to store the solution
                    solution = {}

                    # Initialize all vertices with time 0
                    vertex_times = {vertex: 0.0 for vertex in self.stn.vertices}

                    # Set start vertex to time 0
                    for edge in self.stn.edges:
                        if hasattr(edge['vertex_a'], 'agent') and edge['vertex_a'].agent == 'start':
                            vertex_times[edge['vertex_a']] = 0.0

                    # Topological sort of vertices (simplified)
                    # We'll just process edges in order, which works for most cases
                    for _ in range(5):  # Repeat a few times to propagate constraints
                        for edge in self.stn.edges:
                            # Skip edges with None vertices (absolute time constraints)
                            if edge['vertex_a'] is None or edge['vertex_b'] is None:
                                continue

                            # Get current times
                            time_a = vertex_times[edge['vertex_a']]
                            time_b = vertex_times[edge['vertex_b']]

                            # Get bounds
                            lb, ub = edge['bound']

                            # Enforce lower bound
                            if time_b < time_a + lb:
                                vertex_times[edge['vertex_b']] = time_a + lb

                            # Enforce upper bound
                            if time_b > time_a + ub:
                                # This is a conflict - we'll try to resolve by pushing earlier vertices earlier
                                if time_a > 0:
                                    # Can we push vertex_a earlier?
                                    vertex_times[edge['vertex_a']] = max(0, time_a - (time_b - time_a - ub))
                                else:
                                    # If we can't push earlier, push later
                                    vertex_times[edge['vertex_b']] = time_a + ub

                    # Convert to solution format
                    for vertex in self.stn.vertices:
                        if hasattr(vertex, 'agent') and vertex.agent not in ['start', 'end']:
                            if vertex.agent not in solution:
                                solution[vertex.agent] = []

                            if hasattr(vertex, 'location') and vertex.location is not None:
                                solution[vertex.agent].append({
                                    'x': vertex.location.x,
                                    'y': vertex.location.y,
                                    't': vertex_times[vertex]
                                })

                    # Sort each agent's waypoints by time
                    for agent in solution:
                        solution[agent].sort(key=lambda wp: wp['t'])

                    print(f"Generated schedule for {len(solution)} agents")
                    return solution

            # Create and use custom optimizer
            custom_optimizer = CustomOptimizer(stn)
            solution = custom_optimizer.generate_schedule()
            print("Custom temporal plan optimization successful")
            return solution
        except Exception as e:
            print(f"Custom optimization failed: {e}")

            # Last resort: return a simple solution based on original schedule
            print("Using last resort: simple time-scaled solution")

            # Create a very simple solution by scaling the original schedule
            solution = {}
            for agent, waypoints in stn.tpg.schedule.items():
                solution[agent] = []
                for i, wp in enumerate(waypoints):
                    # Scale time to ensure it's feasible
                    # Use a conservative velocity (half of max)
                    if i > 0:
                        prev_wp = waypoints[i-1]
                        dx = wp['x'] - prev_wp['x']
                        dy = wp['y'] - prev_wp['y']
                        dist = (dx*dx + dy*dy) ** 0.5

                        # Calculate time based on distance and velocity
                        v = UAV_PARAMS['v_max'] * 0.5
                        dt = dist / v if v > 0 else 1.0

                        # Add to previous time
                        t = solution[agent][-1]['t'] + dt
                    else:
                        # First waypoint starts at time 0
                        t = 0.0

                    solution[agent].append({
                        'x': wp['x'],
                        'y': wp['y'],
                        't': t
                    })

            print("Simple time-scaled solution created")
            return solution


def check_trajectory_avoids_obstacles(trajectory, dynamic_obstacles, grid_seq):
    """
    Check if a trajectory avoids all dynamic obstacles

    Args:
        trajectory: The trajectory to check
        dynamic_obstacles: Dictionary of dynamic obstacles with their schedules
        grid_seq: Grid sequence with dynamic obstacles

    Returns:
        avoids_obstacles: Boolean indicating if the trajectory avoids all obstacles
        collision_points: List of (waypoint, time) pairs where collisions occur
    """
    collision_points = []

    # For each waypoint in the trajectory
    for i, (wp, t) in enumerate(zip(trajectory.waypoints, trajectory.times)):
        # Get the grid cell coordinates
        x, y = int(wp.x), int(wp.y)

        # Find the appropriate grid frame for this time
        # Apply the same time scaling as used when creating dynamic obstacles
        grid_frame = min(int(t / (GRID_DT * RADAR_ZONE_SPEED_FACTOR)), len(grid_seq) - 1)

        # Check if this cell contains a dynamic obstacle at this time
        if 0 <= x < grid_seq[0].shape[0] and 0 <= y < grid_seq[0].shape[1]:
            cell_value = grid_seq[grid_frame][x, y]
            if 0.05 < cell_value < 0.95:  # Dynamic obstacle
                collision_points.append((wp, t))

    return len(collision_points) == 0, collision_points


def adjust_trajectory_for_obstacles(trajectory, collision_points, dynamic_obstacles, grid_seq):
    """
    Adjust a trajectory to avoid dynamic obstacles

    Args:
        trajectory: The trajectory to adjust
        collision_points: List of (waypoint, time) pairs where collisions occur
        dynamic_obstacles: Dictionary of dynamic obstacles with their schedules
        grid_seq: Grid sequence with dynamic obstacles

    Returns:
        success: Boolean indicating if the adjustment was successful
    """
    if not collision_points:
        return True

    # Group collision points by waypoint to identify radar zones that are traversed for extended periods
    collision_by_location = {}
    for wp, t in collision_points:
        loc = (int(wp.x), int(wp.y))
        if loc not in collision_by_location:
            collision_by_location[loc] = []
        collision_by_location[loc].append((wp, t))

    # Strategy 1: Speed adjustment - move faster through radar zones
    # For locations with multiple collision points (extended stay in radar zone)
    for loc, collisions in collision_by_location.items():
        if len(collisions) > 1:
            # Sort collisions by time
            collisions.sort(key=lambda c: c[1])

            # Find the first and last waypoint indices
            first_wp, first_time = collisions[0]
            last_wp, last_time = collisions[-1]

            first_idx = trajectory.waypoints.index(first_wp)
            last_idx = trajectory.waypoints.index(last_wp)

            # Calculate the time spent in the radar zone
            time_in_zone = last_time - first_time

            # Aim to reduce time in zone by 75% (move 4x faster through the zone)
            target_time = time_in_zone * 0.25

            # Calculate the time compression factor
            if time_in_zone > 0:
                compression_factor = target_time / time_in_zone

                # Apply compression to all waypoints in the zone
                original_times = trajectory.times.copy()
                time_before_zone = original_times[first_idx]

                # Compress times within the zone
                for i in range(first_idx, last_idx + 1):
                    relative_time = original_times[i] - time_before_zone
                    trajectory.times[i] = time_before_zone + relative_time * compression_factor

                # Shift all subsequent waypoints
                time_shift = (original_times[last_idx] - trajectory.times[last_idx])
                for i in range(last_idx + 1, len(trajectory.times)):
                    trajectory.times[i] -= time_shift

                # Recompute dynamics
                trajectory.compute_dynamics()

                # Check if the trajectory is still dynamically feasible
                if not trajectory.is_dynamically_feasible(UAV_PARAMS):
                    # If not, revert the changes
                    trajectory.times = original_times
                    trajectory.compute_dynamics()
                else:
                    # Check if we've resolved all collisions
                    avoids_obstacles, remaining_collisions = check_trajectory_avoids_obstacles(
                        trajectory, dynamic_obstacles, grid_seq)
                    if avoids_obstacles:
                        return True

    # Strategy 2: Timing adjustment - find safe time windows to pass through radar zones
    for wp, collision_time in collision_points:
        x, y = int(wp.x), int(wp.y)

        # Find a safe time window
        safe_time = None
        for t in range(len(grid_seq)):
            if 0 <= x < grid_seq[0].shape[0] and 0 <= y < grid_seq[0].shape[1]:
                cell_value = grid_seq[t][x, y]
                if cell_value <= 0.05 or cell_value >= 0.95:  # No dynamic obstacle
                    safe_time = float(t)
                    break

        if safe_time is not None:
            # Find the waypoint index
            wp_idx = trajectory.waypoints.index(wp)

            # Adjust the time for this waypoint and all subsequent waypoints
            time_shift = safe_time - collision_time
            original_times = trajectory.times.copy()

            for i in range(wp_idx, len(trajectory.times)):
                trajectory.times[i] += time_shift

            # Recompute dynamics
            trajectory.compute_dynamics()

            # Check if the trajectory is still dynamically feasible
            if not trajectory.is_dynamically_feasible(UAV_PARAMS):
                # If not, revert the changes
                trajectory.times = original_times
                trajectory.compute_dynamics()
            else:
                # Check if we've resolved all collisions
                avoids_obstacles, remaining_collisions = check_trajectory_avoids_obstacles(
                    trajectory, dynamic_obstacles, grid_seq)
                if avoids_obstacles:
                    return True

    # Strategy 3: Path modification - add waypoints to go around radar zones
    # For each collision point, try to find a detour
    for wp, collision_time in collision_points:
        x, y = int(wp.x), int(wp.y)
        wp_idx = trajectory.waypoints.index(wp)

        # Skip if this is the first or last waypoint (can't modify start/goal)
        if wp_idx == 0 or wp_idx == len(trajectory.waypoints) - 1:
            continue

        # Get previous and next waypoints
        prev_wp = trajectory.waypoints[wp_idx - 1]
        next_wp = trajectory.waypoints[wp_idx + 1]

        # Try to find a safe detour point
        for dx in [-2, -1, 1, 2]:
            for dy in [-2, -1, 1, 2]:
                detour_x = x + dx
                detour_y = y + dy

                # Check if the detour point is within bounds and obstacle-free
                if (0 <= detour_x < grid_seq[0].shape[0] and
                    0 <= detour_y < grid_seq[0].shape[1]):

                    # Check if this point is free at the collision time
                    # Apply the same time scaling as used when creating dynamic obstacles
                    grid_frame = min(int(collision_time / (GRID_DT * RADAR_ZONE_SPEED_FACTOR)), len(grid_seq) - 1)
                    cell_value = grid_seq[grid_frame][detour_x, detour_y]

                    if cell_value <= 0.05 or cell_value >= 0.95:  # No dynamic obstacle
                        # Create a new waypoint for the detour
                        detour_wp = Waypoint(detour_x, detour_y)

                        # Calculate appropriate time for the detour
                        # Interpolate based on distance
                        d1 = math.sqrt((detour_x - prev_wp.x)**2 + (detour_y - prev_wp.y)**2)
                        d2 = math.sqrt((next_wp.x - detour_x)**2 + (next_wp.y - detour_y)**2)
                        d_total = d1 + d2
                        d_original = math.sqrt((next_wp.x - prev_wp.x)**2 + (next_wp.y - prev_wp.y)**2)

                        # If the detour isn't too long, use it
                        if d_total < d_original * 1.5:  # Allow up to 50% longer path
                            # Calculate time for the detour
                            t_prev = trajectory.times[wp_idx - 1]
                            t_next = trajectory.times[wp_idx + 1]
                            detour_time = t_prev + (t_next - t_prev) * (d1 / d_total)

                            # Create a copy of the trajectory
                            new_trajectory = Trajectory(trajectory.agent_id)

                            # Add all waypoints up to the collision
                            for i in range(wp_idx):
                                new_trajectory.add_waypoint(trajectory.waypoints[i], trajectory.times[i])

                            # Add the detour waypoint
                            new_trajectory.add_waypoint(detour_wp, detour_time)

                            # Add all waypoints after the collision
                            for i in range(wp_idx + 1, len(trajectory.waypoints)):
                                new_trajectory.add_waypoint(trajectory.waypoints[i], trajectory.times[i])

                            # Compute dynamics for the new trajectory
                            new_trajectory.compute_dynamics()

                            # Check if the new trajectory is dynamically feasible
                            if new_trajectory.is_dynamically_feasible(UAV_PARAMS):
                                # Check if it avoids all obstacles
                                avoids_obstacles, new_collisions = check_trajectory_avoids_obstacles(
                                    new_trajectory, dynamic_obstacles, grid_seq)

                                if avoids_obstacles or len(new_collisions) < len(collision_points):
                                    # Update the original trajectory
                                    trajectory.waypoints = new_trajectory.waypoints
                                    trajectory.times = new_trajectory.times
                                    _apply_kinematic_interpolation(trajectory,
                                                                   trajectory.agent_id,
                                                                   f"agent{trajectory.agent_id}",
                                                                   grid_seq)

                                    # If we've resolved all collisions, return success
                                    if avoids_obstacles:
                                        return True

                                    # Otherwise, recursively try to fix the remaining collisions
                                    if adjust_trajectory_for_obstacles(trajectory, new_collisions,
                                                                      dynamic_obstacles, grid_seq):
                                        return True

    # If all strategies failed, return False
    return False


def _apply_kinematic_interpolation(trajectory, agent_id, agent_name, grid_seq=None):
    """
    Apply kinematically feasible interpolation to a trajectory with continuous motion constraints
    while preserving the original timing relationships and accounting for drone dynamics

    Args:
        trajectory: The trajectory to interpolate
        agent_id: The agent ID
        agent_name: The agent name
        grid_seq: Grid sequence with dynamic obstacles (optional)

    Returns:
        success: Boolean indicating if the interpolation was successful
    """
    try:
        # Extract waypoints and times
        waypoints = trajectory.waypoints
        times = trajectory.times

        if len(waypoints) < 2:
            print(f"Not enough waypoints for interpolation for {agent_name}")
            return False

        # Calculate total path length and analyze original timing
        total_distance = 0
        original_velocities = []
        segment_lengths = []
        segment_directions = []
        turn_angles = []

        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i+1]
            dx = wp2.x - wp1.x
            dy = wp2.y - wp1.y
            segment_length = math.sqrt(dx*dx + dy*dy)
            segment_lengths.append(segment_length)
            total_distance += segment_length

            # Calculate segment direction
            if segment_length > 0:
                direction = (dx/segment_length, dy/segment_length)
            else:
                direction = (0, 0)
            segment_directions.append(direction)

            # Calculate turn angle between segments
            if i > 0:
                prev_dir = segment_directions[i-1]
                curr_dir = direction

                # Calculate dot product
                dot_product = prev_dir[0]*curr_dir[0] + prev_dir[1]*curr_dir[1]
                # Clamp to [-1, 1] to avoid numerical issues
                dot_product = max(-1.0, min(1.0, dot_product))

                # Calculate angle in radians
                angle = math.acos(dot_product)
                turn_angles.append(angle)
            else:
                turn_angles.append(0)

            # Calculate original velocity for this segment
            if i < len(times) - 1:
                dt = times[i+1] - times[i]
                if dt > 0 and segment_length > 0:
                    velocity = segment_length / dt
                    original_velocities.append(velocity)
                else:
                    # Handle zero time difference (waiting) by using minimum velocity
                    original_velocities.append(UAV_PARAMS['v_min'])
            else:
                # Default velocity if timing information is missing
                original_velocities.append(UAV_PARAMS['v_min'])

        # Add a final turn angle for completeness
        turn_angles.append(0)

        # Analyze velocity distribution
        if original_velocities:
            avg_velocity = sum(original_velocities) / len(original_velocities)
            max_velocity = max(original_velocities)
            min_velocity = min(v for v in original_velocities if v > 0)

            print(f"Original velocity stats for {agent_name}:")
            print(f"  Average: {avg_velocity:.2f}, Min: {min_velocity:.2f}, Max: {max_velocity:.2f}")

            # Identify waiting segments (very low velocity)
            waiting_segments = [i for i, v in enumerate(original_velocities) if v < 0.1]
            if waiting_segments:
                print(f"  Found {len(waiting_segments)} waiting segments")

            # Calculate kinematically feasible velocities based on turn angles
            kinematic_velocities = []

            # Look ahead and behind for better velocity planning
            look_ahead = 2  # Number of segments to look ahead for upcoming turns
            look_behind = 2  # Number of segments to look behind for recent turns

            for i, v in enumerate(original_velocities):
                # Get the turn angle at the start and end of this segment
                start_angle = turn_angles[i]
                end_angle = turn_angles[i+1] if i+1 < len(turn_angles) else 0

                # Look ahead for upcoming turns
                upcoming_angles = []
                for j in range(1, look_ahead + 1):
                    if i + j < len(turn_angles):
                        upcoming_angles.append(turn_angles[i + j])

                # Look behind for recent turns
                recent_angles = []
                for j in range(1, look_behind + 1):
                    if i - j >= 0:
                        recent_angles.append(turn_angles[i - j])

                # Calculate maximum turn angle in current segment
                max_turn_angle = max(start_angle, end_angle)

                # Consider upcoming turns for deceleration
                if upcoming_angles:
                    max_upcoming_angle = max(upcoming_angles)
                    # If an upcoming turn is sharp, start slowing down now
                    if max_upcoming_angle > 0.5:  # About 30 degrees
                        # The closer the turn, the more we slow down
                        for j, angle in enumerate(upcoming_angles):
                            if angle > 0.5:
                                # Deceleration factor based on distance to turn
                                decel_factor = 1.0 - (0.7 / (j + 1)) * (angle / math.pi)
                                max_turn_angle = max(max_turn_angle, angle * decel_factor)

                # Consider recent turns for acceleration
                if recent_angles and max(recent_angles) > 0.5:
                    # Just completed a turn, gradually accelerate
                    accel_factor = 0.5 + 0.5 * (i / (len(original_velocities) - 1))  # Increases with distance from turn
                    max_turn_angle = max(max_turn_angle, max(recent_angles) * (1 - accel_factor))

                # Calculate velocity factor based on turn angle
                # More sophisticated model: v_max = omega_max * r
                # For a given angle theta, r = d / (2 * sin(theta/2)) where d is the chord length
                if max_turn_angle > 0.001:
                    # Calculate minimum turn radius based on UAV dynamics
                    min_radius = segment_lengths[i] / (2 * math.sin(max_turn_angle / 2)) if max_turn_angle < math.pi else 0

                    # Calculate maximum velocity based on turn radius and max angular velocity
                    v_max_turn = UAV_PARAMS['omega_max'] * min_radius if min_radius > 0 else UAV_PARAMS['v_min']

                    # Ensure it's within UAV constraints
                    v_max_turn = max(UAV_PARAMS['v_min'], min(v_max_turn, UAV_PARAMS['v_max']))

                    # Use the more conservative of the two approaches
                    angle_factor = 1.0 - max_turn_angle / math.pi
                    angle_factor = max(0.3, angle_factor)  # Don't go below 30% of max speed
                    v_max_angle = UAV_PARAMS['v_max'] * angle_factor

                    v_max = min(v_max_turn, v_max_angle)
                else:
                    # No significant turn, use maximum velocity
                    v_max = UAV_PARAMS['v_max']

                # Calculate velocity considering both original velocity and turn constraints
                if i in waiting_segments:
                    # For waiting segments, use a higher velocity to reduce waiting time
                    # but still respect kinematic constraints
                    kinematic_v = min(UAV_PARAMS['v_max'] * 0.8, v_max)
                else:
                    # For normal segments, try to preserve original velocity but respect turn constraints
                    kinematic_v = min(v, v_max)

                # Apply velocity smoothing to avoid sudden changes
                if i > 0 and kinematic_velocities:
                    prev_v = kinematic_velocities[-1]
                    # Limit acceleration/deceleration to 30% change per segment
                    max_change = 0.3 * prev_v
                    if kinematic_v > prev_v + max_change:
                        kinematic_v = prev_v + max_change  # Limit acceleration
                    elif kinematic_v < prev_v - max_change:
                        kinematic_v = prev_v - max_change  # Limit deceleration

                # Ensure velocity is within UAV constraints
                kinematic_v = max(UAV_PARAMS['v_min'], min(kinematic_v, UAV_PARAMS['v_max']))
                kinematic_velocities.append(kinematic_v)

                # Debug output for significant turns or velocity changes
                if max_turn_angle > 0.5 or (i > 0 and abs(kinematic_velocities[-2] - kinematic_v) > 0.2):
                    print(f"  Segment {i}: Turn angle {math.degrees(max_turn_angle):.1f}°, "
                          f"velocity adjusted from {v:.2f} to {kinematic_v:.2f}")
        else:
            # Default to a reasonable velocity if we couldn't calculate original velocities
            kinematic_velocities = [UAV_PARAMS['v_max'] * 0.7] * len(segment_lengths)

        # Create a new trajectory with kinematically feasible timing
        adjusted = Trajectory(agent_id)

        # Check for duplicate waypoints in the original trajectory
        has_duplicates = False
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i-1].x
            dy = waypoints[i].y - waypoints[i-1].y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-6:
                has_duplicates = True
                print(f"  Warning: Found duplicate waypoints at index {i-1} and {i} in trajectory for {agent_name}")

        if has_duplicates:
            print(f"  Removing duplicate waypoints for {agent_name}")
            # Create a filtered list of waypoints and times
            filtered_waypoints = [waypoints[0]]
            filtered_times = [0.0]  # First waypoint starts at time 0

            for i in range(1, len(waypoints)):
                prev_wp = filtered_waypoints[-1]
                curr_wp = waypoints[i]
                dx = curr_wp.x - prev_wp.x
                dy = curr_wp.y - prev_wp.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist >= 1e-6:  # Not a duplicate
                    filtered_waypoints.append(curr_wp)
                    if i < len(times):
                        filtered_times.append(times[i])
                    else:
                        # If we don't have a time for this waypoint, estimate it
                        filtered_times.append(filtered_times[-1] + 1.0)

            # Use the filtered waypoints
            waypoints = filtered_waypoints
            times = filtered_times
            print(f"  After filtering: {len(waypoints)} waypoints remain for {agent_name}")

            # Recalculate segment lengths and velocities after filtering
            segment_lengths = []
            original_velocities = []
            segment_directions = []
            turn_angles = []

            for i in range(len(waypoints) - 1):
                wp1 = waypoints[i]
                wp2 = waypoints[i+1]
                dx = wp2.x - wp1.x
                dy = wp2.y - wp1.y
                segment_length = math.sqrt(dx*dx + dy*dy)
                segment_lengths.append(segment_length)

                # Calculate segment direction
                if segment_length > 0:
                    direction = (dx/segment_length, dy/segment_length)
                else:
                    direction = (0, 0)
                segment_directions.append(direction)

                # Calculate turn angle between segments
                if i > 0:
                    prev_dir = segment_directions[i-1]
                    curr_dir = direction

                    # Calculate dot product
                    dot_product = prev_dir[0]*curr_dir[0] + prev_dir[1]*curr_dir[1]
                    # Clamp to [-1, 1] to avoid numerical issues
                    dot_product = max(-1.0, min(1.0, dot_product))

                    # Calculate angle in radians
                    angle = math.acos(dot_product)
                    turn_angles.append(angle)
                else:
                    turn_angles.append(0)

                # Calculate original velocity for this segment
                if i < len(times) - 1:
                    dt = times[i+1] - times[i]
                    if dt > 0 and segment_length > 0:
                        velocity = segment_length / dt
                        original_velocities.append(velocity)
                    else:
                        # Handle zero time difference (waiting) by using minimum velocity
                        original_velocities.append(UAV_PARAMS['v_min'])
                else:
                    # Default velocity if timing information is missing
                    original_velocities.append(UAV_PARAMS['v_min'])

            # Add a final turn angle for completeness
            turn_angles.append(0)

            # Recalculate kinematic velocities
            kinematic_velocities = []
            for i, v in enumerate(original_velocities):
                # Get the turn angle at the start and end of this segment
                start_angle = turn_angles[i]
                end_angle = turn_angles[i+1] if i+1 < len(turn_angles) else 0

                # Calculate maximum safe velocity based on turn angles
                max_turn_angle = max(start_angle, end_angle)

                # Calculate velocity factor based on turn angle
                angle_factor = 1.0 - max_turn_angle / math.pi
                angle_factor = max(0.3, angle_factor)  # Don't go below 30% of max speed

                # Calculate velocity considering both original velocity and turn constraints
                kinematic_v = min(v, UAV_PARAMS['v_max'] * angle_factor)

                # Ensure velocity is within UAV constraints
                kinematic_v = max(UAV_PARAMS['v_min'], min(kinematic_v, UAV_PARAMS['v_max']))
                kinematic_velocities.append(kinematic_v)

                # Debug output for significant turns
                if max_turn_angle > 0.5:  # About 30 degrees
                    print(f"  Segment {i}: Turn angle {math.degrees(max_turn_angle):.1f}°, "
                          f"velocity adjusted from {v:.2f} to {kinematic_v:.2f}")

            # If we have too few waypoints after filtering, add some intermediate points
            if len(waypoints) < 3 and len(waypoints) > 1:
                print(f"  Adding intermediate waypoints for {agent_name} to ensure smooth interpolation")
                # Add a midpoint between start and goal
                mid_wp = Waypoint(
                    (waypoints[0].x + waypoints[-1].x) / 2.0,
                    (waypoints[0].y + waypoints[-1].y) / 2.0
                )
                mid_time = (times[0] + times[-1]) / 2.0

                # Insert the midpoint
                waypoints = [waypoints[0], mid_wp, waypoints[-1]]
                times = [times[0], mid_time, times[-1]]

        # First waypoint starts at time 0
        adjusted.add_waypoint(waypoints[0], 0.0)

        # Calculate times based on kinematic velocities
        current_time = 0.0

        # Make sure we have at least one segment length
        if not segment_lengths:
            print(f"  Warning: No segment lengths calculated for {agent_name}")
            if len(waypoints) >= 2:
                # Calculate at least one segment length
                wp1 = waypoints[0]
                wp2 = waypoints[1]
                dx = wp2.x - wp1.x
                dy = wp2.y - wp1.y
                segment_length = math.sqrt(dx*dx + dy*dy)
                segment_lengths.append(segment_length)

                # Add a default velocity
                kinematic_velocities = [UAV_PARAMS['v_max'] * 0.7]
                print(f"  Added default segment length {segment_length:.2f} with velocity {kinematic_velocities[0]:.2f}")
            else:
                print(f"  Error: Not enough waypoints to calculate segment lengths for {agent_name}")
                return False

        # Check if we have enough waypoints
        if len(waypoints) <= len(segment_lengths):
            print(f"  Warning: Not enough waypoints ({len(waypoints)}) for segments ({len(segment_lengths)}) in {agent_name}")
            # Add missing waypoints if needed
            while len(waypoints) <= len(segment_lengths):
                # Create a new waypoint by extrapolating from the last two
                if len(waypoints) >= 2:
                    last_wp = waypoints[-1]
                    prev_wp = waypoints[-2]
                    dx = last_wp.x - prev_wp.x
                    dy = last_wp.y - prev_wp.y
                    new_wp = Waypoint(last_wp.x + dx, last_wp.y + dy)
                    waypoints.append(new_wp)
                    print(f"  Added extrapolated waypoint at ({new_wp.x}, {new_wp.y})")
                else:
                    # If we don't have enough waypoints to extrapolate, duplicate the last one
                    last_wp = waypoints[-1]
                    new_wp = Waypoint(last_wp.x, last_wp.y)
                    waypoints.append(new_wp)
                    print(f"  Added duplicate waypoint at ({new_wp.x}, {new_wp.y})")

        # Make sure we have enough kinematic velocities
        if len(kinematic_velocities) < len(segment_lengths):
            print(f"  Warning: Not enough velocities ({len(kinematic_velocities)}) for segments ({len(segment_lengths)}) in {agent_name}")
            # Add default velocities
            while len(kinematic_velocities) < len(segment_lengths):
                kinematic_velocities.append(UAV_PARAMS['v_max'] * 0.7)

        # Now add waypoints with adjusted times
        for i in range(len(segment_lengths)):
            # Time = distance / velocity
            if i < len(kinematic_velocities):
                velocity = kinematic_velocities[i]
            else:
                velocity = UAV_PARAMS['v_max'] * 0.7
                print(f"  Warning: Using default velocity for segment {i}")

            segment_time = segment_lengths[i] / velocity if velocity > 0 else 0.1
            current_time += segment_time

            # Make sure we don't go out of bounds
            if i+1 < len(waypoints):
                # Add waypoint with adjusted time
                adjusted.add_waypoint(waypoints[i+1], current_time)
            else:
                print(f"  Warning: Skipping waypoint {i+1} (out of range) for {agent_name}")

        # Check if the adjusted trajectory avoids dynamic obstacles
        if grid_seq is not None:
            # Create a temporary trajectory for checking
            temp_trajectory = Trajectory(agent_id)
            for wp, t in zip(adjusted.waypoints, adjusted.times):
                temp_trajectory.add_waypoint(wp, t)

            # Check for collisions with dynamic obstacles
            has_collisions = False
            for i, (wp, t) in enumerate(zip(temp_trajectory.waypoints, temp_trajectory.times)):
                # Get the grid cell coordinates
                x, y = int(wp.x), int(wp.y)

                # Find the appropriate grid frame for this time
                # Apply the same time scaling as used when creating dynamic obstacles
                grid_frame = min(int(t / (GRID_DT * RADAR_ZONE_SPEED_FACTOR)), len(grid_seq) - 1)

                # Check if this cell contains a dynamic obstacle at this time
                if 0 <= x < grid_seq[0].shape[0] and 0 <= y < grid_seq[0].shape[1]:
                    cell_value = grid_seq[grid_frame][x, y]
                    if 0.05 < cell_value < 0.95:  # Dynamic obstacle
                        has_collisions = True
                        print(f"  Warning: Adjusted trajectory for {agent_name} collides with dynamic obstacle at time {t:.2f}")
                        break

            if has_collisions:
                # Try to adjust the timing to avoid collisions
                print(f"  Attempting to adjust timing to avoid dynamic obstacles for {agent_name}")

                # Scale the entire trajectory to be faster or slower
                for scale_factor in [0.8, 0.9, 1.1, 1.2, 0.7, 1.3]:
                    # Create a scaled trajectory
                    scaled = Trajectory(agent_id)
                    scaled.add_waypoint(adjusted.waypoints[0], 0.0)

                    for i in range(1, len(adjusted.waypoints)):
                        scaled_time = adjusted.times[i] * scale_factor
                        scaled.add_waypoint(adjusted.waypoints[i], scaled_time)

                    # Check for collisions
                    has_collisions = False
                    for i, (wp, t) in enumerate(zip(scaled.waypoints, scaled.times)):
                        x, y = int(wp.x), int(wp.y)
                        # Apply the same time scaling as used when creating dynamic obstacles
                        grid_frame = min(int(t / (GRID_DT * RADAR_ZONE_SPEED_FACTOR)), len(grid_seq) - 1)

                        if 0 <= x < grid_seq[0].shape[0] and 0 <= y < grid_seq[0].shape[1]:
                            cell_value = grid_seq[grid_frame][x, y]
                            if 0.05 < cell_value < 0.95:  # Dynamic obstacle
                                has_collisions = True
                                break

                    if not has_collisions:
                        print(f"  Successfully avoided dynamic obstacles by scaling time by {scale_factor}")
                        adjusted = scaled
                        break

                if has_collisions:
                    print(f"  Warning: Could not avoid all dynamic obstacles for {agent_name}")

        # Now create a densely interpolated trajectory using cubic spline interpolation
        # ------------------------------------------------------------------
        # ONE global spline with timing that respects original schedule
        # ------------------------------------------------------------------
        pts_xy = [(w.x, w.y) for w in adjusted.waypoints]
        original_times = adjusted.times

        # Use our improved spline function that respects original timing and avoids obstacles
        nsamples = 250
        interp_x, interp_y, t_dense = make_global_spline(
            pts_xy,
            UAV_PARAMS['v_max'],
            UAV_PARAMS['omega_max'],
            nsamples=nsamples,
            original_times=original_times,
            grid_seq=grid_seq  # Pass grid_seq to check for collisions during interpolation
        )

        # Print timing information for debugging
        print(f"  Original timing: start={original_times[0]:.2f}, end={original_times[-1]:.2f}, duration={original_times[-1]-original_times[0]:.2f}")
        print(f"  Adjusted timing: start={t_dense[0]:.2f}, end={t_dense[-1]:.2f}, duration={t_dense[-1]-t_dense[0]:.2f}")

        interpolated = Trajectory(agent_id)
        for xx, yy, tt in zip(interp_x, interp_y, t_dense):
            interpolated.add_waypoint(Waypoint(float(xx), float(yy)), float(tt))
        # ---------- end Catmull-Rom block ----------

        # Replace the original trajectory with the interpolated one
        trajectory.waypoints = interpolated.waypoints
        trajectory.times = interpolated.times

        # Calculate and store velocities for visualization
        trajectory.velocities = []
        for i in range(len(trajectory.waypoints) - 1):
            wp1 = trajectory.waypoints[i]
            wp2 = trajectory.waypoints[i+1]
            t1 = trajectory.times[i]
            t2 = trajectory.times[i+1]

            dx = wp2.x - wp1.x
            dy = wp2.y - wp1.y
            dt = t2 - t1

            if dt > 0:
                vx = dx / dt
                vy = dy / dt
            else:
                vx = 0
                vy = 0

            trajectory.velocities.append((vx, vy))

        # Add a final velocity (same as the last segment)
        if trajectory.velocities:
            trajectory.velocities.append(trajectory.velocities[-1])
        else:
            trajectory.velocities.append((0, 0))

        print(f"Successfully applied kinematically feasible interpolation to trajectory for {agent_name}")
        return True
    except Exception as e:
        print(f"Kinematic interpolation failed for {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        # Keep the original trajectory
        return False





def step6_generate_final_trajectories(schedule, uav_params, dynamic_obstacles=None, grid_seq=None):
    """
    Step 6: Generate final trajectories with control inputs and validate against UAV dynamics

    Args:
        schedule: Optimized schedule from the temporal plan
        uav_params: UAV dynamics parameters
        dynamic_obstacles: Dictionary of dynamic obstacles with their schedules
        grid_seq: Grid sequence with dynamic obstacles

    Returns:
        result_trajectories: List of final trajectories with control inputs
    """
    print("Step 6: Generating final trajectories...")

    # Initialize the list to store final trajectories
    result_trajectories = []

    for agent_name, waypoints in schedule.items():
        try:
            # Extract agent ID from name
            agent_id = int(agent_name.replace('agent', ''))

            # Create a new trajectory
            trajectory = Trajectory(agent_id)

            # Sort waypoints by time
            waypoints.sort(key=lambda wp: wp['t'])

            # Add waypoints to trajectory
            for wp in waypoints:
                waypoint = Waypoint(wp['x'], wp['y'])
                trajectory.add_waypoint(waypoint, wp['t'])

            # Apply path smoothing using kinematic interpolation
            if len(trajectory.waypoints) >= 2:
                try:
                    # Try to apply kinematic interpolation with dynamic obstacle avoidance
                    _apply_kinematic_interpolation(trajectory, agent_id, agent_name, grid_seq)
                except Exception as e:
                    print(f"Path smoothing failed for {agent_name}: {e}. Using original trajectory.")

            # Compute dynamics
            trajectory.compute_dynamics()

            # Check if the trajectory is dynamically feasible
            if trajectory.is_dynamically_feasible(uav_params):
                print(f"Trajectory for {agent_name} is dynamically feasible")
            else:
                print(f"Warning: Initial trajectory for {agent_name} is not dynamically feasible")

            # Check if the trajectory avoids dynamic obstacles
            if dynamic_obstacles and grid_seq is not None:
                avoids_obstacles, collision_points = check_trajectory_avoids_obstacles(
                    trajectory, dynamic_obstacles, grid_seq)
                if not avoids_obstacles:
                    print(f"Warning: Trajectory for {agent_name} collides with dynamic obstacles at {len(collision_points)} points")
                    # Attempt to adjust the trajectory to avoid obstacles
                    if adjust_trajectory_for_obstacles(trajectory, collision_points, dynamic_obstacles, grid_seq):
                        print(f"Successfully adjusted trajectory for {agent_name} to avoid dynamic obstacles")
                    else:
                        print(f"Failed to adjust trajectory for {agent_name} to avoid all dynamic obstacles")

                # Apply a more sophisticated adjustment to ensure continuous motion
                # This approach scales the entire trajectory to meet velocity constraints

                # First, check if there are any velocity issues
                min_velocity = float('inf')
                max_velocity = 0
                for i in range(len(trajectory.velocities)):
                    vx, vy = trajectory.velocities[i]
                    v_mag = math.sqrt(vx*vx + vy*vy)
                    min_velocity = min(min_velocity, v_mag) if v_mag > 0 else min_velocity
                    max_velocity = max(max_velocity, v_mag)

                print(f"  Velocity range: {min_velocity:.2f} to {max_velocity:.2f} (constraints: {uav_params['v_min']:.2f} to {uav_params['v_max']:.2f})")

                # Determine if we need to scale the trajectory
                need_scaling = False
                scale_factor = 1.0

                if max_velocity > uav_params['v_max']:
                    # Need to slow down (stretch time)
                    scale_factor = max(scale_factor, max_velocity / uav_params['v_max'] * 1.05)  # Add 5% margin
                    need_scaling = True
                    print(f"  Need to slow down by factor of {scale_factor:.2f}")

                if min_velocity < uav_params['v_min'] and min_velocity > 0:
                    # Need to speed up (compress time)
                    temp_factor = uav_params['v_min'] / min_velocity * 1.05  # Add 5% margin
                    # Only apply if it doesn't conflict with max velocity constraint
                    if max_velocity * temp_factor <= uav_params['v_max']:
                        scale_factor = 1.0 / temp_factor  # Invert because we're scaling time
                        need_scaling = True
                        print(f"  Need to speed up by factor of {1.0/scale_factor:.2f}")

                # Check for turn rate issues
                max_turn_rate = 0
                for turn_rate in trajectory.turn_rates:
                    max_turn_rate = max(max_turn_rate, abs(turn_rate))

                if max_turn_rate > uav_params['omega_max']:
                    # Need to slow down turns
                    turn_scale = max_turn_rate / uav_params['omega_max'] * 1.05  # Add 5% margin
                    scale_factor = max(scale_factor, turn_scale)
                    need_scaling = True
                    print(f"  Need to slow down turns by factor of {turn_scale:.2f}")

                if need_scaling:
                    # Apply scaling to the entire trajectory
                    # We'll create a new trajectory with adjusted timing
                    adjusted_trajectory = Trajectory(agent_id)

                    # Keep the first waypoint at time 0
                    adjusted_trajectory.add_waypoint(trajectory.waypoints[0], 0.0)

                    # Scale all other waypoint times
                    for i in range(1, len(trajectory.waypoints)):
                        wp = trajectory.waypoints[i]
                        original_dt = trajectory.times[i] - trajectory.times[0]
                        adjusted_time = trajectory.times[0] + original_dt * scale_factor
                        adjusted_trajectory.add_waypoint(wp, adjusted_time)

                    # Replace the trajectory
                    trajectory.waypoints = adjusted_trajectory.waypoints
                    trajectory.times = adjusted_trajectory.times

                    # Recompute dynamics
                    trajectory.compute_dynamics()

                    # Check if the adjusted trajectory is feasible
                    if trajectory.is_dynamically_feasible(uav_params):
                        print(f"  Adjusted trajectory for {agent_name} is now dynamically feasible")
                    else:
                        print(f"  Warning: Adjusted trajectory for {agent_name} is still not dynamically feasible")

                        # As a last resort, relax the constraints slightly for visualization purposes
                        print(f"  Relaxing constraints for visualization purposes")
                else:
                    print(f"  No scaling needed for trajectory of {agent_name}")

            # Add the trajectory to our result list
            result_trajectories.append(trajectory)
        except Exception as e:
            print(f"Error processing trajectory for {agent_name}: {e}")

    print(f"Generated {len(result_trajectories)} final trajectories")
    return result_trajectories

def visualize_trajectories(trajectories, grid_seq, output_dir="trajectory_visualization"):
    """
    Visualize the final trajectories

    Args:
        trajectories: List of trajectories
        grid_seq: The dynamic map sequence
        output_dir: Directory to save the visualization
    """
    print("Visualizing trajectories...")

    # Memory management: Limit the number of frames to process to avoid memory issues
    max_frames = 500  # Limit to 500 frames to prevent memory issues
    if grid_seq is not None and len(grid_seq) > max_frames:
        print(f"Warning: Limiting visualization to {max_frames} frames to prevent memory issues")
        grid_seq = grid_seq[:max_frames]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine the maximum time across all trajectories
    max_time = 0
    for trajectory in trajectories:
        if trajectory.times:
            max_time = max(max_time, trajectory.times[-1])

    # Create a figure
    plt.figure(figsize=(10, 10))

    # Plot the static map with dynamic obstacles
    static_map = grid_seq[0].copy()
    ax = plt.gca()
    plt.imshow(static_map, cmap='binary', origin='lower')
    lock_axes(ax, static_map)          # <<< one-liner does all three things

    # Overlay dynamic obstacles from different frames with transparency
    if len(grid_seq) > 1:
        # Create a composite map showing dynamic obstacle paths
        dynamic_overlay = np.zeros_like(static_map, dtype=float)

        # Combine all frames to show obstacle paths
        for frame in grid_seq[1:]:
            # Add obstacle positions from this frame (assuming obstacles are 0s)
            obstacle_positions = (frame == 0) & (static_map != 0)  # Only dynamic obstacles
            dynamic_overlay[obstacle_positions] += 0.2  # Add with transparency

        # Normalize and apply a colormap
        if np.max(dynamic_overlay) > 0:
            dynamic_overlay = dynamic_overlay / np.max(dynamic_overlay)
            plt.imshow(dynamic_overlay, cmap='Reds', alpha=0.5, origin='lower')

    # Define colors for different agents
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray']

    # Plot the trajectories
    for i, trajectory in enumerate(trajectories):
        # Extract x and y coordinates
        x = [wp.x for wp in trajectory.waypoints]
        y = [wp.y for wp in trajectory.waypoints]

        # Get color for this agent
        color = colors[i % len(colors)]

        # Plot the trajectory
        plt.plot(y, x, '-', linewidth=2, color=color, label=f"Agent {trajectory.agent_id}")

        # Plot the start (green) and goal (red) positions with black edge for visibility
        # Make them larger (markersize=15 instead of 10)
        plt.plot(y[0], x[0], 'o', color='green', markersize=15, markeredgecolor='black')
        plt.plot(y[-1], x[-1], 'o', color='red', markersize=15, markeredgecolor='black')

        # Add circles around start and goal positions for better visibility (+5 pixels)
        start_circle = plt.Circle((y[0], x[0]), 5, fill=False, color='green', linestyle='-', linewidth=2)
        goal_circle = plt.Circle((y[-1], x[-1]), 5, fill=False, color='red', linestyle='-', linewidth=2)
        plt.gca().add_patch(start_circle)
        plt.gca().add_patch(goal_circle)

        # Add agent ID labels for better start-goal pair clarification
        plt.text(y[0], x[0], f"S{trajectory.agent_id}", color='white', fontsize=12,
                 ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle="round", fc="green", alpha=0.7))
        plt.text(y[-1], x[-1], f"G{trajectory.agent_id}", color='white', fontsize=12,
                 ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle="round", fc="red", alpha=0.7))

    plt.title("UAV Trajectories")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "trajectories.png"))

    # Create an animation
    fig, ax = plt.subplots(figsize=(10, 10))

    # Number of frames in the animation - increased for smoother motion
    num_frames = 300  # Increased from 100 to 300 for smoother animation
    time_step = max_time / num_frames if max_time > 0 else 1.0

    for frame in range(num_frames):
        ax.clear()

        # Current time
        current_time = frame * time_step

        # Get the appropriate frame from grid_seq based on current time with interpolation
        # Assuming grid_seq has frames at regular intervals
        if len(grid_seq) > 1:
            # Calculate which frames to interpolate between, ensuring we stay within bounds
            # Map the current animation time to the grid sequence time
            # Apply the same time scaling factor used for dynamic obstacles
            # Since dynamic obstacles are created with t = grid_idx * GRID_DT * RADAR_ZONE_SPEED_FACTOR,
            # we need to divide current_time by (GRID_DT * RADAR_ZONE_SPEED_FACTOR) to get the grid index
            frame_idx = current_time / (GRID_DT * RADAR_ZONE_SPEED_FACTOR)

            # Ensure we stay within bounds
            frame_idx_floor = min(int(frame_idx), len(grid_seq) - 1)
            frame_idx_ceil = min(frame_idx_floor + 1, len(grid_seq) - 1)

            # Get the interpolation factor
            alpha = frame_idx - frame_idx_floor

            if frame_idx_floor == frame_idx_ceil or alpha < 0.001:
                # Just use the exact frame if we're at an integer time or very close
                ax.imshow(grid_seq[frame_idx_floor], cmap='binary', origin='lower')
                lock_axes(ax, static_map)
            else:
                # Interpolate between frames for smoother transitions
                # Convert to float arrays for interpolation
                frame1 = grid_seq[frame_idx_floor].astype(float)
                frame2 = grid_seq[frame_idx_ceil].astype(float)

                # Linear interpolation between frames
                interpolated_frame = frame1 * (1 - alpha) + frame2 * alpha

                # Display the interpolated frame
                ax.imshow(interpolated_frame, cmap='binary', origin='lower')
                lock_axes(ax, static_map)
        else:
            # Plot the static map if no dynamic sequence
            ax.imshow(static_map, cmap='binary', origin='lower')
            lock_axes(ax, static_map)

        # Define colors for different agents (same as in the static plot)
        colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray']

        # Plot the trajectories up to the current time
        for i, trajectory in enumerate(trajectories):
            # Extract all waypoints and times for this trajectory
            all_x = [wp.x for wp in trajectory.waypoints]
            all_y = [wp.y for wp in trajectory.waypoints]
            all_t = trajectory.times

            # Get color for this agent
            color = colors[i % len(colors)]

            if all_x and all_y:
                # Extract points up to current time for the path
                x_path = []
                y_path = []
                for wp, t in zip(trajectory.waypoints, trajectory.times):
                    if t <= current_time:
                        x_path.append(wp.x)
                        y_path.append(wp.y)

                # Plot the trajectory path up to current time
                if x_path and y_path:
                    ax.plot(y_path, x_path, '-', linewidth=2, color=color, label=f"Agent {trajectory.agent_id}")

                # Always show start and goal positions for reference
                ax.plot(all_y[0], all_x[0], 'o', color='green', markersize=12, markeredgecolor='black')
                ax.plot(all_y[-1], all_x[-1], 'o', color='red', markersize=12, markeredgecolor='black')

                # Add circles around start and goal positions for better visibility (+5 pixels)
                start_circle = plt.Circle((all_y[0], all_x[0]), 5, fill=False, color='green', linestyle='-', linewidth=2)
                goal_circle = plt.Circle((all_y[-1], all_x[-1]), 5, fill=False, color='red', linestyle='-', linewidth=2)
                ax.add_patch(start_circle)
                ax.add_patch(goal_circle)

                # Add agent ID labels
                ax.text(all_y[0], all_x[0], f"S{trajectory.agent_id}", color='white', fontsize=10,
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle="round", fc="green", alpha=0.7))
                ax.text(all_y[-1], all_x[-1], f"G{trajectory.agent_id}", color='white', fontsize=10,
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle="round", fc="red", alpha=0.7))

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
                        if next_t > prev_t:  # Avoid division by zero
                            alpha = (current_time - prev_t) / (next_t - prev_t)

                            # Interpolate position
                            current_x = all_x[prev_idx] * (1 - alpha) + all_x[next_idx] * alpha
                            current_y = all_y[prev_idx] * (1 - alpha) + all_y[next_idx] * alpha

                            # Plot the current interpolated position with agent's color
                            ax.plot(current_y, current_x, 'o', color=color, markersize=12, markeredgecolor='black')

                            # Add a circle around the current position for better visibility
                            current_circle = plt.Circle((current_y, current_x), 5, fill=False, color=color, linestyle='-', linewidth=2)
                            ax.add_patch(current_circle)

                            # Add orientation indicator (heading arrow)
                            # Calculate heading direction from velocity
                            if next_idx < len(all_x):
                                # Calculate velocity vector
                                vel_x = all_x[next_idx] - all_x[prev_idx]
                                vel_y = all_y[next_idx] - all_y[prev_idx]

                                # Normalize to get direction
                                vel_mag = math.sqrt(vel_x**2 + vel_y**2)
                                if vel_mag > 0.001:  # Avoid division by zero
                                    vel_x /= vel_mag
                                    vel_y /= vel_mag

                                    # Scale for visualization
                                    arrow_scale = 3.0
                                    vel_x *= arrow_scale
                                    vel_y *= arrow_scale

                                    # Plot heading arrow with agent's color
                                    ax.arrow(current_y, current_x, vel_y, vel_x,
                                             head_width=0.8, head_length=1.2,
                                             fc=color, ec=color, alpha=0.8)
                    elif x_path and y_path:
                        # We're at or after the last waypoint
                        ax.plot(y_path[-1], x_path[-1], 'o', color=color, markersize=12, markeredgecolor='black')

                        # Add a circle around the final position for better visibility
                        final_circle = plt.Circle((y_path[-1], x_path[-1]), 5, fill=False, color=color, linestyle='-', linewidth=2)
                        ax.add_patch(final_circle)

                        # Add orientation indicator for the final position if possible
                        if len(x_path) >= 2:
                            # Use the direction from the second-to-last to last point
                            vel_x = x_path[-1] - x_path[-2]
                            vel_y = y_path[-1] - y_path[-2]

                            # Normalize to get direction
                            vel_mag = math.sqrt(vel_x**2 + vel_y**2)
                            if vel_mag > 0.001:  # Avoid division by zero
                                vel_x /= vel_mag
                                vel_y /= vel_mag

                                # Scale for visualization
                                arrow_scale = 3.0
                                vel_x *= arrow_scale
                                vel_y *= arrow_scale

                                # Plot heading arrow with agent's color
                                ax.arrow(y_path[-1], x_path[-1], vel_y, vel_x,
                                         head_width=0.8, head_length=1.2,
                                         fc=color, ec=color, alpha=0.8)

        ax.set_title(f"Time: {current_time:.2f}")
        ax.grid(True)

        # Save the frame
        plt.savefig(os.path.join(output_dir, f"frame_{frame:03d}.png"))

    print(f"Visualization saved to {output_dir}")

    # Create a GIF animation from the frames
    try:
        # Use imageio v2 to avoid deprecation warnings
        import imageio.v2 as imageio

        # Get all frame files
        frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.png')])

        if frame_files:
            # Create a GIF
            gif_path = os.path.join(output_dir, "agent_movements.gif")

            # Read images
            images = [imageio.imread(f) for f in frame_files]

            # Save as GIF with smoother animation
            # Using a duration of 0.05 seconds per frame (20 fps) for smoother animation
            imageio.mimsave(gif_path, images, duration=0.05)  # 20 frames per second

            print(f"Animated GIF created successfully: {gif_path}")

            # Also create a higher quality MP4 video if ffmpeg is available
            try:
                import subprocess
                import shutil

                # Check if ffmpeg is available
                ffmpeg_available = shutil.which("ffmpeg") is not None

                if not ffmpeg_available:
                    print("ffmpeg not found in PATH. Skipping MP4 video creation.")
                    print("To create MP4 videos, please install ffmpeg: apt-get install -y ffmpeg")
                else:
                    # Create a temporary directory for frames with sequential numbering
                    temp_dir = os.path.join(output_dir, "temp_frames")
                    os.makedirs(temp_dir, exist_ok=True)

                    # Copy frames to temp directory with sequential numbering
                    for i, img in enumerate(images):
                        temp_frame = os.path.join(temp_dir, f"frame_{i:05d}.png")
                        imageio.imwrite(temp_frame, img)

                    # Use ffmpeg to create an MP4 video (higher quality than GIF)
                    video_path = os.path.join(output_dir, "agent_movements.mp4")
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",  # Overwrite output file if it exists
                        "-framerate", "30",  # 30 fps for smooth video
                        "-i", os.path.join(temp_dir, "frame_%05d.png"),
                        "-c:v", "libx264",
                        "-profile:v", "high",
                        "-crf", "20",  # Quality (lower is better, 18-28 is good)
                        "-pix_fmt", "yuv420p",
                        video_path
                    ]

                    # Run ffmpeg
                    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if result.returncode == 0:
                        print(f"MP4 video created successfully: {video_path}")
                    else:
                        print(f"Failed to create MP4 video. ffmpeg error: {result.stderr.decode('utf-8')}")

                    # Clean up temporary files
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)

            except Exception as e:
                print(f"Failed to create MP4 video: {e}")
        else:
            print("No frames found to create animation")
    except Exception as e:
        print(f"Failed to create animation: {e}")

def debug_dynamic_obstacles(grid_seq, output_dir="dynamic_obstacles_debug"):
    """
    Create a debug visualization of dynamic obstacles

    Args:
        grid_seq: The dynamic map sequence
        output_dir: Directory to save the visualization
    """
    # Skip visualization to save memory
    print("Debug visualization of dynamic obstacles skipped to save memory")
    return


def process_drone_paths(map_path, num_agents, num_obstacles, output_dir="drone_solution"):
    """
    Process drone paths through all 6 steps

    Args:
        map_path: Path to the static map
        num_agents: Number of agents
        num_obstacles: Number of dynamic obstacles
        output_dir: Directory to save the output

    Returns:
        final_trajectories: List of final trajectories with control inputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create debug directory
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Create a performance log file
    perf_log_path = os.path.join(output_dir, "performance_log.txt")
    with open(perf_log_path, 'w') as f:
        f.write("Performance Analysis Log\n")
        f.write("======================\n\n")

    # Import time for performance measurement
    import time

    # Step 1: Generate initial discrete paths
    print("\nStep 1: Generating initial discrete paths...")
    start_time = time.time()
    paths, grid_seq, start_list, goal_list = step1_generate_initial_paths(
        map_path, num_agents, num_obstacles, output_dir
    )
    step1_time = time.time() - start_time
    print(f"Step 1 completed in {step1_time:.2f} seconds")

    # Skip debug visualization of dynamic obstacles to save memory
    debug_time = 0
    print("Skipping debug visualization to save memory")

    # Step 2: Convert discrete paths to schedule format
    print("\nStep 2: Converting discrete paths to schedule format...")
    start_time = time.time()
    schedule = step2_convert_to_schedule(paths)
    step2_time = time.time() - start_time
    print(f"Step 2 completed in {step2_time:.2f} seconds")

    # Save the schedule to a YAML file
    schedule_path = os.path.join(output_dir, "initial_schedule.yaml")
    with open(schedule_path, 'w') as f:
        yaml.dump({"schedule": schedule}, f, default_flow_style=False)

    # Step 3: Build Temporal Plan Graph
    print("\nStep 3: Building Temporal Plan Graph...")
    start_time = time.time()
    tpg = step3_build_temporal_plan_graph(schedule, UAV_PARAMS)
    step3_time = time.time() - start_time
    print(f"Step 3 completed in {step3_time:.2f} seconds")

    # Step 4: Construct Simple Temporal Network with dynamic obstacles
    print("\nStep 4: Constructing Simple Temporal Network...")
    start_time = time.time()

    # Extract dynamic obstacles from the grid sequence
    # In GridMapEnv, dynamic obstacles are represented as values between 0.1-0.9
    dynamic_obstacles = {}

    # Create dynamic obstacle trajectories from the grid sequence
    if grid_seq is not None and len(grid_seq) > 1:
        # Identify dynamic obstacles by finding cells that change value between 0.1-0.9
        for t in range(len(grid_seq)):
            grid = grid_seq[t]

            # Find cells with values between 0.1 and 0.9 (dynamic obstacles)
            dynamic_mask = np.logical_and(grid > 0.05, grid < 0.95)
            obstacle_positions = np.where(dynamic_mask)

            for i in range(len(obstacle_positions[0])):
                x, y = obstacle_positions[0][i], obstacle_positions[1][i]
                obstacle_id = f"dyn_obs_{x}_{y}"

                if obstacle_id not in dynamic_obstacles:
                    dynamic_obstacles[obstacle_id] = []

                # Add position with timestamp - scaled to make obstacles move slower
                # Apply the radar zone speed factor to make radar zones move slower than agents
                dynamic_obstacles[obstacle_id].append({
                    'x': x,
                    'y': y,
                    't': float(t) * GRID_DT * RADAR_ZONE_SPEED_FACTOR  # Apply radar zone speed factor
                })

    # Filter out obstacles that don't move
    moving_obstacles = {}
    for obs_id, positions in dynamic_obstacles.items():
        if len(positions) > 1:
            # Check if the obstacle actually moves
            unique_positions = set((pos['x'], pos['y']) for pos in positions)
            if len(unique_positions) > 1:
                moving_obstacles[obs_id] = positions

    print(f"Identified {len(moving_obstacles)} moving dynamic obstacles")

    # Construct the Simple Temporal Network
    stn = step4_construct_simple_temporal_network(tpg, UAV_PARAMS, moving_obstacles)
    step4_time = time.time() - start_time
    print(f"Step 4 completed in {step4_time:.2f} seconds")

    # ------------------------------------------------------------------
    # Prevent the optimiser from squeezing the plan unrealistically
    # ------------------------------------------------------------------
    for e in stn.edges:
        lb, ub = e['bound']
        e['bound'][1] = max(ub, lb * 1.3)      # ≤ 30 % slack above min travel time

    # Step 5: Optimize temporal plan
    print("\nStep 5: Optimizing temporal plan...")
    start_time = time.time()
    solution = step5_optimize_temporal_plan(stn)
    step5_time = time.time() - start_time
    print(f"Step 5 completed in {step5_time:.2f} seconds")

    if solution:
        # Save the optimized schedule to a YAML file
        optimized_schedule_path = os.path.join(output_dir, "optimized_schedule.yaml")
        with open(optimized_schedule_path, 'w') as f:
            yaml.dump(solution, f, default_flow_style=False)

        # Step 6: Generate final trajectories with dynamic obstacle avoidance
        print("\nStep 6: Generating final trajectories...")
        start_time = time.time()
        final_trajectories = step6_generate_final_trajectories(solution, UAV_PARAMS, moving_obstacles, grid_seq)
        step6_time = time.time() - start_time
        print(f"Step 6 completed in {step6_time:.2f} seconds")

        # Create visualization directories
        grid_vis_dir = os.path.join(output_dir, "grid_visualization")
        continuous_vis_dir = os.path.join(output_dir, "continuous_visualization")
        os.makedirs(grid_vis_dir, exist_ok=True)
        os.makedirs(continuous_vis_dir, exist_ok=True)

        # Only create the final solution visualization to save memory
        print("\nCreating final solution visualization only...")
        vis_start_time = time.time()

        # Skip traditional grid-based visualization to save memory
        grid_vis_time = 0

        # Continuous space visualization (final solution only)
        print("\nCreating continuous space visualization...")
        cont_vis_time = 0
        try:
            # Import the continuous visualization module
            from continuous_visualization import visualize_continuous_space

            # Create continuous space visualization with optimized parameters
            cont_vis_start_time = time.time()

            # Optimize the grid sequence for visualization
            # Balance between memory usage and smooth radar zone movement
            if len(grid_seq) > 500:
                # For very large sequences, use every 3rd frame
                vis_grid_seq = grid_seq[::3]
                print(f"Optimized grid sequence for visualization: reduced from {len(grid_seq)} to {len(vis_grid_seq)} frames")
            elif len(grid_seq) > 100:
                # For medium sequences, use every 2nd frame
                vis_grid_seq = grid_seq[::2]
                print(f"Optimized grid sequence for visualization: reduced from {len(grid_seq)} to {len(vis_grid_seq)} frames")
            else:
                # For small sequences, use all frames for maximum smoothness
                vis_grid_seq = grid_seq
                print(f"Using all {len(grid_seq)} frames for visualization")

            # Force garbage collection before visualization
            import gc
            gc.collect()

            # Check if the map is too large for full visualization
            map_height, map_width = grid_seq[0].shape
            map_size = map_height * map_width

            # For very large maps, use a simplified visualization approach
            if map_size > 70000:  # For maps larger than 100x100
                print(f"Large map detected ({map_height}x{map_width}). Using simplified visualization to save memory.")

                # Create a simplified version of the visualization that uses less memory
                def simplified_visualize_continuous_space(trajectories, grid_seq, output_dir, uav_params):
                    """Simplified version of continuous space visualization that uses less memory"""
                    print("Using simplified continuous space visualization...")

                    # Create output directory
                    os.makedirs(output_dir, exist_ok=True)

                    # Create a single static visualization instead of an animation
                    plt.figure(figsize=(10, 10), dpi=100)

                    # Get the map dimensions
                    height, width = grid_seq[0].shape

                    # Draw static obstacles from the first frame
                    static_map = grid_seq[0].copy()
                    plt.imshow(static_map, cmap='gray_r', origin='upper', alpha=0.5)

                    # Define colors for different agents
                    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray']

                    # Plot each trajectory
                    for i, trajectory in enumerate(trajectories):
                        # Get color for this agent
                        agent_color = colors[i % len(colors)]

                        # Get waypoints
                        x = [wp.x for wp in trajectory.waypoints]
                        y = [wp.y for wp in trajectory.waypoints]

                        # Plot the trajectory
                        plt.plot(y, x, '-', color=agent_color, linewidth=2, alpha=0.8)

                        # Plot start and goal
                        plt.plot(y[0], x[0], 'o', color='green', markersize=10)
                        plt.plot(y[-1], x[-1], 'o', color='red', markersize=10)

                        # Add agent ID labels
                        plt.text(y[0], x[0], f"S{trajectory.agent_id}", color='white', fontsize=10,
                                ha='center', va='center', fontweight='bold',
                                bbox=dict(boxstyle="round", fc="green", alpha=0.7))
                        plt.text(y[-1], x[-1], f"G{trajectory.agent_id}", color='white', fontsize=10,
                                ha='center', va='center', fontweight='bold',
                                bbox=dict(boxstyle="round", fc="red", alpha=0.7))

                    # Add title and grid
                    plt.title("Continuous Space Trajectory Visualization", fontsize=16)
                    plt.grid(True, linestyle='--', alpha=0.3)

                    # Save the figure
                    plt.savefig(os.path.join(output_dir, "continuous_trajectories.png"), dpi=150)
                    plt.close()

                    print(f"Simplified visualization saved to {output_dir}")

                    # Skip animation for large maps
                    print("Animation skipped for large map to save memory")

                # Use the simplified visualization
                simplified_visualize_continuous_space(final_trajectories, vis_grid_seq, continuous_vis_dir, UAV_PARAMS)
            else:
                # For smaller maps, use the original visualization with optimized parameters
                print("Using the original continuous visualization with optimized parameters...")

                # Call the original visualization function with optimized grid sequence
                visualize_continuous_space(final_trajectories, vis_grid_seq, continuous_vis_dir, UAV_PARAMS)

            cont_vis_time = time.time() - cont_vis_start_time
            print(f"Continuous space visualization completed in {cont_vis_time:.2f} seconds")
            return

            # The following function is kept as a fallback but not used
            def unused_optimized_visualize_continuous_space(trajectories, grid_seq, output_dir, uav_params):
                """Performance-optimized version of continuous space visualization (unused)"""
                print("Using performance-optimized continuous space visualization...")

                # Create output directory
                os.makedirs(output_dir, exist_ok=True)

                # Determine the time range for visualization
                min_time = float('inf')
                max_time = 0

                for traj in trajectories:
                    if traj.times:
                        min_time = min(min_time, traj.times[0])
                        max_time = max(max_time, traj.times[-1])

                # Ensure we have a valid time range
                if min_time == float('inf') or max_time <= min_time:
                    min_time = 0
                    max_time = 100

                # Limit the number of frames to generate
                max_frames = 100  # Limit to 100 frames to prevent memory issues

                # Calculate time step based on max_frames
                time_step = (max_time - min_time) / max_frames

                # Generate frames at regular intervals
                frame_times = [min_time + i * time_step for i in range(max_frames)]

                # Create a figure for visualization
                plt.figure(figsize=(10, 10))

                # Generate frames
                for frame, current_time in enumerate(frame_times):
                    # Clear the figure for each frame
                    plt.clf()

                    # Create a new axis
                    ax = plt.gca()

                    # Get the grid dimensions
                    grid_height, grid_width = grid_seq[0].shape

                    # Set the axis limits
                    ax.set_xlim(0, grid_width)
                    ax.set_ylim(0, grid_height)

                    # Find the appropriate grid frame for this time
                    # Apply the same time scaling as used when creating dynamic obstacles
                    grid_frame = min(int(current_time / (GRID_DT * RADAR_ZONE_SPEED_FACTOR)), len(grid_seq) - 1)

                    # Display the grid as a background
                    grid = grid_seq[grid_frame]

                    # Create a custom colormap for the grid
                    # White for free space, black for static obstacles, green for dynamic obstacles
                    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green'])
                    bounds = [0, 0.05, 0.95, 1.0]
                    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

                    # Display the grid
                    ax.imshow(grid, cmap=cmap, norm=norm, origin='upper', alpha=0.5)

                    # Plot each trajectory
                    for i, traj in enumerate(trajectories):
                        # Skip if trajectory is empty
                        if not traj.waypoints or not traj.times:
                            continue

                        # Get agent color
                        color = plt.cm.tab10(i % 10)

                        # Find the position and velocity at the current time
                        pos_x, pos_y = None, None
                        vel_x, vel_y = 0, 0
                        speed = 0

                        # If current_time is before the trajectory starts, use the first position
                        if current_time <= traj.times[0]:
                            pos_x, pos_y = traj.waypoints[0].x, traj.waypoints[0].y
                            # Use initial velocity if available
                            if hasattr(traj, 'velocities') and traj.velocities:
                                vel_x, vel_y = traj.velocities[0][0], traj.velocities[0][1]
                                speed = (vel_x**2 + vel_y**2)**0.5

                        # If current_time is after the trajectory ends, use the last position
                        elif current_time >= traj.times[-1]:
                            pos_x, pos_y = traj.waypoints[-1].x, traj.waypoints[-1].y
                            # Use final velocity if available
                            if hasattr(traj, 'velocities') and traj.velocities:
                                vel_x, vel_y = traj.velocities[-1][0], traj.velocities[-1][1]
                                speed = (vel_x**2 + vel_y**2)**0.5

                        # Otherwise, interpolate between waypoints
                        else:
                            # Find the two waypoints that bracket the current time
                            for j in range(len(traj.times) - 1):
                                if traj.times[j] <= current_time <= traj.times[j+1]:
                                    # Interpolate between the two waypoints
                                    t1, t2 = traj.times[j], traj.times[j+1]
                                    x1, y1 = traj.waypoints[j].x, traj.waypoints[j].y
                                    x2, y2 = traj.waypoints[j+1].x, traj.waypoints[j+1].y

                                    # Linear interpolation
                                    alpha = (current_time - t1) / (t2 - t1) if t2 > t1 else 0
                                    pos_x = x1 + alpha * (x2 - x1)
                                    pos_y = y1 + alpha * (y2 - y1)

                                    # Calculate velocity
                                    if hasattr(traj, 'velocities') and len(traj.velocities) > j:
                                        vx1, vy1 = traj.velocities[j][0], traj.velocities[j][1]
                                        if len(traj.velocities) > j+1:
                                            vx2, vy2 = traj.velocities[j+1][0], traj.velocities[j+1][1]
                                            # Interpolate velocity
                                            vel_x = vx1 + alpha * (vx2 - vx1)
                                            vel_y = vy1 + alpha * (vy2 - vy1)
                                        else:
                                            vel_x, vel_y = vx1, vy1

                                        speed = (vel_x**2 + vel_y**2)**0.5
                                    else:
                                        # Calculate velocity from position change
                                        dt = t2 - t1
                                        if dt > 0:
                                            vel_x = (x2 - x1) / dt
                                            vel_y = (y2 - y1) / dt
                                            speed = ((x2 - x1)**2 + (y2 - y1)**2)**0.5 / dt
                                    break

                        # Plot the agent position and velocity
                        if pos_x is not None and pos_y is not None:
                            # Plot the agent as a circle with size based on speed
                            # Normalize speed to UAV parameters
                            norm_speed = min(1.0, speed / uav_params['v_max']) if uav_params['v_max'] > 0 else 0.5

                            # Base circle size
                            circle_size = 1.0

                            # Plot the agent as a circle
                            circle = plt.Circle((pos_y, pos_x), circle_size, color=color, alpha=0.8)
                            ax.add_patch(circle)

                            # Add speed indicator circles
                            if speed > 0.1:
                                # Add concentric circles to indicate speed
                                for r in [1.5, 2.0, 2.5]:
                                    if norm_speed >= r / 3.0:
                                        speed_circle = plt.Circle((pos_y, pos_x), circle_size * r,
                                                                color=color, alpha=0.2, fill=False)
                                        ax.add_patch(speed_circle)

                            # Add velocity vector
                            if speed > 0.1:
                                # Scale velocity vector for visualization
                                scale = 2.0
                                arrow_vel_x = vel_x * scale / speed
                                arrow_vel_y = vel_y * scale / speed

                                # Plot heading arrow with agent's color
                                ax.arrow(pos_y, pos_x, arrow_vel_y, arrow_vel_x,
                                        head_width=0.8, head_length=1.2,
                                        fc=color, ec=color, alpha=0.8)

                            # Add agent ID
                            ax.text(pos_y, pos_x, f"{traj.agent_id}", color='white',
                                    fontsize=8, ha='center', va='center')

                    # Set title and grid
                    ax.set_title(f"Time: {current_time:.2f}")
                    ax.grid(True)

                    # Save the frame
                    plt.savefig(os.path.join(output_dir, f"frame_{frame:03d}.png"))

                    # Print progress every 10 frames
                    if frame % 10 == 0:
                        print(f"Generated frame {frame}/{max_frames}")

                print(f"Visualization saved to {output_dir}")

                # Create a GIF animation from the frames
                try:
                    # Use imageio v2 to avoid deprecation warnings
                    import imageio.v2 as imageio

                    # Get all frame files
                    frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)
                                         if f.startswith('frame_') and f.endswith('.png')])

                    if frame_files:
                        # Create a GIF
                        gif_path = os.path.join(output_dir, "agent_movements.gif")

                        # Read images in batches to reduce memory usage
                        batch_size = 10
                        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
                            for i in range(0, len(frame_files), batch_size):
                                batch = frame_files[i:i+batch_size]
                                for f in batch:
                                    writer.append_data(imageio.imread(f))

                                # Force garbage collection after each batch
                                import gc
                                gc.collect()

                        print(f"Animated GIF created successfully: {gif_path}")
                except Exception as e:
                    print(f"Failed to create animation: {e}")

            # The visualization is already done above, this is just a placeholder
            # to maintain the code structure
            pass
        except Exception as e:
            print(f"Error creating continuous space visualization: {e}")
            print("Falling back to traditional visualization")

            # If continuous visualization fails, fall back to traditional visualization
            print("Creating traditional grid-based visualization as fallback...")
            vis_start_time = time.time()

            # Use a simplified version of the traditional visualization
            def simplified_visualize_trajectories(trajectories, grid_seq, output_dir):
                """Simplified version of traditional visualization to prevent memory issues"""
                print("Using simplified traditional visualization...")

                # Create output directory
                os.makedirs(output_dir, exist_ok=True)

                # Only create a single frame showing the final positions
                plt.figure(figsize=(10, 10))

                # Get the grid dimensions
                grid_height, grid_width = grid_seq[0].shape

                # Display the static grid
                plt.imshow(grid_seq[0], cmap='gray_r', origin='upper', alpha=0.5)

                # Plot each trajectory's final position
                for i, traj in enumerate(trajectories):
                    if not traj.waypoints:
                        continue

                    # Get agent color
                    color = plt.cm.tab10(i % 10)

                    # Get final position
                    final_pos = traj.waypoints[-1]

                    # Plot the final position
                    plt.scatter(final_pos.y, final_pos.x, color=color, s=100, marker='o')

                    # Add agent ID
                    plt.text(final_pos.y, final_pos.x, f"{traj.agent_id}", color='white',
                            fontsize=8, ha='center', va='center')

                # Set title and grid
                plt.title("Final Positions")
                plt.grid(True)

                # Save the image
                plt.savefig(os.path.join(output_dir, "final_positions.png"))
                print(f"Final positions saved to {output_dir}/final_positions.png")

            # Use the simplified traditional visualization
            simplified_visualize_trajectories(final_trajectories, grid_seq, grid_vis_dir)

            grid_vis_time = time.time() - vis_start_time
            print(f"Grid-based visualization completed in {grid_vis_time:.2f} seconds")

        # Write performance log
        with open(perf_log_path, 'a') as f:
            f.write(f"Step 1 (Generate initial paths): {step1_time:.2f} seconds\n")
            f.write(f"Step 2 (Convert to schedule): {step2_time:.2f} seconds\n")
            f.write(f"Step 3 (Build TPG): {step3_time:.2f} seconds\n")
            f.write(f"Step 4 (Construct STN): {step4_time:.2f} seconds\n")
            f.write(f"Step 5 (Optimize temporal plan): {step5_time:.2f} seconds\n")
            f.write(f"Step 6 (Generate final trajectories): {step6_time:.2f} seconds\n")
            f.write(f"Grid visualization: {grid_vis_time:.2f} seconds\n")
            f.write(f"Continuous visualization: {cont_vis_time:.2f} seconds\n\n")

            # Calculate total time and percentages
            total_time = step1_time + step2_time + step3_time + step4_time + step5_time + step6_time
            f.write(f"Total processing time (excluding visualization): {total_time:.2f} seconds\n\n")

            f.write("Percentage breakdown:\n")
            f.write(f"Step 1: {(step1_time/total_time)*100:.1f}%\n")
            f.write(f"Step 2: {(step2_time/total_time)*100:.1f}%\n")
            f.write(f"Step 3: {(step3_time/total_time)*100:.1f}%\n")
            f.write(f"Step 4: {(step4_time/total_time)*100:.1f}%\n")
            f.write(f"Step 5: {(step5_time/total_time)*100:.1f}%\n")
            f.write(f"Step 6: {(step6_time/total_time)*100:.1f}%\n")

            # Print the most computationally expensive step
            max_time = max(step1_time, step2_time, step3_time, step4_time, step5_time, step6_time)
            if max_time == step1_time:
                f.write("\nMost computationally expensive step: Step 1 (Generate initial paths)\n")
                f.write("This is likely due to the path planning algorithm (SIPPS) which has to search through the state space.\n")
            elif max_time == step2_time:
                f.write("\nMost computationally expensive step: Step 2 (Convert to schedule)\n")
            elif max_time == step3_time:
                f.write("\nMost computationally expensive step: Step 3 (Build TPG)\n")
            elif max_time == step4_time:
                f.write("\nMost computationally expensive step: Step 4 (Construct STN)\n")
            elif max_time == step5_time:
                f.write("\nMost computationally expensive step: Step 5 (Optimize temporal plan)\n")
                f.write("This is likely due to the optimization algorithm which has to solve a complex temporal constraint problem.\n")
            elif max_time == step6_time:
                f.write("\nMost computationally expensive step: Step 6 (Generate final trajectories)\n")
                f.write("This is likely due to the kinematic interpolation and collision checking with dynamic obstacles.\n")

        print(f"\nPerformance analysis saved to {perf_log_path}")
        return final_trajectories
    else:
        print("Failed to find a feasible solution")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate dynamically feasible trajectories for fixed-wing UAVs")
    parser.add_argument("--map", type=str, default="GridBasedPathPlanning/RL_Environment/data/cleaned_empty/empty-48-48-random-10_60_agents.png", help="Path to the static map")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--obstacles", type=int, default=5, help="Number of dynamic obstacles")
    parser.add_argument("--output", type=str, default="drone_solution", help="Output directory")

    args = parser.parse_args()

    # Process drone paths
    process_drone_paths(args.map, args.agents, args.obstacles, args.output)

if __name__ == "__main__":
    main()
