#!/usr/bin/env python3
"""
Continuous Space Visualization for Dynamic Drone Solution

This module provides enhanced visualization capabilities for drone trajectories
in a continuous space, making it easier to understand kinematic feasibility.
Uses spline interpolation to create smooth, realistic drone trajectories.
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow, FancyArrowPatch
from matplotlib.collections import PatchCollection, LineCollection
import imageio.v2 as imageio
import subprocess
import shutil
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.distance import euclidean
import colorsys
import psutil  # For memory usage tracking

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

# Add the current directory to the path to ensure we can import from dynamic_drone_solution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try to import parameters from dynamic_drone_solution
try:
    from dynamic_drone_solution import GRID_DT, RADAR_ZONE_SPEED_FACTOR, UAV_PARAMS
    print(f"Successfully imported parameters from dynamic_drone_solution: GRID_DT={GRID_DT}, RADAR_ZONE_SPEED_FACTOR={RADAR_ZONE_SPEED_FACTOR}")
except ImportError as e:
    print(f"Warning: Could not import parameters from dynamic_drone_solution: {e}")
    # Default values if import fails
    GRID_DT = 4.0
    RADAR_ZONE_SPEED_FACTOR = 3.0

def create_spline_trajectory(trajectory, num_points=50):  # Reduced from 100 to 50 points
    """
    Create a smooth spline-based trajectory from waypoints

    Args:
        trajectory: Original trajectory with waypoints
        num_points: Number of points to generate along the spline (reduced for memory efficiency)

    Returns:
        spline_x: X coordinates along the spline
        spline_y: Y coordinates along the spline
        spline_t: Time points along the spline
        spline_v: Velocities along the spline
        spline_a: Accelerations along the spline
        spline_curvature: Curvature along the spline
    """
    # Force garbage collection before starting
    import gc
    gc.collect()
    # Extract waypoints and times
    x = np.array([wp.x for wp in trajectory.waypoints])
    y = np.array([wp.y for wp in trajectory.waypoints])
    t = np.array(trajectory.times)

    # Check for duplicate points and handle them
    if len(x) > 1:
        # Find duplicate points (where both x and y are the same)
        duplicates = []
        for i in range(1, len(x)):
            if abs(x[i] - x[i-1]) < 1e-6 and abs(y[i] - y[i-1]) < 1e-6:
                duplicates.append(i)

        if duplicates:
            print(f"Found {len(duplicates)} duplicate points in trajectory for agent {trajectory.agent_id}")

            # Filter out duplicates
            mask = np.ones(len(x), dtype=bool)
            for i in duplicates:
                mask[i] = False

            # Keep only non-duplicate points
            x = x[mask]
            y = y[mask]
            t = t[mask]

            print(f"After filtering duplicates: {len(x)} points remain")

            # If we have too few points after filtering, add some intermediate points
            if len(x) < 3 and len(x) > 1:
                print(f"Adding intermediate points to ensure smooth interpolation")
                # Add a midpoint between start and end
                mid_x = (x[0] + x[-1]) / 2.0
                mid_y = (y[0] + y[-1]) / 2.0
                mid_t = (t[0] + t[-1]) / 2.0

                # Insert the midpoint
                x = np.array([x[0], mid_x, x[-1]])
                y = np.array([y[0], mid_y, y[-1]])
                t = np.array([t[0], mid_t, t[-1]])

    # Ensure times are strictly increasing
    if len(t) > 1:
        for i in range(1, len(t)):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + 0.01  # Add a small increment to ensure strict monotonicity

    if len(x) < 4:
        # For trajectories with few points, use linear interpolation with more points
        # to ensure we have enough for a smooth curve
        t_new = np.linspace(t[0], t[-1], num_points)
        x_interp = interp1d(t, x, kind='linear' if len(x) > 1 else 'nearest')
        y_interp = interp1d(t, y, kind='linear' if len(y) > 1 else 'nearest')

        spline_x = x_interp(t_new)
        spline_y = y_interp(t_new)
        spline_t = t_new

        # Calculate velocities (first derivative)
        if len(x) > 1:
            dx_dt = np.gradient(spline_x, spline_t)
            dy_dt = np.gradient(spline_y, spline_t)
            spline_v = np.sqrt(dx_dt**2 + dy_dt**2)

            # Calculate accelerations (second derivative)
            d2x_dt2 = np.gradient(dx_dt, spline_t)
            d2y_dt2 = np.gradient(dy_dt, spline_t)
            spline_a = np.sqrt(d2x_dt2**2 + d2y_dt2**2)

            # Calculate curvature
            # κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
            denominator = (dx_dt**2 + dy_dt**2)**(3/2)
            # Avoid division by zero
            denominator = np.where(denominator < 1e-10, 1e-10, denominator)
            spline_curvature = numerator / denominator
        else:
            spline_v = np.zeros_like(spline_t)
            spline_a = np.zeros_like(spline_t)
            spline_curvature = np.zeros_like(spline_t)
    else:
        # For trajectories with enough points, use cubic spline interpolation
        # Create a parameter along the curve based on cumulative distance
        # This helps create more natural splines than using time directly
        dist = np.zeros(len(x))

        # Check for very small segments and add a minimum distance
        has_small_segments = False
        for i in range(1, len(x)):
            segment_length = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
            if segment_length < 1e-6:
                has_small_segments = True
                # Add a tiny offset to avoid zero-length segments
                segment_length = 1e-6
            dist[i] = dist[i-1] + segment_length

        # Normalize the distance parameter
        if dist[-1] > 0:
            u = dist / dist[-1]
        else:
            # If total distance is zero, use evenly spaced points
            u = np.linspace(0, 1, len(x))

        # Ensure u is strictly increasing
        for i in range(1, len(u)):
            if u[i] <= u[i-1]:
                u[i] = u[i-1] + 1e-6

        # Create a new parameter for interpolation
        u_new = np.linspace(0, 1, num_points)

        try:
            # Use cubic spline interpolation
            cs_x = CubicSpline(u, x)
            cs_y = CubicSpline(u, y)
            cs_t = CubicSpline(u, t)
        except Exception as e:
            print(f"Error creating splines: {e}")
            print(f"Falling back to linear interpolation")

            # Fall back to linear interpolation
            cs_x = interp1d(u, x, kind='linear')
            cs_y = interp1d(u, y, kind='linear')
            cs_t = interp1d(u, t, kind='linear')

        # Evaluate the splines
        spline_x = cs_x(u_new)
        spline_y = cs_y(u_new)
        spline_t = cs_t(u_new)

        # Calculate derivatives based on the type of interpolation
        if hasattr(cs_x, 'derivative'):
            # For CubicSpline objects
            # Calculate first derivatives
            dx_du = cs_x.derivative()(u_new)
            dy_du = cs_y.derivative()(u_new)
            dt_du = cs_t.derivative()(u_new)

            # Calculate second derivatives
            d2x_du2 = cs_x.derivative(2)(u_new)
            d2y_du2 = cs_y.derivative(2)(u_new)
        else:
            # For interp1d objects, use finite differences
            # Calculate first derivatives
            du = u_new[1] - u_new[0] if len(u_new) > 1 else 0.01
            dx_du = np.gradient(spline_x, du)
            dy_du = np.gradient(spline_y, du)
            dt_du = np.gradient(spline_t, du)

            # Calculate second derivatives
            d2x_du2 = np.gradient(dx_du, du)
            d2y_du2 = np.gradient(dy_du, du)

        # Calculate velocities in terms of time
        # v = ds/dt = (ds/du) / (dt/du)
        ds_du = np.sqrt(dx_du**2 + dy_du**2)
        dt_du = np.maximum(dt_du, 1e-10)  # Avoid division by zero
        spline_v = ds_du / dt_du

        # Calculate accelerations
        # a = d²s/dt² = d/dt(ds/dt) = d/du(ds/dt) * du/dt = d/du(ds/du / dt/du) * du/dt
        # Simplified approximation:
        spline_a = np.sqrt(d2x_du2**2 + d2y_du2**2) / dt_du**2

        # Calculate curvature
        # κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx_du * d2y_du2 - dy_du * d2x_du2)
        denominator = (dx_du**2 + dy_du**2)**(3/2)
        # Avoid division by zero
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)
        spline_curvature = numerator / denominator

    return spline_x, spline_y, spline_t, spline_v, spline_a, spline_curvature


def visualize_continuous_space(trajectories, grid_seq, output_dir, uav_params):
    """
    Visualize trajectories in a continuous space with velocity-coded paths
    using spline interpolation for smooth, realistic drone trajectories

    Args:
        trajectories: List of trajectories
        grid_seq: Grid sequence with dynamic obstacles
        output_dir: Directory to save the visualization
        uav_params: UAV dynamics parameters
    """
    # Check if the map is too large for full visualization
    map_height, map_width = grid_seq[0].shape
    map_size = map_height * map_width

    # For extremely large maps, use a more efficient approach
    if map_size > 50000:  # For maps larger than ~220x220
        print(f"Map size ({map_height}x{map_width}) is extremely large. Using optimized visualization for large maps.")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Calculate downsample factor based on map size
        downsample_factor = max(1, int(max(map_height, map_width) / 200))
        print(f"Downsampling map by factor of {downsample_factor} for visualization")

        # Create a downsampled version of the grid sequence
        downsampled_grid_seq = []
        for grid in grid_seq[::3]:  # Use every 3rd frame to save memory
            downsampled = np.zeros((map_height // downsample_factor, map_width // downsample_factor), dtype=np.float32)
            for i in range(map_height // downsample_factor):
                for j in range(map_width // downsample_factor):
                    # Take the maximum value in the block to ensure obstacles are preserved
                    block = grid[i*downsample_factor:(i+1)*downsample_factor,
                                j*downsample_factor:(j+1)*downsample_factor]
                    downsampled[i, j] = np.max(block)
            downsampled_grid_seq.append(downsampled)

        print(f"Downsampled grid sequence from {len(grid_seq)} to {len(downsampled_grid_seq)} frames")

        # Create spline trajectories for each agent
        spline_trajectories = []
        for trajectory in trajectories:
            # Create a smooth spline trajectory with fewer points to save memory
            spline_x, spline_y, spline_t, spline_v, _, _ = create_spline_trajectory(
                trajectory, num_points=50)  # Use fewer points for large maps

            spline_trajectories.append({
                'agent_id': trajectory.agent_id,
                'x': spline_x,
                'y': spline_y,
                't': spline_t,
                'v': spline_v
            })

        # Determine the maximum time across all trajectories
        max_time = 0
        for traj in spline_trajectories:
            if len(traj['t']) > 0:
                max_time = max(max_time, traj['t'][-1])

        # Create a static visualization first
        plt.figure(figsize=(10, 10), dpi=80)

        # Draw the downsampled static map
        plt.imshow(downsampled_grid_seq[0], cmap='gray_r', origin='upper', alpha=0.5)

        # Define colors for different agents
        colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray']

        # Plot each trajectory
        for i, traj in enumerate(spline_trajectories):
            # Get color for this agent
            agent_color = colors[i % len(colors)]

            # Plot the trajectory
            plt.plot(traj['y'] / downsample_factor, traj['x'] / downsample_factor, '-',
                    color=agent_color, linewidth=2, alpha=0.8)

            # Plot start and end points
            start_x, start_y = traj['x'][0] / downsample_factor, traj['y'][0] / downsample_factor
            end_x, end_y = traj['x'][-1] / downsample_factor, traj['y'][-1] / downsample_factor

            plt.plot(start_y, start_x, 'o', color='green', markersize=8)
            plt.plot(end_y, end_x, 'o', color='red', markersize=8)

            # Add agent ID
            plt.text(start_y, start_x, f"S{traj['agent_id']}", color='white', fontsize=8,
                    ha='center', va='center', fontweight='bold')
            plt.text(end_y, end_x, f"G{traj['agent_id']}", color='white', fontsize=8,
                    ha='center', va='center', fontweight='bold')

        plt.title("Large Map Trajectory Visualization")
        plt.savefig(os.path.join(output_dir, "large_map_visualization.png"))
        plt.close()

        # Create a simplified animation with fewer frames
        num_frames = min(30, len(downsampled_grid_seq))  # Limit to 30 frames
        frame_times = np.linspace(0, max_time, num_frames)

        print(f"Creating simplified animation with {num_frames} frames...")

        # Generate frames
        for frame, current_time in enumerate(frame_times):
            # Create a figure
            plt.figure(figsize=(10, 10), dpi=80)

            # Get the appropriate grid frame for this time
            grid_frame = min(int(current_time / (GRID_DT * RADAR_ZONE_SPEED_FACTOR)), len(downsampled_grid_seq) - 1)

            # Draw the grid
            plt.imshow(downsampled_grid_seq[grid_frame], cmap='gray_r', origin='upper', alpha=0.5)

            # Plot each trajectory up to the current time
            for i, traj in enumerate(spline_trajectories):
                # Get color for this agent
                agent_color = colors[i % len(colors)]

                # Find points up to current time
                mask = traj['t'] <= current_time
                if np.any(mask):
                    x_path = traj['x'][mask] / downsample_factor
                    y_path = traj['y'][mask] / downsample_factor

                    # Plot the path
                    plt.plot(y_path, x_path, '-', color=agent_color, linewidth=2, alpha=0.8)

                    # Plot the current position
                    current_x = x_path[-1]
                    current_y = y_path[-1]
                    plt.plot(current_y, current_x, 'o', color=agent_color, markersize=8)

                    # Add agent ID
                    plt.text(current_y, current_x, f"{traj['agent_id']}", color='white', fontsize=8,
                            ha='center', va='center', fontweight='bold')

            # Add title with current time
            plt.title(f"Time: {current_time:.2f}")

            # Save the frame
            plt.savefig(os.path.join(frames_dir, f"frame_{frame:03d}.png"))
            plt.close()

            # Force garbage collection
            if frame % 5 == 0:
                import gc
                gc.collect()
                print(f"Frame {frame}/{num_frames} - Memory usage: {get_memory_usage():.1f} MB")

        # Create a GIF animation
        try:
            # Get all frame files
            frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                                if f.startswith('frame_') and f.endswith('.png')])

            if frame_files:
                # Create a GIF
                gif_path = os.path.join(output_dir, "large_map_animation.gif")

                # Process images in batches to reduce memory usage
                batch_size = 5

                # Create a GIF writer
                with imageio.get_writer(gif_path, mode='I', duration=1.0) as writer:  # 5 frames per second
                    for i in range(0, len(frame_files), batch_size):
                        # Process a batch of frames
                        batch = frame_files[i:i+batch_size]

                        # Read and write each image in the batch
                        for f in batch:
                            writer.append_data(imageio.imread(f))

                        # Force garbage collection after each batch
                        import gc
                        gc.collect()

                print(f"Large map animation created: {gif_path}")
            else:
                print("No frames found to create animation")
        except Exception as e:
            print(f"Failed to create animation: {e}")

        return
    try:
        print("Visualizing trajectories in continuous space...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get the map dimensions
        height, width = grid_seq[0].shape

        # Create a figure with white background for continuous space visualization
        # Use even dimensions for video compatibility
        # Reduced figure size from 12x12 to 10x10 to save memory
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white', dpi=100)  # Lower DPI to save memory
        ax.set_facecolor('white')

        # Force garbage collection
        import gc
        gc.collect()

        # Set bounds slightly larger than the map
        ax.set_xlim(-1, width + 1)
        ax.set_ylim(-1, height + 1)

        # Draw a light grid for reference
        ax.grid(True, linestyle='--', alpha=0.3)

        # Draw static obstacles from the first frame
        static_map = grid_seq[0].copy()
        for y in range(height):
            for x in range(width):
                if static_map[y, x] > 0.95:  # Static obstacle
                    rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black', alpha=0.7)
                    ax.add_patch(rect)

        # Define colors for different agents
        colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray']

        # Create a colormap for velocity visualization
        velocity_cmap = plt.cm.viridis

        # Plot the trajectories with spline interpolation and velocity-coded colors
        for i, trajectory in enumerate(trajectories):
            # Get color for this agent
            agent_color = colors[i % len(colors)]

            # Create a smooth spline trajectory with fewer points to save memory
            spline_x, spline_y, spline_t, spline_v, spline_a, spline_curvature = create_spline_trajectory(
                trajectory, num_points=100)  # Reduced from 200 to 100 points to save memory

            # Get the original waypoints for reference
            orig_x = [wp.x for wp in trajectory.waypoints]
            orig_y = [wp.y for wp in trajectory.waypoints]

            # Normalize velocities for colormapping
            if np.max(spline_v) > np.min(spline_v):
                norm_v = (spline_v - np.min(spline_v)) / (np.max(spline_v) - np.min(spline_v))
            else:
                norm_v = np.ones_like(spline_v) * 0.5

            # Create a line collection for the spline with color gradient based on velocity
            points = np.array([spline_y, spline_x]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a color array for the segments
            colors_array = np.array([velocity_cmap(v) for v in norm_v[:-1]])

            # Create the line collection
            lc = LineCollection(segments, colors=colors_array, linewidth=4, alpha=0.8)
            line = ax.add_collection(lc)

            # Plot the start (green) and goal (red) positions with black edge for visibility
            ax.plot(orig_y[0], orig_x[0], 'o', color='green', markersize=10, markeredgecolor='black')
            ax.plot(orig_y[-1], orig_x[-1], 'o', color='red', markersize=10, markeredgecolor='black')

            # Add agent ID labels for better start-goal pair clarification
            ax.text(orig_y[0], orig_x[0], f"S{trajectory.agent_id}", color='white', fontsize=10,
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round", fc="green", alpha=0.7))
            ax.text(orig_y[-1], orig_x[-1], f"G{trajectory.agent_id}", color='white', fontsize=10,
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round", fc="red", alpha=0.7))

            # Visualize turning constraints at key points where curvature is high
            # Find points with high curvature (turns)
            high_curvature_indices = np.where(spline_curvature > np.percentile(spline_curvature, 75))[0]

            # Sample a subset of these points to avoid cluttering
            sample_indices = high_curvature_indices[::max(1, len(high_curvature_indices) // 5)]

            for idx in sample_indices:
                if idx < len(spline_v):
                    v_mag = spline_v[idx]

                    # Calculate minimum turning radius based on velocity and max turn rate
                    min_radius = v_mag / uav_params['omega_max'] if uav_params['omega_max'] > 0 and v_mag > 0 else 0

                    # Draw a circle representing the minimum turning radius
                    if min_radius > 0.5:  # Only draw if radius is significant
                        circle = plt.Circle((spline_y[idx], spline_x[idx]), min_radius, fill=False,
                                          color=agent_color, linestyle='--', alpha=0.3)
                        ax.add_patch(circle)

                        # Add a small marker to show the point
                        ax.plot(spline_y[idx], spline_x[idx], 'o', color=agent_color, markersize=4)

            # Visualize acceleration/deceleration with arrows
            # Sample points along the trajectory
            sample_indices = np.linspace(0, len(spline_t)-1, 10, dtype=int)

            for idx in sample_indices:
                if idx > 0 and idx < len(spline_t)-1:
                    # Calculate direction vector
                    dx = spline_x[idx+1] - spline_x[idx-1]
                    dy = spline_y[idx+1] - spline_y[idx-1]

                    # Normalize
                    mag = np.sqrt(dx**2 + dy**2)
                    if mag > 0.001:
                        dx /= mag
                        dy /= mag

                        # Scale arrow by velocity
                        arrow_scale = 0.5 + 1.5 * norm_v[idx]  # Scale between 0.5 and 2.0

                        # Create an arrow
                        arrow = FancyArrowPatch(
                            (spline_y[idx], spline_x[idx]),
                            (spline_y[idx] + dy * arrow_scale, spline_x[idx] + dx * arrow_scale),
                            arrowstyle='-|>', color=agent_color, alpha=0.7,
                            mutation_scale=10 + 10 * norm_v[idx]  # Scale arrowhead size by velocity
                        )
                        ax.add_patch(arrow)

        # Add a colorbar for velocity
        sm = plt.cm.ScalarMappable(cmap=velocity_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Velocity (units/time)')

        # Add legend for agents
        agent_patches = []
        for i, trajectory in enumerate(trajectories):
            color = colors[i % len(colors)]
            agent_patches.append(plt.Line2D([0], [0], color=color, lw=2, label=f"Agent {trajectory.agent_id}"))

        # Add legend for start/goal
        start_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                                markeredgecolor='black', markersize=10, label='Start')
        goal_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                               markeredgecolor='black', markersize=10, label='Goal')

        # Combine all legend elements
        legend_elements = agent_patches + [start_patch, goal_patch]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.title("Continuous Space Trajectory Visualization", fontsize=16)

        # Save the figure
        plt.savefig(os.path.join(output_dir, "continuous_trajectories.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create an animated visualization
        create_continuous_animation(trajectories, grid_seq, output_dir, uav_params)

        print(f"Continuous space visualization saved to {output_dir}")
    except Exception as e:
        print(f"Error creating continuous space visualization: {e}")
        print("Falling back to traditional visualization only")

def create_continuous_animation(trajectories, grid_seq, output_dir, uav_params):
    """
    Create an animated visualization of trajectories in continuous space
    using spline interpolation for smooth, realistic drone trajectories

    Args:
        trajectories: List of trajectories
        grid_seq: Grid sequence with dynamic obstacles
        output_dir: Directory to save the visualization
        uav_params: UAV dynamics parameters
    """
    # Check if the map is too large for animation
    map_height, map_width = grid_seq[0].shape
    map_size = map_height * map_width

    if map_size > 20000:  # For maps larger than ~140x140
        print(f"Map size ({map_height}x{map_width}) is too large for animation. Skipping animation to save memory.")
        return
    try:
        print("Creating continuous space animation...")

        # Get the map dimensions
        height, width = grid_seq[0].shape

        # Determine the maximum time across all trajectories
        max_time = 0
        for trajectory in trajectories:
            if trajectory.times:
                max_time = max(max_time, trajectory.times[-1])

        # Number of frames in the animation - increased for smoother visualization
        num_frames = 120  # Increased from 60 to 120 for smoother animation (4 seconds at 30fps)
        time_step = max_time / num_frames if max_time > 0 else 1.0

        # Print memory usage information
        print(f"Creating animation with {num_frames} frames...")
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"Current memory usage: {memory_info.rss / (1024 * 1024):.1f} MB")
        except ImportError:
            print("psutil not available, skipping memory usage reporting")

        # Force garbage collection before starting animation
        import gc
        gc.collect()

        # Define colors for different agents
        colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray']

        # Create a colormap for velocity visualization
        velocity_cmap = plt.cm.viridis

        # Create frames directory
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Precompute spline trajectories for all agents
        spline_trajectories = []
        for trajectory in trajectories:
            # Create a smooth spline trajectory
            spline_x, spline_y, spline_t, spline_v, spline_a, spline_curvature = create_spline_trajectory(
                trajectory, num_points=200)  # More points for smoother curves

            spline_trajectories.append({
                'x': spline_x,
                'y': spline_y,
                't': spline_t,
                'v': spline_v,
                'a': spline_a,
                'curvature': spline_curvature,
                'agent_id': trajectory.agent_id
            })

        for frame in range(num_frames):
            # Current time
            current_time = frame * time_step

            # Create a figure with white background - use fixed dimensions for video compatibility
            # Reduced figure size and DPI to save memory
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='white', dpi=80)
            ax.set_facecolor('white')

            # Report memory usage periodically
            if frame % 10 == 0:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    print(f"Frame {frame}/{num_frames} - Memory usage: {memory_info.rss / (1024 * 1024):.1f} MB")
                except ImportError:
                    print(f"Frame {frame}/{num_frames}")

            # Set bounds slightly larger than the map
            ax.set_xlim(-1, width + 1)
            ax.set_ylim(-1, height + 1)

            # Draw a light grid for reference
            ax.grid(True, linestyle='--', alpha=0.3)

            # Draw static obstacles
            for y in range(height):
                for x in range(width):
                    if grid_seq[0][y, x] > 0.95:  # Static obstacle
                        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black', alpha=0.7)
                        ax.add_patch(rect)

            # Draw dynamic obstacles at current time
            if len(grid_seq) > 1:
                # Map the current animation time to the grid sequence time
                # Use the parameters imported at the module level
                time_scaling_factor = GRID_DT * RADAR_ZONE_SPEED_FACTOR  # Use the same scaling as in the simulation

                # Only print this once at the beginning of the animation
                if frame == 0:
                    print(f"Using time_scaling_factor={time_scaling_factor} for animation")
                    print(f"Grid sequence length: {len(grid_seq)}, max_time: {max_time}")

                # Calculate the frame index in the grid sequence that corresponds to the current animation time
                # This is the critical formula that ensures synchronization between agent and obstacle timing
                # The current_time is in agent time units, we need to convert it to grid sequence time units
                # Since dynamic obstacles are created with t = grid_idx * GRID_DT * RADAR_ZONE_SPEED_FACTOR,
                # we need to divide current_time by (GRID_DT * RADAR_ZONE_SPEED_FACTOR) to get the grid index
                frame_idx = min(current_time / time_scaling_factor, len(grid_seq) - 1)

                # Get the appropriate frame
                frame_idx_floor = int(frame_idx)
                frame_idx_ceil = min(frame_idx_floor + 1, len(grid_seq) - 1)
                alpha = frame_idx - frame_idx_floor

                # Get the current grid state
                if frame_idx_floor == frame_idx_ceil or alpha < 0.001:
                    current_grid = grid_seq[frame_idx_floor]
                else:
                    # Interpolate between frames
                    current_grid = grid_seq[frame_idx_floor] * (1 - alpha) + grid_seq[frame_idx_ceil] * alpha

                # Draw dynamic obstacles
                for y in range(height):
                    for x in range(width):
                        if 0.05 < current_grid[y, x] < 0.95:  # Dynamic obstacle
                            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='lightgreen', alpha=0.7)
                            ax.add_patch(rect)

            # Plot the trajectories up to the current time using splines
            for i, spline_traj in enumerate(spline_trajectories):
                # Get color for this agent
                agent_color = colors[i % len(colors)]

                # Get the original trajectory for reference points
                trajectory = trajectories[i]
                orig_x = [wp.x for wp in trajectory.waypoints]
                orig_y = [wp.y for wp in trajectory.waypoints]

                # Always show start and goal positions
                ax.plot(orig_y[0], orig_x[0], 'o', color='green', markersize=10, markeredgecolor='black')
                ax.plot(orig_y[-1], orig_x[-1], 'o', color='red', markersize=10, markeredgecolor='black')

                # Add agent ID labels
                ax.text(orig_y[0], orig_x[0], f"S{spline_traj['agent_id']}", color='white', fontsize=10,
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle="round", fc="green", alpha=0.7))
                ax.text(orig_y[-1], orig_x[-1], f"G{spline_traj['agent_id']}", color='white', fontsize=10,
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle="round", fc="red", alpha=0.7))

                # Get spline data
                spline_x = spline_traj['x']
                spline_y = spline_traj['y']
                spline_t = spline_traj['t']
                spline_v = spline_traj['v']

                # Find points up to current time
                mask = spline_t <= current_time
                if np.any(mask):
                    x_path = spline_x[mask]
                    y_path = spline_y[mask]
                    v_path = spline_v[mask]
                    t_path = spline_t[mask]

                    # Normalize velocities for colormapping
                    if len(v_path) > 0 and np.max(v_path) > np.min(v_path):
                        norm_v = (v_path - np.min(v_path)) / (np.max(v_path) - np.min(v_path))
                    else:
                        norm_v = np.ones_like(v_path) * 0.5

                    # Create a line collection for the spline with color gradient based on velocity
                    if len(x_path) >= 2:
                        points = np.array([y_path, x_path]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)

                        # Create a color array for the segments
                        colors_array = np.array([velocity_cmap(v) for v in norm_v[:-1]])

                        # Create the line collection
                        lc = LineCollection(segments, colors=colors_array, linewidth=4, alpha=0.8)
                        line = ax.add_collection(lc)

                    # Find the current position (last point in the path)
                    if len(x_path) > 0:
                        current_x = x_path[-1]
                        current_y = y_path[-1]

                        # Plot the current position
                        ax.plot(current_y, current_x, 'o', color=agent_color, markersize=10, markeredgecolor='black')

                        # Add orientation indicator (heading arrow)
                        # Find the direction at the current point
                        if len(x_path) >= 2:
                            # Use the last two points to determine direction
                            dx = x_path[-1] - x_path[-2]
                            dy = y_path[-1] - y_path[-2]

                            # Normalize to get direction
                            vel_mag = np.sqrt(dx**2 + dy**2)
                            if vel_mag > 0.001:  # Avoid division by zero
                                dx /= vel_mag
                                dy /= vel_mag

                                # Scale arrow by velocity
                                v_idx = len(norm_v) - 1
                                arrow_scale = 0.5 + 2.5 * norm_v[v_idx] if v_idx >= 0 else 1.5

                                # Create an arrow
                                arrow = FancyArrowPatch(
                                    (current_y, current_x),
                                    (current_y + dy * arrow_scale, current_x + dx * arrow_scale),
                                    arrowstyle='-|>', color=agent_color, alpha=0.7,
                                    mutation_scale=10 + 10 * norm_v[v_idx] if v_idx >= 0 else 15
                                )
                                ax.add_patch(arrow)

                                # Calculate minimum turning radius based on current velocity
                                v_current = v_path[-1] if len(v_path) > 0 else 0
                                min_radius = v_current / uav_params['omega_max'] if uav_params['omega_max'] > 0 and v_current > 0 else 0

                                # Draw turning radius circle if significant
                                if min_radius > 0.5:
                                    circle = plt.Circle((current_y, current_x), min_radius, fill=False,
                                                      color=agent_color, linestyle='--', alpha=0.3)
                                    ax.add_patch(circle)

            # Add a colorbar for velocity
            sm = plt.cm.ScalarMappable(cmap=velocity_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Velocity (units/time)')

            # Add title with current time
            ax.set_title(f"Time: {current_time:.2f} / {max_time:.2f}", fontsize=16)

            # Save the frame with fixed dimensions for video compatibility
            # Don't use tight_layout() as it can create variable-sized images
            # Instead, use a fixed figure size and fixed margins
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

            # Save with fixed dimensions to ensure compatibility with video creation
            # Use lower DPI to save memory
            plt.savefig(os.path.join(frames_dir, f"frame_{frame:03d}.png"),
                       dpi=80,  # Reduced from 150 to 80
                       bbox_inches=None,  # Don't use bbox_inches='tight'
                       pad_inches=0.1)

            # Close the figure and force garbage collection
            plt.close(fig)
            fig = None
            ax = None

            # Force garbage collection every few frames
            if frame % 5 == 0:
                import gc
                gc.collect()

        # Create a GIF animation from the frames
        try:
            # Get all frame files
            frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])

            if frame_files:
                # Create a GIF
                gif_path = os.path.join(output_dir, "continuous_animation.gif")

                # Process images in batches to reduce memory usage
                batch_size = 10

                # Create a GIF writer
                with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:  # 10 frames per second
                    for i in range(0, len(frame_files), batch_size):
                        # Process a batch of frames
                        batch = frame_files[i:i+batch_size]

                        # Read and write each image in the batch
                        for f in batch:
                            writer.append_data(imageio.imread(f))

                        # Force garbage collection after each batch
                        import gc
                        gc.collect()

                        # Print progress
                        print(f"Processed frames {i} to {min(i+batch_size, len(frame_files))} of {len(frame_files)}")

                print(f"Continuous space animation created: {gif_path}")

                # Also create an MP4 video if ffmpeg is available
                try:
                    # Check if ffmpeg is available
                    ffmpeg_available = shutil.which("ffmpeg") is not None

                    if ffmpeg_available:
                        # Create a temporary directory for frames with sequential numbering
                        temp_dir = os.path.join(output_dir, "temp_frames")
                        os.makedirs(temp_dir, exist_ok=True)

                        # Copy frames to temp directory with sequential numbering
                        # Process in batches to reduce memory usage
                        for i, frame_file in enumerate(frame_files):
                            # Create a symbolic link instead of copying to save memory
                            temp_frame = os.path.join(temp_dir, f"frame_{i:05d}.png")
                            try:
                                # Try to create a symbolic link first (more efficient)
                                os.symlink(os.path.abspath(frame_file), temp_frame)
                            except (OSError, AttributeError):
                                # If symlink fails, fall back to copying
                                shutil.copy(frame_file, temp_frame)

                            # Force garbage collection periodically
                            if i % 20 == 0:
                                gc.collect()

                        # Use ffmpeg to create an MP4 video
                        video_path = os.path.join(output_dir, "continuous_animation.mp4")
                        ffmpeg_cmd = [
                            "ffmpeg", "-y",  # Overwrite output file if it exists
                            "-framerate", "30",  # 30 fps for smooth video
                            "-i", os.path.join(temp_dir, "frame_%05d.png"),
                            "-c:v", "libx264",
                            "-profile:v", "high",
                            "-crf", "20",  # Quality (lower is better, 18-28 is good)
                            "-pix_fmt", "yuv420p",
                            # Ensure dimensions are even (required for h264)
                            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1:1",
                            video_path
                        ]

                        # Run ffmpeg
                        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        if result.returncode == 0:
                            print(f"Continuous space MP4 video created: {video_path}")
                        else:
                            print(f"Failed to create MP4 video. ffmpeg error: {result.stderr.decode('utf-8')}")

                        # Clean up temporary files
                        for file in os.listdir(temp_dir):
                            os.remove(os.path.join(temp_dir, file))
                        os.rmdir(temp_dir)
                    else:
                        print("ffmpeg not found. Skipping MP4 video creation.")
                except Exception as e:
                    print(f"Failed to create MP4 video: {e}")
            else:
                print("No frames found to create animation")
        except Exception as e:
            print(f"Failed to create animation: {e}")
    except Exception as e:
        print(f"Error creating continuous animation: {e}")
        print("Animation creation failed, but static visualization may still be available")
