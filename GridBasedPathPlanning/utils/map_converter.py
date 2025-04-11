import numpy as np

def convert_grid_seq_for_mylns2(grid_seq):
    """
    Convert a grid sequence from GridBasedPathPlanning to a format suitable for mylns2.
    
    Args:
        grid_seq: 3D tensor [frames, height, width] from grid.GenerateGridSequence()
        
    Returns:
        3D tensor [frames, height, width] where:
        - 0 represents free cells
        - 1 represents static obstacles
        - 2 represents dynamic obstacles
    """
    frames, height, width = grid_seq.shape
    converted_seq = np.zeros((frames, height, width), dtype=np.int32)
    
    # Extract static obstacles from the first frame
    # In GridBasedPathPlanning, static obstacles have value 1
    static_obstacles = (grid_seq[0] == 1)
    
    for t in range(frames):
        # Start with free cells everywhere
        converted_seq[t] = 0
        
        # Mark static obstacles
        converted_seq[t][static_obstacles] = 1
        
        # Mark dynamic obstacles (any cell with value > 1 in grid_seq)
        # In GridBasedPathPlanning, dynamic obstacles have values >= 2
        dynamic_obstacles = (grid_seq[t] >= 2)
        converted_seq[t][dynamic_obstacles] = 2
    
    return converted_seq

def convert_mapf_map_for_gridenv(mapf_map):
    """
    Convert a map from MAPF environment (-1 for obstacles, 0 for free)
    to GridBasedPathPlanning format (0 for free, 1 for obstacles)
    
    Args:
        mapf_map: 2D array where -1 represents obstacles in mapf_gym
        
    Returns:
        2D array where 1 represents obstacles and 0 represents free cells
    """
    converted_map = np.zeros_like(mapf_map, dtype=np.int32)
    converted_map[mapf_map == -1] = 1
    return converted_map
