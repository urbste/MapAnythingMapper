from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class SLAMConfig:
    """Configuration settings for the Visual SLAM system."""
    # --- Core Paths ---
    image_folder: str = ""
    video_path: str = ""
    telemetry_file: Optional[str] = None
    cam_dist_path: Optional[str] = None
    output_dir: str = None # If None, will be created relative to image_folder
    database_path: str = None # If None, will be created relative to image_folder
    
    # --- Sequence Processing ---
    start_time_seconds: float = 0.0 # Start time in seconds for video processing
    end_time_seconds: float = -1.0 # End time in seconds for video processing (-1 for entire video)
    chunk_size: int = 30
    overlap_size: int = 2
    stride: int = 3
    
    # --- Camera Intrinsics ---
    resolution_set: int = 518
    
    # --- Relocalization ---
    relocalization_db_path: str = None # Path to an existing slam_database.pt for relocalization
    create_debug_matches: bool = False # If True, saves images of the relocalization matches
    relocalization_threshold_percent: float = 10 # Threshold for the relocalization matches in percent
    max_relocalization_threshold: float = 0.23
    relocalization_frame_overlap: int = 4 # Number of frames to overlap at start and end of a localized sequence
    keypoint_matching_threshold: float = 0.2 # Confidence threshold for keypoint matches in relocalization
    min_keypoint_matches: int = 1000 # Minimum number of keypoint matches required for relocalization
    spatial_distance_threshold_m: float = 1.0 # Max spatial distance (meters) for accepting a reference pose as prior
    num_reference_priors: int = 3 # Number of consecutive reference frames to add as priors when within distance threshold
    
    # --- Point Cloud Generation ---
    max_depth_for_point_cloud: float = 5.0 # Max depth in meters to include in the final PCL
    point_cloud_confidence_threshold: float = 0.5 # Confidence threshold for including points in the PCL

    # --- Debugging ---
    create_debug_chunk_visualizations: bool = False # If True, saves a PLY for each chunk's PCL and trajectory
