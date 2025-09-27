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
    start_frame: int = 0
    max_frames: int = -1 # -1 for all frames
    chunk_size: int = 30
    overlap_size: int = 2
    stride: int = 3
    
    # --- Camera Intrinsics ---
    resolution_set: int = 518
    org_image_size: Tuple[int, int] = (960, 540)
    base_intrinsics: Tuple[float, float, float, float] = (491.0157875755563, 491.0157875755563, 475.701166946744, 279.55005345005975)
    
    # --- Relocalization ---
    relocalization_db_path: str = None # Path to an existing slam_database.pt for relocalization
    create_debug_matches: bool = False # If True, saves images of the relocalization matches
    relocalization_threshold_percent: float = 10 # Threshold for the relocalization matches in percent
    max_relocalization_threshold: float = 0.23
    relocalization_frame_overlap: int = 10 # Number of frames to overlap at start and end of a localized sequence
    
    # --- Point Cloud Generation ---
    max_depth_for_point_cloud: float = 5.0 # Max depth in meters to include in the final PCL
    point_cloud_confidence_threshold: float = 0.5 # Confidence threshold for including points in the PCL

    # --- Debugging ---
    create_debug_chunk_visualizations: bool = False # If True, saves a PLY for each chunk's PCL and trajectory

    # --- Pre-processing ---
    crop_bottom_percent: float = 0.0 # Percentage of the image to crop from the bottom before processing
