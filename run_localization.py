import os
import argparse
from slam_config import SLAMConfig
from visual_slam import VisualSLAM
from point_cloud_processor import PointCloudProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main(args):
    # --- 1. Configure and run SLAM for localization ---
    config = SLAMConfig(
        video_path=args.video_path,
        telemetry_file=args.telemetry_file,
        cam_dist_path=args.cam_dist_path,
        output_dir=args.output_dir,
        database_path=args.database_path,
        relocalization_db_path=args.relocalization_db_path,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        stride=args.stride,
        create_debug_chunk_visualizations=args.debug_viz,
        point_cloud_confidence_threshold=args.pcl_confidence,
    )

    print("--- Running Localization on New Sequence ---")
    slam_system = VisualSLAM(config)
    slam_system.extend_map_with_sequence()
    
    # --- 2. Post-processing and Visualization ---
    if args.visualize:
        print("\n\n--- Visualizing Combined Results ---")
        
        processor = PointCloudProcessor()
        
        # --- Process and visualize the original reference map ---
        pcd_ref, traj_ref = processor.generate_pcd_and_trajectory(
            chunk_dirs=[args.ref_chunk_dir], 
            trajectory_color=[1, 0, 0], # Red for reference
            config=config # Config is mainly for PCL settings, can be reused
        )
        
        # --- Process and visualize the new localized map ---
        pcd_loc, traj_loc = processor.generate_pcd_and_trajectory(
            chunk_dirs=[config.output_dir], 
            trajectory_color=[0, 0, 1], # Blue for localized
            config=config
        )
        
        # --- Visualize both together ---
        geometries = [g for g in [pcd_ref, traj_ref, pcd_loc, traj_loc] if g]
        if geometries:
            processor.visualize_geometries(geometries)
        else:
            print("No geometries to visualize.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLAM localization on a new video sequence against an existing map.")
    
    # --- Input for Localization Sequence ---
    parser.add_argument("--video_path", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Query/run2/run2.MP4", 
        help="Path to the new video file to localize.")
    parser.add_argument("--telemetry_file", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Query/run2/run2.json", help="Path to the telemetry file for the new video.")
    parser.add_argument("--cam_dist_path", type=str,default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/cam_calib.json", help="Path to camera distortion calibration file.")

    # --- Reference Map ---
    parser.add_argument("--relocalization_db_path", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/_slam_results/slam_database.pt", help="Path to the slam_database.pt from the original map.")
    parser.add_argument("--ref_chunk_dir", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/_slam_results/chunks", help="Path to the output 'chunks' directory of the original map for visualization.")

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Query/run2/_slam_results", help="Directory to save output chunks for the new sequence.")
    parser.add_argument("--database_path", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Query/run2/_slam_results/slam_database.pt", help="Path to save the new SLAM database for the localized sequence.")

    # --- Sequence Control ---
    parser.add_argument("--start_frame", type=int, default=200, help="Start frame for video processing.")
    parser.add_argument("--max_frames", type=int, default=1000, help="Maximum number of frames to process (-1 for all).")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame.")

    # --- SLAM Parameters ---
    parser.add_argument("--chunk_size", type=int, default=50, help="Size of each processing chunk.")
    parser.add_argument("--overlap_size", type=int, default=1, help="Number of overlapping frames between chunks.")
    parser.add_argument("--pcl_confidence", type=float, default=0.8, help="Confidence threshold for points in the point cloud.")
    
    # --- Visualization ---
    parser.add_argument("--visualize", action='store_true', help="Enable post-processing and visualization of the combined results.")
    parser.add_argument("--debug_viz", action='store_true', help="Enable saving of debug visualizations for each chunk of the new sequence.")

    args = parser.parse_args()
    main(args)
