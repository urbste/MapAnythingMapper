import os
import time
import argparse
from slam_config import SLAMConfig
from visual_slam import VisualSLAM
from point_cloud_processor import PointCloudProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main(args):
    # --- 1. Configure and run SLAM ---
    config = SLAMConfig(
        image_folder=args.image_folder,
        video_path=args.video_path,
        telemetry_file=args.telemetry_file,
        cam_dist_path=args.cam_dist_path,
        output_dir=args.output_dir,
        database_path=args.database_path,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        stride=args.stride,
        create_debug_chunk_visualizations=args.debug_viz,
        point_cloud_confidence_threshold=args.pcl_confidence,
    )

    print("--- Running SLAM ---")
    slam_system = VisualSLAM(config)
    slam_system.run_slam_for_sequence()
    
    # --- 2. Post-processing and Visualization ---
    if args.visualize:
        print("\n\n--- Visualizing Results ---")
        
        processor = PointCloudProcessor()
        
        results_dir = os.path.dirname(config.database_path)
        os.makedirs(results_dir, exist_ok=True)
        
        pcd, traj, normal_viz = processor.generate_pcd_and_trajectory(
            chunk_dirs=[config.output_dir], 
            trajectory_color=[1, 0, 0],
            config=config
        )
        
        if pcd:
            pcd_path = os.path.join(results_dir, "final_point_cloud.ply")
            processor.save_ply(pcd, pcd_path)
            print(f"Saved point cloud to {pcd_path}")

        if traj:
            traj_path = os.path.join(results_dir, "final_trajectory.ply")
            processor.save_trajectory_as_ply(traj, traj_path)
            print(f"Saved trajectory to {traj_path}")

        geometries = [g for g in [pcd, traj, normal_viz] if g]
        if geometries:
            processor.visualize_geometries(geometries)
        else:
            print("No geometries to visualize.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Visual SLAM on a video or image sequence.")
    
    # --- Input ---
    parser.add_argument("--video_path", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/run1.MP4", help="Path to the input video file.")
    parser.add_argument("--image_folder", type=str, default="", help="Path to the folder of input images.")
    parser.add_argument("--telemetry_file", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/run1.json", help="Path to the telemetry file.")
    parser.add_argument("--cam_dist_path", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/cam_calib.json", help="Path to camera distortion calibration file.")

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/_slam_results", help="Directory to save output chunks. If not set, it's created in a _slam_results folder next to the input.")
    parser.add_argument("--database_path", type=str, default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/_slam_results/slam_database.pt", help="Path to save the SLAM database. If not set, it's created in a _slam_results folder.")

    # --- Sequence Control ---
    parser.add_argument("--start_frame", type=int, default=500, help="Start frame for video processing.")
    parser.add_argument("--max_frames", type=int, default=500, help="Maximum number of frames to process (-1 for all).")
    parser.add_argument("--stride", type=int, default=3, help="Process every Nth frame.")

    # --- SLAM Parameters ---
    parser.add_argument("--chunk_size", type=int, default=30, help="Size of each processing chunk.")
    parser.add_argument("--overlap_size", type=int, default=1, help="Number of overlapping frames between chunks.")
    parser.add_argument("--pcl_confidence", type=float, default=0.8, help="Confidence threshold for points in the point cloud.")
    
    # --- Visualization ---
    parser.add_argument("--visualize", action='store_true',default=True, help="Enable post-processing and visualization of the results.")
    parser.add_argument("--debug_viz", action='store_true',default=True, help="Enable saving of debug visualizations for each chunk.")

    args = parser.parse_args()

    if not args.video_path and not args.image_folder:
        raise ValueError("Either --video_path or --image_folder must be provided.")

    main(args)
