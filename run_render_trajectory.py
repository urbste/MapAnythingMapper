"""
Example script to render reconstructed trajectory overlay onto video.
"""

import argparse
from render_trajectory import render_trajectory_to_video


def main():
    parser = argparse.ArgumentParser(
        description="Render SLAM trajectory overlay onto video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render reference reconstruction
  python run_render_trajectory.py --video /path/to/reference.mp4 --output_dir /path/to/reference/_slam_results
  
  # Render query localization
  python run_render_trajectory.py --video /path/to/query.mp4 --output_dir /path/to/query/_slam_results --output_video /path/to/query_overlay.mp4
"""
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing SLAM reconstruction data (_slam_results folder with chunk_XXXX subdirectories)"
    )
    
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Path for output video (default: <output_dir>/trajectory_overlay.mp4)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("TRAJECTORY RENDERING")
    print("="*70)
    print(f"Input video:       {args.video}")
    print(f"Reconstruction:    {args.output_dir}")
    if args.output_video:
        print(f"Output video:      {args.output_video}")
    print("="*70)
    print()
    
    render_trajectory_to_video(
        video_path=args.video,
        output_dir=args.output_dir,
        output_video_path=args.output_video
    )


if __name__ == "__main__":
    main()


