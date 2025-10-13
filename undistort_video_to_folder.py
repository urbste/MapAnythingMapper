#!/usr/bin/env python3
"""
Script to undistort a video and save frames to a folder.

This script takes a video file and a camera calibration file, undistorts each frame,
and saves the undistorted frames as image files in a specified output directory.
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from utils.undistortion import create_undistortion_maps_from_file


def get_video_info(video_path):
    """
    Get information about the video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    
    cap.release()
    return info


def undistort_video_to_folder(
    video_path,
    cam_dist_path,
    output_dir,
    start_time=0.0,
    end_time=-1.0,
    stride=1,
    scale=1.0,
    target_size=None,
    frame_prefix="frame",
    image_format="jpg",
    reverse_names=False
):
    """
    Undistort a video and save frames to a folder.
    
    Args:
        video_path: Path to the video file
        cam_dist_path: Path to camera distortion calibration JSON file
        output_dir: Directory to save undistorted frames
        start_time: Start time in seconds (default: 0.0)
        end_time: End time in seconds (default: -1.0 for entire video)
        stride: Process every Nth frame (default: 1)
        scale: Scale factor for camera parameters (default: 1.0)
        target_size: Target size (height, width) for undistorted frames (default: None)
        frame_prefix: Prefix for saved frame filenames (default: "frame")
        image_format: Image format for saved frames (default: "png")
        reverse_names: Reverse frame numbering (start high, count down to 0) (default: False)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get video information
    print(f"Loading video: {video_path}")
    video_info = get_video_info(video_path)
    print(f"Video info: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f} FPS, {video_info['total_frames']} frames")
    
    # Calculate frame range
    start_frame = int(start_time * video_info['fps'])
    if end_time < 0:
        end_frame = video_info['total_frames']
    else:
        end_frame = min(int(end_time * video_info['fps']), video_info['total_frames'])
    
    total_frames_to_process = (end_frame - start_frame + stride - 1) // stride
    print(f"Processing frames {start_frame} to {end_frame} with stride {stride} ({total_frames_to_process} frames)")
    
    # Create undistortion maps
    print(f"\nLoading camera calibration from: {cam_dist_path}")
    undistortion_maps = create_undistortion_maps_from_file(cam_dist_path, scale=scale)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    print(f"\nProcessing and saving frames to: {output_dir}")
    frame_count = 0
    saved_count = 0
    
    # Calculate reverse naming counter if needed
    reverse_counter = total_frames_to_process - 1 if reverse_names else None
    
    with tqdm(total=total_frames_to_process, desc="Undistorting frames") as pbar:
        bad_frames = 0
        while True:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Check if we've reached the end frame
            if current_frame >= end_frame:
                break
            
            # Read frame
            
            ret, frame = cap.read()
            if not ret:
                bad_frames += 1
                if bad_frames > 100:
                    print("Too many bad frames. Stopping.")
                    break
                continue
            
            
            # Process only every stride-th frame
            if frame_count % stride == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply undistortion
                undistorted_frame = undistortion_maps.undistort_image(frame_rgb, target_size=target_size)
                
                # Convert back to BGR for saving
                undistorted_bgr = cv2.cvtColor(undistorted_frame, cv2.COLOR_RGB2BGR)
                
                # Determine frame number for filename
                if reverse_names:
                    frame_number = reverse_counter
                    reverse_counter -= 1
                else:
                    frame_number = current_frame
                
                # Save frame
                frame_filename = f"{frame_prefix}_{frame_number:06d}.{image_format}"
                frame_path = output_path / frame_filename
                cv2.imwrite(str(frame_path), undistorted_bgr)
                
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()
    
    print(f"\n✓ Successfully saved {saved_count} undistorted frames to {output_dir}")
    
    # Save metadata
    metadata_path = output_path / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Video file: {video_path}\n")
        f.write(f"Camera calibration: {cam_dist_path}\n")
        f.write(f"Original video size: {video_info['width']}x{video_info['height']}\n")
        f.write(f"Original FPS: {video_info['fps']:.2f}\n")
        f.write(f"Total frames in video: {video_info['total_frames']}\n")
        f.write(f"Start frame: {start_frame}\n")
        f.write(f"End frame: {end_frame}\n")
        f.write(f"Stride: {stride}\n")
        f.write(f"Frames saved: {saved_count}\n")
        f.write(f"Scale factor: {scale}\n")
        if target_size:
            f.write(f"Target size: {target_size[0]}x{target_size[1]}\n")
        f.write(f"Image format: {image_format}\n")
        f.write(f"Reverse naming: {reverse_names}\n")
    
    print(f"✓ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Undistort a video and save frames to a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Undistort entire video
  python undistort_video_to_folder.py --video input.mp4 --cam_dist cam.json --output_dir frames/
  
  # Undistort specific time range with stride
  python undistort_video_to_folder.py --video input.mp4 --cam_dist cam.json --output_dir frames/ \\
      --start_time 5.0 --end_time 15.0 --stride 2
  
  # Save as JPEG with custom prefix
  python undistort_video_to_folder.py --video input.mp4 --cam_dist cam.json --output_dir frames/ \\
      --image_format jpg --frame_prefix undistorted
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--video", 
        type=str, 
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--cam_dist", 
        type=str, 
        required=True,
        help="Path to camera distortion calibration JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save undistorted frames"
    )
    
    # Optional arguments for video processing
    parser.add_argument(
        "--start_time", 
        type=float, 
        default=0.0,
        help="Start time in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--end_time", 
        type=float, 
        default=-1.0,
        help="End time in seconds (default: -1.0 for entire video)"
    )
    parser.add_argument(
        "--stride", 
        type=int, 
        default=1,
        help="Process every Nth frame (default: 1, process all frames)"
    )
    
    # Optional arguments for camera and output
    parser.add_argument(
        "--scale", 
        type=float, 
        default=1.0,
        help="Scale factor for camera parameters (default: 1.0)"
    )
    parser.add_argument(
        "--target_width", 
        type=int, 
        default=None,
        help="Target width for undistorted frames (default: None, use camera default)"
    )
    parser.add_argument(
        "--target_height", 
        type=int, 
        default=None,
        help="Target height for undistorted frames (default: None, use camera default)"
    )
    parser.add_argument(
        "--frame_prefix", 
        type=str, 
        default="frame",
        help="Prefix for saved frame filenames (default: 'frame')"
    )
    parser.add_argument(
        "--image_format", 
        type=str, 
        default="jpg",
        choices=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Image format for saved frames (default: 'png')"
    )
    parser.add_argument(
        "--reverse_names",
        action="store_true",
        help="Reverse frame numbering (start from highest number, count down to 0)"
    )
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = None
    if args.target_width is not None and args.target_height is not None:
        target_size = (args.target_height, args.target_width)
    elif args.target_width is not None or args.target_height is not None:
        print("Warning: Both --target_width and --target_height must be specified. Ignoring target size.")
    
    # Run undistortion
    undistort_video_to_folder(
        video_path=args.video,
        cam_dist_path=args.cam_dist,
        output_dir=args.output_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        stride=args.stride,
        scale=args.scale,
        target_size=target_size,
        frame_prefix=args.frame_prefix,
        image_format=args.image_format,
        reverse_names=args.reverse_names
    )


if __name__ == "__main__":
    main()

