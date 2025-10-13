"""
Render reconstructed trajectory onto video frames by projecting 3D trajectory to image plane.
Also renders ground normals as arrows.
"""

import os
import cv2
import numpy as np
import torch
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def interpolate_poses_and_normals_full(keyframe_poses, keyframe_normals, keyframe_plane_ds, keyframe_paths, stride):
    """
    Interpolates poses, normals, and plane_ds between keyframes based on detected stride.
    
    Args:
        keyframe_poses: List of poses (torch.Tensor or numpy) for keyframes
        keyframe_normals: List of normals for keyframes
        keyframe_plane_ds: List of plane distances for keyframes
        keyframe_paths: List of frame paths/identifiers for keyframes
        stride: Detected stride between keyframes
        
    Returns:
        Tuple of (interpolated_poses, interpolated_normals, interpolated_plane_ds)
    """
    if stride <= 1 or len(keyframe_poses) < 2:
        return keyframe_poses, keyframe_normals, keyframe_plane_ds
    
    # Extract frame indices from paths
    keyframe_indices = []
    for path in keyframe_paths:
        try:
            base = os.path.basename(path)
            name, _sep, _ext = base.partition('.')
            digits = ''.join([c for c in name if c.isdigit()])
            frame_idx = int(digits)
            keyframe_indices.append(frame_idx)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse frame index from {path}, skipping interpolation")
            return keyframe_poses, keyframe_normals, keyframe_plane_ds
    
    # Convert poses to numpy if needed
    keyframe_poses_np = []
    for p in keyframe_poses:
        if isinstance(p, torch.Tensor):
            p_np = p[0].numpy() if p.ndim == 3 else p.numpy()
        else:
            p_np = p
        keyframe_poses_np.append(p_np)
    
    # Setup SLERP for rotation interpolation
    key_rotations = Rotation.from_matrix([p[:3, :3] for p in keyframe_poses_np])
    slerp = Slerp(keyframe_indices, key_rotations)
    
    # Setup linear interpolation for translations
    key_translations = np.array([p[:3, 3] for p in keyframe_poses_np])
    
    # Determine range for interpolation
    min_idx, max_idx = min(keyframe_indices), max(keyframe_indices)
    interp_indices = np.arange(min_idx, max_idx + 1)
    
    # Interpolate rotations and translations
    interp_rotations = slerp(interp_indices).as_matrix()
    interp_translations = np.array([
        np.interp(interp_indices, keyframe_indices, key_translations[:, i]) 
        for i in range(3)
    ]).T
    
    # Create interpolated poses
    interpolated_poses = []
    for i in range(len(interp_indices)):
        pose = np.eye(4)
        pose[:3, :3] = interp_rotations[i]
        pose[:3, 3] = interp_translations[i]
        interpolated_poses.append(torch.from_numpy(pose).float())
    
    # Interpolate normals if available
    interpolated_normals = []
    if keyframe_normals and len(keyframe_normals) > 0 and keyframe_normals[0] is not None:
        try:
            key_normals_array = np.array([n if isinstance(n, np.ndarray) else n for n in keyframe_normals])
            key_normals_normalized = key_normals_array / (np.linalg.norm(key_normals_array, axis=1, keepdims=True) + 1e-8)
            
            # Convert normals to rotations for SLERP
            reference = np.array([0.0, 0.0, 1.0])
            rotations = []
            for normal in key_normals_normalized:
                v = np.cross(reference, normal)
                s = np.linalg.norm(v)
                c = np.dot(reference, normal)
                
                if s < 1e-6:
                    if c > 0:
                        rotations.append(Rotation.identity())
                    else:
                        rotations.append(Rotation.from_euler('x', 180, degrees=True))
                else:
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
                    rotations.append(Rotation.from_matrix(R))
            
            normal_slerp = Slerp(keyframe_indices, Rotation.from_quat([r.as_quat() for r in rotations]))
            interpolated_rotations = normal_slerp(interp_indices)
            
            interpolated_normals = [rot.apply(reference) for rot in interpolated_rotations]
            interpolated_normals = [n / (np.linalg.norm(n) + 1e-8) for n in interpolated_normals]
            
            print(f"  Interpolated {len(interpolated_normals)} normals from {len(keyframe_normals)} keyframes")
        except Exception as e:
            print(f"  Warning: Could not interpolate normals ({e}), using keyframe values")
            interpolated_normals = keyframe_normals
    else:
        interpolated_normals = keyframe_normals
    
    # Interpolate plane_d values (linear interpolation)
    interpolated_plane_ds = []
    if keyframe_plane_ds and len(keyframe_plane_ds) > 0:
        try:
            key_plane_ds_array = np.array([d for d in keyframe_plane_ds if d is not None])
            if len(key_plane_ds_array) == len(keyframe_indices):
                interpolated_plane_ds = np.interp(interp_indices, keyframe_indices, key_plane_ds_array).tolist()
                print(f"  Interpolated {len(interpolated_plane_ds)} plane_d values")
            else:
                interpolated_plane_ds = keyframe_plane_ds
        except Exception as e:
            print(f"  Warning: Could not interpolate plane_d ({e}), using keyframe values")
            interpolated_plane_ds = keyframe_plane_ds
    else:
        interpolated_plane_ds = keyframe_plane_ds
    
    print(f"  Interpolated {len(interpolated_poses)} poses from {len(keyframe_poses)} keyframes (stride={stride})")
    return interpolated_poses, interpolated_normals, interpolated_plane_ds


def interpolate_all_frames_in_range(chunk_data, stride):
    """
    Interpolate poses, normals, and timestamps for ALL frames in the video range.
    
    Args:
        chunk_data: Loaded chunk data
        stride: Stride used during reconstruction
        
    Returns:
        Interpolated chunk data with poses for every frame
    """
    if stride <= 1:
        return chunk_data
    
    # Collect all keyframe data
    all_poses = []
    all_normals = []
    all_plane_ds = []
    all_frame_pos = []
    all_frametimes = []
    
    for chunk in chunk_data:
        poses = chunk['poses']
        # Convert poses to numpy and ensure (4, 4) shape
        converted_poses = []
        for p in poses:
            if isinstance(p, torch.Tensor):
                p_np = p.cpu().numpy()
                # Squeeze out batch dimension if present
                if p_np.ndim == 3:
                    p_np = p_np[0]
            else:
                p_np = np.array(p)
                if p_np.ndim == 3:
                    p_np = p_np[0]
            converted_poses.append(p_np)
        all_poses.extend(converted_poses)
        
        normals = chunk.get('normals', [])
        if normals:
            normals = [n.cpu().numpy() if isinstance(n, torch.Tensor) else n for n in normals]
        all_normals.extend(normals if normals else [None] * len(poses))
        
        plane_ds = chunk.get('plane_ds', [])
        if plane_ds:
            plane_ds = [d.cpu().item() if isinstance(d, torch.Tensor) else d for d in plane_ds]
        all_plane_ds.extend(plane_ds if plane_ds else [None] * len(poses))
        
        all_frame_pos.extend(chunk.get('frame_pos', []))
        all_frametimes.extend(chunk.get('frametimes_ns', []))
    
    if not all_frame_pos or len(all_frame_pos) < 2:
        print("Warning: No frame positions found, cannot interpolate")
        return chunk_data
    
    # Determine the full frame range
    min_frame = min(all_frame_pos)
    max_frame = max(all_frame_pos)
    total_frames = max_frame - min_frame + 1
    
    print(f"Interpolating from {len(all_poses)} keyframes to {total_frames} total frames...")
    print(f"  Frame range: [{min_frame}, {max_frame}], stride={stride}")
    
    # Create keyframe indices (positions in the full frame range)
    keyframe_indices = [fp - min_frame for fp in all_frame_pos]
    
    # Setup SLERP for rotation interpolation
    key_rotations = Rotation.from_matrix([p[:3, :3] for p in all_poses])
    slerp = Slerp(keyframe_indices, key_rotations)
    
    # Setup linear interpolation for translations
    key_translations = np.array([p[:3, 3] for p in all_poses])
    
    # Interpolate for all frames in range
    all_frame_indices = np.arange(0, total_frames)
    interp_rotations = slerp(all_frame_indices).as_matrix()
    interp_translations = np.array([
        np.interp(all_frame_indices, keyframe_indices, key_translations[:, i]) 
        for i in range(3)
    ]).T
    
    # Create interpolated poses
    interpolated_poses = []
    for i in range(len(all_frame_indices)):
        pose = np.eye(4)
        pose[:3, :3] = interp_rotations[i]
        pose[:3, 3] = interp_translations[i]
        interpolated_poses.append(pose)
    
    # Interpolate normals
    interpolated_normals = []
    if all_normals and any(n is not None for n in all_normals):
        try:
            valid_normals = [n for n in all_normals if n is not None]
            valid_indices = [i for i, n in enumerate(all_normals) if n is not None]
            
            if len(valid_normals) > 1:
                key_normals_array = np.array(valid_normals)
                key_normals_normalized = key_normals_array / (np.linalg.norm(key_normals_array, axis=1, keepdims=True) + 1e-8)
                
                reference = np.array([0.0, 0.0, 1.0])
                rotations = []
                for normal in key_normals_normalized:
                    v = np.cross(reference, normal)
                    s = np.linalg.norm(v)
                    c = np.dot(reference, normal)
                    
                    if s < 1e-6:
                        if c > 0:
                            rotations.append(Rotation.identity())
                        else:
                            rotations.append(Rotation.from_euler('x', 180, degrees=True))
                    else:
                        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
                        rotations.append(Rotation.from_matrix(R))
                
                valid_keyframe_indices = [keyframe_indices[i] for i in valid_indices]
                normal_slerp = Slerp(valid_keyframe_indices, Rotation.from_quat([r.as_quat() for r in rotations]))
                interpolated_rotations = normal_slerp(all_frame_indices)
                
                interpolated_normals = [rot.apply(reference) for rot in interpolated_rotations]
                interpolated_normals = [n / (np.linalg.norm(n) + 1e-8) for n in interpolated_normals]
                print(f"  Interpolated {len(interpolated_normals)} normals")
        except Exception as e:
            print(f"  Warning: Could not interpolate normals ({e})")
            interpolated_normals = []
    
    # Interpolate plane_d values
    interpolated_plane_ds = []
    if all_plane_ds and any(d is not None for d in all_plane_ds):
        try:
            valid_plane_ds = [d for d in all_plane_ds if d is not None]
            valid_indices = [i for i, d in enumerate(all_plane_ds) if d is not None]
            
            if len(valid_plane_ds) > 1:
                valid_keyframe_indices = [keyframe_indices[i] for i in valid_indices]
                interpolated_plane_ds = np.interp(all_frame_indices, valid_keyframe_indices, valid_plane_ds).tolist()
                print(f"  Interpolated {len(interpolated_plane_ds)} plane_d values")
        except Exception as e:
            print(f"  Warning: Could not interpolate plane_d ({e})")
            interpolated_plane_ds = []
    
    # Interpolate timestamps
    interpolated_frametimes = []
    if all_frametimes and len(all_frametimes) > 1:
        interpolated_frametimes = np.interp(all_frame_indices, keyframe_indices, all_frametimes).astype(np.int64).tolist()
        print(f"  Interpolated {len(interpolated_frametimes)} timestamps")
    
    # Create interpolated frame positions
    interpolated_frame_pos = list(range(min_frame, max_frame + 1))
    
    # Return as a single chunk with all interpolated data
    return [{
        'poses': interpolated_poses,
        'paths': [f"frame_{fp}" for fp in interpolated_frame_pos],
        'frametimes_ns': interpolated_frametimes,
        'frame_pos': interpolated_frame_pos,
        'normals': interpolated_normals,
        'plane_ds': interpolated_plane_ds
    }]


def load_reconstruction_data(output_dir, stride=1, interpolate_all=False):
    """
    Load all chunk data from a SLAM reconstruction.
    
    Args:
        output_dir: Directory containing chunks
        stride: Stride used during reconstruction  
        interpolate_all: If True, interpolate poses for ALL frames in range
    
    Returns:
        List of dicts with 'poses', 'paths', 'frametimes_ns', 'frame_pos', 'normals', 'plane_ds'
    """
    chunk_dirs = sorted(glob.glob(os.path.join(output_dir, "chunk_*")))
    
    all_data = []
    for chunk_dir in chunk_dirs:
        metadata_file = os.path.join(chunk_dir, "metadata.pt")
        if not os.path.exists(metadata_file):
            continue
        
        metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
        
        # Extract poses
        poses = metadata['camera_poses']
        if isinstance(poses[0], torch.Tensor):
            poses = [p.cpu().numpy() if p.ndim == 3 else p.cpu().numpy().reshape(1, 4, 4) 
                     for p in poses]
        
        # Extract normals and plane_ds
        normals = metadata.get('ground_normals', [])
        if normals:
            normals = [n.cpu().numpy() if isinstance(n, torch.Tensor) else n for n in normals]
        
        plane_ds = metadata.get('plane_ds', [])
        if plane_ds:
            plane_ds = [d.cpu().item() if isinstance(d, torch.Tensor) else d for d in plane_ds]
        
        # Get frame identifiers
        paths = metadata.get('image_paths', [])
        frametimes = metadata.get('frametimes_ns', [])
        frame_pos = metadata.get('frame_pos', [])
        
        all_data.append({
            'poses': poses,
            'paths': paths,
            'frametimes_ns': frametimes,
            'frame_pos': frame_pos,
            'normals': normals,
            'plane_ds': plane_ds
        })
    
    # Interpolate for all frames if requested
    if interpolate_all and stride > 1:
        all_data = interpolate_all_frames_in_range(all_data, stride)
    
    return all_data


def extract_trajectory_data(chunk_data):
    """
    Extract all camera positions, normals, plane_ds, and poses from the reconstruction.
    
    Returns:
        Tuple of (positions np.array of shape (N, 3), normals list, plane_ds list, poses list)
    """
    positions = []
    normals = []
    plane_ds = []
    poses = []
    
    for chunk in chunk_data:
        chunk_normals = chunk.get('normals', [])
        chunk_plane_ds = chunk.get('plane_ds', [])
        
        for i, pose in enumerate(chunk['poses']):
            if pose.ndim == 3:
                pos = pose[0, :3, 3]
            else:
                pos = pose[:3, 3]
            positions.append(pos)
            poses.append(pose)
            
            # Get corresponding normal
            if i < len(chunk_normals) and chunk_normals[i] is not None:
                normals.append(chunk_normals[i])
            else:
                normals.append(None)
            
            # Get corresponding plane_d
            if i < len(chunk_plane_ds) and chunk_plane_ds[i] is not None:
                plane_ds.append(chunk_plane_ds[i])
            else:
                plane_ds.append(None)
    
    return np.array(positions), normals, plane_ds, poses


def smooth_trajectory(trajectory_3d, window_length=11, polyorder=3):
    """
    Smooth trajectory using Savitzky-Golay filter.
    
    Args:
        trajectory_3d: (N, 3) array of positions
        window_length: Length of filter window (must be odd)
        polyorder: Order of polynomial to fit
        
    Returns:
        Smoothed trajectory (N, 3)
    """
    if len(trajectory_3d) < window_length:
        # Not enough points to smooth
        return trajectory_3d
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Apply filter to each dimension
    smoothed = np.zeros_like(trajectory_3d)
    for i in range(3):
        smoothed[:, i] = savgol_filter(trajectory_3d[:, i], window_length, polyorder)
    
    return smoothed




def project_trajectory_to_ground(trajectory_3d, ground_normals, camera_poses, plane_ds=None):
    """
    Project trajectory points onto the ground plane by ray-casting from camera along its Y-axis.
    Finds the intersection of the camera's downward direction with the ground plane.
    
    Args:
        trajectory_3d: (N, 3) array of camera positions
        ground_normals: List of (3,) normals, one per trajectory point
        camera_poses: List of (4, 4) camera poses to get camera Y-axis direction
        plane_ds: List of plane distances (optional, will be estimated if not provided)
        
    Returns:
        Projected trajectory (N, 3)
    """
    projected = trajectory_3d.copy()
    
    for i in range(len(trajectory_3d)):
        if i >= len(ground_normals) or ground_normals[i] is None:
            continue
            
        normal = ground_normals[i]
        if isinstance(normal, torch.Tensor):
            normal = normal.cpu().numpy()
        
        # Ensure normal is normalized
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Get camera pose
        if i >= len(camera_poses):
            continue
        pose = camera_poses[i]
        if pose.ndim == 3:
            pose = pose[0]
        
        # Camera Y-axis in world coordinates (downward direction for the camera)
        # In camera frame, Y points down, so we take the second column of rotation matrix
        camera_y_world = pose[:3, 1]
        camera_y_world = camera_y_world / (np.linalg.norm(camera_y_world) + 1e-8)
        
        # Camera position
        cam_pos = trajectory_3d[i]
        
        # Get plane distance d for the plane equation: normal · p + d = 0
        # If we have plane_ds, use it; otherwise estimate from camera position
        if plane_ds is not None and i < len(plane_ds) and plane_ds[i] is not None:
            d = plane_ds[i]
            if isinstance(d, torch.Tensor):
                d = d.cpu().item()
        else:
            # Estimate d: assume ground is about 1.5m below camera
            # d = -normal · point_on_plane
            estimated_ground_point = cam_pos - normal * 1.5
            d = -np.dot(normal, estimated_ground_point)
        
        # Ray-plane intersection
        # Ray: P(t) = cam_pos + t * camera_y_world
        # Plane: normal · P + d = 0
        # Substituting: normal · (cam_pos + t * camera_y_world) + d = 0
        # Solving for t: t = -(normal · cam_pos + d) / (normal · camera_y_world)
        
        denominator = np.dot(normal, camera_y_world)
        
        # Check if ray is parallel to plane (should be rare for ground plane)
        if abs(denominator) < 1e-6:
            # Fallback: project along normal
            projected[i] = cam_pos - normal * 1.5
            continue
        
        t = -(np.dot(normal, cam_pos) + d) / denominator
        
        # Only project if intersection is in front of camera (positive t) and reasonable distance
        if t > 0 and t < 10.0:  # Reasonable distance check (< 10m down)
            intersection = cam_pos + t * camera_y_world
            projected[i] = intersection
        else:
            # Fallback: keep original position or project along normal
            projected[i] = cam_pos - normal * 1.5
    
    return projected


def project_trajectory_to_image(trajectory_3d, current_pose, intrinsics, current_idx=None, max_distance=15.0):
    """
    Project 3D trajectory points to image plane using camera pose and intrinsics.
    
    Args:
        trajectory_3d: (N, 3) array of world coordinates
        current_pose: (4, 4) camera-to-world transformation
        intrinsics: (3, 3) camera intrinsics matrix
        current_idx: Index of current camera in trajectory (optional)
        max_distance: Maximum distance in meters to show trajectory (default: 15.0)
        
    Returns:
        (N, 2) array of pixel coordinates, (N,) array of valid flags (in front of camera and within distance)
    """
    # Convert to camera coordinates
    # We need world-to-camera, so invert the camera-to-world pose
    pose = current_pose[0] if current_pose.ndim == 3 else current_pose
    world_to_cam = np.linalg.inv(pose)
    
    # Get current camera position in world coordinates
    current_pos = pose[:3, 3]
    
    # Calculate 3D distances from current position to all trajectory points
    distances = np.linalg.norm(trajectory_3d - current_pos, axis=1)
    
    # Filter based on distance and trajectory direction (only show upcoming trajectory)
    distance_mask = distances <= max_distance
    
    # If we know the current index, only show trajectory ahead (limited to next 30 poses)
    if current_idx is not None:
        future_mask = np.zeros(len(trajectory_3d), dtype=bool)
        end_idx = min(current_idx + 30, len(trajectory_3d))  # Limit to next 30 poses
        future_mask[current_idx:end_idx] = True  # Only show current and next 30 points
        distance_mask = distance_mask & future_mask
    
    # Transform points to camera space
    traj_homogeneous = np.hstack([trajectory_3d, np.ones((len(trajectory_3d), 1))])
    traj_cam = (world_to_cam @ traj_homogeneous.T).T  # (N, 4)
    traj_cam = traj_cam[:, :3]  # (N, 3)
    
    # Filter points behind camera
    in_front_mask = traj_cam[:, 2] > 0.1  # At least 10cm in front
    
    # Combine all filters
    valid_mask = distance_mask & in_front_mask
    
    # Project to image plane
    traj_2d = (intrinsics @ traj_cam.T).T  # (N, 3)
    traj_2d = traj_2d[:, :2] / (traj_2d[:, 2:3] + 1e-8)  # Normalize by z
    
    return traj_2d, valid_mask


def project_normal_to_image(normal, camera_pose, intrinsics, arrow_length=2.0):
    """
    Project a ground normal vector to image coordinates for visualization.
    
    Args:
        normal: (3,) ground normal in world coordinates
        camera_pose: (4, 4) camera-to-world transformation
        intrinsics: (3, 3) camera intrinsics matrix
        arrow_length: Length of arrow in meters
        
    Returns:
        Tuple of (start_2d, end_2d) pixel coordinates, or (None, None) if behind camera
    """
    if normal is None:
        return None, None
    
    pose = camera_pose[0] if camera_pose.ndim == 3 else camera_pose
    world_to_cam = np.linalg.inv(pose)
    
    # Start point is at camera position
    cam_pos_world = pose[:3, 3]
    
    # End point is along the normal (pointing upward)
    end_point_world = cam_pos_world + normal * arrow_length
    
    # Transform both points to camera space
    start_cam = (world_to_cam @ np.append(cam_pos_world, 1))[:3]
    end_cam = (world_to_cam @ np.append(end_point_world, 1))[:3]
    
    # Check if points are in front of camera
    if start_cam[2] <= 0.1 or end_cam[2] <= 0.1:
        return None, None
    
    # Project to image plane
    start_2d = (intrinsics @ start_cam)
    start_2d = start_2d[:2] / start_2d[2]
    
    end_2d = (intrinsics @ end_cam)
    end_2d = end_2d[:2] / end_2d[2]
    
    return start_2d, end_2d


def draw_trajectory_on_frame(frame, trajectory_2d, valid_mask, current_idx=None, 
                              current_normal=None, intrinsics=None, current_pose=None):
    """
    Draw trajectory as a colored line on the frame, plus ground normal arrow.
    
    Args:
        frame: Image array (H, W, 3)
        trajectory_2d: (N, 2) pixel coordinates
        valid_mask: (N,) boolean mask for valid points
        current_idx: Index of current camera position (will be highlighted)
        current_normal: (3,) ground normal vector for current frame
        intrinsics: (3, 3) camera intrinsics (for normal projection)
        current_pose: (4, 4) camera pose (for normal projection)
        
    Returns:
        frame with trajectory and normal drawn
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw trajectory line
    valid_points = trajectory_2d[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_points) > 1:
        # Draw line segments
        for i in range(len(valid_points) - 1):
            pt1 = tuple(valid_points[i].astype(int))
            pt2 = tuple(valid_points[i + 1].astype(int))
            
            # Check if points are within image bounds
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                
                # Color gradient based on position in the visible trajectory segment
                # Green (near) -> Yellow -> Red (far within 15m)
                if len(valid_indices) > 1:
                    # Ratio from 0 (near) to 1 (far) within visible range
                    ratio = i / max(len(valid_points) - 1, 1)
                    
                    if ratio < 0.5:
                        # Green to yellow
                        color = (0, int(255 * (1 - 2 * ratio)), int(255 * 2 * ratio))
                    else:
                        # Yellow to red
                        color = (0, int(255 * (2 - 2 * ratio)), 255)
                else:
                    color = (0, 255, 0)  # Default green
                
                cv2.line(overlay, pt1, pt2, color, 3, cv2.LINE_AA)
    
    # Draw current position
    if current_idx is not None and current_idx < len(valid_mask) and valid_mask[current_idx]:
        current_pt = tuple(trajectory_2d[current_idx].astype(int))
        if 0 <= current_pt[0] < w and 0 <= current_pt[1] < h:
            cv2.circle(overlay, current_pt, 8, (255, 0, 255), -1)  # Magenta dot
            cv2.circle(overlay, current_pt, 10, (255, 255, 255), 2)  # White outline
    
    # Draw ground normal arrow if available
    if current_normal is not None and intrinsics is not None and current_pose is not None:
        start_2d, end_2d = project_normal_to_image(current_normal, current_pose, intrinsics, arrow_length=1.5)
        
        if start_2d is not None and end_2d is not None:
            start_pt = tuple(start_2d.astype(int))
            end_pt = tuple(end_2d.astype(int))
            
            # Check if arrow points are within bounds
            if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
                0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
                
                # Draw arrow (cyan color for normal)
                cv2.arrowedLine(overlay, start_pt, end_pt, (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.3)
                
                # Add text label
                label_pos = (end_pt[0] + 10, end_pt[1])
                cv2.putText(overlay, "Normal", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Blend overlay with original
    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return result


def synchronize_trajectories(ref_data, query_data, ref_stride=1, query_stride=1):
    """
    Synchronize reference and query trajectories based on timestamps.
    
    Args:
        ref_data: Reference chunk data
        query_data: Query chunk data  
        ref_stride: Stride used for reference
        query_stride: Stride used for query
        
    Returns:
        Tuple of (ref_traj_sync, query_traj_sync, time_diffs, distances, sync_indices)
    """
    # Extract data with interpolation
    ref_traj_3d, ref_normals, ref_plane_ds, ref_poses = extract_trajectory_data(ref_data)
    query_traj_3d, query_normals, query_plane_ds, query_poses = extract_trajectory_data(query_data)
    
    # Extract timestamps
    ref_timestamps = []
    query_timestamps = []
    
    for chunk in ref_data:
        ref_timestamps.extend(chunk.get('frametimes_ns', []))
    
    for chunk in query_data:
        query_timestamps.extend(chunk.get('frametimes_ns', []))
    
    print(f"Reference trajectory: {len(ref_traj_3d)} poses, {len(ref_timestamps)} timestamps")
    print(f"Query trajectory: {len(query_traj_3d)} poses, {len(query_timestamps)} timestamps")
    
    if not ref_timestamps or not query_timestamps:
        print("Warning: Missing timestamps, cannot synchronize trajectories")
        return None, None, None, None, None
    
    # Convert to numpy arrays
    ref_timestamps = np.array(ref_timestamps)
    query_timestamps = np.array(query_timestamps)
    
    # Align both trajectories to start at time 0
    ref_start_time = ref_timestamps[0]
    query_start_time = query_timestamps[0]
    ref_timestamps = ref_timestamps - ref_start_time  # Normalize to start at 0
    query_timestamps = query_timestamps - query_start_time  # Normalize to start at 0
    
    print(f"Aligned reference trajectory to start at time 0 (original start: {ref_start_time / 1e9:.2f}s)")
    print(f"Aligned query trajectory to start at time 0 (original start: {query_start_time / 1e9:.2f}s)")
    
    # Find overlapping time range
    time_start = max(ref_timestamps[0], query_timestamps[0])
    time_end = min(ref_timestamps[-1], query_timestamps[-1])
    
    print(f"Time overlap: {(time_end - time_start) / 1e9:.2f} seconds")
    
    # Find synchronized indices
    ref_sync_indices = []
    query_sync_indices = []
    time_diffs = []
    distances = []
    
    for i, ref_t in enumerate(ref_timestamps):
        if ref_t < time_start or ref_t > time_end:
            continue
        
        # Find closest query timestamp
        time_diff = np.abs(query_timestamps - ref_t)
        closest_idx = np.argmin(time_diff)
        
        if time_diff[closest_idx] < 100e6:  # Within 100ms
            ref_sync_indices.append(i)
            query_sync_indices.append(closest_idx)
            
            # Calculate 3D distance
            dist = np.linalg.norm(ref_traj_3d[i] - query_traj_3d[closest_idx])
            distances.append(dist)
            
            # Time difference (positive means query is ahead)
            t_diff = (query_timestamps[closest_idx] - ref_t) / 1e9  # Convert to seconds
            time_diffs.append(t_diff)
    
    print(f"Synchronized {len(ref_sync_indices)} poses")
    
    # Convert lists to numpy arrays
    time_diffs_arr = np.array(time_diffs)
    distances_arr = np.array(distances)
    
    print(f"Average distance: {np.mean(distances_arr):.2f}m (min={np.min(distances_arr):.2f}m, max={np.max(distances_arr):.2f}m)")
    print(f"Average time difference: {np.mean(np.abs(time_diffs_arr)):.2f}s (range: [{np.min(time_diffs_arr):.2f}s, {np.max(time_diffs_arr):.2f}s])")
    
    if len(time_diffs_arr) > 0:
        ref_ahead_count = np.sum(time_diffs_arr < 0)
        query_ahead_count = np.sum(time_diffs_arr > 0)
        print(f"  Reference ahead in {ref_ahead_count} frames, Query ahead in {query_ahead_count} frames")
    
    return (ref_sync_indices, query_sync_indices, time_diffs_arr, 
            distances_arr, (ref_traj_3d, query_traj_3d, ref_poses, query_poses))


def project_ground_circle_to_image(position_3d, normal, camera_pose, intrinsics, radius=0.5, num_points=20):
    """
    Project a circle on the ground plane to image coordinates.
    
    Args:
        position_3d: (3,) position on ground in world coordinates
        normal: (3,) ground normal at that position
        camera_pose: (4, 4) camera-to-world transformation
        intrinsics: (3, 3) camera intrinsics matrix
        radius: Circle radius in meters
        num_points: Number of points to approximate the circle
        
    Returns:
        List of (x, y) pixel coordinates for the circle, or None if not visible
    """
    if normal is None:
        return None
    
    pose = camera_pose[0] if camera_pose.ndim == 3 else camera_pose
    world_to_cam = np.linalg.inv(pose)
    
    # Normalize normal
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    
    # Create two orthogonal vectors in the ground plane
    # Use arbitrary vector that's not parallel to normal
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent1 = np.cross(normal, arbitrary)
    tangent1 = tangent1 / (np.linalg.norm(tangent1) + 1e-8)
    tangent2 = np.cross(normal, tangent1)
    tangent2 = tangent2 / (np.linalg.norm(tangent2) + 1e-8)
    
    # Generate circle points on the ground plane
    circle_points_3d = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        point = position_3d + radius * (np.cos(angle) * tangent1 + np.sin(angle) * tangent2)
        circle_points_3d.append(point)
    
    # Transform to camera space and project
    circle_points_2d = []
    all_visible = True
    for point_3d in circle_points_3d:
        point_cam = (world_to_cam @ np.append(point_3d, 1))[:3]
        
        # Check if in front of camera
        if point_cam[2] <= 0.1:
            all_visible = False
            continue
        
        # Project to image
        point_2d = (intrinsics @ point_cam)
        point_2d = point_2d[:2] / point_2d[2]
        circle_points_2d.append(point_2d)
    
    return circle_points_2d if all_visible and len(circle_points_2d) > 3 else None


def draw_comparison_overlay(frame, ref_traj_2d, query_traj_2d, ref_valid, query_valid,
                            time_diff, distance, ref_ahead, current_ref_idx=None, current_query_idx=None,
                            ref_normal=None, intrinsics=None, current_ref_pose=None,
                            query_pos_3d=None, query_normal=None):
    """
    Draw both trajectories and comparison metrics on the frame.
    
    Args:
        frame: Image array
        ref_traj_2d: Reference trajectory in 2D
        query_traj_2d: Query trajectory in 2D
        ref_valid: Valid mask for reference
        query_valid: Valid mask for query
        time_diff: Time difference in seconds
        distance: Distance in meters
        ref_ahead: Boolean, True if reference is ahead
        current_ref_idx: Current index in reference trajectory
        current_query_idx: Current index in query trajectory
        ref_normal: Ground normal (optional)
        intrinsics: Camera intrinsics (optional)
        current_ref_pose: Camera pose (optional)
        query_pos_3d: Query rider's 3D position (optional, for ground circle)
        query_normal: Query rider's ground normal (optional, for ground circle)
        
    Returns:
        Frame with overlays
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw reference trajectory (RED)
    ref_points = ref_traj_2d[ref_valid]
    if len(ref_points) > 1:
        for i in range(len(ref_points) - 1):
            pt1 = tuple(ref_points[i].astype(int))
            pt2 = tuple(ref_points[i + 1].astype(int))
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(overlay, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)  # Red
    
    # Draw query trajectory (BLUE)
    query_points = query_traj_2d[query_valid]
    if len(query_points) > 1:
        for i in range(len(query_points) - 1):
            pt1 = tuple(query_points[i].astype(int))
            pt2 = tuple(query_points[i + 1].astype(int))
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(overlay, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)  # Blue
    
    # Draw current positions
    if current_ref_idx is not None and current_ref_idx < len(ref_valid) and ref_valid[current_ref_idx]:
        ref_pt = tuple(ref_traj_2d[current_ref_idx].astype(int))
        if 0 <= ref_pt[0] < w and 0 <= ref_pt[1] < h:
            cv2.circle(overlay, ref_pt, 10, (0, 0, 255), -1)  # Red dot
            cv2.circle(overlay, ref_pt, 12, (255, 255, 255), 2)  # White outline
    
    if current_query_idx is not None and current_query_idx < len(query_valid) and query_valid[current_query_idx]:
        query_pt = tuple(query_traj_2d[current_query_idx].astype(int))
        if 0 <= query_pt[0] < w and 0 <= query_pt[1] < h:
            cv2.circle(overlay, query_pt, 10, (255, 0, 0), -1)  # Blue dot
            cv2.circle(overlay, query_pt, 12, (255, 255, 255), 2)  # White outline
    
    # Draw ground normal if available
    if ref_normal is not None and intrinsics is not None and current_ref_pose is not None:
        start_2d, end_2d = project_normal_to_image(ref_normal, current_ref_pose, intrinsics, arrow_length=1.5)
        if start_2d is not None and end_2d is not None:
            start_pt = tuple(start_2d.astype(int))
            end_pt = tuple(end_2d.astype(int))
            if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
                0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
                cv2.arrowedLine(overlay, start_pt, end_pt, (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.3)
    
    # Draw query rider position as circle on ground
    if query_pos_3d is not None and query_normal is not None and intrinsics is not None and current_ref_pose is not None:
        circle_2d = project_ground_circle_to_image(
            query_pos_3d, query_normal, current_ref_pose, intrinsics, radius=0.6, num_points=24
        )
        
        if circle_2d is not None and len(circle_2d) > 0:
            # Draw filled circle on ground
            circle_pts = np.array([[int(pt[0]), int(pt[1])] for pt in circle_2d], dtype=np.int32)
            
            # Check if at least some points are within bounds
            valid_pts = [(pt[0] >= 0 and pt[0] < w and pt[1] >= 0 and pt[1] < h) for pt in circle_pts]
            if any(valid_pts):
                # Draw filled circle
                cv2.fillPoly(overlay, [circle_pts], (255, 0, 0), lineType=cv2.LINE_AA)  # Blue for query
                # Draw circle outline
                cv2.polylines(overlay, [circle_pts], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    
    # Blend trajectory overlay
    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Add text overlay with comparison metrics
    # Create semi-transparent background for text
    info_overlay = result.copy()
    box_height = 120
    cv2.rectangle(info_overlay, (10, 10), (400, box_height), (0, 0, 0), -1)
    result = cv2.addWeighted(info_overlay, 0.6, result, 0.4, 0)
    
    # Add text
    y_pos = 35
    line_height = 30
    
    # Legend
    cv2.putText(result, "Reference", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "Query", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 0, 0), 2, cv2.LINE_AA)
    y_pos += line_height
    
    # Distance
    cv2.putText(result, f"Distance: {distance:.2f}m", (20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y_pos += line_height
    
    # Time difference
    leader_text = "Reference" if ref_ahead else "Query"
    time_text = f"{leader_text} ahead: {abs(time_diff):.2f}s"
    color = (0, 255, 0) if ref_ahead else (0, 165, 255)  # Green if ref ahead, orange if query ahead
    cv2.putText(result, time_text, (20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return result


def render_trajectory_to_video(video_path, output_dir, output_video_path=None, 
                               smooth=True, project_to_ground=True, 
                               smooth_window=11, max_distance=15.0, stride=1, 
                               query_output_dir=None, query_stride=1):
    """
    Main function to render trajectory onto video.
    Optionally compares with a query trajectory for race comparison.
    
    Args:
        video_path: Path to input video (reference run)
        output_dir: Directory containing reference SLAM reconstruction chunks
        output_video_path: Path for output video (defaults to output_dir/trajectory_overlay.mp4)
        smooth: Whether to smooth the trajectory (default: True)
        project_to_ground: Whether to project trajectory to ground plane (default: True)
        smooth_window: Window length for smoothing filter (default: 11)
        max_distance: Maximum distance in meters to show trajectory (default: 15.0)
        stride: Frame stride used during reference SLAM processing (for interpolation)
        query_output_dir: Optional path to query trajectory chunks for comparison
        query_stride: Frame stride used during query SLAM processing
    """
    # Determine if we're in comparison mode
    comparison_mode = query_output_dir is not None
    
    # Load reference reconstruction data (with full interpolation for all frames)
    print("Loading reference reconstruction data...")
    ref_chunk_data = load_reconstruction_data(output_dir, stride=stride, interpolate_all=True)
    if not ref_chunk_data:
        print(f"Error: No reconstruction data found in {output_dir}")
        return
    
    # Load query data if in comparison mode
    query_chunk_data = None
    sync_data = None
    if comparison_mode:
        print(f"\nLoading query reconstruction data for comparison...")
        query_chunk_data = load_reconstruction_data(query_output_dir, stride=query_stride, interpolate_all=True)
        if not query_chunk_data:
            print(f"Error: No query data found in {query_output_dir}")
            comparison_mode = False
        else:
            print("\nSynchronizing trajectories...")
            sync_result = synchronize_trajectories(ref_chunk_data, query_chunk_data, stride, query_stride)
            if sync_result[0] is not None:
                sync_data = sync_result
            else:
                print("Warning: Could not synchronize trajectories, disabling comparison mode")
                comparison_mode = False
    
    # Extract reference trajectory data
    print("\nExtracting reference trajectory and normals...")
    trajectory_3d, trajectory_normals, trajectory_plane_ds, trajectory_poses = extract_trajectory_data(ref_chunk_data)
    print(f"  Loaded trajectory with {len(trajectory_3d)} poses")
    num_normals = sum(1 for n in trajectory_normals if n is not None)
    if num_normals > 0:
        print(f"  Found {num_normals} ground normals")
    num_plane_ds = sum(1 for d in trajectory_plane_ds if d is not None)
    if num_plane_ds > 0:
        print(f"  Found {num_plane_ds} plane distances")
    
    # Extract query trajectory data if in comparison mode
    query_trajectory_3d = None
    query_trajectory_normals = None
    query_trajectory_plane_ds = None
    query_trajectory_poses = None
    
    if comparison_mode and query_chunk_data:
        print("\nExtracting query trajectory and normals...")
        query_trajectory_3d, query_trajectory_normals, query_trajectory_plane_ds, query_trajectory_poses = extract_trajectory_data(query_chunk_data)
        print(f"  Loaded query trajectory with {len(query_trajectory_3d)} poses")
        num_query_normals = sum(1 for n in query_trajectory_normals if n is not None)
        if num_query_normals > 0:
            print(f"  Found {num_query_normals} ground normals")
    
    # Extract synchronized data if in comparison mode
    if comparison_mode and sync_data:
        ref_sync_idx, query_sync_idx, time_diffs, distances, traj_data = sync_data
        ref_traj_full, query_traj_full, ref_poses_full, query_poses_full = traj_data
    
    # Smooth reference trajectory
    if smooth:
        print("\nSmoothing reference trajectory...")
        trajectory_3d = smooth_trajectory(trajectory_3d, window_length=smooth_window, polyorder=3)
        print(f"  ✓ Applied Savitzky-Golay filter (window={smooth_window})")
    else:
        print("  Skipping trajectory smoothing")
    
    # Project reference trajectory to ground if normals are available
    if project_to_ground and any(n is not None for n in trajectory_normals):
        print("Projecting reference trajectory to ground via camera Y-axis ray-casting...")
        trajectory_3d_ground = project_trajectory_to_ground(
            trajectory_3d, trajectory_normals, trajectory_poses, trajectory_plane_ds
        )
        
        # Calculate average projection distance
        projection_dists = np.linalg.norm(trajectory_3d - trajectory_3d_ground, axis=1)
        avg_distance = np.mean(projection_dists[projection_dists > 0])
        print(f"  ✓ Average projection distance: {avg_distance:.3f}m")
        
        trajectory_3d = trajectory_3d_ground
    elif project_to_ground:
        print("  Warning: No ground plane information found for reference, using original trajectory")
    else:
        print("  Skipping ground plane projection for reference")
    
    # Smooth and project query trajectory if in comparison mode
    if comparison_mode and query_trajectory_3d is not None:
        if smooth:
            print("\nSmoothing query trajectory...")
            query_trajectory_3d = smooth_trajectory(query_trajectory_3d, window_length=smooth_window, polyorder=3)
            print(f"  ✓ Applied Savitzky-Golay filter (window={smooth_window})")
        
        if project_to_ground and any(n is not None for n in query_trajectory_normals):
            print("Projecting query trajectory to ground via camera Y-axis ray-casting...")
            query_trajectory_3d_ground = project_trajectory_to_ground(
                query_trajectory_3d, query_trajectory_normals, query_trajectory_poses, query_trajectory_plane_ds
            )
            
            # Calculate average projection distance
            projection_dists = np.linalg.norm(query_trajectory_3d - query_trajectory_3d_ground, axis=1)
            avg_distance = np.mean(projection_dists[projection_dists > 0])
            print(f"  ✓ Average projection distance: {avg_distance:.3f}m")
            
            query_trajectory_3d = query_trajectory_3d_ground
        elif project_to_ground:
            print("  Warning: No ground plane information found for query, using original trajectory")
        
        # Update sync data with projected trajectories and recalculate distances
        if sync_data:
            ref_traj_full = trajectory_3d
            query_traj_full = query_trajectory_3d
            
            # Recalculate 3D distances after ground projection
            recalc_distances = []
            for i, ref_idx in enumerate(ref_sync_idx):
                query_idx = query_sync_idx[i]
                if ref_idx < len(ref_traj_full) and query_idx < len(query_traj_full):
                    dist = np.linalg.norm(ref_traj_full[ref_idx] - query_traj_full[query_idx])
                    recalc_distances.append(dist)
                else:
                    recalc_distances.append(0.0)
            
            print(f"  Recalculated distances after projection: mean={np.mean(recalc_distances):.2f}m, max={np.max(recalc_distances):.2f}m")
            
            sync_data = (ref_sync_idx, query_sync_idx, time_diffs, np.array(recalc_distances), 
                        (ref_traj_full, query_traj_full, ref_poses_full, query_poses_full))
    
    # Get frame mapping (frame_pos -> trajectory index) and find start/end frames
    frame_to_traj_idx = {}
    traj_idx = 0
    min_frame_pos = float('inf')
    max_frame_pos = 0
    has_frame_pos_data = False
    
    for chunk in ref_chunk_data:
        if chunk['frame_pos']:
            has_frame_pos_data = True
            for i, frame_pos in enumerate(chunk['frame_pos']):
                frame_to_traj_idx[frame_pos] = traj_idx
                min_frame_pos = min(min_frame_pos, frame_pos)
                max_frame_pos = max(max_frame_pos, frame_pos)
                traj_idx += 1
        else:
            # No frame_pos data, just count poses
            traj_idx += len(chunk['poses'])
    
    if not has_frame_pos_data:
        print("Warning: No frame_pos data found in chunks!")
        print("  This typically happens when processing image folders instead of videos.")
        print("  Cannot render to video without frame position information.")
        return
    
    print(f"Frame range in reconstruction: [{min_frame_pos}, {max_frame_pos}]")
    print(f"Total trajectory poses: {len(trajectory_3d)}")
    print(f"Frame positions with poses: {len(frame_to_traj_idx)}")
    
    # Check for gaps in frame coverage
    if len(frame_to_traj_idx) < len(trajectory_3d):
        print(f"  Warning: Frame mapping incomplete! {len(trajectory_3d) - len(frame_to_traj_idx)} poses without frame_pos")
    
    # Check if we have pose for every frame in range (indicates proper interpolation)
    expected_frames = max_frame_pos - min_frame_pos + 1
    actual_frames = len(frame_to_traj_idx)
    if actual_frames < expected_frames:
        missing_frames = expected_frames - actual_frames
        print(f"  Note: Missing {missing_frames} frames in range (stride was used without full interpolation)")
        print(f"        Trajectory will only render on frames with poses: every ~{expected_frames // actual_frames} frames")
    
    # Try to load intrinsics from slam_database.pt first
    intrinsics = None
    recon_height, recon_width = None, None
    
    database_path = os.path.join(output_dir, "slam_database.pt")
    if os.path.exists(database_path):
        print("Loading data from slam_database.pt...")
        database = torch.load(database_path, map_location="cpu", weights_only=False)
        if isinstance(database, list) and len(database) > 0:
            first_entry = database[0]
            
            # Get intrinsics
            if 'intrinsics' in first_entry:
                intrinsics = first_entry['intrinsics']
                if isinstance(intrinsics, torch.Tensor):
                    intrinsics = intrinsics.cpu().numpy()
                if intrinsics.ndim == 3:
                    intrinsics = intrinsics[0]
                print(f"✓ Found intrinsics in database")
            
            # Get reconstruction resolution from depth_z shape
            if 'depth_z' in first_entry:
                depth_z = first_entry['depth_z']
                if hasattr(depth_z, 'shape') and len(depth_z.shape) >= 3:
                    # depth_z shape is [1, H, W, 1] or [H, W, 1]
                    if depth_z.ndim == 4:
                        recon_height, recon_width = depth_z.shape[1], depth_z.shape[2]
                    else:
                        recon_height, recon_width = depth_z.shape[0], depth_z.shape[1]
                    print(f"✓ Found reconstruction resolution from depth_z: {recon_width}x{recon_height}")
    
    # Fallback: try to load from chunk metadata if we're still missing data
    if intrinsics is None or recon_width is None:
        print("Checking chunk metadata for missing data...")
        first_metadata = torch.load(
            os.path.join(sorted(glob.glob(os.path.join(output_dir, "chunk_*")))[0], "metadata.pt"),
            map_location="cpu",
            weights_only=False
        )
        
        if recon_width is None and 'image_shape' in first_metadata:
            recon_height, recon_width = first_metadata['image_shape']
            print(f"✓ Found reconstruction resolution in chunk metadata: {recon_width}x{recon_height}")
        
        if intrinsics is None and 'intrinsics' in first_metadata:
            intrinsics = first_metadata['intrinsics']
            if isinstance(intrinsics, torch.Tensor):
                intrinsics = intrinsics.cpu().numpy()
            if intrinsics.ndim == 3:
                intrinsics = intrinsics[0]
            print(f"✓ Found intrinsics in chunk metadata")
    
    # Last resort: use default intrinsics
    if intrinsics is None:
        print("Warning: No intrinsics found, using default")
        if recon_width and recon_height:
            # Create default intrinsics based on reconstruction resolution
            fx = fy = max(recon_width, recon_height)
            cx, cy = recon_width / 2, recon_height / 2
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        else:
            intrinsics = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    
    print(f"Using intrinsics:\n{intrinsics}")
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video: {video_width}x{video_height} @ {fps} fps, {total_frames} frames")
    
    # Determine output resolution (use reconstruction resolution if available)
    if recon_width and recon_height:
        output_width, output_height = recon_width, recon_height
        needs_resize = (video_width != recon_width or video_height != recon_height)
        if needs_resize:
            print(f"Will resize frames from {video_width}x{video_height} to {output_width}x{output_height}")
    else:
        output_width, output_height = video_width, video_height
        needs_resize = False
        print("Warning: No reconstruction resolution found, using original video resolution")
    
    # Seek to start frame
    if min_frame_pos > 0:
        print(f"Seeking to start frame {min_frame_pos}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame_pos)
    
    # Setup output video
    if output_video_path is None:
        output_video_path = os.path.join(output_dir, "trajectory_overlay.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    
    print(f"Rendering trajectory overlay to: {output_video_path}")
    print(f"Output resolution: {output_width}x{output_height}")
    
    # Process each frame
    frame_idx = min_frame_pos  # Start from the first reconstructed frame
    num_frames_to_process = max_frame_pos - min_frame_pos + 1
    pbar = tqdm(total=num_frames_to_process, desc="Rendering")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > max_frame_pos:
            break
        
        # Resize frame if needed
        if needs_resize:
            frame = cv2.resize(frame, (output_width, output_height))
        
        # Check if this frame has a trajectory pose
        if frame_idx in frame_to_traj_idx:
            traj_idx = frame_to_traj_idx[frame_idx]
            
            # Get current pose
            current_pose = None
            pose_count = 0
            for chunk in ref_chunk_data:
                if traj_idx < pose_count + len(chunk['poses']):
                    current_pose = chunk['poses'][traj_idx - pose_count]
                    break
                pose_count += len(chunk['poses'])
            
            if current_pose is not None:
                if comparison_mode and sync_data:
                    # Comparison mode: draw both trajectories
                    # Find corresponding synchronized index
                    sync_idx_in_list = None
                    corresponding_query_idx = None
                    
                    if traj_idx in ref_sync_idx:
                        list_idx = ref_sync_idx.index(traj_idx)
                        sync_idx_in_list = list_idx
                        corresponding_query_idx = query_sync_idx[list_idx]
                    
                    # Project both trajectories
                    ref_traj_2d, ref_valid = project_trajectory_to_image(
                        ref_traj_full, current_pose, intrinsics,
                        current_idx=traj_idx, max_distance=max_distance
                    )
                    
                    query_traj_2d, query_valid = project_trajectory_to_image(
                        query_traj_full, current_pose, intrinsics,
                        current_idx=None, max_distance=max_distance
                    )
                    
                    # Get comparison metrics
                    if sync_idx_in_list is not None:
                        time_diff = time_diffs[sync_idx_in_list]
                        distance = distances[sync_idx_in_list]
                        ref_ahead = time_diff < 0  # Negative means ref is ahead in time
                    else:
                        time_diff = 0
                        distance = 0
                        ref_ahead = True
                    
                    # Get current normal
                    current_normal = trajectory_normals[traj_idx] if traj_idx < len(trajectory_normals) else None
                    
                    # Get query rider position and normal for ground circle
                    query_pos_3d = None
                    query_normal = None
                    if corresponding_query_idx is not None and corresponding_query_idx < len(query_traj_full):
                        query_pos_3d = query_traj_full[corresponding_query_idx]
                        if corresponding_query_idx < len(query_trajectory_normals):
                            query_normal = query_trajectory_normals[corresponding_query_idx]
                    
                    # Draw comparison overlay
                    frame = draw_comparison_overlay(
                        frame, ref_traj_2d, query_traj_2d, ref_valid, query_valid,
                        time_diff, distance, ref_ahead,
                        current_ref_idx=traj_idx, current_query_idx=corresponding_query_idx,
                        ref_normal=current_normal, intrinsics=intrinsics, current_ref_pose=current_pose,
                        query_pos_3d=query_pos_3d, query_normal=query_normal
                    )
                else:
                    # Single trajectory mode
                    traj_2d, valid_mask = project_trajectory_to_image(
                        trajectory_3d, current_pose, intrinsics, 
                        current_idx=traj_idx, 
                        max_distance=max_distance
                    )
                    
                    # Get current normal
                    current_normal = trajectory_normals[traj_idx] if traj_idx < len(trajectory_normals) else None
                    
                    # Draw trajectory and normal
                    frame = draw_trajectory_on_frame(
                        frame, traj_2d, valid_mask, traj_idx,
                        current_normal=current_normal,
                        intrinsics=intrinsics,
                        current_pose=current_pose
                    )
 
        cv2.imshow("Trajectory Overlay", frame)
        cv2.waitKey(30)
        # Write frame
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Trajectory overlay saved to: {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render SLAM trajectory onto video with optional comparison")
    
    # Reference trajectory
    parser.add_argument("--video", type=str,  default="/home/steffen/Data/GPStrava/Nina/Reference/nina1.MP4", 
                       help="Path to reference video")
    parser.add_argument("--output_dir", type=str,  default="/home/steffen/Data/GPStrava/Nina/Reference/nina1/_slam_results",
                       help="Directory containing reference SLAM reconstruction (_slam_results)")
    parser.add_argument("--stride", type=int, default=2,
                       help="Frame stride used during reference SLAM (for pose interpolation, default: 1)")
    
    # Query trajectory (optional, for comparison)
    parser.add_argument("--query_output_dir", type=str, default="/home/steffen/Data/GPStrava/Nina/Query/nina2/_slam_results",
                       help="Optional: Directory containing query SLAM reconstruction for comparison")
    parser.add_argument("--query_stride", type=int, default=2,
                       help="Frame stride used during query SLAM (default: 1)")
    
    # Output
    parser.add_argument("--output_video", type=str, default=None,
                       help="Path for output video (default: output_dir/trajectory_overlay.mp4)")
    
    # Visualization options
    parser.add_argument("--max_distance", type=float, default=15.0,
                       help="Maximum distance in meters to show trajectory (default: 15.0)")
    parser.add_argument("--no_smooth", action='store_true',
                       help="Disable trajectory smoothing")
    parser.add_argument("--no_ground_projection", action='store_true',
                       help="Disable ground plane projection")
    
    args = parser.parse_args()
    
    render_trajectory_to_video(
        args.video, args.output_dir, args.output_video, 
        smooth=not args.no_smooth,
        project_to_ground=not args.no_ground_projection,
        max_distance=args.max_distance,
        stride=args.stride,
        query_output_dir=args.query_output_dir,
        query_stride=args.query_stride
    )

