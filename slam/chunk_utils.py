"""Utilities for saving and visualizing SLAM chunks and trajectories."""

import os
import glob
import torch
import numpy as np
import open3d as o3d
from typing import List, Dict, Optional
from scipy.spatial.transform import Rotation, Slerp


def save_chunk(
    chunk_idx: int,
    predictions: List[Dict],
    all_poses: List[torch.Tensor],
    all_paths: List[str],
    is_reconstructed_flags: List[bool],
    all_normals: List[Optional[np.ndarray]],
    all_plane_ds: List[Optional[float]],
    output_dir: str,
    max_depth: float,
    video_path: Optional[str] = None,
    frametimes_ns: Optional[List[int]] = None,
    frame_pos: Optional[List[int]] = None,
    intrinsics: Optional[np.ndarray] = None,
    image_shape: Optional[tuple] = None
):
    """
    Saves the processed data for a chunk to a .pt file.
    
    Args:
        chunk_idx: Index of the chunk
        predictions: List of prediction dictionaries from the model
        all_poses: List of camera poses
        all_paths: List of image paths/identifiers
        is_reconstructed_flags: List of boolean flags indicating if frame was reconstructed
        all_normals: List of ground normal vectors (can be None)
        all_plane_ds: List of plane distances (can be None)
        output_dir: Directory to save chunk data
        max_depth: Maximum depth for point cloud filtering
        video_path: Path to video file (if processing video)
        frametimes_ns: List of frame timestamps in nanoseconds (optional)
        frame_pos: List of frame positions in video (optional)
    """
    if not predictions:
        return

    # Points, colors, and confidences are only from the *reconstructed* views
    all_pts3d = torch.cat([p['pts3d'].reshape(-1, 3) for p in predictions], dim=0)
    all_pts3d_cam = torch.cat([p['pts3d_cam'].reshape(-1, 3) for p in predictions], dim=0)
    all_colors = torch.cat([p['img_no_norm'].reshape(-1, 3) for p in predictions], dim=0)
    all_masks = torch.cat([p['mask'].reshape(-1) for p in predictions], dim=0)

    # Apply masks to filter points before saving
    all_pts3d = all_pts3d[all_masks]
    all_pts3d_cam = all_pts3d_cam[all_masks]
    all_colors = all_colors[all_masks]

    # Apply depth filter
    depth_mask = all_pts3d_cam[:, 2] < max_depth
    all_pts3d = all_pts3d[depth_mask]
    all_colors = all_colors[depth_mask]
    
    chunk_dir = os.path.join(output_dir, f"chunk_{chunk_idx:04d}")
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Save point cloud data as a compressed numpy array for efficiency
    pcd_path = os.path.join(chunk_dir, "point_cloud.npz")
    np.savez_compressed(
        pcd_path,
        pts3d=all_pts3d.cpu().numpy(),
        colors=(all_colors * 255).byte().cpu().numpy()
    )

    # Save metadata separately
    metadata = {
        'camera_poses': [p.cpu() for p in all_poses],
        'is_reconstructed': is_reconstructed_flags,
        'ground_normals': all_normals,
        'plane_ds': all_plane_ds,
    }
    if not video_path:
        metadata['image_paths'] = all_paths
    if frametimes_ns:
        metadata['frametimes_ns'] = frametimes_ns
    if frame_pos:
        metadata['frame_pos'] = frame_pos
    if intrinsics is not None:
        metadata['intrinsics'] = intrinsics
    if image_shape is not None:
        metadata['image_shape'] = image_shape
        
    torch.save(metadata, os.path.join(chunk_dir, "metadata.pt"))
    print(f"Saved chunk data for chunk {chunk_idx} to {chunk_dir}")


def save_chunk_with_interpolation(
    chunk_idx: int,
    chunk_start_idx: int,
    chunk_end_idx: int,
    all_processed_paths: List[str],
    predictions: List[Dict],
    pred_paths: List[str],
    all_normals: List[Optional[np.ndarray]],
    all_plane_ds: List[Optional[float]],
    output_dir: str,
    max_depth: float,
    stride: int,
    video_path: Optional[str] = None,
    frametimes_ns: Optional[List[int]] = None,
    frame_pos_buffer: Optional[List[int]] = None,
    intrinsics: Optional[np.ndarray] = None,
    image_shape: Optional[tuple] = None
):
    """
    Saves keyframe poses and normals for a chunk without interpolation.
    Interpolation will be done in the visualization code based on frame IDs and stride.
    
    Args:
        chunk_idx: Index of the chunk
        chunk_start_idx: Start index in the full processed paths list
        chunk_end_idx: End index in the full processed paths list
        all_processed_paths: Full list of all processed frame paths
        predictions: List of prediction dictionaries from the model
        pred_paths: Paths corresponding to the predictions
        all_normals: List of ground normal vectors
        all_plane_ds: List of plane distances
        output_dir: Directory to save chunk data
        max_depth: Maximum depth for point cloud filtering
        stride: Frame stride used during processing
        video_path: Path to video file (if processing video)
        frametimes_ns: List of frame timestamps in nanoseconds
        frame_pos_buffer: List of frame positions in video
    """
    # Save only keyframe poses (no interpolation)
    poses_to_save = [p['camera_poses'] for p in predictions]
    paths_to_save = pred_paths
    is_reconstructed = [True] * len(predictions)  # All saved poses are keyframes
    
    print(f"  Chunk {chunk_idx}: Saving {len(poses_to_save)} keyframe poses (stride={stride})")
    
    frametimes_ns_to_save = []
    frame_pos_to_save = []
    if video_path and frametimes_ns and frame_pos_buffer:
        # When processing a video, paths are identifiers like "frame_123"
        # Extract the frame index from each path and get corresponding metadata
        for path in paths_to_save:
            try:
                base = os.path.basename(path)
                name, _sep, _ext = base.partition('.')
                digits = ''.join([c for c in name if c.isdigit()])
                frame_idx_from_path = int(digits)
                
                # The frame index in the path corresponds to the index in frame_pos_buffer/frametimes_ns
                if frame_idx_from_path < len(frame_pos_buffer) and frame_idx_from_path < len(frametimes_ns):
                    frame_pos_to_save.append(frame_pos_buffer[frame_idx_from_path])
                    frametimes_ns_to_save.append(frametimes_ns[frame_idx_from_path])
                else:
                    print(f"Warning: frame index {frame_idx_from_path} out of bounds for buffers (len={len(frame_pos_buffer)})")
                    frame_pos_to_save.append(frame_idx_from_path)
                    frametimes_ns_to_save.append(0)
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse frame index from path '{path}'. Skipping metadata. Error: {e}")

    # Save stride information so visualization code can interpolate
    save_chunk(
        chunk_idx, 
        predictions, 
        poses_to_save, 
        paths_to_save, 
        is_reconstructed, 
        all_normals,  # Save only keyframe normals
        all_plane_ds,  # Save only keyframe plane_ds
        output_dir,
        max_depth,
        video_path,
        frametimes_ns_to_save, 
        frame_pos_to_save,
        intrinsics,
        image_shape
    )


def save_chunks_for_visualization(output_dir: str, alignment_transform: Optional[torch.Tensor]):
    """
    Saves the point clouds for each frame for later visualization.
    Applies alignment transform to all saved chunks.
    
    Args:
        output_dir: Directory containing chunk subdirectories
        alignment_transform: 4x4 transformation matrix to apply (can be None)
    """
    print("Saving point clouds for visualization...")
    chunk_dirs = sorted(glob.glob(os.path.join(output_dir, "chunk_*")))
    transform_np = alignment_transform.cpu().numpy() if alignment_transform is not None else np.eye(4)

    for chunk_dir in chunk_dirs:
        try:
            metadata_file = os.path.join(chunk_dir, "metadata.pt")
            pcd_file = os.path.join(chunk_dir, "point_cloud.npz")

            if not (os.path.exists(metadata_file) and os.path.exists(pcd_file)):
                continue
            
            # Load data
            metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
            pcd_data = np.load(pcd_file)

            # --- Transform Poses ---
            original_poses = metadata['camera_poses']
            aligned_poses = [torch.from_numpy(transform_np) @ pose for pose in original_poses]
            metadata['camera_poses'] = aligned_poses

            # --- Transform Normals ---
            if 'ground_normals' in metadata:
                R_align = transform_np[:3, :3]
                original_normals = [n.numpy() if isinstance(n, torch.Tensor) else n for n in metadata['ground_normals']]
                aligned_normals = [R_align @ n if n is not None else None for n in original_normals]
                metadata['ground_normals'] = aligned_normals
            
            metadata['is_aligned'] = True
            torch.save(metadata, metadata_file)

            # --- Transform Points ---
            pts3d = pcd_data['pts3d']
            pts_hom = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
            aligned_pts3d = (transform_np @ pts_hom.T).T[:, :3]

            np.savez_compressed(
                pcd_file,
                pts3d=aligned_pts3d,
                colors=pcd_data['colors']
            )
        except Exception as e:
            print(f"Warning: Could not align chunk {os.path.basename(chunk_dir)}: {e}")
    print("Finished aligning chunk files.")


def save_debug_chunk_visualization(chunk_idx: int, predictions: List[Dict], output_dir: str, max_depth: float):
    """
    Saves the point cloud and trajectory for a single chunk for debugging.
    
    Args:
        chunk_idx: Index of the chunk
        predictions: List of prediction dictionaries from the model
        output_dir: Base output directory (debug_chunks will be created inside)
        max_depth: Maximum depth for point cloud filtering
    """
    if not predictions:
        return

    debug_dir = os.path.join(output_dir, "debug_chunks")
    os.makedirs(debug_dir, exist_ok=True)

    # --- Create Point Cloud ---
    all_pts3d = torch.cat([p['pts3d'].reshape(-1, 3) for p in predictions], dim=0)
    all_pts3d_cam = torch.cat([p['pts3d_cam'].reshape(-1, 3) for p in predictions], dim=0)
    all_colors = torch.cat([p['img_no_norm'].reshape(-1, 3) for p in predictions], dim=0)
    all_masks = torch.cat([p['mask'].reshape(-1) for p in predictions], dim=0)

    # Apply confidence filter
    all_pts3d = all_pts3d[all_masks]
    all_pts3d_cam = all_pts3d_cam[all_masks]
    all_colors = all_colors[all_masks]

    # Apply depth filter (using world Z coordinate)
    depth_mask = all_pts3d_cam[:, 2] < max_depth
    all_pts3d = all_pts3d[depth_mask]
    all_colors = all_colors[depth_mask]

    if all_pts3d.shape[0] == 0:
        print(f"Debug chunk {chunk_idx}: No points left after filtering.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts3d.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(all_colors.cpu().numpy())
    
    pcd_path = os.path.join(debug_dir, f"debug_chunk_{chunk_idx:04d}_pcd.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)

    # --- Estimate and Visualize Ground Normal ---
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        ground_normal = np.array([a, b, c])
        
        # Ensure the normal points upwards (positive Y)
        if ground_normal[1] < 0:
            ground_normal = -ground_normal

        print(f"Debug chunk {chunk_idx}: Estimated ground normal: {ground_normal}")

        # Get a point on the plane to draw the normal from (using the pcd center)
        center_point = pcd.get_center()
        
        # Create a line representing the normal
        normal_line = o3d.geometry.LineSet()
        normal_line.points = o3d.utility.Vector3dVector([center_point, center_point + ground_normal * 2.0]) # 2-meter long line
        normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        normal_line.colors = o3d.utility.Vector3dVector([[1, 0, 1]]) # Magenta

        normal_path = os.path.join(debug_dir, f"debug_chunk_{chunk_idx:04d}_normal.ply")
        o3d.io.write_line_set(normal_path, normal_line)

    except Exception as e:
        print(f"Debug chunk {chunk_idx}: Could not estimate ground normal. {e}")

    # --- Create Trajectory ---
    camera_poses = [p['camera_poses'][0].cpu().numpy() for p in predictions]
    camera_positions = [pose[:3, 3] for pose in camera_poses]
    lines = [[i, i + 1] for i in range(len(camera_positions) - 1)]
    
    if not lines:
        print(f"Debug chunk {chunk_idx}: Not enough poses for a trajectory.")
        return

    # Save trajectory as a dense point cloud PLY for easy visualization
    points_on_lines = []
    for i in range(len(camera_positions) - 1):
        start = camera_positions[i]
        end = camera_positions[i+1]
        # Add 100 points per line segment
        points_on_lines.extend(np.linspace(start, end, 100))
    
    if not points_on_lines:
        return
        
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(points_on_lines)
    traj_pcd.paint_uniform_color([1, 0, 0]) # Red

    traj_path = os.path.join(debug_dir, f"debug_chunk_{chunk_idx:04d}_traj.ply")
    o3d.io.write_point_cloud(traj_path, traj_pcd)

    print(f"Saved debug visualization for chunk {chunk_idx} to {debug_dir}")

