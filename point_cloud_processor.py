import torch
import glob
import os
from typing import List, Tuple, Optional
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp
from slam_config import SLAMConfig


def interpolate_poses_and_normals(
    keyframe_poses: List[torch.Tensor], 
    keyframe_normals: List[np.ndarray],
    keyframe_plane_ds: List[float],
    keyframe_paths: List[str],
    stride: int
) -> Tuple[List[torch.Tensor], List[np.ndarray], List[float]]:
    """
    Interpolates poses and normals between keyframes based on detected stride.
    
    Args:
        keyframe_poses: List of poses for keyframes
        keyframe_normals: List of normals for keyframes
        keyframe_plane_ds: List of plane distances for keyframes
        keyframe_paths: List of frame paths/identifiers for keyframes
        stride: Detected stride between keyframes
        
    Returns:
        Tuple of (interpolated_poses, interpolated_normals, interpolated_plane_ds)
    """
    if stride <= 1 or len(keyframe_poses) < 2:
        # No interpolation needed
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
            # If we can't parse indices, return keyframes as-is
            print(f"Warning: Could not parse frame index from {path}, skipping interpolation")
            return keyframe_poses, keyframe_normals, keyframe_plane_ds
    
    # Convert poses to numpy for interpolation
    keyframe_poses_np = [p[0].numpy() if p.ndim == 3 else p.numpy() for p in keyframe_poses]
    
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
        interpolated_poses.append(torch.from_numpy(pose).float().unsqueeze(0))
    
    # Interpolate normals if available
    interpolated_normals = []
    if keyframe_normals and len(keyframe_normals) > 0 and keyframe_normals[0] is not None:
        try:
            # Convert normals to numpy
            key_normals_np = []
            for n in keyframe_normals:
                if isinstance(n, torch.Tensor):
                    key_normals_np.append(n.cpu().numpy())
                elif isinstance(n, np.ndarray):
                    key_normals_np.append(n)
                else:
                    key_normals_np.append(np.array(n))
            
            key_normals_array = np.array(key_normals_np)
            key_normals_normalized = key_normals_array / (np.linalg.norm(key_normals_array, axis=1, keepdims=True) + 1e-8)
            
            # Convert normals to rotations for SLERP
            reference = np.array([0.0, 0.0, 1.0])
            rotations = []
            for normal in key_normals_normalized:
                v = np.cross(reference, normal)
                s = np.linalg.norm(v)
                c = np.dot(reference, normal)
                
                if s < 1e-6:  # Vectors are parallel
                    if c > 0:
                        rotations.append(Rotation.identity())
                    else:
                        rotations.append(Rotation.from_euler('x', 180, degrees=True))
                else:
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
                    rotations.append(Rotation.from_matrix(R))
            
            # Interpolate rotations
            normal_slerp = Slerp(keyframe_indices, Rotation.from_quat([r.as_quat() for r in rotations]))
            interpolated_rotations = normal_slerp(interp_indices)
            
            # Apply rotations to reference vector
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
            key_plane_ds_np = []
            for d in keyframe_plane_ds:
                if isinstance(d, torch.Tensor):
                    key_plane_ds_np.append(d.cpu().item() if d.numel() == 1 else d.cpu().numpy())
                else:
                    key_plane_ds_np.append(d)
            
            key_plane_ds_array = np.array(key_plane_ds_np)
            interpolated_plane_ds = np.interp(interp_indices, keyframe_indices, key_plane_ds_array).tolist()
            
            print(f"  Interpolated {len(interpolated_plane_ds)} plane_d values from {len(keyframe_plane_ds)} keyframes")
        except Exception as e:
            print(f"  Warning: Could not interpolate plane_d ({e}), using keyframe values")
            interpolated_plane_ds = keyframe_plane_ds
    else:
        interpolated_plane_ds = keyframe_plane_ds
    
    print(f"  Interpolated {len(interpolated_poses)} poses from {len(keyframe_poses)} keyframes (stride={stride})")
    return interpolated_poses, interpolated_normals, interpolated_plane_ds


def create_normal_geometries(camera_poses, normals, plane_ds, line_length=1.0, color=[0, 1, 0]):
    """Creates a LineSet geometry for visualizing normal vectors."""
    lines = []
    points = []
    colors = []
    
    if normals is None or camera_poses is None:
        return None

    for i, pose in enumerate(camera_poses):
        if i >= len(normals) or normals[i] is None:
            continue
            
        cam_pos = pose[0, :3, 3].numpy()
        normal = normals[i]
        if isinstance(normal, torch.Tensor):
            normal = normal.numpy()
        
        # Start point is the camera position
        start_point = cam_pos
        # End point is along the normal vector
        end_point = start_point + normal * line_length
        
        points.append(start_point)
        points.append(end_point)
        lines.append([len(points)-2, len(points)-1])
        colors.append(color)

    if not points:
        return None

    normal_viz = o3d.geometry.LineSet()
    normal_viz.points = o3d.utility.Vector3dVector(points)
    normal_viz.lines = o3d.utility.Vector2iVector(lines)
    normal_viz.colors = o3d.utility.Vector3dVector(colors)
    return normal_viz


class PointCloudProcessor:
    def __init__(self):
        pass

    def generate_pcd_and_trajectory(self, chunk_dirs: List[str], trajectory_color: List[float], config: 'SLAMConfig'):
        """
        Generates a combined point cloud and camera trajectory from processed chunk directories.
        Interpolates camera poses and normals if a stride was used during processing.
        """
        all_downsampled_pts = []
        all_downsampled_colors = []
        all_camera_poses = []
        all_normals = []
        all_plane_ds = []
        all_paths = []
        processed_frames = set() # Use a set for efficient de-duplication of frames/images

        print("Loading and processing chunks...")
        
        # --- Determine the base directory for results (e.g., to find alignment file) ---
        # We assume all chunk_dirs are inside a single parent experiment folder.
        # e.g., /path/to/results/chunks_0, /path/to/results/chunks_1 -> base is /path/to/results
        base_dir = ""
        if chunk_dirs:
            base_dir = os.path.dirname(chunk_dirs[0])
        
        is_pre_aligned = False
        # Check the first chunk's metadata to see if the data is already aligned.
        first_metadata_path = sorted(glob.glob(os.path.join(chunk_dirs[0], "*", "metadata.pt")))
        if first_metadata_path:
            try:
                first_metadata = torch.load(first_metadata_path[0], map_location="cpu", weights_only=False)
                if first_metadata.get('is_aligned', False):
                    is_pre_aligned = True
                    print("Chunk data is already aligned. PointCloudProcessor will not apply an additional transform.")
            except Exception:
                pass # Fail silently if metadata can't be read

        for chunk_dir in chunk_dirs:
            for metadata_file in sorted(glob.glob(os.path.join(chunk_dir, "*", "metadata.pt"))):
                try:
                    pcd_file = os.path.join(os.path.dirname(metadata_file), "point_cloud.npz")
                    if not os.path.exists(pcd_file):
                        continue

                    metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
                    pcd_data = np.load(pcd_file)
                    
                    pts3d = pcd_data['pts3d']
                    colors = pcd_data['colors']
                    poses = metadata['camera_poses']
                    normals = metadata.get('ground_normals', [])
                    plane_ds = metadata.get('plane_ds', [])
                    
                    # De-duplicate poses based on image path or frame position
                    if 'image_paths' in metadata:
                        paths = metadata['image_paths']
                        for i, (path, pose) in enumerate(zip(paths, poses)):
                            if path not in processed_frames:
                                all_camera_poses.append(pose)
                                all_paths.append(path)
                                if i < len(normals):
                                    all_normals.append(normals[i])
                                if i < len(plane_ds):
                                    all_plane_ds.append(plane_ds[i])
                                processed_frames.add(path)
                    elif 'frame_pos' in metadata:
                        frame_positions = metadata['frame_pos']
                        paths = metadata.get('image_paths', [f"frame_{fp}" for fp in frame_positions])
                        for i, (frame_pos, pose) in enumerate(zip(frame_positions, poses)):
                            if frame_pos not in processed_frames:
                                all_camera_poses.append(pose)
                                all_paths.append(paths[i] if i < len(paths) else f"frame_{frame_pos}")
                                if i < len(normals):
                                    all_normals.append(normals[i])
                                if i < len(plane_ds):
                                    all_plane_ds.append(plane_ds[i])
                                processed_frames.add(frame_pos)
                    else:
                        # Fallback for old metadata without paths or frame_pos
                        all_camera_poses.extend(poses)
                        if 'ground_normals' in metadata:
                            all_normals.extend(metadata['ground_normals'])
                        if 'plane_ds' in metadata:
                            all_plane_ds.extend(metadata['plane_ds'])

                except Exception as e:
                    print(f"Could not load or process chunk files in {os.path.dirname(metadata_file)}: {e}")
                    continue

                # Voxel downsample per chunk to manage memory
                pcd_chunk = o3d.geometry.PointCloud()
                pcd_chunk.points = o3d.utility.Vector3dVector(pts3d)
                pcd_chunk.colors = o3d.utility.Vector3dVector(colors / 255.0)
                downsampled_pcd_chunk = pcd_chunk.voxel_down_sample(voxel_size=0.05)
                
                if len(downsampled_pcd_chunk.points) > 0:
                    all_downsampled_pts.append(np.asarray(downsampled_pcd_chunk.points))
                    all_downsampled_colors.append(np.asarray(downsampled_pcd_chunk.colors))
                
        if not all_camera_poses:
            print("No valid poses found to generate trajectory.")
            # Still try to return a point cloud if one was generated
            if not all_downsampled_pts:
                 return None, None, None
            else:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_downsampled_pts, axis=0))
                pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_downsampled_colors, axis=0))
                pcd = pcd.voxel_down_sample(voxel_size=0.02)
                return pcd, None, None

        if not all_downsampled_pts:
            print("No points to generate point cloud.")
            return None, None, None

        # --- Interpolate poses and normals if stride was used ---
        stride = config.stride if hasattr(config, 'stride') else 1
        if stride > 1 and len(all_camera_poses) > 1 and len(all_paths) > 0:
            print(f"Detected stride={stride}, performing interpolation...")
            all_camera_poses, all_normals, all_plane_ds = interpolate_poses_and_normals(
                all_camera_poses, all_normals, all_plane_ds, all_paths, stride
            )

        # --- Load and Apply Final Alignment Transform (if not pre-aligned) ---
        alignment_transform = None
        alignment_file = os.path.join(base_dir, "alignment.pt")
        if os.path.exists(alignment_file) and not is_pre_aligned:
            try:
                alignment_transform = torch.load(alignment_file, map_location="cpu", weights_only=False)
                print(f"Loaded and will apply alignment transform from {alignment_file}")
            except Exception as e:
                print(f"Could not load alignment file {alignment_file}: {e}")

        # Create Final Point Cloud
        final_pts = np.concatenate(all_downsampled_pts, axis=0)
        final_colors = np.concatenate(all_downsampled_colors, axis=0)

        # Apply alignment to point cloud
        if alignment_transform is not None:
            pts_hom = np.hstack((final_pts, np.ones((final_pts.shape[0], 1))))
            final_pts = (alignment_transform @ pts_hom.T).T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_pts)
        pcd.colors = o3d.utility.Vector3dVector(final_colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        print(f"Generated point cloud with {len(pcd.points)} points.")
        
        # Create Trajectory from the dense poses loaded directly from chunks
        final_poses_for_trajectory = [pose[0].numpy() if pose.ndim == 3 else pose.numpy() for pose in all_camera_poses]
        
        # Apply alignment to trajectory poses
        if alignment_transform is not None:
            final_poses_for_trajectory = [alignment_transform @ pose for pose in final_poses_for_trajectory]
            
            # Also apply alignment to normals
            if all_normals and len(all_normals) > 0:
                R_align = alignment_transform[:3, :3].numpy() if isinstance(alignment_transform, torch.Tensor) else alignment_transform[:3, :3]
                aligned_normals = []
                for n in all_normals:
                    if n is not None:
                        if isinstance(n, torch.Tensor):
                            n = n.cpu().numpy()
                        aligned_normals.append(R_align @ n)
                    else:
                        aligned_normals.append(None)
                all_normals = aligned_normals

        camera_positions = [pose[:3, 3] for pose in final_poses_for_trajectory]
        lines = [[i, i + 1] for i in range(len(camera_positions) - 1)]
        trajectory = o3d.geometry.LineSet()
        trajectory.points = o3d.utility.Vector3dVector(camera_positions)
        trajectory.lines = o3d.utility.Vector2iVector(lines)
        trajectory.colors = o3d.utility.Vector3dVector([trajectory_color for _ in range(len(lines))])
        
        # Create Normal Geometries (convert poses back to tensors for compatibility)
        camera_poses_for_normals = [torch.from_numpy(pose).float().unsqueeze(0) for pose in final_poses_for_trajectory]
        normal_viz = create_normal_geometries(camera_poses_for_normals, all_normals, all_plane_ds)
        
        return pcd, trajectory, normal_viz

    def save_ply(self, pcd, filename):
        """Saves a point cloud to a PLY file."""
        if pcd and not pcd.is_empty():
            o3d.io.write_point_cloud(filename, pcd)
            print(f"Saved point cloud to {filename}")

    def save_trajectory_as_ply(self, trajectory, filename, num_points_per_line=100):
        """Saves a trajectory (LineSet) as a colored point cloud PLY file."""
        if not trajectory or not trajectory.has_lines():
            print("Trajectory is empty, cannot save.")
            return

        points = np.asarray(trajectory.points)
        lines = np.asarray(trajectory.lines)
        colors = np.asarray(trajectory.colors)

        all_line_points = []
        all_line_colors = []

        for i, line in enumerate(lines):
            start_point = points[line[0]]
            end_point = points[line[1]]
            color = colors[i]
            
            # Create interpolated points along the line segment
            line_points = np.linspace(start_point, end_point, num_points_per_line)
            line_colors = np.tile(color, (num_points_per_line, 1))
            
            all_line_points.append(line_points)
            all_line_colors.append(line_colors)

        if not all_line_points:
            return

        # Create a new point cloud from the interpolated line points
        trajectory_pcd = o3d.geometry.PointCloud()
        trajectory_pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_line_points, axis=0))
        trajectory_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_line_colors, axis=0))
        
        self.save_ply(trajectory_pcd, filename)

    def visualize_geometries(self, geometries):
        """Displays a list of Open3D geometries."""
        print("Displaying visualization. Close the window to exit.")
        o3d.visualization.draw_geometries(geometries)
