import torch
import glob
import os
from typing import List
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp
from slam_config import SLAMConfig


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
        Also interpolates camera poses if a stride was used during processing.
        """
        all_downsampled_pts = []
        all_downsampled_colors = []
        all_camera_poses = []
        all_normals = []
        all_plane_ds = []
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
                    
                    # De-duplicate poses based on image path or frame position
                    if 'image_paths' in metadata:
                        paths = metadata['image_paths']
                        for path, pose in zip(paths, poses):
                            if path not in processed_frames:
                                all_camera_poses.append(pose)
                                processed_frames.add(path)
                    elif 'frame_pos' in metadata:
                        frame_positions = metadata['frame_pos']
                        for frame_pos, pose in zip(frame_positions, poses):
                            if frame_pos not in processed_frames:
                                all_camera_poses.append(pose)
                                processed_frames.add(frame_pos)
                    else:
                        # Fallback for old metadata without paths or frame_pos
                        all_camera_poses.extend(poses)

                    # --- Load Normals ---
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
        final_poses_for_trajectory = [pose[0].numpy() for pose in all_camera_poses]
        
        # Apply alignment to trajectory poses
        if alignment_transform is not None:
            final_poses_for_trajectory = [alignment_transform @ pose for pose in final_poses_for_trajectory]

        camera_positions = [pose[:3, 3] for pose in final_poses_for_trajectory]
        lines = [[i, i + 1] for i in range(len(camera_positions) - 1)]
        trajectory = o3d.geometry.LineSet()
        trajectory.points = o3d.utility.Vector3dVector(camera_positions)
        trajectory.lines = o3d.utility.Vector2iVector(lines)
        trajectory.colors = o3d.utility.Vector3dVector([trajectory_color for _ in range(len(lines))])
        
        # Create Normal Geometries
        normal_viz = create_normal_geometries(all_camera_poses, all_normals, all_plane_ds)
        
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
