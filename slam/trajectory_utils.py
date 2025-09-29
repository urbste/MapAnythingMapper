import os
import numpy as np
import open3d as o3d
import torch
from typing import List, Dict


def save_query_trajectory_and_normals(database: List[Dict], database_path: str):
    if not database:
        return

    camera_positions = []
    per_frame_normals = []
    for entry in database:
        pose = entry['camera_poses']
        if isinstance(pose, torch.Tensor):
            pose_np = pose[0].detach().cpu().numpy()
        else:
            pose_np = pose[0]
        camera_positions.append(pose_np[:3, 3])
        nrm = entry.get('ground_normal', None)
        if isinstance(nrm, torch.Tensor):
            nrm = nrm.detach().cpu().numpy()
        per_frame_normals.append(nrm)

    if len(camera_positions) < 2:
        return

    steps = [np.linalg.norm(camera_positions[i+1] - camera_positions[i]) for i in range(len(camera_positions)-1)]
    median_step = np.median(steps) if steps else 1.0
    normal_length = float(max(0.25 * median_step, 0.1))

    # Trajectory dense points
    num_points_per_segment = 100
    traj_pts = []
    traj_cols = []
    traj_color = np.array([0.0, 0.0, 1.0])
    for i in range(len(camera_positions) - 1):
        start = camera_positions[i]
        end = camera_positions[i + 1]
        seg_points = np.linspace(start, end, num_points_per_segment)
        seg_colors = np.tile(traj_color, (num_points_per_segment, 1))
        traj_pts.append(seg_points)
        traj_cols.append(seg_colors)

    if not traj_pts:
        return

    traj_pts = np.concatenate(traj_pts, axis=0)
    traj_cols = np.concatenate(traj_cols, axis=0)

    out_dir = os.path.dirname(database_path) if database_path else "."
    os.makedirs(out_dir, exist_ok=True)

    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(traj_pts)
    traj_pcd.colors = o3d.utility.Vector3dVector(traj_cols)
    traj_path = os.path.join(out_dir, "query_trajectory_only.ply")
    o3d.io.write_point_cloud(traj_path, traj_pcd)

    # Normals
    normal_pts = []
    normal_cols = []
    normal_color = np.array([0.0, 1.0, 0.0])
    for i, pos in enumerate(camera_positions):
        nrm = per_frame_normals[i] if i < len(per_frame_normals) else None
        if nrm is None:
            continue
        end = pos + nrm * normal_length
        seg_points = np.linspace(pos, end, 20)
        seg_colors = np.tile(normal_color, (seg_points.shape[0], 1))
        normal_pts.append(seg_points)
        normal_cols.append(seg_colors)

    if normal_pts:
        normals_pcd = o3d.geometry.PointCloud()
        normals_pcd.points = o3d.utility.Vector3dVector(np.concatenate(normal_pts, axis=0))
        normals_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(normal_cols, axis=0))
        normals_path = os.path.join(out_dir, "query_normals.ply")
        o3d.io.write_point_cloud(normals_path, normals_pcd)

