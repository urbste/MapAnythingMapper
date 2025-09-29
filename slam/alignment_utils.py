import numpy as np
import torch
from typing import List, Dict
from scipy.spatial.transform import Rotation

from utils.math import get_z_zero_vec


def align_reconstruction_to_gravity(database: List[Dict], gravity, frametimes_ns, device) -> torch.Tensor:
    if gravity is None:
        return None
    if not database:
        return None

    grav_dir_w_target = np.array([0, 0, -1])
    all_rotvecs = []

    for entry in database:
        path = entry['image_path']
        try:
            # Extract frame index from image path like '.../frame_123.jpg'
            import os
            base = os.path.basename(path)
            name, _sep, _ext = base.partition('.')
            digits = ''.join([c for c in name if c.isdigit()])
            frame_idx = int(digits)
            if frame_idx >= len(frametimes_ns):
                continue
            tns = frametimes_ns[frame_idx]
            if tns not in gravity:
                continue
            g_c = np.array(gravity[tns])

            # Pose is T_w_c
            T_w_c = entry['camera_poses'].to(device)

            # We need R_c_w for the rotation
            T_c_w = torch.linalg.inv(T_w_c)
            R_c_w = T_c_w[0, :3, :3].cpu().numpy()

            # Target gravity vector in camera coordinates
            g_target_c = R_c_w @ grav_dir_w_target

            R_align_c, _ = Rotation.align_vectors(g_c, g_target_c)
            all_rotvecs.append(R_align_c.as_rotvec())
        except Exception:
            continue

    if not all_rotvecs:
        return None

    mean_rotvec = np.mean(np.array(all_rotvecs), axis=0)
    R_align = Rotation.from_rotvec(mean_rotvec).as_matrix()

    transform = torch.eye(4, device=device)
    transform[:3, :3] = torch.linalg.inv(torch.from_numpy(R_align).float())
    return transform


def align_reconstruction_to_gps_direction(database: List[Dict], gps_ned, frametimes_ns, zero_z: bool, device) -> torch.Tensor:
    if gps_ned is None:
        return None
    if len(database) < 2:
        return None

    try:
        # Get start and end poses and corresponding timestamps
        start_entry = database[0]
        end_entry = database[-1]

        import os
        def _idx_from_path(p):
            base = os.path.basename(p)
            name, _sep, _ext = base.partition('.')
            digits = ''.join([c for c in name if c.isdigit()])
            return int(digits)

        start_idx = _idx_from_path(start_entry['image_path'])
        end_idx = _idx_from_path(end_entry['image_path'])

        start_tns = frametimes_ns[start_idx]
        end_tns = frametimes_ns[end_idx]
        if start_tns not in gps_ned or end_tns not in gps_ned:
            return None

        # Get GPS positions
        gps_start = np.array(gps_ned[start_tns])
        gps_end = np.array(gps_ned[end_tns])

        # Get SLAM camera positions
        cam_pose_start = start_entry['camera_poses'][0].cpu().numpy()
        cam_pose_end = end_entry['camera_poses'][0].cpu().numpy()
        cam_position_start = cam_pose_start[:3, 3]
        cam_position_end = cam_pose_end[:3, 3]

        # Calculate direction vectors
        vec_gps = gps_end - gps_start
        vec_sfm = cam_position_end - cam_position_start
        if np.linalg.norm(vec_sfm) < 1e-6:
            return None

        # Scale from full 3D vectors
        scale = np.linalg.norm(vec_gps) / np.linalg.norm(vec_sfm)

        # Optionally constrain rotation to XY plane
        vec_gps_for_rot = get_z_zero_vec(vec_gps) if zero_z else vec_gps
        vec_sfm_for_rot = get_z_zero_vec(vec_sfm) if zero_z else vec_sfm

        # Rotation to align SLAM vector with GPS vector
        R_align, _ = Rotation.align_vectors(vec_sfm_for_rot, vec_gps_for_rot)
        R_align = R_align.as_matrix().T

        transform = torch.eye(4, device=device)
        transform[:3, :3] = scale * torch.from_numpy(R_align).float()
        return transform
    except Exception:
        return None


def get_up_vector_residuals(aligned_database: List[Dict], gravity, frametimes_ns) -> float:
    """Average residual between measured gravity and world's up-vector after alignment."""
    if gravity is None:
        return -1.0
    all_residuals = []
    grav_dir_w_target = np.array([0, 0, -1])
    for entry in aligned_database:
        path = entry.get('image_path')
        if not path:
            continue
        try:
            import os
            base = os.path.basename(path)
            name, _sep, _ext = base.partition('.')
            digits = ''.join([c for c in name if c.isdigit()])
            frame_idx = int(digits)
            if frame_idx >= len(frametimes_ns):
                continue
            tns = frametimes_ns[frame_idx]
            if tns not in gravity:
                continue
            g_c_measured = np.array(gravity[tns])

            T_w_c = entry['camera_poses'].cpu().numpy()
            R_w_c = T_w_c[0, :3, :3]
            g_c_aligned = R_w_c.T @ grav_dir_w_target
            all_residuals.append(np.sqrt((g_c_measured[2] - g_c_aligned[2]) ** 2))
        except Exception:
            continue
    if not all_residuals:
        return -1.0
    return float(np.mean(all_residuals))

