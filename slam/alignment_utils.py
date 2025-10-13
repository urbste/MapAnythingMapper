"""
Utilities for aligning reconstructions, including similarity transforms.
"""

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def align_reconstruction_to_gravity(poses, gravity_vector):
    """
    Aligns a reconstruction so that gravity_vector points down (-Z).
    
    Args:
        poses: List of 4x4 camera-to-world pose matrices
        gravity_vector: 3D vector representing the gravity direction
    
    Returns:
        aligned_poses: List of transformed poses
        transform: 4x4 alignment transform applied
    """
    gravity_vector = gravity_vector / (np.linalg.norm(gravity_vector) + 1e-8)
    target_down = np.array([0.0, 0.0, -1.0])
    
    v = np.cross(gravity_vector, target_down)
    s = np.linalg.norm(v)
    c = np.dot(gravity_vector, target_down)
    
    if s < 1e-6:
        if c > 0:
            rotation_matrix = np.eye(3)
        else:
            rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    
    aligned_poses = []
    for pose in poses:
        if isinstance(pose, torch.Tensor):
            pose_np = pose.cpu().numpy()
            aligned = transform @ pose_np
            aligned_poses.append(torch.from_numpy(aligned).to(pose.device))
        else:
            aligned_poses.append(transform @ pose)
    
    return aligned_poses, transform


def align_reconstruction_to_gps_direction(poses, gps_vectors, target_direction=None):
    """
    Aligns reconstruction to match GPS heading direction.
    
    Args:
        poses: List of 4x4 camera-to-world pose matrices
        gps_vectors: List of 3D GPS displacement vectors in NED coordinates
        target_direction: Optional 3D target direction vector
    
    Returns:
        aligned_poses: List of transformed poses
        transform: 4x4 alignment transform applied
    """
    if target_direction is None:
        target_direction = np.array([1.0, 0.0, 0.0])
    
    avg_direction = np.mean(gps_vectors, axis=0)
    avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-8)
    
    v = np.cross(avg_direction, target_direction)
    s = np.linalg.norm(v)
    c = np.dot(avg_direction, target_direction)
    
    if s < 1e-6:
        if c > 0:
            rotation_matrix = np.eye(3)
        else:
            rotation_matrix = -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    
    aligned_poses = []
    for pose in poses:
        if isinstance(pose, torch.Tensor):
            pose_np = pose.cpu().numpy()
            aligned = transform @ pose_np
            aligned_poses.append(torch.from_numpy(aligned).to(pose.device))
        else:
            aligned_poses.append(transform @ pose)
    
    return aligned_poses, transform


def compute_similarity_transform(src_poses, dst_poses):
    """
    Computes a similarity transform (rotation, translation, scale) that aligns
    src_poses to dst_poses using Umeyama's algorithm.
    
    This uses multiple pose correspondences to estimate scale in addition to
    rotation and translation.
    
    Args:
        src_poses: List of N source 4x4 pose matrices (predicted poses)
        dst_poses: List of N destination 4x4 pose matrices (reference/ground truth poses)
    
    Returns:
        transform: 4x4 similarity transform matrix [sR | t; 0 0 0 1]
        scale: Estimated scale factor
    """
    assert len(src_poses) == len(dst_poses), "Must have equal number of source and destination poses"
    assert len(src_poses) >= 2, "Need at least 2 pose correspondences for similarity transform"
    
    # Extract camera centers (translation components)
    src_points = []
    dst_points = []
    
    for src_pose, dst_pose in zip(src_poses, dst_poses):
        if isinstance(src_pose, torch.Tensor):
            src_pose = src_pose.cpu().numpy()
        if isinstance(dst_pose, torch.Tensor):
            dst_pose = dst_pose.cpu().numpy()
        
        # Handle batch dimensions - squeeze to (4, 4) if needed
        if src_pose.ndim == 3:
            src_pose = src_pose.squeeze(0)
        if dst_pose.ndim == 3:
            dst_pose = dst_pose.squeeze(0)
        
        # Extract translation (camera center)
        src_points.append(src_pose[:3, 3])
        dst_points.append(dst_pose[:3, 3])
    
    src_points = np.array(src_points)  # (N, 3)
    dst_points = np.array(dst_points)  # (N, 3)
    
    # Debug: Print shapes
    print(f"  Debug: src_points shape: {src_points.shape}, dst_points shape: {dst_points.shape}")
    
    # Ensure we have the right shape
    if src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError(f"src_points should be (N, 3), got {src_points.shape}")
    if dst_points.ndim != 2 or dst_points.shape[1] != 3:
        raise ValueError(f"dst_points should be (N, 3), got {dst_points.shape}")
    
    # Umeyama's algorithm for similarity transform estimation
    # Reference: "Least-squares estimation of transformation parameters between two point patterns"
    
    # 1. Compute centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    
    # 2. Center the points
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # 3. Compute scale
    src_var = np.sum(src_centered ** 2) / len(src_points)
    
    # 4. Compute cross-covariance matrix
    H = src_centered.T @ dst_centered / len(src_points)
    
    # 5. Compute rotation using SVD
    U, S, Vt = np.linalg.svd(H)
    R_matrix = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T
    
    # 6. Compute scale (S is a 1D array of singular values, sum them)
    scale = np.sum(S) / src_var if src_var > 1e-8 else 1.0
    
    # 7. Compute translation
    t = dst_centroid - scale * R_matrix @ src_centroid
    
    # 8. Build 4x4 similarity transform
    transform = np.eye(4)
    transform[:3, :3] = scale * R_matrix
    transform[:3, 3] = t
    
    return transform, scale


def compute_rigid_transform_single_pose(src_pose, dst_pose):
    """
    Computes a rigid transform (no scale) from a single pose correspondence.
    This is the original method used in the code.
    
    Args:
        src_pose: 4x4 source pose matrix (predicted)
        dst_pose: 4x4 destination pose matrix (reference/ground truth)
    
    Returns:
        transform: 4x4 rigid transform matrix
    """
    if isinstance(src_pose, torch.Tensor):
        device = src_pose.device
        src_np = src_pose.cpu().numpy()
        dst_np = dst_pose.cpu().numpy() if isinstance(dst_pose, torch.Tensor) else dst_pose
        
        transform_np = dst_np @ np.linalg.inv(src_np)
        return torch.from_numpy(transform_np).float().to(device)
    else:
        return dst_pose @ np.linalg.inv(src_pose)
