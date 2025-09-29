import numpy as np
import open3d as o3d
import torch
from typing import List, Dict, Optional, Tuple


def estimate_ground_plane(
    predictions: List[Dict],
    distance_threshold: float = 0.2,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Estimate ground plane normal and plane distance using RANSAC on the merged
    points of a chunk's predictions.

    Returns (normal_tensor, plane_d_tensor) or (None, None) on failure.
    """
    try:
        if not predictions:
            return None, None

        # Build a combined point cloud for plane estimation
        chunk_pcd_pts = torch.cat([p['pts3d'].reshape(-1, 3) for p in predictions], dim=0)
        chunk_pcd_masks = torch.cat([p['mask'].reshape(-1) for p in predictions], dim=0)
        chunk_pcd_pts = chunk_pcd_pts[chunk_pcd_masks]

        if chunk_pcd_pts.shape[0] <= 1000:
            return None, None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(chunk_pcd_pts.cpu().numpy())
        pcd = pcd.voxel_down_sample(voxel_size=0.1)

        plane_model, _ = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        if normal[1] < 0:
            normal = -normal
            d = -d

        dev = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        ground_normal = torch.from_numpy(normal).float().to(dev)
        plane_d = torch.tensor(d).float().to(dev)
        return ground_normal, plane_d
    except Exception:
        return None, None


