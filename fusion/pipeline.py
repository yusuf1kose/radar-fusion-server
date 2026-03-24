import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from typing import List, Optional
from ingestion.log_replayer import RadarFrame


def frames_to_pointcloud(frames, z_min=-2.0, z_max=0.0):
    all_points = []
    for frame in frames:
        if frame is None:
            continue
        for pt in frame.point_cloud:
            if z_min <= pt[2] <= z_max:
                all_points.append(pt)
    pcd = o3d.geometry.PointCloud()
    if all_points:
        pcd.points = o3d.utility.Vector3dVector(np.array(all_points, dtype=np.float64))
    return pcd


def dbscan_filter(pcd, eps=0.15, min_samples=3):
    if len(pcd.points) < min_samples:
        return pcd
    points = np.asarray(pcd.points)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    mask = labels >= 0
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(points[mask])
    return filtered


def fuse_pointclouds(pcds):
    non_empty = [p for p in pcds if len(p.points) > 0]
    if not non_empty:
        return o3d.geometry.PointCloud()
    if len(non_empty) == 1:
        return non_empty[0]
    all_pts = np.vstack([np.asarray(p.points) for p in non_empty])
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(all_pts)
    return merged


def project_to_2d(pcd):
    if len(pcd.points) == 0:
        return np.array([])
    points = np.asarray(pcd.points)
    return points[:, :2]