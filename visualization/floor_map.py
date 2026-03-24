import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
from ingestion.log_replayer import sync_frames
from fusion.pipeline import frames_to_pointcloud, dbscan_filter, fuse_pointclouds, project_to_2d
from sklearn.cluster import DBSCAN


def detect_objects(floor_pts: np.ndarray, eps: float = 0.3, min_samples: int = 3):
    if len(floor_pts) < min_samples:
        return [], []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(floor_pts)
    centroids = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = floor_pts[labels == label]
        centroids.append(cluster.mean(axis=0))
    return centroids, labels


def run_visualization(log_dir: str = "data/log1", delay: float = 0.1):
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 4)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Live 2D Radar Floor Map")
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#333333")

    scatter = ax.scatter([], [], c="#00ff88", s=15, alpha=0.6, label="Points")
    centroid_scatter = ax.scatter([], [], c="#ff4444", s=200, marker="x", linewidths=2, label="Objects")
    ax.legend(facecolor="#222222", labelcolor="white")

    frame_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white", fontsize=9, va="top")

    for i, (f1, f2, f3) in enumerate(sync_frames(log_dir)):
        pcds = [frames_to_pointcloud([f], z_min=-2.0, z_max=0.0) for f in [f1, f2, f3]]
        filtered = [dbscan_filter(p) for p in pcds]
        fused = fuse_pointclouds(filtered)
        floor_pts = project_to_2d(fused)

        if len(floor_pts) > 0:
            scatter.set_offsets(floor_pts)
            centroids, _ = detect_objects(floor_pts)
            if centroids:
                centroid_scatter.set_offsets(np.array(centroids))
            else:
                centroid_scatter.set_offsets(np.empty((0, 2)))
        else:
            scatter.set_offsets(np.empty((0, 2)))
            centroid_scatter.set_offsets(np.empty((0, 2)))

        frame_text.set_text(f"Frame: {i}  |  Points: {len(floor_pts)}  |  Objects: {len(centroids) if len(floor_pts) > 0 else 0}")
        plt.pause(delay)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visualization()