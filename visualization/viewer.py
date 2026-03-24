import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from collections import deque

sys.path.insert(0, ".")
from ingestion.log_replayer import sync_frames
from fusion.pipeline import frames_to_pointcloud, dbscan_filter, fuse_pointclouds, project_to_2d
from sklearn.cluster import DBSCAN

# Known sensor positions in meters (from logs/README.md triangle layout)
# Placing angle3 at origin (bottom center), angle1 top-left, angle2 top-right
SENSOR_POSITIONS = {
    "Sensor 1": (-1.09, 2.45),
    "Sensor 2": (1.09, 2.19),
    "Sensor 3": (0.0, 0.0),
}

ROOM_WIDTH = 6.0   # x: -3 to 3
ROOM_HEIGHT = 4.5  # y: -0.5 to 4

TRAIL_LENGTH = 20  # frames to keep in trail


def detect_objects(floor_pts, eps=0.3, min_samples=3):
    if len(floor_pts) < min_samples:
        return [], np.array([])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(floor_pts)
    centroids = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = floor_pts[labels == label]
        centroids.append(cluster.mean(axis=0))
    return centroids, labels


def draw_sensor_triangle(ax):
    positions = list(SENSOR_POSITIONS.values())
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            ax.plot(
                [positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]],
                color="#334455", linewidth=1, linestyle="--", zorder=1
            )
    for name, (x, y) in SENSOR_POSITIONS.items():
        ax.plot(x, y, "^", color="#4488ff", markersize=12, zorder=5)
        ax.text(x, y + 0.15, name, color="#4488ff", fontsize=8,
                ha="center", va="bottom", zorder=6)


def run_viewer(log_dir: str = "data/log1", delay: float = 0.08):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 4)
    ax.set_xlabel("X (meters)", color="#8b949e")
    ax.set_ylabel("Y (meters)", color="#8b949e")
    ax.set_title("Distributed mmWave Radar — 2D Floor Map", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    ax.grid(True, color="#21262d", linewidth=0.5, zorder=0)

    # Room boundary
    room_rect = patches.Rectangle((-2.5, -0.2), 5, 4,
                                   linewidth=1.5, edgecolor="#30363d",
                                   facecolor="none", linestyle="-", zorder=1)
    ax.add_patch(room_rect)
    ax.text(-2.4, 3.7, "Coverage Area (5m × 4m)", color="#30363d", fontsize=8)

    draw_sensor_triangle(ax)

    # Live elements
    point_scatter = ax.scatter([], [], c="#00ff88", s=18, alpha=0.7, zorder=3, label="Radar Points")
    centroid_scatter = ax.scatter([], [], c="#ff4444", s=300, marker="o",
                                  facecolors="none", edgecolors="#ff4444",
                                  linewidths=2, zorder=6, label="Detected Person")
    centroid_x_scatter = ax.scatter([], [], c="#ff4444", s=80, marker="x",
                                    linewidths=2, zorder=7)

    trail_lines = []
    trail = deque(maxlen=TRAIL_LENGTH)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00ff88", markersize=8, label="Radar Points"),
        Line2D([0], [0], marker="o", color="#ff4444", markerfacecolor="none", markersize=10, label="Detected Person"),
        Line2D([0], [0], marker="^", color="#4488ff", markerfacecolor="#4488ff", markersize=8, label="Radar Sensor"),
    ]
    ax.legend(handles=legend_elements, facecolor="#161b22", labelcolor="white",
              loc="upper right", fontsize=8, framealpha=0.9)

    info_box = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                       color="white", fontsize=9, va="top",
                       bbox=dict(boxstyle="round", facecolor="#161b22", alpha=0.8))

    for i, (f1, f2, f3) in enumerate(sync_frames(log_dir)):
        pcds = [frames_to_pointcloud([f], z_min=-2.0, z_max=0.0) for f in [f1, f2, f3]]
        filtered = [dbscan_filter(p) for p in pcds]
        fused = fuse_pointclouds(filtered)
        floor_pts = project_to_2d(fused)

        centroids = []
        if len(floor_pts) > 0:
            point_scatter.set_offsets(floor_pts)
            centroids, _ = detect_objects(floor_pts)
        else:
            point_scatter.set_offsets(np.empty((0, 2)))

        # Draw trail
        for line in trail_lines:
            line.remove()
        trail_lines.clear()

        if centroids:
            pos = np.array(centroids[0])
            trail.append(pos)
            centroid_scatter.set_offsets([pos])
            centroid_x_scatter.set_offsets([pos])

            if len(trail) > 1:
                trail_arr = np.array(trail)
                for t in range(1, len(trail_arr)):
                    alpha = t / len(trail_arr) * 0.6
                    line, = ax.plot(
                        trail_arr[t-1:t+1, 0],
                        trail_arr[t-1:t+1, 1],
                        color="#ff8800", alpha=alpha, linewidth=1.5, zorder=4
                    )
                    trail_lines.append(line)
        else:
            centroid_scatter.set_offsets(np.empty((0, 2)))
            centroid_x_scatter.set_offsets(np.empty((0, 2)))

        sensors_active = sum(f is not None for f in [f1, f2, f3])
        info_box.set_text(
            f"Frame: {i:3d}  |  Active Sensors: {sensors_active}/3\n"
            f"Points: {len(floor_pts):3d}  |  Objects Detected: {len(centroids)}"
        )

        plt.pause(delay)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_viewer()