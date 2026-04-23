import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from collections import deque
from scipy.spatial.distance import cdist

sys.path.insert(0, ".")
from ingestion.log_replayer import sync_frames
from fusion.pipeline import frames_to_pointcloud, fuse_pointclouds, project_to_2d
from sklearn.cluster import DBSCAN

SENSOR_POSITIONS = {
    "Sensor 1": (-1.09, 2.45),
    "Sensor 2": ( 1.09, 2.19),
    "Sensor 3": ( 0.00, 0.00),
}

TRAIL_LENGTH = 30
MOVEMENT_THRESHOLD = 0.02
STATIC_CONFIRM_FRAMES = 20


def get_clusters(floor_pts, eps=0.30, min_samples=3):
    if len(floor_pts) < min_samples:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(floor_pts)
    result = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        mask = labels == lbl
        result.append((floor_pts[mask].mean(axis=0), floor_pts[mask]))
    return result


def draw_sensor_triangle(ax):
    pos = list(SENSOR_POSITIONS.values())
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                    color="#334455", lw=1, ls="--", zorder=1)
    for name, (x, y) in SENSOR_POSITIONS.items():
        ax.plot(x, y, "^", color="#4488ff", ms=12, zorder=5)
        ax.text(x, y+0.1, name, color="#4488ff", fontsize=8,
                ha="center", va="bottom", zorder=6)


def run_viewer(log_dir: str = "data/sim1", delay: float = 0.1):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.3, 3.5)
    ax.set_xlabel("X (meters)", color="#8b949e")
    ax.set_ylabel("Y (meters)", color="#8b949e")
    ax.set_title("Distributed mmWave Radar — 2D Floor Map", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, color="#21262d", lw=0.5, zorder=0)
    ax.add_patch(patches.Rectangle((-2.0, -0.1), 4.0, 3.5,
                                    lw=1.5, edgecolor="#30363d", facecolor="none", zorder=1))
    draw_sensor_triangle(ax)

    person_pts_sc  = ax.scatter([], [], c="#00ff88", s=20, alpha=0.8, zorder=3)
    static_pts_sc  = ax.scatter([], [], c="#ffcc00", s=20, alpha=0.8, zorder=3)
    person_ring_sc = ax.scatter([], [], s=320, marker="o",
                                facecolors="none", edgecolors="#ff4444", lw=2, zorder=6)
    object_ring_sc = ax.scatter([], [], s=280, marker="s",
                                facecolors="none", edgecolors="#ffcc00", lw=2, zorder=6)

    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#00ff88", ms=8, label="Moving Points"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#ffcc00", ms=8, label="Static Points"),
        Line2D([0],[0], marker="o", color="#ff4444", markerfacecolor="none", ms=10, label="Person"),
        Line2D([0],[0], marker="s", color="#ffcc00", markerfacecolor="none", ms=10, label="Object"),
        Line2D([0],[0], marker="^", color="#4488ff", markerfacecolor="#4488ff", ms=8, label="Radar Sensor"),
    ]
    ax.legend(handles=legend_elements, facecolor="#161b22", labelcolor="white",
              loc="upper right", fontsize=8, framealpha=0.9)

    info_box = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white",
                       fontsize=9, va="top",
                       bbox=dict(boxstyle="round", facecolor="#161b22", alpha=0.8))

    trail_artists = []
    label_artists = []
    trail = deque(maxlen=TRAIL_LENGTH)
    tracks = []

    for frame_idx, (f1, f2, f3) in enumerate(sync_frames(log_dir)):
        # KEY FIX: fuse all 3 sensors together FIRST, no per-sensor DBSCAN
        fused_pcd = frames_to_pointcloud([f1, f2, f3], z_min=0.1, z_max=1.7)
        floor_pts = project_to_2d(fused_pcd)

        for a in trail_artists + label_artists:
            a.remove()
        trail_artists.clear()
        label_artists.clear()

        clusters = get_clusters(floor_pts) if len(floor_pts) >= 3 else []

        # Match clusters to tracks
        if tracks and clusters:
            track_pos = np.array([t["centroid"] for t in tracks])
            clust_pos = np.array([c for c, _ in clusters])
            dists = cdist(clust_pos, track_pos)
            matched_t = set()
            matched_c = set()
            for ci in range(len(clusters)):
                ti = int(np.argmin(dists[ci]))
                if dists[ci][ti] < 0.5 and ti not in matched_t:
                    c, pts = clusters[ci]
                    moved = np.linalg.norm(c - tracks[ti]["centroid"])
                    tracks[ti]["centroid"] = c.copy()
                    tracks[ti]["pts"] = pts
                    tracks[ti]["missed"] = 0
                    if moved < MOVEMENT_THRESHOLD:
                        tracks[ti]["static_count"] += 1
                    else:
                        tracks[ti]["static_count"] = 0
                    matched_t.add(ti)
                    matched_c.add(ci)
            for ti, t in enumerate(tracks):
                if ti not in matched_t:
                    t["missed"] += 1
            for ci, (c, pts) in enumerate(clusters):
                if ci not in matched_c:
                    tracks.append({"centroid": c.copy(), "pts": pts,
                                   "static_count": 0, "missed": 0})
            tracks = [t for t in tracks if t["missed"] < 4]
        elif clusters:
            tracks = [{"centroid": c.copy(), "pts": pts,
                       "static_count": 0, "missed": 0}
                      for c, pts in clusters]
        else:
            for t in tracks:
                t["missed"] += 1
            tracks = [t for t in tracks if t["missed"] < 4]

        person_pts, static_pts = [], []
        person_centers, object_centers = [], []

        for t in tracks:
            if t["static_count"] >= STATIC_CONFIRM_FRAMES:
                static_pts.extend(t["pts"].tolist())
                object_centers.append(t["centroid"])
            else:
                person_pts.extend(t["pts"].tolist())
                person_centers.append(t["centroid"])

        person_pts_sc.set_offsets(np.array(person_pts) if person_pts else np.empty((0,2)))
        static_pts_sc.set_offsets(np.array(static_pts) if static_pts else np.empty((0,2)))

        if person_centers:
            person_ring_sc.set_offsets(np.array(person_centers))
            moving = [t for t in tracks if t["static_count"] < STATIC_CONFIRM_FRAMES]
            if moving:
                best = min(moving, key=lambda t: t["static_count"])
                trail.append(best["centroid"].copy())
            if len(trail) > 1:
                ta = np.array(trail)
                for i in range(1, len(ta)):
                    alpha = (i / len(ta)) * 0.75
                    ln, = ax.plot(ta[i-1:i+1, 0], ta[i-1:i+1, 1],
                                  color="#ff8800", alpha=alpha, lw=2, zorder=4)
                    trail_artists.append(ln)
        else:
            person_ring_sc.set_offsets(np.empty((0,2)))

        if object_centers:
            object_ring_sc.set_offsets(np.array(object_centers))
            for oc in object_centers:
                lbl = ax.text(oc[0], oc[1] + 0.15, "Object",
                              color="#ffcc00", fontsize=7, ha="center", zorder=7)
                label_artists.append(lbl)
        else:
            object_ring_sc.set_offsets(np.empty((0,2)))

        sensors_active = sum(f is not None for f in [f1, f2, f3])
        info_box.set_text(
            f"Frame: {frame_idx:3d}  |  Sensors: {sensors_active}/3\n"
            f"Points: {len(floor_pts):3d}  |  "
            f"Person: {len(person_centers)}  |  Objects: {len(object_centers)}"
        )

        plt.pause(delay)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_viewer()
