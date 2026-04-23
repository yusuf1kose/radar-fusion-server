"""
3D mmWave Radar Simulator
Person walks slowly around two static obstacles inside the sensor triangle.
Obstacles have near-zero noise so they register as stationary.
"""

import json
import math
import random
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

SENSOR_POSITIONS = {
    1: np.array([-1.09, 2.45, 1.19]),
    2: np.array([ 1.09, 2.19, 1.19]),
    3: np.array([ 0.00, 0.00, 1.19]),
}

FRAME_RATE = 9.5
PERSON_NOISE = 0.04     # realistic body movement noise
OBSTACLE_NOISE = 0.005  # near-zero — obstacles don't move
_ANCHOR_EPOCH = datetime(2005, 6, 12, 0, 0, 0, tzinfo=timezone.utc).timestamp()

# Fixed obstacle surface points — precomputed once, reused every frame with tiny noise
OBSTACLES = [
    {"cx": -0.5, "cy": 1.8, "w": 0.7, "d": 0.3, "h": 0.4},
    {"cx":  0.6, "cy": 2.0, "w": 0.3, "d": 0.3, "h": 0.5},
]

# Precompute fixed obstacle surface points
def precompute_obstacle_pts(obstacles, n_per_obs=12, seed=0):
    rng = np.random.RandomState(seed)
    all_pts = []
    for obs in obstacles:
        pts = []
        for _ in range(n_per_obs):
            face = rng.randint(0, 4)
            if face == 0:
                x = rng.uniform(obs["cx"] - obs["w"]/2, obs["cx"] + obs["w"]/2)
                y = obs["cy"] + obs["d"]/2
            elif face == 1:
                x = rng.uniform(obs["cx"] - obs["w"]/2, obs["cx"] + obs["w"]/2)
                y = obs["cy"] - obs["d"]/2
            elif face == 2:
                x = obs["cx"] - obs["w"]/2
                y = rng.uniform(obs["cy"] - obs["d"]/2, obs["cy"] + obs["d"]/2)
            else:
                x = obs["cx"] + obs["w"]/2
                y = rng.uniform(obs["cy"] - obs["d"]/2, obs["cy"] + obs["d"]/2)
            z = rng.uniform(0.15, obs["h"])
            pts.append(np.array([x, y, z]))
        all_pts.append(pts)
    return all_pts  # list of lists of np.array


FIXED_OBS_PTS = precompute_obstacle_pts(OBSTACLES)

WAYPOINTS = [
    ( 0.0, 0.5),
    ( 0.0, 1.0),
    ( 0.3, 1.5),
    ( 0.3, 2.2),
    ( 0.0, 2.8),
    (-0.3, 2.2),
    (-0.3, 1.5),
    ( 0.0, 1.0),
    ( 0.0, 0.5),
]


def run_simulation(duration=40.0, output_dir="data/sim1", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    px, py = WAYPOINTS[0]
    wp_idx = 1
    speed = 0.25
    dt = 1.0 / FRAME_RATE
    n_frames = int(duration * FRAME_RATE)

    files = {s: open(f"{output_dir}/angle{s}.jsonl", "w") for s in [1, 2, 3]}
    counters = {s: 0 for s in [1, 2, 3]}

    print(f"Simulating {n_frames} frames ({duration:.0f}s) → {output_dir}")
    t0 = time.time()

    for fi in range(n_frames):
        # Step person
        target = WAYPOINTS[wp_idx % len(WAYPOINTS)]
        dx, dy = target[0] - px, target[1] - py
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 0.07:
            wp_idx += 1
        else:
            px += (dx / dist) * speed * dt
            py += (dy / dist) * speed * dt

        ts = _ANCHOR_EPOCH + fi * dt

        for sid in [1, 2, 3]:
            spos = SENSOR_POSITIONS[sid]
            local_pts = []

            # Person body — 18 points with realistic noise
            for _ in range(18):
                theta = random.uniform(0, 2 * math.pi)
                r = 0.18 * random.uniform(0.5, 1.0)
                frac = random.uniform(0.2, 1.0)
                wp = np.array([px + r * math.cos(theta),
                               py + r * math.sin(theta),
                               frac * 1.75])
                if np.linalg.norm(wp - spos) <= 3.5:
                    local_pts.append(wp - spos + np.random.normal(0, PERSON_NOISE, 3))

            # Obstacles — use FIXED precomputed points + tiny noise
            for obs_pts in FIXED_OBS_PTS:
                for wp in obs_pts:
                    if np.linalg.norm(wp - spos) <= 3.5:
                        # Tiny noise — centroid will barely move between frames
                        local_pts.append(wp - spos + np.random.normal(0, OBSTACLE_NOISE, 3))

            files[sid].write(json.dumps({
                "timestamp": round(ts, 6),
                "frame_number": counters[sid],
                "point_cloud": [[round(float(v), 6) for v in p] for p in local_pts],
            }) + "\n")
            counters[sid] += 1

    for f in files.values():
        f.close()
    print(f"Done in {time.time()-t0:.2f}s — {counters[1]} frames per sensor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration",   type=float, default=40.0)
    ap.add_argument("--output-dir", type=str,   default="data/sim1")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()
    run_simulation(args.duration, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
