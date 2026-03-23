import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional


@dataclass
class RadarFrame:
    sensor_id: int
    timestamp: float
    frame_number: int
    point_cloud: List[List[float]]


def load_jsonl(path: Path, sensor_id: int) -> List[RadarFrame]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frames.append(RadarFrame(
                sensor_id=sensor_id,
                timestamp=obj["timestamp"],
                frame_number=obj["frame_number"],
                point_cloud=obj["point_cloud"],
            ))
    frames.sort(key=lambda f: f.timestamp)
    return frames


def sync_frames(
    log_dir: str = "data/log1",
    sync_window: float = 0.15,
) -> Iterator[Tuple[Optional[RadarFrame], Optional[RadarFrame], Optional[RadarFrame]]]:
    base = Path(log_dir)
    frames1 = load_jsonl(base / "angle1.jsonl", sensor_id=1)
    frames2 = load_jsonl(base / "angle2.jsonl", sensor_id=2)
    frames3 = load_jsonl(base / "angle3.jsonl", sensor_id=3)

    print(f"Loaded: {len(frames1)} frames (sensor1), "
          f"{len(frames2)} (sensor2), {len(frames3)} (sensor3)")

    i1, i2, i3 = 0, 0, 0

    while i1 < len(frames1) or i2 < len(frames2) or i3 < len(frames3):
        candidates = []
        if i1 < len(frames1): candidates.append(frames1[i1].timestamp)
        if i2 < len(frames2): candidates.append(frames2[i2].timestamp)
        if i3 < len(frames3): candidates.append(frames3[i3].timestamp)

        if not candidates:
            break

        anchor = min(candidates)

        f1 = frames1[i1] if i1 < len(frames1) and abs(frames1[i1].timestamp - anchor) <= sync_window else None
        f2 = frames2[i2] if i2 < len(frames2) and abs(frames2[i2].timestamp - anchor) <= sync_window else None
        f3 = frames3[i3] if i3 < len(frames3) and abs(frames3[i3].timestamp - anchor) <= sync_window else None

        if f1: i1 += 1
        if f2: i2 += 1
        if f3: i3 += 1

        yield f1, f2, f3


if __name__ == "__main__":
    print("Testing log replayer...")
    count = 0
    for f1, f2, f3 in sync_frames("data/log1"):
        sensors_present = sum(f is not None for f in [f1, f2, f3])
        pts1 = len(f1.point_cloud) if f1 else 0
        pts2 = len(f2.point_cloud) if f2 else 0
        pts3 = len(f3.point_cloud) if f3 else 0
        print(f"Frame group {count:3d} | sensors: {sensors_present}/3 | "
              f"pts: {pts1:3d} + {pts2:3d} + {pts3:3d} = {pts1+pts2+pts3:3d}")
        count += 1
        if count >= 10:
            break
    print(f"\nShowing first 10 frame groups")