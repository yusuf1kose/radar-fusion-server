# radar-fusion-server

# Distributed mmWave Radar Network for Real-Time 2D Floor Visualization

A distributed radar sensing system that fuses data from three Texas Instruments IWR6843AOP radar nodes to generate real-time 2D floor visualizations. The system provides privacy-preserving spatial awareness that operates in low-light environments and privacy-sensitive areas without capturing any visual imagery.

## Team

- Yusuf Kose (yusuf1kose)
- Ian Posada Kim (Ijkkim)
- Filho Tran (filhotran)
- Ihsan Mehtabuddin (Nhiblam)

**Advisor:** Professor Mahima Agumbe Suresh

## Prerequisites

- Python 3.11
- Git
- AWS EC2 instance (for live deployment)
- TI IWR6843AOP radar hardware (for live mode; sample logs included for local testing)

## Installation
```bash
# Clone the repository
git clone https://github.com/yusuf1kose/radar-fusion-server.git
cd radar-fusion-server

# Create virtual environment
py -3.11 -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

No environment variables required for local replay mode. For live AWS deployment, update `config.yaml`:
```yaml
data:
  log_dir: data
  log_session: log1

fusion:
  dbscan_eps: 0.15
  dbscan_min_samples: 3

database:
  path: radar.db

server:
  host: 0.0.0.0
  port: 5000
```

## Running the Application

**Local replay mode (no hardware needed):**
```bash
# Run the full 2D floor map viewer using sample log data
python -m visualization.viewer

# Test the fusion pipeline only
python -m fusion.pipeline

# Test the log replayer only
python -m ingestion.log_replayer
```

**Initialize the database:**
```bash
python ingestion/database.py
```

## Usage

When you run `python -m visualization.viewer` a window will open showing:

- **Blue triangles** — the 3 radar sensor positions in their real-world triangle formation
- **Green dots** — fused radar point cloud after DBSCAN noise filtering
- **Red circle** — detected person/object centroid
- **Orange trail** — movement history of the detected object over the last 20 frames
- **Info panel** — live frame count, active sensors, point count, and objects detected

The system replays the sample 20-second log session at approximately 9-10 Hz matching the original recording speed.

## Project Structure
```
radar-fusion-server/
├── data/
│   ├── log1/          # Sample session 1 (3 x ~20s JSONL files)
│   └── log2/          # Sample session 2 (redundant capture)
├── ingestion/
│   ├── log_replayer.py   # Syncs and replays 3-sensor JSONL logs by timestamp
│   └── database.py       # SQLite schema and insert helpers (WAL mode)
├── fusion/
│   └── pipeline.py       # DBSCAN filtering, point cloud fusion, 2D projection
├── visualization/
│   ├── viewer.py         # Full 2D floor map viewer with sensor layout and trails
│   └── floor_map.py      # Simple floor map (development reference)
├── tests/
├── config.yaml
└── requirements.txt
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'ingestion'`** — Run scripts with `-m` flag from the project root: `python -m visualization.viewer`

**`venv` not activating on Windows** — Use `.\venv\Scripts\Activate.ps1` instead of `venv\Scripts\activate`

**Viewer closes immediately** — Make sure you are in the project root directory when running commands
