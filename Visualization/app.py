"""
Minimal visualization layer server with realtime push, detector selection API,
and placeholder root-cause endpoints.
"""

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    import zmq
    import zmq.asyncio

    ZMQ_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ZMQ_AVAILABLE = False

# Default endpoints from Design.md communicator layer
DEFAULT_VIS_ENDPOINT = os.getenv("VIS_ENDPOINT", "tcp://localhost:5556")

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "realtime.log"

# In-memory detector config (can be extended to persist or notify detection layer)
DETECTOR_CONFIG: Dict[str, bool] = {
    "statistical": True,
    "comparison": True,
    "ml": True,
}

# Async queue for realtime data fan-out to WebSocket clients
realtime_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )


app = FastAPI(title="Visualization Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/detection/config")
async def get_detection_config() -> Dict[str, bool]:
    """Return current detector enablement."""
    return DETECTOR_CONFIG


@app.put("/api/detection/config")
async def update_detection_config(config: Dict[str, bool]) -> Dict[str, bool]:
    """Update detector enablement (statistical, comparison, ml)."""
    for key in ["statistical", "comparison", "ml"]:
        if key in config:
            DETECTOR_CONFIG[key] = bool(config[key])
    logging.info("Updated detector config: %s", DETECTOR_CONFIG)
    return DETECTOR_CONFIG


@app.get("/api/root_cause/{anomaly_id}")
async def get_root_cause_placeholder(anomaly_id: str) -> Dict:
    """Placeholder root-cause response."""
    return {
        "anomaly_id": anomaly_id,
        "status": "pending",
        "message": "Root cause analysis placeholder. To be implemented.",
        "root_causes": [],
    }


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """Realtime data push to clients."""
    await websocket.accept()
    try:
        while True:
            data = await realtime_queue.get()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover
        logging.error("WebSocket error: %s", exc)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def zmq_receive_loop(endpoint: str):
    """Background ZeroMQ SUB loop to receive realtime data from communicator."""
    if not ZMQ_AVAILABLE:
        logging.warning("pyzmq not available; realtime data disabled")
        return

    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect(endpoint)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe all
    logging.info("ZMQ subscriber connected to %s", endpoint)

    try:
        while True:
            msg = socket.recv()
            try:
                text = msg.decode("utf-8")
                # Push to queue (drop if full)
                try:
                    realtime_queue.put_nowait(text)
                except asyncio.QueueFull:
                    logging.warning("Realtime queue full; dropping message")
                # Append to log file
                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(text + "\n")
            except Exception as exc:  # pragma: no cover
                logging.error("Failed to process message: %s", exc)
    except Exception as exc:  # pragma: no cover
        logging.error("ZMQ loop error: %s", exc)
    finally:
        socket.close()
        ctx.term()


def start_background_zmq_thread():
    """Start ZMQ subscriber in separate thread."""
    thread = threading.Thread(
        target=zmq_receive_loop, args=(DEFAULT_VIS_ENDPOINT,), daemon=True
    )
    thread.start()
    return thread


def main():
    setup_logging()
    start_background_zmq_thread()

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        reload=False,
    )


if __name__ == "__main__":
    main()

