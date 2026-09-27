"""
Robust RTSP → YOLO → MPEG-TS/UDP pipeline for QGroundControl.

Industry-standard pattern:
  [Capture thread]  → jitter buffer → [Inference loop] → latest frame
  [Output thread]   → fixed-FPS encoder (repeats last frame during RTSP gaps)

Capture uses FFmpeg (not OpenCV) for RTSP — better timeouts and reconnect behavior.
"""

import os
import subprocess
import threading
import queue
import time
import warnings

import numpy as np
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RTSP_URL   = "rtsp://192.168.144.25:8554/main.264"
UDP_OUT    = "udp://127.0.0.1:5600"
WIDTH      = 640
HEIGHT     = 480
FPS        = 30
CONF       = 0.25
IMGSZ      = 640

JITTER_BUFFER   = 3      # absorb brief RTSP jitter (frames)
STALL_SEC       = 3.0    # no frame this long → reconnect RTSP
RECONNECT_MIN   = 1.0    # backoff floor (seconds)
RECONNECT_MAX   = 10.0   # backoff ceiling (seconds)
# ─────────────────────────────────────────────────────────────────────────────

FRAME_BYTES = WIDTH * HEIGHT * 3
stop_event  = threading.Event()

# Shared between inference and output threads
frame_lock      = threading.Lock()
latest_output   = None          # BGR uint8 (H, W, 3)
stream_healthy  = threading.Event()


def rtsp_ffmpeg_cmd():
    """FFmpeg ingest — TCP RTSP; stall recovery handled by watchdog (not -stimeout)."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-an", "-sn",
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-r", str(FPS),
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "pipe:1",
    ]


def udp_ffmpeg_cmd():
    """MPEG-TS over UDP for QGC (0.0.0.0:5600 in QGC settings)."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-g", str(FPS), "-bf", "0",
        "-b:v", "2M",
        "-f", "mpegts",
        UDP_OUT,
    ]


def _put_drop_oldest(q: queue.Queue, item) -> None:
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put(item)


def capture_thread(raw_queue: queue.Queue) -> None:
    """Dedicated producer: reads RTSP via FFmpeg, auto-reconnects with backoff."""
    backoff = RECONNECT_MIN

    while not stop_event.is_set():
        print("[capture] Connecting to RTSP...")
        # stderr must not use PIPE without a reader — buffer fills and FFmpeg blocks.
        proc = subprocess.Popen(
            rtsp_ffmpeg_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        last_frame_at = [time.time()]
        watchdog_stop = threading.Event()
        backoff = RECONNECT_MIN

        def stall_watchdog():
            while not watchdog_stop.is_set() and proc.poll() is None:
                if time.time() - last_frame_at[0] > STALL_SEC:
                    print("[capture] Stall detected, reconnecting...")
                    proc.terminate()
                    break
                watchdog_stop.wait(0.5)

        watchdog = threading.Thread(target=stall_watchdog, daemon=True)
        watchdog.start()

        try:
            while not stop_event.is_set():
                raw = proc.stdout.read(FRAME_BYTES)
                if len(raw) != FRAME_BYTES:
                    code = proc.poll()
                    print(f"[capture] Stream ended ({len(raw)}/{FRAME_BYTES} bytes, exit={code}).")
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3).copy()
                _put_drop_oldest(raw_queue, frame)
                last_frame_at[0] = time.time()
                stream_healthy.set()

        finally:
            watchdog_stop.set()
            watchdog.join(timeout=1)
            stream_healthy.clear()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        if stop_event.is_set():
            break

        print(f"[capture] Reconnecting in {backoff:.1f}s...")
        stop_event.wait(backoff)
        backoff = min(RECONNECT_MAX, backoff * 1.5)

    print("[capture] Exited.")


def output_thread(encoder: subprocess.Popen) -> None:
    """
    Fixed-FPS pusher — keeps QGC stream alive by repeating the last frame
    when inference or RTSP briefly stalls (common drone/broadcast pattern).
    """
    interval = 1.0 / FPS
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    while not stop_event.is_set():
        t0 = time.perf_counter()

        with frame_lock:
            frame = latest_output

        if frame is None:
            frame = blank

        try:
            encoder.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("[output] Encoder pipe closed.")
            stop_event.set()
            break

        elapsed = time.perf_counter() - t0
        stop_event.wait(max(0.0, interval - elapsed))

    print("[output] Exited.")


def set_latest_output(frame: np.ndarray) -> None:
    global latest_output
    with frame_lock:
        latest_output = frame


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}")

weights = r"weights\visdrone-yolov8s.pt"
model   = YOLO(weights, task="detect")
model.to("cuda")

dummy = torch.zeros(1, 3, IMGSZ, IMGSZ, device="cuda")
model.model(dummy)
print("Model warmed up on:", next(model.model.parameters()).device)

raw_queue = queue.Queue(maxsize=JITTER_BUFFER)

encoder = subprocess.Popen(
    udp_ffmpeg_cmd(),
    stdin=subprocess.PIPE,
)

threads = [
    threading.Thread(target=capture_thread, args=(raw_queue,), name="capture", daemon=True),
    threading.Thread(target=output_thread, args=(encoder,), name="output", daemon=True),
]
for t in threads:
    t.start()

print("Waiting for first RTSP frame...")
while raw_queue.empty() and not stop_event.is_set():
    time.sleep(0.05)

if stop_event.is_set():
    print("Could not start stream. Exiting.")
    exit(1)

print("Starting inference loop...")
frame_count = 0
infer_count = 0
t_start     = time.time()
fps_actual  = 0.0

try:
    while not stop_event.is_set():
        try:
            frame = raw_queue.get(timeout=1.0)
        except queue.Empty:
            # Jitter buffer empty — output thread keeps sending last frame to QGC
            if not stream_healthy.is_set():
                print("[inference] RTSP reconnecting (QGC still gets last frame)...")
            continue

        # Skip inference when behind — always prefer the freshest frame
        if not raw_queue.empty():
            try:
                while True:
                    frame = raw_queue.get_nowait()
            except queue.Empty:
                pass

        results = model.predict(
            frame,
            conf=CONF,
            imgsz=IMGSZ,
            device="cuda",
            quantize=16,
            max_det=100,
            verbose=False,
        )
        annotated = results[0].plot()
        set_latest_output(annotated)

        infer_count += 1
        frame_count += 1
        if frame_count % 30 == 0:
            fps_actual = infer_count / (time.time() - t_start)
            healthy = "ok" if stream_healthy.is_set() else "reconnecting"
            print(f"FPS: {fps_actual:.1f} | RTSP: {healthy} | buffer: {raw_queue.qsize()}")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    stop_event.set()
    if encoder.stdin:
        encoder.stdin.close()
    encoder.wait()
    for t in threads:
        t.join(timeout=3)
    print("Clean shutdown complete.")
