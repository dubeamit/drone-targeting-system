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
from collections import Counter
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from mavlink_telemetry import MAVLinkTelemetry
from ray_projection import CameraProjection

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RTSP_URL   = "rtsp://192.168.144.25:8554/main.264"
UDP_OUT    = "udp://127.0.0.1:5600"
WIDTH      = 1280
HEIGHT     = 700
FPS        = 30
CONF       = 0.25
IMGSZ      = 1280
GROUND_ELEVATION_MSL = 0.0

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

current_targets = []  # list of dicts: {"bbox": [x1, y1, x2, y2], "lat": lat, "lon": lon, "alt": alt}
telemetry = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for t in current_targets:
            x1, y1, x2, y2 = t["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                class_name = t.get("class_name", "Unknown")
                print(f"\n[Operator] Selected '{class_name}' at Lat: {t['lat']:.6f}, Lon: {t['lon']:.6f}")
                if telemetry:
                    telemetry.send_target_command(t['lat'], t['lon'], t['alt'])
                break


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

# Initialize Telemetry & Projection
# telemetry = MAVLinkTelemetry(connection_string="udpin:localhost:14550")
telemetry = MAVLinkTelemetry(connection_string="udpin:127.0.0.1:14551")

telemetry.start()
projection = CameraProjection(
    native_width=2560, native_height=1400,
    fov_h_deg=71.5, fov_v_deg=43.0,
    max_zoom=10.0,
    inference_width=WIDTH, inference_height=HEIGHT
)

cv2.namedWindow("Operator View")
cv2.setMouseCallback("Operator View", mouse_callback)

weights = os.path.join("weights", "custom_yolov8s.pt")
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
            
            # Keep OpenCV window responsive during stalls
            with frame_lock:
                if latest_output is not None:
                    cv2.imshow("Operator View", latest_output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
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
        
        frame_timestamp = time.time()
        tel_state = telemetry.get_state_at_time(frame_timestamp)
        
        annotated = frame.copy()
        new_targets = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            
            # Bottom center of bounding box
            u = (x1 + x2) / 2.0
            v = y2
            
            gps_target = projection.pixel_to_gps(u, v, tel_state, GROUND_ELEVATION_MSL)
            
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{name} {conf:.2f}"
            
            if gps_target:
                new_targets.append({
                    "bbox": [x1, y1, x2, y2],
                    "lat": gps_target["lat"],
                    "lon": gps_target["lon"],
                    "alt": gps_target["alt_msl"],
                    "class_name": name
                })
                label += f" | {gps_target['distance_m']:.1f}m"
                gps_str = f"{gps_target['lat']:.5f}, {gps_target['lon']:.5f}"
                cv2.putText(annotated, gps_str, (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.putText(annotated, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        current_targets = new_targets
        
        # Telemetry overlay
        overlay = f"Alt: {tel_state['alt_rel']:.1f}m | Pitch: {tel_state['gimbal_pitch']:.1f} | Zoom: {tel_state['zoom_level']}x"
        cv2.putText(annotated, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        set_latest_output(annotated)
        
        cv2.imshow("Operator View", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        infer_count += 1
        frame_count += 1
        if frame_count % 30 == 0:
            fps_actual = infer_count / (time.time() - t_start)
            healthy = "ok" if stream_healthy.is_set() else "reconnecting"
            if len(new_targets) > 0:
                counts = Counter(t["class_name"] for t in new_targets)
                count_str = ", ".join([f"{c}x {k}" for k, c in counts.items()])
                print(f"[TRACK Frame {frame_count:04d} | {fps_actual:4.1f} FPS | RTSP: {healthy}] Detected ({len(new_targets)}): {count_str}")
            else:
                print(f"[SCAN  Frame {frame_count:04d} | {fps_actual:4.1f} FPS | RTSP: {healthy}] Scanning... (0 targets)")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    stop_event.set()
    if encoder.stdin:
        encoder.stdin.close()
    encoder.wait()
    if telemetry:
        telemetry.stop()
    for t in threads:
        t.join(timeout=3)
    cv2.destroyAllWindows()
    print("Clean shutdown complete.")
