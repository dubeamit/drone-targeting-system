#!/usr/bin/env python3
"""
Drone Targeting System - Comprehensive Test & Demonstration Pipeline
====================================================================
Features:
  1. Multi-Model Support: easily switch between models via CLI:
     --model v9e   (weights/custom_yolov9e.pt, 350MB, high-res military target detector)
     --model 26s   (runs/finetune/yolo26s/weights/best.pt, 20MB, 24-class with drone/bird)
     --model v8s   (runs/finetune/yolov8s/weights/best.pt, 22MB, baseline)
     or any path to a .pt file.
  2. Multi-Video Support: pass any video filename from videos/ or full path:
     --source tank_and_army.mp4
     --source apache_attack_helicopter.mp4
     --source drones.mp4
     --source indian_army_parade.mp4
     --source IAF_sukhoi_su-30mki.mp4
     --source 0 (webcam)
  3. Telemetry Modes:
     --mock       (Default: zero-setup simulated drone attitude & GPS)
     --ardupilot  (Connects live to ArduPilot SITL / MAVProxy on udp:127.0.0.1:14551)
  4. Real-time 3D Ray-Projection: calculates ground GPS coordinates for every detection.
  5. Interactive Targeting: Left-click any bounding box to dispatch GUIDED command.
  6. Dedicated Fixed-FPS QGC Streaming:
     Runs a steady-clock output thread feeding H.264 MPEG-TS to udp://127.0.0.1:5600.
     Guarantees QGroundControl's GStreamer demuxer stays locked without freezing.
"""

import argparse
from collections import Counter
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR / "qgc"))
from ray_projection import CameraProjection

# Friendly Model Shortcuts
MODEL_PRESETS = {
    "v9e": CURRENT_DIR / "weights" / "custom_yolov9e.pt",
    "yolov9e": CURRENT_DIR / "weights" / "custom_yolov9e.pt",
    "26s": CURRENT_DIR / "runs" / "finetune" / "yolo26s" / "weights" / "best.pt",
    "yolo26s": CURRENT_DIR / "runs" / "finetune" / "yolo26s" / "weights" / "best.pt",
    "v8s": CURRENT_DIR / "runs" / "finetune" / "yolov8s" / "weights" / "best.pt",
    "yolov8s": CURRENT_DIR / "runs" / "finetune" / "yolov8s" / "weights" / "best.pt",
}

# ── Global Shared State ───────────────────────────────────────────────────────
state_lock = threading.Lock()
current_targets = []
active_lock_target = None
status_banner = "Click any bounding box to lock target & guide drone"
banner_expiry = 0.0
telemetry_client = None

# QGC Output Thread Shared State
output_lock = threading.Lock()
latest_hud_frame = None
stop_event = threading.Event()


def mouse_callback(event, x, y, flags, param):
    """Operator Click Callback: Left-click on any detected box to engage target."""
    global active_lock_target, status_banner, banner_expiry, telemetry_client
    if event == cv2.EVENT_LBUTTONDOWN:
        with state_lock:
            selected = None
            for t in current_targets:
                x1, y1, x2, y2 = t["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected = t
                    break

            if selected:
                active_lock_target = selected
                gps = selected["gps"]
                cls = selected["class_name"]
                conf = selected["conf"]
                dist = selected["dist"]

                print("\n" + "=" * 80)
                print("🎯 [OPERATOR TARGET ACQUIRED & ENGAGED]")
                print(f"Target:       {cls.upper()} ({conf*100:.1f}%)")
                if gps:
                    print(f"Target GPS:   Lat: {gps['lat']:.6f}°, Lon: {gps['lon']:.6f}° | Alt: {gps['alt_msl']:.1f}m")
                    print(f"Slant Range:  {dist:.1f} meters")
                else:
                    print("Target GPS:   Skyward / horizon (no ground intersection)")

                # If connected to ArduPilot, send real MAVLink command
                if telemetry_client and hasattr(telemetry_client, "send_target_command") and gps:
                    telemetry_client.send_target_command(gps["lat"], gps["lon"], gps["alt_msl"])
                    print("[MAVLink SITL] Dispatched real GUIDED waypoint command to ArduPilot!")
                else:
                    print("[MAVLink Mock] Dispatched simulated GUIDED command to Autopilot!")
                print("=" * 80 + "\n")

                if gps:
                    status_banner = f"LOCKED: {cls.upper()} -> GUIDED WAYPOINT SENT ({gps['lat']:.5f}, {gps['lon']:.5f})"
                else:
                    status_banner = f"LOCKED: {cls.upper()}"
                banner_expiry = time.time() + 4.0
            else:
                active_lock_target = None
                status_banner = "No target at click location"
                banner_expiry = time.time() + 2.0


class MockTelemetry:
    """Simulates realistic live drone and gimbal attitude telemetry."""
    def __init__(self, lat=28.6139, lon=77.2090, alt_rel=50.0, gimbal_pitch=-45.0):
        self.base_lat = lat
        self.base_lon = lon
        self.alt_rel = alt_rel
        self.gimbal_pitch = gimbal_pitch
        self.yaw = 90.0  # Heading East
        self.t0 = time.time()

    def get_state(self):
        dt = time.time() - self.t0
        heading = (self.yaw + math.sin(dt * 0.1) * 5.0) % 360.0
        pitch_fluct = self.gimbal_pitch + math.sin(dt * 0.2) * 1.0

        return {
            "lat": self.base_lat,
            "lon": self.base_lon,
            "alt_msl": self.alt_rel + 15.0,
            "alt_rel": self.alt_rel,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": heading,
            "gimbal_pitch": pitch_fluct,
            "gimbal_yaw": 0.0,
            "zoom_level": 1.0,
            "timestamp": time.time(),
            "source": "MOCK"
        }

    def stop(self):
        pass


class ArduPilotTelemetryWrapper:
    """Connects to real ArduPilot SITL / MAVProxy on UDP port 14551."""
    def __init__(self, connection_string="udpin:127.0.0.1:14551"):
        try:
            from mavlink_telemetry import MAVLinkTelemetry
            self.client = MAVLinkTelemetry(connection_string=connection_string)
            self.connected = self.client.start()
        except Exception as e:
            print(f"[MAVLink Warning] Could not connect to ArduPilot ({e}). Falling back to Mock.")
            self.client = None
            self.connected = False
            self.mock = MockTelemetry()

    def get_state(self):
        if self.connected and self.client:
            st = self.client.get_current_state()
            st["source"] = "ARDUPILOT_SITL"
            return st
        return self.mock.get_state()

    def send_target_command(self, lat, lon, alt):
        if self.connected and self.client:
            self.client.send_target_command(lat, lon, alt)

    def stop(self):
        if self.connected and self.client:
            self.client.stop()


def make_ffmpeg_udp_cmd(ffmpeg_bin: str, width: int, height: int, fps: int, udp_url: str, mode: str = "mpegts"):
    """
    Builds FFmpeg streaming command:
      - mode='mpegts': streams MPEG-TS to udp://127.0.0.1:5600 (for QGC 'MPEG-TS Video Stream')
      - mode='rtp' / 'h264': streams RTP H.264 to rtp://127.0.0.1:5600 (for QGC 'UDP h.264 Video Stream')
    """
    if mode in ("rtp", "h264"):
        target_url = "rtp://127.0.0.1:5600" if "rtp://" not in udp_url else udp_url
        return [
            ffmpeg_bin,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", str(fps), "-bf", "0",
            "-b:v", "2M",
            "-f", "rtp",
            target_url,
        ]
    else:
        return [
            ffmpeg_bin,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", str(fps), "-bf", "0",
            "-b:v", "2M",
            "-f", "mpegts",
            udp_url,
        ]


def qgc_output_thread_func(encoder: subprocess.Popen, fps: float, width: int = 1280, height: int = 700):
    """
    Exact output_thread from client qgc/inference_stream.py:
    High-precision monotonic fixed-FPS pusher that repeats the last frame
    to keep QGroundControl stream alive.
    """
    interval = 1.0 / float(fps)
    blank = np.zeros((height, width, 3), dtype=np.uint8)

    while not stop_event.is_set():
        t0 = time.perf_counter()

        with output_lock:
            frame = latest_hud_frame

        if frame is None:
            frame = blank

        try:
            encoder.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            break

        elapsed = time.perf_counter() - t0
        stop_event.wait(max(0.0, interval - elapsed))


def draw_hud(frame, tel, targets, active_target, banner_text, fps, model_name):
    """Draws defense-grade HUD overlay with telemetry and targeting reticles."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Top Telemetry Bar
    bar_h = 42
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    source_tag = tel.get("source", "MOCK")
    tel_str = (
        f"[{model_name}] ALT: {tel['alt_rel']:.1f}m | "
        f"PITCH: {tel['gimbal_pitch']:.1f}° | "
        f"HDG: {tel['yaw']:.0f}° | "
        f"GPS: {tel['lat']:.5f}°, {tel['lon']:.5f}° | "
        f"TEL: {source_tag} | FPS: {fps:.1f}"
    )
    cv2.putText(frame, tel_str, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 2, cv2.LINE_AA)

    # 2. Draw Bounding Boxes and Ray-Projected Target Info
    for t in targets:
        x1, y1, x2, y2 = t["bbox"]
        cls = t["class_name"]
        conf = t["conf"]
        gps = t["gps"]
        dist = t["dist"]

        is_locked = (active_target and active_target["bbox"] == t["bbox"])
        color = (0, 0, 255) if is_locked else (0, 255, 0)
        thick = 3 if is_locked else 2

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # Label tag
        label = f"{cls} {conf*100:.0f}%"
        if dist is not None:
            label += f" | {dist:.0f}m"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lw + 6, max(22, y1)), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw ground target crosshair and GPS tag below box
        cx = int((x1 + x2) / 2)
        by = min(h - 5, y2 + 12)
        cv2.drawMarker(frame, (cx, y2), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        if gps:
            coord_str = f"({gps['lat']:.5f}, {gps['lon']:.5f})"
            cv2.putText(frame, coord_str, (max(5, cx - 60), by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # If locked, draw crosshair reticle on target center
        if is_locked:
            tcx = int((x1 + x2) / 2)
            tcy = int((y1 + y2) / 2)
            cv2.circle(frame, (tcx, tcy), 24, (0, 0, 255), 2)
            cv2.putText(frame, "LOCKED", (x1, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # 3. Bottom Status Banner
    bot_y = h - 14
    cv2.rectangle(frame, (0, h - 34), (w, h), (15, 15, 15), -1)
    banner_color = (0, 220, 255) if "LOCKED" in banner_text else (200, 200, 200)
    cv2.putText(frame, banner_text, (12, bot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, banner_color, 1, cv2.LINE_AA)


def resolve_video_source(src_input: str) -> str:
    """Helper to resolve video file from current dir or videos/ directory."""
    try:
        int(src_input)
        return src_input
    except ValueError:
        pass

    if src_input.startswith("rtsp://") or src_input.startswith("http://"):
        return src_input

    p = Path(src_input)
    if p.exists():
        return str(p)

    in_videos = CURRENT_DIR / "videos" / src_input
    if in_videos.exists():
        return str(in_videos)

    return src_input


def resolve_model_path(model_input: str) -> Path:
    """Helper to resolve model shortcuts like 'v9e', '26s', 'v8s' or direct paths."""
    key = model_input.lower().strip()
    if key in MODEL_PRESETS:
        return MODEL_PRESETS[key]

    p = Path(model_input)
    if p.exists():
        return p

    in_weights = CURRENT_DIR / "weights" / model_input
    if in_weights.exists():
        return in_weights

    return p


def main():
    parser = argparse.ArgumentParser(description="Drone Targeting System - Comprehensive Test Pipeline")
    parser.add_argument("--source", default="drone_video.mp4",
                        help="Video filename, path, webcam index (0), or RTSP URL. Auto-checks 'videos/' folder.")
    parser.add_argument("--model", default="v9e",
                        help="Model preset ('v9e', '26s', 'v8s') or path to .pt file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (e.g. 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--ardupilot", action="store_true", help="Connect live to ArduPilot SITL / MAVProxy on UDP 14551")
    parser.add_argument("--udp", default="udp://127.0.0.1:5600", help="UDP stream URL for QGroundControl")
    parser.add_argument("--stream-mode", choices=["mpegts", "rtp", "h264"], default="mpegts",
                        help="Streaming container: 'mpegts' (for QGC MPEG-TS) or 'rtp'/'h264' (for QGC UDP h.264)")
    parser.add_argument("--ffmpeg-bin", default="/usr/bin/ffmpeg", help="Path to system ffmpeg binary")
    parser.add_argument("--no-qgc", action="store_true", help="Disable QGC UDP stream broadcast")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop video file when finished")
    args = parser.parse_args()

    # 1. Resolve Video Source
    resolved_source = resolve_video_source(args.source)
    source_is_index = False
    try:
        source_val = int(resolved_source)
        source_is_index = True
    except ValueError:
        source_val = resolved_source
        if not source_val.startswith("rtsp://") and not Path(source_val).exists():
            print(f"Error: Source video file '{source_val}' not found.")
            print(f"Available videos in 'videos/': {[v.name for v in (CURRENT_DIR / 'videos').glob('*.mp4')]}")
            sys.exit(1)

    # 2. Resolve Model Checkpoint
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        print(f"Warning: Model '{model_path}' not found. Falling back to custom_yolov9e.pt")
        model_path = MODEL_PRESETS["v9e"]

    # 3. Print Banner
    print("\n" + "=" * 70)
    print("🚁 DRONE TARGETING SYSTEM - LOCAL TEST & BENCHMARK PIPELINE")
    print("=" * 70)
    print(f"  • Video Source:      {source_val}")
    print(f"  • Model Checkpoint:  {model_path.name} ({model_path})")
    print(f"  • Inference Device:  {args.device}")
    print(f"  • QGC UDP Output:    {'Disabled (--no-qgc)' if args.no_qgc else args.udp}")
    print(f"  • Telemetry Source:  {'ArduPilot SITL (UDP 14551)' if args.ardupilot else 'Simulated Mock Autopilot'}")
    print("=" * 70 + "\n")

    # 4. Load YOLO Model
    print("Loading YOLO detector...")
    model = YOLO(str(model_path))
    model.to(args.device)
    print(f"Model '{model_path.name}' loaded successfully! ({len(model.names)} classes detected)\n")

    # 5. Open Video Capture
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Failed to open video source: {source_val}")
        sys.exit(1)

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Input Stream Opened: {in_w}x{in_h} @ {fps_in:.1f} FPS")

    # Check CUDA Availability
    if not torch.cuda.is_available() and args.device == "cpu":
        print("\n" + "!" * 75)
        print("⚠️  WARNING: PYTORCH CANNOT DETECT CUDA (Running on CPU = ~1.5 FPS!)")
        print("    On Ubuntu laptops, this happens after sleep/suspend.")
        print("    To restore GPU instantly, run: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm")
        print("!" * 75 + "\n")

    # Exact Client Operator Resolution (1280x700 proven in qgc/inference_stream.py)
    out_w, out_h = 1280, 700

    # 6. Initialize Camera 3D Ray-Projection
    projection = CameraProjection(
        native_width=in_w,
        native_height=in_h,
        fov_h_deg=70.0,
        fov_v_deg=45.0,
        max_zoom=10.0,
        inference_width=out_w,
        inference_height=out_h,
    )

    # 7. Initialize Telemetry (Mock vs ArduPilot SITL)
    global telemetry_client
    if args.ardupilot:
        telemetry_client = ArduPilotTelemetryWrapper(connection_string="udpin:127.0.0.1:14551")
    else:
        telemetry_client = MockTelemetry(lat=28.6139, lon=77.2090, alt_rel=50.0, gimbal_pitch=-45.0)

    # 8. Initialize Dedicated FFmpeg Output Thread for QGC
    ffmpeg_proc = None
    qgc_thread = None
    if not args.no_qgc:
        try:
            cmd = make_ffmpeg_udp_cmd(args.ffmpeg_bin, out_w, out_h, 30, args.udp, mode=args.stream_mode)
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            qgc_thread = threading.Thread(
                target=qgc_output_thread_func,
                args=(ffmpeg_proc, 30.0, out_w, out_h),
                name="qgc_encoder",
                daemon=True
            )
            qgc_thread.start()
            print(f"[FFmpeg] Dedicated 30 FPS encoder ({args.stream_mode.upper()}) streaming to {args.udp} for QGroundControl...")
        except Exception as e:
            print(f"[FFmpeg Warning] Could not start FFmpeg stream: {e}")
            ffmpeg_proc = None

    # 9. OpenCV UI Window
    window_name = f"Operator View - {model_path.stem}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, out_w, out_h)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nControls:")
    print("  [Left-Click] Click on any detected object box to select target & send GUIDED waypoint")
    print("  [Space]      Pause / Resume video")
    print("  [Q / Esc]    Quit test pipeline\n")

    frame_count = 0
    t_start = time.time()
    last_log_time = 0.0
    fps_display = float(fps_in)
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if not source_is_index and not args.no_loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("End of video stream.")
                        break

                frame_resized = cv2.resize(frame, (out_w, out_h))

                # Fetch Telemetry
                tel = telemetry_client.get_state()

                # Run YOLO Inference
                results = model.predict(frame_resized, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)

                # Process Detections & Compute Ray-Projection GPS
                new_targets = []
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = model.names.get(cls_id, f"cls_{cls_id}")
                        conf_val = float(box.conf[0].cpu().numpy())

                        # Ground contact point
                        bx = (xyxy[0] + xyxy[2]) / 2.0
                        by = float(xyxy[3])

                        gps_result = projection.pixel_to_gps(bx, by, tel, ground_elevation_msl=0.0)
                        dist = gps_result["distance_m"] if gps_result else None

                        new_targets.append({
                            "bbox": xyxy,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "conf": conf_val,
                            "gps": gps_result,
                            "dist": dist
                        })

                with state_lock:
                    current_targets.clear()
                    current_targets.extend(new_targets)

                frame_count += 1
                now = time.time()
                elapsed = now - t_start
                if elapsed > 0:
                    fps_display = frame_count / elapsed

                # Console logging for terminal visibility & video screen recordings (Kazam / terminal)
                # Adaptive update: every ~0.5s or every frame if FPS is low (< 4 FPS)
                if (now - last_log_time >= 0.5) or (fps_display < 4.0):
                    last_log_time = now
                    if len(new_targets) > 0:
                        counts = Counter(t["class_name"] for t in new_targets)
                        count_str = ", ".join([f"{c}x {k}" for k, c in counts.items()])
                        top_details = "; ".join([
                            f"{t['class_name']} {t['conf']*100:.0f}%" + (f" @ {t['dist']:.0f}m" if t['dist'] else "")
                            for t in new_targets[:4]
                        ])
                        if len(new_targets) > 4:
                            top_details += f" (+{len(new_targets)-4} more)"
                        lock_status = f" | [LOCKED: {active_lock_target['class_name']}]" if active_lock_target else ""
                        print(f"[TRACK Frame {frame_count:04d} | {fps_display:4.1f} FPS] Detected ({len(new_targets)}): {count_str} | {top_details}{lock_status}")
                    else:
                        lock_status = f" | [LOCKED: {active_lock_target['class_name']}]" if active_lock_target else ""
                        print(f"[SCAN  Frame {frame_count:04d} | {fps_display:4.1f} FPS] Scanning... (0 targets in FOV){lock_status}")

            # Determine status banner
            global status_banner, banner_expiry
            curr_time = time.time()
            active_banner = status_banner if curr_time < banner_expiry else "Click any bounding box to lock target & guide drone"

            # Draw HUD
            hud_frame = frame_resized.copy()
            draw_hud(hud_frame, tel, current_targets, active_lock_target, active_banner, fps_display, model_path.stem)

            # Update shared frame for dedicated QGC output thread
            global latest_hud_frame
            with output_lock:
                latest_hud_frame = hud_frame

            # Display Operator Window
            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Exiting test pipeline...")
                break
            elif key == ord(' '):
                paused = not paused
                print("[Status] Paused" if paused else "[Status] Resumed")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")

    finally:
        stop_event.set()
        cap.release()
        if qgc_thread:
            qgc_thread.join(timeout=2)
        if ffmpeg_proc:
            try:
                if ffmpeg_proc.stdin:
                    ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=2)
            except Exception:
                ffmpeg_proc.kill()
        if telemetry_client:
            telemetry_client.stop()
        cv2.destroyAllWindows()
        print("Pipeline shutdown clean.")


if __name__ == "__main__":
    main()
