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


def qgc_output_thread_func(encoder: subprocess.Popen, fps: float, width: int = 1280, height: int = 720):
    """
    High-precision monotonic fixed-FPS pusher that repeats the last frame
    to keep QGroundControl stream alive. Automatically formats portrait frames for 16:9 QGC UDP.
    """
    interval = 1.0 / float(fps)
    blank = np.zeros((height, width, 3), dtype=np.uint8)

    while not stop_event.is_set():
        t0 = time.perf_counter()

        with output_lock:
            frame = latest_hud_frame

        if frame is None:
            out_frame = blank
        elif frame.shape[1] != width or frame.shape[0] != height:
            fh, fw = frame.shape[:2]
            scale_qgc = min(width / fw, height / fh)
            nw = int(round(fw * scale_qgc))
            nh = int(round(fh * scale_qgc))
            resized_qgc = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            out_frame = np.zeros((height, width, 3), dtype=np.uint8)
            px = (width - nw) // 2
            py = (height - nh) // 2
            out_frame[py:py + nh, px:px + nw] = resized_qgc
        else:
            out_frame = frame

        try:
            encoder.stdin.write(out_frame.tobytes())
        except (BrokenPipeError, OSError):
            break

        elapsed = time.perf_counter() - t0
        stop_event.wait(max(0.0, interval - elapsed))


def draw_corner_brackets(img, x1, y1, x2, y2, color, thickness=1, length=9):
    """Draws sleek military corner brackets instead of solid obstructing rectangles."""
    bw = x2 - x1
    bh = y2 - y1
    l = max(3, min(length, bw // 3, bh // 3))
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness, cv2.LINE_AA)


def draw_hud(frame, tel, targets, active_target, banner_text, fps, model_name, is_recording=False, declutter_mode="TACTICAL", conf_threshold=0.40):
    """
    Draws defense-grade HUD overlay with tactical reticles and clean non-colliding typography.
    Avoids blinding white coordinate blurs by showing target GPS solely on the active locked target.
    """
    h, w = frame.shape[:2]

    # 1. Top Telemetry Bar (Dark semi-transparent glassmorphism)
    bar_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (12, 16, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    source_tag = tel.get("source", "MOCK")
    tel_str = (
        f"[{model_name.upper()}] ALT: {tel['alt_rel']:.1f}m | "
        f"PITCH: {tel['gimbal_pitch']:.1f}° | "
        f"HDG: {tel['yaw']:.0f}° | "
        f"FPS: {fps:.1f} | CONF: {conf_threshold:.2f} | MODE: {declutter_mode}"
    )
    cv2.putText(frame, tel_str, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

    # Top-Right Recording Indicator
    if is_recording:
        cv2.circle(frame, (w - 75, 18), 5, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 64, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

    # 2. Draw Targets (Tactical Corner Brackets & Anti-Collision Reticles)
    for t in targets:
        x1, y1, x2, y2 = t["bbox"]
        cls = t["class_name"]
        conf = t["conf"]
        gps = t["gps"]
        dist = t["dist"]
        bw = x2 - x1
        bh = y2 - y1

        is_locked = (active_target and active_target["bbox"] == t["bbox"])

        if is_locked:
            # === ACTIVE LOCKED TARGET (Prominent Red Engagement Reticle) ===
            lock_color = (0, 30, 255)
            draw_corner_brackets(frame, x1, y1, x2, y2, lock_color, thickness=2, length=14)

            # Central Tracking Circle & Crosshair
            tcx = int((x1 + x2) / 2)
            tcy = int((y1 + y2) / 2)
            cv2.circle(frame, (tcx, tcy), 18, lock_color, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (tcx, tcy), lock_color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)

            # Top Header Tag
            top_label = f"LOCKED: {cls.upper()} ({conf*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(top_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            ly1 = max(bar_h + 4, y1 - 24)
            cv2.rectangle(frame, (x1, ly1), (x1 + tw + 8, ly1 + th + 6), lock_color, -1)
            cv2.putText(frame, top_label, (x1 + 4, ly1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            # Bottom GPS Telemetry Tag
            if gps:
                coord_str = f"LAT: {gps['lat']:.5f}°  LON: {gps['lon']:.5f}° | RNG: {dist:.0f}m"
                (cw, ch), _ = cv2.getTextSize(coord_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                by1 = min(h - 40, y2 + 6)
                cv2.rectangle(frame, (x1 - 2, by1), (x1 + cw + 8, by1 + ch + 8), (10, 10, 10), -1)
                cv2.rectangle(frame, (x1 - 2, by1), (x1 + cw + 8, by1 + ch + 8), lock_color, 1)
                cv2.putText(frame, coord_str, (x1 + 3, by1 + ch + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        else:
            # === UNLOCKED TARGET (Tactical Corner Brackets & Anti-Collision Tags) ===
            reticle_color = (0, 240, 170)  # Modern military cyan/emerald
            draw_corner_brackets(frame, x1, y1, x2, y2, reticle_color, thickness=1, length=8)

            # Decide whether to render text based on declutter mode and object scale
            should_show_label = False
            if declutter_mode == "FULL":
                should_show_label = True
            elif declutter_mode == "TACTICAL":
                # Show labels for vehicles, aircraft, artillery, drones, or larger boxes (prevents crowd collisions)
                is_high_value = any(k in cls for k in ["vehicle", "tank", "helicopter", "airplane", "truck", "drone", "artillery", "carrier"])
                if is_high_value or bh >= 40:
                    should_show_label = True

            if should_show_label:
                # Clean, compact label
                tag_text = f"{cls} {conf*100:.0f}%"
                if dist is not None and bh >= 50:
                    tag_text += f" | {dist:.0f}m"
                (lw, lh), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                ty1 = max(bar_h + 2, y1 - lh - 5)

                # Semi-transparent dark pill background (prevents overlapping solid blocks)
                sub_w = min(w - x1, lw + 6)
                sub_h = lh + 4
                if sub_w > 0 and sub_h > 0 and ty1 + sub_h <= h:
                    sub = frame[ty1:ty1 + sub_h, x1:x1 + sub_w]
                    black_rect = np.zeros_like(sub)
                    cv2.addWeighted(sub, 0.35, black_rect, 0.65, 0, sub)
                cv2.putText(frame, tag_text, (x1 + 3, ty1 + lh), cv2.FONT_HERSHEY_SIMPLEX, 0.36, reticle_color, 1, cv2.LINE_AA)

    # 3. Bottom Status Banner
    bot_y = h - 10
    cv2.rectangle(frame, (0, h - 28), (w, h), (12, 16, 20), -1)
    banner_color = (0, 220, 255) if "LOCKED" in banner_text else (190, 190, 190)
    cv2.putText(frame, banner_text, (10, bot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, banner_color, 1, cv2.LINE_AA)


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

    in_output = CURRENT_DIR / "output_video" / src_input
    if in_output.exists():
        return str(in_output)

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
                        help="Video filename, path, webcam index (0), or RTSP URL. Auto-checks 'videos/' and 'output_video/'.")
    parser.add_argument("--model", default="v9e",
                        help="Model preset ('v9e', '26s', 'v8s') or path to .pt file.")
    parser.add_argument("--conf", type=float, default=0.40, help="Initial confidence threshold (default: 0.40, adjust live with [ and ])")
    parser.add_argument("--mode", choices=["tactical", "full", "minimal"], default="tactical",
                        help="HUD Declutter Mode: 'tactical' (clean reticles & vehicle labels), 'minimal' (brackets only), 'full' (all labels)")
    parser.add_argument("--show-all-labels", "--all-labels", action="store_true",
                        help="Force all labels and confidence scores to appear on every detected bounding box (overrides decluttering)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution (e.g. 640 for speed, 1024/1280 for tiny aerial targets)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--ardupilot", action="store_true", help="Connect live to ArduPilot SITL / MAVProxy on UDP 14551")
    parser.add_argument("--udp", default="udp://127.0.0.1:5600", help="UDP stream URL for QGroundControl")
    parser.add_argument("--stream-mode", choices=["mpegts", "rtp", "h264"], default="mpegts",
                        help="Streaming container: 'mpegts' (for QGC MPEG-TS) or 'rtp'/'h264' (for QGC UDP h.264)")
    parser.add_argument("--ffmpeg-bin", default="/usr/bin/ffmpeg", help="Path to system ffmpeg binary")
    parser.add_argument("--no-qgc", action="store_true", help="Disable QGC UDP stream broadcast")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop video file when finished")
    parser.add_argument("--save", action="store_true", help="Record and save HUD video to output_video/")
    parser.add_argument("--save-path", default=None, help="Custom output filename or path for saved annotated video")
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

    # Adaptive Operator Canvas Resolution:
    # If video is portrait (in_h > in_w, e.g. 1080x1920), adapt display vertically to fill 100% of window with ZERO black sidebars!
    if in_h > in_w:
        out_h = min(in_h, 960)
        out_w = int(round(in_w * (out_h / in_h)))
    else:
        out_w = min(in_w, 1280)
        out_h = int(round(in_h * (out_w / in_w)))

    # Standard resolution for QGC streaming
    qgc_w, qgc_h = 1280, 720

    # 6. Initialize Camera 3D Ray-Projection (using native camera optical parameters)
    projection = CameraProjection(
        native_width=in_w,
        native_height=in_h,
        fov_h_deg=70.0,
        fov_v_deg=45.0,
        max_zoom=10.0,
        inference_width=in_w,
        inference_height=in_h,
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
            cmd = make_ffmpeg_udp_cmd(args.ffmpeg_bin, qgc_w, qgc_h, 30, args.udp, mode=args.stream_mode)
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            qgc_thread = threading.Thread(
                target=qgc_output_thread_func,
                args=(ffmpeg_proc, 30.0, qgc_w, qgc_h),
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
    print("  [Left-Click] Select & Lock Target (dispatches GUIDED waypoint)")
    print("  [D]          Toggle Declutter Mode (TACTICAL -> MINIMAL -> FULL)")
    print("  [[ / ]]      Decrease / Increase Confidence Threshold (- / +)")
    print("  [R]          Start / Stop Video Recording to output_video/")
    print("  [Space]      Pause / Resume video")
    print("  [Q / Esc]    Quit test pipeline\n")

    current_conf = float(args.conf)
    declutter_mode = "FULL" if args.show_all_labels else args.mode.upper()

    # 10. Video Recording / Writer
    video_writer = None
    recording = args.save
    recorded_frames = 0
    save_file_path = None

    def get_or_create_writer():
        nonlocal video_writer, save_file_path
        if video_writer is not None:
            return video_writer
        out_dir = CURRENT_DIR / "output_video"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_path:
            p = Path(args.save_path)
            if p.is_absolute():
                save_file_path = p
            elif str(p).startswith("output_video/"):
                save_file_path = CURRENT_DIR / p
            else:
                save_file_path = out_dir / p
        else:
            src_stem = Path(resolved_source).stem if not source_is_index else f"cam_{source_val}"
            save_file_path = out_dir / f"{src_stem}_{model_path.stem}_annotated.mp4"

        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = float(fps_in) if (fps_in and fps_in > 0) else 30.0
        video_writer = cv2.VideoWriter(str(save_file_path), fourcc, fps_out, (out_w, out_h))
        if not video_writer.isOpened():
            # Fallback to XVID / avc1 if mp4v fails
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(str(save_file_path), fourcc, fps_out, (out_w, out_h))
        print(f"🎥 [VideoWriter] Initialized recording ({out_w}x{out_h} @ {fps_out:.1f} FPS) -> {save_file_path}\n")
        return video_writer

    if recording:
        get_or_create_writer()

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

                # 1. Aspect-Ratio Preserving Letterbox Canvas for Display & QGC Stream (1280x720)
                in_h, in_w = frame.shape[:2]
                scale = min(out_w / in_w, out_h / in_h)
                new_w = int(round(in_w * scale))
                new_h = int(round(in_h * scale))
                pad_x = (out_w - new_w) // 2
                pad_y = (out_h - new_h) // 2

                frame_scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = frame_scaled

                # 2. Fetch Telemetry
                tel = telemetry_client.get_state()

                # 3. Run Native YOLO Inference (avoids aspect-ratio distortion, preserves small targets)
                results = model.predict(frame, conf=current_conf, imgsz=args.imgsz, device=args.device, agnostic_nms=True, iou=0.45, verbose=False)

                # 4. Process Detections, Map Bounding Boxes to Display Canvas & Compute Ray-Projection GPS
                new_targets = []
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        xyxy_native = box.xyxy[0].cpu().numpy().tolist()
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = model.names.get(cls_id, f"cls_{cls_id}")
                        conf_val = float(box.conf[0].cpu().numpy())

                        # Map bounding box coordinates onto 1280x720 Operator Display Canvas
                        cx1 = int(round(xyxy_native[0] * scale + pad_x))
                        cy1 = int(round(xyxy_native[1] * scale + pad_y))
                        cx2 = int(round(xyxy_native[2] * scale + pad_x))
                        cy2 = int(round(xyxy_native[3] * scale + pad_y))
                        bbox_canvas = [cx1, cy1, cx2, cy2]

                        # Ground contact point in native camera coordinates for physically accurate 3D ray projection
                        bx_native = (xyxy_native[0] + xyxy_native[2]) / 2.0
                        by_native = float(xyxy_native[3])

                        gps_result = projection.pixel_to_gps(bx_native, by_native, tel, ground_elevation_msl=0.0)
                        dist = gps_result["distance_m"] if gps_result else None

                        new_targets.append({
                            "bbox": bbox_canvas,
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
            hud_frame = canvas.copy()
            draw_hud(hud_frame, tel, current_targets, active_lock_target, active_banner, fps_display, model_path.stem, is_recording=recording, declutter_mode=declutter_mode, conf_threshold=current_conf)

            # Record frame to file if active
            if recording:
                writer = get_or_create_writer()
                if writer is not None and writer.isOpened():
                    writer.write(hud_frame)
                    recorded_frames += 1

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
            elif key == ord('r') or key == ord('R'):
                recording = not recording
                if recording:
                    get_or_create_writer()
                    status_banner = f"REC ACTIVE: {save_file_path.name}"
                    banner_expiry = time.time() + 4.0
                    print(f"\n🔴 [REC] Started recording to: {save_file_path}")
                else:
                    status_banner = f"REC PAUSED ({recorded_frames} frames)"
                    banner_expiry = time.time() + 4.0
                    print(f"\n⏸️  [REC] Paused recording ({recorded_frames} frames captured)")
            elif key == ord('[') or key == ord('-'):
                current_conf = max(0.10, round(current_conf - 0.05, 2))
                status_banner = f"CONFIDENCE THRESHOLD: {current_conf:.2f}"
                banner_expiry = time.time() + 3.0
                print(f"[Controls] Confidence lowered to {current_conf:.2f}")
            elif key == ord(']') or key == ord('+') or key == ord('='):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                status_banner = f"CONFIDENCE THRESHOLD: {current_conf:.2f}"
                banner_expiry = time.time() + 3.0
                print(f"[Controls] Confidence raised to {current_conf:.2f}")
            elif key == ord('d') or key == ord('D'):
                modes = ["TACTICAL", "MINIMAL", "FULL"]
                idx = (modes.index(declutter_mode) + 1) % len(modes)
                declutter_mode = modes[idx]
                status_banner = f"HUD DECLUTTER MODE: {declutter_mode}"
                banner_expiry = time.time() + 3.0
                print(f"[Controls] HUD Mode changed to: {declutter_mode}")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")

    finally:
        stop_event.set()
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"🎬 [VideoWriter] Successfully saved {recorded_frames} frames to: {save_file_path}")
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
