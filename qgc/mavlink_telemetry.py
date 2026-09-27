import threading
import time
import math
import logging
from pymavlink import mavutil

logger = logging.getLogger("MAVLinkTelemetry")

class MAVLinkTelemetry:
    def __init__(self, connection_string="udpin:localhost:14550", baudrate=57600):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.mav = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Telemetry State Buffer
        # We will keep the last N states to allow time-syncing with delayed video frames
        self.history = []
        self.history_max_len = 100
        
        # Current state
        self.current_state = {
            "lat": 0.0,
            "lon": 0.0,
            "alt_msl": 0.0,
            "alt_rel": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "gimbal_pitch": 0.0,
            "gimbal_yaw": 0.0,
            "zoom_level": 1.0,
            "timestamp": 0.0
        }

    def start(self):
        self.running = True
        try:
            logger.info(f"Connecting to MAVLink on: {self.connection_string} at {self.baudrate} baud")
            self.mav = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate)
            # Send request for data streams
            self.mav.wait_heartbeat(timeout=5.0)
            logger.info("MAVLink Heartbeat received! Requesting data streams...")
            self.mav.mav.request_data_stream_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
            )
        except Exception as e:
            logger.error(f"Failed to connect to MAVLink: {e}.")
            return False

        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.mav:
            self.mav.close()
        logger.info("MAVLink telemetry stopped.")

    def _recv_loop(self):
        while self.running:
            try:
                msg = self.mav.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue
                
                msg_type = msg.get_type()
                updated = False
                
                with self.lock:
                    if msg_type == 'GLOBAL_POSITION_INT':
                        self.current_state["lat"] = msg.lat / 1e7
                        self.current_state["lon"] = msg.lon / 1e7
                        self.current_state["alt_msl"] = msg.alt / 1000.0
                        self.current_state["alt_rel"] = msg.relative_alt / 1000.0
                        updated = True
                        
                    elif msg_type == 'ATTITUDE':
                        self.current_state["roll"] = math.degrees(msg.roll)
                        self.current_state["pitch"] = math.degrees(msg.pitch)
                        yaw_deg = math.degrees(msg.yaw)
                        self.current_state["yaw"] = yaw_deg if yaw_deg >= 0 else yaw_deg + 360.0
                        updated = True
                        
                    elif msg_type == 'MOUNT_STATUS':
                        # Mount status usually gives pitch, roll, yaw in centi-degrees
                        self.current_state["gimbal_pitch"] = msg.pointing_a / 100.0
                        self.current_state["gimbal_yaw"] = msg.pointing_c / 100.0
                        updated = True
                        
                    elif msg_type == 'GIMBAL_DEVICE_ATTITUDE_STATUS':
                        # Quaternion to Euler
                        q = msg.q
                        # Pitch
                        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
                        pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)
                        self.current_state["gimbal_pitch"] = math.degrees(pitch)
                        # Yaw
                        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
                        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
                        self.current_state["gimbal_yaw"] = math.degrees(math.atan2(siny_cosp, cosy_cosp))
                        updated = True
                        
                    elif msg_type == 'CAMERA_SETTINGS':
                        self.current_state["zoom_level"] = msg.zoomLevel
                        updated = True

                    if updated:
                        self.current_state["timestamp"] = time.time()
                        # Deep copy the state into history
                        self.history.append(dict(self.current_state))
                        if len(self.history) > self.history_max_len:
                            self.history.pop(0)
                            
            except Exception as e:
                logger.debug(f"Error in MAVLink loop: {e}")

    def get_state_at_time(self, timestamp):
        """Finds the telemetry state closest to the given timestamp."""
        with self.lock:
            if not self.history:
                return dict(self.current_state)
            
            # Find the state with the minimum time difference
            best_state = min(self.history, key=lambda x: abs(x["timestamp"] - timestamp))
            return dict(best_state)

    def send_target_command(self, lat, lon, alt_msl):
        """Sends a SET_POSITION_TARGET_GLOBAL_INT command."""
        if not self.mav:
            return False
            
        logger.info(f"Sending drone to Target -> Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt_msl:.1f}")
        
        # Coordinate frame: MAV_FRAME_GLOBAL_INT (absolute altitude MSL)
        # Type mask: Ignore velocity and acceleration (0b0000111111111000)
        try:
            self.mav.mav.set_position_target_global_int_send(
                0, # time_boot_ms
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                0b0000111111111000,
                int(lat * 1e7),
                int(lon * 1e7),
                alt_msl,
                0, 0, 0, # vx, vy, vz
                0, 0, 0, # afx, afy, afz
                0, 0 # yaw, yaw_rate
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send target command: {e}")
            return False
