import numpy as np
import math
import logging

logger = logging.getLogger("RayProjection")

def rotation_matrix_x(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ], dtype=np.float64)

def rotation_matrix_y(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ], dtype=np.float64)

def rotation_matrix_z(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float64)

class CameraProjection:
    def __init__(self, 
                 native_width=2560, native_height=1400, 
                 fov_h_deg=71.5, fov_v_deg=43.0, 
                 max_zoom=10.0,
                 inference_width=1280, inference_height=700):
        
        self.native_width = native_width
        self.native_height = native_height
        
        self.fov_h_rad = math.radians(fov_h_deg)
        self.fov_v_rad = math.radians(fov_v_deg)
        
        self.max_zoom = max_zoom
        
        # Scaling factor from inference resolution to native resolution
        self.scale_x = native_width / float(inference_width)
        self.scale_y = native_height / float(inference_height)
        
        # Camera (Z forward, X right, Y down) to Gimbal (X forward, Y right, Z down)
        self.R_c2g = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float64)

    def pixel_to_gps(self, pixel_u, pixel_v, telemetry_state, ground_elevation_msl=0.0):
        """
        Converts pixel coordinates (u, v) from inference image to real-world GPS.
        Assumes flat terrain at ground_elevation_msl.
        """
        # 1. Un-scale pixel to native camera resolution
        u_native = pixel_u * self.scale_x
        v_native = pixel_v * self.scale_y
        
        zoom = telemetry_state.get("zoom_level", 1.0)
        
        # 2. Calculate effective focal lengths with zoom
        # Linear zoom approximation for FOV: effective_fov = fov / zoom
        # Which translates to focal_length * zoom
        f_x = (self.native_width / (2.0 * math.tan(self.fov_h_rad / 2.0))) * zoom
        f_y = (self.native_height / (2.0 * math.tan(self.fov_v_rad / 2.0))) * zoom
        
        # 3. Ray in camera frame
        x_c = (u_native - self.native_width / 2.0) / f_x
        y_c = (v_native - self.native_height / 2.0) / f_y
        z_c = 1.0
        
        v_c = np.array([x_c, y_c, z_c], dtype=np.float64)
        v_c = v_c / np.linalg.norm(v_c)
        
        # 4. Rotate from Camera to Gimbal frame
        v_g = self.R_c2g.dot(v_c)
        
        # 5. Rotate from Gimbal to Earth (NED) frame
        # Standard stabilized gimbal: 
        # Pitch is absolute (from horizon), roll is 0 (stabilized), yaw is relative to drone heading
        g_pitch_rad = math.radians(telemetry_state["gimbal_pitch"])
        g_yaw_rad = math.radians(telemetry_state["gimbal_yaw"])
        
        d_roll_rad = math.radians(telemetry_state["roll"])
        d_pitch_rad = math.radians(telemetry_state["pitch"])
        d_yaw_rad = math.radians(telemetry_state["yaw"])
        
        # Absolute Gimbal Yaw = Drone Yaw + Relative Gimbal Yaw
        abs_yaw_rad = d_yaw_rad + g_yaw_rad
        
        # Rotation Matrix: Gimbal to Earth
        R_g2e = rotation_matrix_z(abs_yaw_rad).dot(
            rotation_matrix_y(g_pitch_rad)
        )
        
        # Ray in Earth (NED) frame
        v_e = R_g2e.dot(v_g)
        
        # 6. Intersect ray with ground plane
        v_N, v_E, v_D = v_e
        
        # Drone position in NED (Lat, Lon origin; Alt down)
        drone_alt_msl = telemetry_state["alt_msl"]
        
        P_cam_ned = np.array([0.0, 0.0, -drone_alt_msl], dtype=np.float64)
        
        # If the ray is pointing upwards or parallel to ground, no intersection
        if v_D <= 0.0001:
            logger.debug("Ray points skyward or parallel to ground. No intersection.")
            return None
            
        # Ground plane is at D = -ground_elevation_msl
        # Solve: P_cam_ned[2] + t * v_D = -ground_elevation_msl
        t = (-ground_elevation_msl - P_cam_ned[2]) / v_D
        
        if t < 0:
            logger.debug("Intersection is behind the camera.")
            return None
            
        # Target position in NED (offset from Drone in meters)
        N_target = t * v_N
        E_target = t * v_E
        
        # 7. Convert NED offset to GPS coordinates
        R_earth = 6378137.0
        
        d_lat_rad = N_target / R_earth
        d_lon_rad = E_target / (R_earth * math.cos(math.radians(telemetry_state["lat"])))
        
        target_lat = telemetry_state["lat"] + math.degrees(d_lat_rad)
        target_lon = telemetry_state["lon"] + math.degrees(d_lon_rad)
        target_alt = ground_elevation_msl
        
        distance = math.sqrt(N_target**2 + E_target**2 + (drone_alt_msl - ground_elevation_msl)**2)
        
        return {
            "lat": target_lat,
            "lon": target_lon,
            "alt_msl": target_alt,
            "distance_m": distance
        }
