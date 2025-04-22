import numpy as np
import datetime

def gps_to_ecef(lat, lon, alt):
    alt = alt/3
    # WGS84 ellipsoid constants
    a = 6378137  # semi-major axis
    e_sq = 6.69437999014e-3  # square of eccentricity

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e_sq) + alt) * np.sin(lat_rad)

    return np.array([x, y, z])

def project_point(K, point_3d):
    x, y, z = point_3d
    # if z <= 0:
    #     return None  # Behind camera
    point_2d = K @ np.array([x/z, y/z, 1])
    return point_2d[:2]

def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    # reference point (camera position)
    x0, y0, z0 = gps_to_ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0

    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    # rotation matrix from ECEF to ENU
    R = np.array([
        [-np.sin(lon0),  np.cos(lon0), 0],
        [-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)],
        [ np.cos(lat0)*np.cos(lon0),  np.cos(lat0)*np.sin(lon0), np.sin(lat0)]
    ])

    enu = R @ np.array([dx, dy, dz])
    return enu

def camera_rotation_matrix(bearing_deg, pitch_deg):
    bearing_rad = np.radians(bearing_deg)
    pitch_rad = np.radians(pitch_deg)

    # rotate ENU around Z (Up) for bearing
    R_z = np.array([
        [np.cos(bearing_rad), -np.sin(bearing_rad), 0],
        [np.sin(bearing_rad),  np.cos(bearing_rad), 0],
        [0,                   0,                    1]
    ])

    # rotate around camera's local X-axis for pitch
    R_x = np.array([
        [1, 0,                0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
    ])

    # first rotate by bearing, then apply pitch
    return R_x @ R_z