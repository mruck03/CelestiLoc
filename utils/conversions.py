from astropy.io import fits
import os
import numpy as np
import cv2
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import open3d as o3d
from image_filter import K
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, minimize
from astropy.time import Time
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import copy
from scipy.spatial import cKDTree

def viz_projection(image_pts, proj_pts, image=None, figsize=(8, 8)):
    """
    Visualize the projection difference between actual and projected image points.

    Args:
        image_pts (N, 2): Actual image points (observed).
        proj_pts (N, 2): Projected points using estimated pose.
        image: Optional image to overlay the points on (as np.array).
        figsize: Size of the plot.
    """
    image_pts = np.array(image_pts)
    proj_pts = np.array(proj_pts)

    if image is not None:
        img_viz = image.copy()
        for (x_obs, y_obs), (x_proj, y_proj) in zip(image_pts, proj_pts):
            cv2.circle(img_viz, (int(x_obs), int(y_obs)), 4, (0, 255, 0), -1)  # green: actual
            cv2.circle(img_viz, (int(x_proj), int(y_proj)), 4, (0, 0, 255), -1)  # red: projected
            cv2.line(img_viz, (int(x_obs), int(y_obs)), (int(x_proj), int(y_proj)), (255, 255, 0), 1)

        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
        plt.title("Projection Visualization (Green = Actual, Red = Projected)")
        plt.axis('off')
        plt.show()

    else:
        plt.figure(figsize=figsize)
        plt.scatter(image_pts[:, 0], image_pts[:, 1], color='green', label='Observed')
        plt.scatter(proj_pts[:, 0], proj_pts[:, 1], color='red', label='Projected')
        for (xo, yo), (xp, yp) in zip(image_pts, proj_pts):
            plt.plot([xo, xp], [yo, yp], color='gray', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title("Projection Residuals")
        plt.show()

def celestial_to_cartesian(ra, dec, dist):
    x = dist * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = dist * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z = dist * np.sin(np.deg2rad(dec))
    return x, y, z

def celestial_to_ecef(star_data, time_str="2025-04-01T04:00:00"):

    obs_time = Time(time_str)

    gst = obs_time.sidereal_time('mean', longitude=0)  # GST in hours
    gst_rad = gst.to(u.rad)

    rotation_matrix = np.array([
        [ np.cos(-gst_rad), -np.sin(-gst_rad), 0],
        [ np.sin(-gst_rad),  np.cos(-gst_rad), 0],
        [ 0,                0,                1]
    ])

    x, y, z = celestial_to_cartesian(star_data[:, 2], star_data[:, 3], star_data[:, 4])

    star_coords = np.vstack((x, y, z))

    star_coords_adj = rotation_matrix @ star_coords

    return star_coords_adj.T

def plot_3d_pose(dataset, quat=None, tvec=None):
    
    #Vizualize
    if np.all(quat) != None:
        T = quaternion_to_transformation_matrix(quat, tvec/np.linalg.norm(tvec))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
        frame.transform(T)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])

    
    all_obj_norm = []

    for entry in dataset:
        star_data = entry["star_data"]
        obs_time = entry["obs_time"]

        print(star_data.shape)

        star_data[:, 4] /= star_data[:, 4].max()
        star_data[:, 4] += 10
        stars_metric_norm = celestial_to_ecef(star_data, time_str=obs_time.isot)

        all_obj_norm.append(stars_metric_norm)

    all_obj_norm = np.vstack(all_obj_norm)
    pc_adj = all_obj_norm

    pcd_adj = o3d.geometry.PointCloud()
    pcd_adj.points = o3d.utility.Vector3dVector(pc_adj)
    pcd_adj.colors = o3d.utility.Vector3dVector(np.ones((len(pc_adj), 3)) * [0, 0, 1])

    earth = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    

    # Create coordinate frame (origin)
    if np.all(quat) != None:
        o3d.visualization.draw_geometries([frame, axis, pcd_adj, earth])
    else:
        o3d.visualization.draw_geometries([axis, pcd_adj, earth])


def quaternion_to_transformation_matrix(quaternion, translation):
    """
    Converts a quaternion and a translation vector into a 4x4 transformation matrix.
    
    Args:
        quaternion: (w, x, y, z) tuple or list
        translation: (tx, ty, tz) tuple or list
    
    Returns:
        4x4 NumPy array representing the transformation matrix.
    """
    # Convert quaternion to rotation matrix
    r = R.from_quat(quaternion)  # SciPy uses (x, y, z, w)
    R_matrix = r.as_matrix()  # 3x3 rotation matrix
    
    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = translation
    
    return T

def lla_to_ecef(lat, lon, alt_m=0.0):
    """
    Convert latitude, longitude, and altitude to ECEF (Earth-Centered, Earth-Fixed) coordinates.
    
    Parameters:
        lat (float): Latitude in radians
        lon (float): Longitude in radians
        alt_m (float): Altitude in meters (default is 0 for sea level)
        
    Returns:
        x, y, z (float): ECEF coordinates in meters
    """
    # WGS84 ellipsoid constants
    a = 6378137.0          # Equatorial radius in meters
    f = 1 / 298.257223563  # Flattening
    e2 = f * (2 - f)       # Square of eccentricity

    lon = ((lon + np.pi) % (2*np.pi)) - np.pi

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)  # Radius of curvature

    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = ((1 - e2) * N + alt_m) * np.sin(lat)

    return np.array([x, y, z])