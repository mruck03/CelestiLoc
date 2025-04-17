from astropy.io import fits
import os
import numpy as np
import cv2
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, minimize
from astropy.time import Time
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import copy
from scipy.spatial import cKDTree

from utils.load_data import *
from utils.conversions import *

def residual_quaternion_t(params, quat, img_pts, star_pts, camera_matrix):
    """
    Computes the residuals between transformed points in frame A and points in frame B using a quaternion.

    Args:
        params (np.ndarray): Optimization parameters (first 4 values are quaternion, last 3 are translation).
        img_pts (np.ndarray): Array of shape (N, 2) representing indexes of image
        points_b (np.ndarray): Array of shape (N, 3) representing points of stars in world frame

    Returns:
        np.ndarray: Residuals (differences) between transformed points_a and points_b.
    """
    # Extract quaternion and translation parameters
    lat, lon = params


    t = lla_to_ecef(lat, lon)
    cam_2_world = quaternion_to_transformation_matrix(quat, t)

    world_2_cam = np.linalg.inv(cam_2_world)

    star_pts_homo = np.vstack((star_pts.T, np.ones(star_pts.shape[0])))
    star_pts_cam = world_2_cam[:3, :] @ star_pts_homo

    # Get normalized vectors from camera to world star points
    star_rays_cam = star_pts_cam.T / np.linalg.norm(star_pts_cam.T, axis=1, keepdims=True)

    # Back-project image points into camera rays
    img_pts_homo = np.vstack((img_pts.T, np.ones(img_pts.shape[0])))
    img_pts_3D = np.linalg.inv(camera_matrix) @ img_pts_homo

    img_rays = img_pts_3D.T / np.linalg.norm(img_pts_3D.T, axis=1, keepdims=True)

    # Compute angle between rays and star vectors
    dot_products = np.sum(star_rays_cam * img_rays, axis=1)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

    # print(f"Current Lat: {np.degrees(lat)}, Lon: {np.degrees(lon)}, Residual: {np.sum(angles**2)}")

    proj_points = camera_matrix @ star_pts_cam[:3, :]
    proj_points = (proj_points[:2, :] / proj_points[2, :]).T

    # Compute reprojection errors
    errors = np.linalg.norm(img_pts - proj_points, axis=1)

    # Compute the mean error
    sum_error = np.sum(errors)

    return sum_error #np.sum(angles**2)# or np.mean(angles) if needed


def is_camera_facing_down(quat, lat, lon):

    # Get ECEF unit "up" direction at this lat/lon
    up_vector = lla_to_ecef(lat, lon)
    up_vector /= np.linalg.norm(up_vector)

    # Get camera's forward direction in ECEF
    rot = R.from_quat(quat)
    cam_forward = rot.apply(np.array([0, 0, 1]))  # camera looks along +Z

    # If dot < 0 → facing into Earth
    dot = cam_forward @ up_vector
    return dot < 0

def solve_rotation_svd(image_points, star_points, camera_matrix, tvec):
    """
    Solves for optimal rotation that aligns camera rays to star directions using SVD.

    Parameters:
        image_points: (N, 2) 2D image coordinates (undistorted)
        star_points: (N, 3) 3D star positions in ECEF (already aligned to time)
        camera_matrix: (3, 3) camera intrinsics

    Returns:
        Quaternion representing rotation from camera to world (ECEF)
    """
    N = image_points.shape[0]

    # Step 1: Back-project image points to normalized rays in camera frame
    img_pts_h = np.hstack([image_points, np.ones((N, 1))])  # (N, 3)
    cam_rays = (np.linalg.inv(camera_matrix) @ img_pts_h.T).T  # (N, 3)
    cam_rays /= np.linalg.norm(cam_rays, axis=1, keepdims=True)

    # Step 2: Transform Stars into Camera frame and Normalize star vectors
    tvec = np.asarray(tvec).reshape(1, 3)
    star_dirs = star_points - tvec
    star_dirs /= np.linalg.norm(star_dirs, axis=1, keepdims=True)

    # Step 3: Compute SVD of covariance matrix
    H = cam_rays.T @ star_dirs  # 3x3
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    # Step 4: Convert to quaternion
    quat = R.from_matrix(R_opt).as_quat()  # (x, y, z, w)
    return quat

def wrap_longitude(lon):
    # Wrap to [-π, π]
    return ((lon + np.pi) % (2 * np.pi)) - np.pi

def clamp_latitude(lat):
    # Clamp to [-π/2, π/2]
    return np.clip(lat, -np.pi / 2, np.pi / 2)

def filter_visible_stars(lat, lon, star_points_ecef):
    # 1. Observer ECEF position
    obs_ecef = lla_to_ecef(lat, lon)

    # 2. Vector to each star from observer
    vectors_to_stars = star_points_ecef - obs_ecef  # shape (N, 3)
    vectors_to_stars /= np.linalg.norm(vectors_to_stars, axis=1, keepdims=True)

    # 3. Observer "up" direction
    observer_up = obs_ecef / np.linalg.norm(obs_ecef)

    # 4. Compute dot product
    dot_products = vectors_to_stars @ observer_up

    # 5. Keep stars with positive dot product (i.e., above the horizon)
    visible_mask = dot_products > 0

    return visible_mask


class Particle:
    def __init__(self, lat, lon, pitch, bearing, roll, quat=np.array([0, 0, 0, 1]), weight=1.0):
        self.lat = lat
        self.lon = lon
        self.pitch = pitch
        self.bearing = bearing
        self.roll = roll
        self.weight = weight
        self.quat = quat / np.linalg.norm(quat)

class ParticleFilter:
    def __init__(self, pitch, bearing, roll, num_particles, image_points, star_points, camera_matrix, percent_visible=0.8):
        self.particles = []
        min_visible = max(int(percent_visible * len(star_points)), 3)
        while len(self.particles) < num_particles:
            #Sample latitude and lon uniformly over the globe
            lat = np.arcsin(np.random.uniform(-1, 1)) 
            lon = np.random.uniform(-np.pi, np.pi)

            visible_mask = filter_visible_stars(lat, lon, star_points)
            if np.sum(visible_mask) < min_visible:
                continue


            camera_rotation = R.from_euler('zyx', [bearing, roll, pitch], degrees=False).as_matrix()
            # Combine camera-to-ENU rotation (rotated direction of camera in ENU)
            camera_in_enu =  camera_rotation  @ axis_to_cam @ np.eye(3)


            quat = R.from_matrix(enu_axes(lat, lon) @ camera_in_enu).as_quat()
            # tvec = lla_to_ecef(lat, lon)
            # quat = solve_rotation_svd(image_points, star_points, camera_matrix, tvec)
            # quat = quat / np.linalg.norm(quat)

            # if is_camera_facing_down(quat, lat, lon):
            #     continue
            
            self.particles.append(Particle(lat, lon, pitch, bearing, roll, quat))

    def predict(self, lat_std, lon_std, pitch_std, bearing_std, roll_std, image_points, star_points, camera_matrix, percent_visible=0.8):

        min_visible = max(int(percent_visible * len(star_points)), 3)
        for p in self.particles:
            lat = clamp_latitude(p.lat + np.random.normal(0, lat_std))
            lon = wrap_longitude(p.lon + np.random.normal(0, lon_std))

            visible_mask = filter_visible_stars(lat, lon, star_points)
            if np.sum(visible_mask) < min_visible:
                continue

            # Solve for rotation at proposed position
            pitch = p.pitch + np.random.normal(0, pitch_std)
            bearing = p.bearing + np.random.normal(0, bearing_std)
            roll = p.roll + np.random.normal(0, roll_std)
            camera_rotation = R.from_euler('zyx', [bearing, roll, pitch], degrees=False).as_matrix()
            # Combine camera-to-ENU rotation (rotated direction of camera in ENU)
            camera_in_enu =  camera_rotation  @ axis_to_cam @ np.eye(3)


            quat = R.from_matrix(enu_axes(lat, lon) @ camera_in_enu).as_quat()
            # tvec = lla_to_ecef(lat, lon)
            # quat = solve_rotation_svd(image_points, star_points, camera_matrix, tvec)
            # quat = quat / np.linalg.norm(quat)

            # # Reject if camera would be facing into Earth
            if is_camera_facing_down(quat, lat, lon):
                continue

            # Accept
            p.lat = lat
            p.lon = lon
            p.pitch = pitch
            p.bearing = bearing
            p.roll = roll
            p.quat = quat


    def update_weights(self, image_points, star_points, camera_matrix, scale=1):
        weights = []
        # for p in tqdm(self.particles, desc="Updating particles"):
        residuals = []

        for p in self.particles:            
            residual = residual_quaternion_t([p.lat, p.lon], p.quat, image_points, star_points, camera_matrix) #TODO: Change to Reprojection Error!
            residuals.append(residual)

        residuals = np.array(residuals)
        res_min = np.min(residuals)
        res_max = np.max(residuals)
        res_range = res_max - res_min if res_max != res_min else 1.0

        # print(f"res_range: {res_range}")

        residuals = (residuals - res_min) / res_range
        weights = np.exp(-residuals * scale)

        for i, p in enumerate(self.particles):
            p.weight = weights[i]

        total_weight = np.sum(weights)
        if total_weight == 0:
            # Avoid division by zero — assign equal weights
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)
        else:
            # Normalize with NumPy for accuracy
            normalized_weights = np.array([w / total_weight for w in weights])
            for i, p in enumerate(self.particles):
                p.weight = normalized_weights[i]


    def N_eff(self):
        weights = np.array([p.weight for p in self.particles], dtype=np.float64)
        total = np.sum(weights)

        if total == 0 or np.isnan(total):
            return 0.0

        weights /= total
        return 1.0 / np.sum(weights**2)

    def resample(self):
        """
        Low variance (systematic) resampling for particle filter.
        """
        N = len(self.particles)
        weights = np.array([p.weight for p in self.particles])
        weights /= np.sum(weights)  # Normalize to sum to 1

        # Cumulative sum
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # Ensure total sum is exactly 1.0

        r = np.random.uniform(0, 1/N)
        indexes = np.zeros(N, dtype=int)

        i, j = 0, 0
        for m in range(N):
            u = r + m / N
            while u > cumulative_sum[j]:
                j += 1
            indexes[i] = j
            i += 1

        # Resample by deep-copying selected particles
        self.particles = [copy.deepcopy(self.particles[idx]) for idx in indexes]


    def estimate(self):
        lats = np.array([p.lat for p in self.particles])
        lons = np.array([p.lon for p in self.particles])
        weights = np.array([p.weight for p in self.particles])

        # Weighted mean latitude (not wrapped — use regular average)
        mean_lat = np.average(lats, weights=weights)

        # Wrapped average for longitude
        sin_lon = np.sin(lons)
        cos_lon = np.cos(lons)
        mean_sin = np.average(sin_lon, weights=weights)
        mean_cos = np.average(cos_lon, weights=weights)
        mean_lon = np.arctan2(mean_sin, mean_cos)  # Result ∈ [-π, π]

        return mean_lat, mean_lon

def visualize_particles(particles, step=None, estimate=None):
    """
    Visualize particles on a 2D map projection of the Earth.
    
    Parameters:
        particles (list): List of Particle objects.
        step (int): Optional. Current iteration number for labeling.
        estimate (tuple): Optional. Estimated (lat, lon) to mark.
    """
    lats = np.array([np.degrees(p.lat) for p in particles])
    lons = np.array([np.degrees(p.lon) for p in particles])
    weights = np.array([p.weight for p in particles])

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_facecolor("black")

    # Plot particles
    plt.scatter(lons, lats, c=weights, cmap='cool', s=20, alpha=0.8, edgecolor='k', linewidths=0.3)

    # Plot estimated location
    if estimate is not None:
        est_lat_deg = np.degrees(estimate[0])
        est_lon_deg = np.degrees(estimate[1])
        plt.scatter(est_lon_deg, est_lat_deg, color='yellow', s=80, marker='*', label='Estimated Pose')

    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    title = f"Particle Distribution"
    if step is not None:
        title += f" - Step {step}"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def visualize_pitch_yaw_chart(particles, step=None):
    """
    Plots particles' pitch vs yaw as a scatter plot, colored by weight.
    """
    yaws = np.array([np.degrees(p.bearing) for p in particles])     # in degrees
    pitches = np.array([np.degrees(p.pitch) for p in particles])  # in degrees
    weights = np.array([p.weight for p in particles])

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(yaws, pitches, c=weights, cmap='plasma', s=40, edgecolor='k', alpha=0.8)

    plt.colorbar(sc, label='Particle Weight')
    plt.xlabel("Yaw (degrees)")
    plt.ylabel("Pitch (degrees)")
    title = "Pitch vs Yaw of Particles"
    if step is not None:
        title += f" - Step {step}"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()