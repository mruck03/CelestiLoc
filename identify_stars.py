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
from utils.solver import *

    
#Camera Parameter - Cales Camera (1920x1080)
#Intrisics
K = np.array([[1395.14165465379, 0, 980.742176449420],
            [0, 1387.40627391815, 528.268905594248],
            [0, 0, 1]])

#Distortion
D = np.array([-0.391719060689475, 0.133262225859808, 0, 0])


def main():
    image_names = ["2025-04-07T055924556Z", "2025-04-08T085807477Z", "2025-04-07T045859245Z", "2025-04-15T015950238Z", "2025-04-12T085905112Z"]
    # image_names = ["2025-04-08T085807477Z"]
    fits_dir = "fits_files"
    txt_path = os.path.join(fits_dir, "timestamp.txt")

    dataset = load_star_data_batch(image_names, fits_dir, txt_path)


    all_obj_pts = []
    all_img_pts = []

    for entry in dataset:
        star_data = entry["star_data"]
        obs_time = entry["obs_time"]
        
        print(star_data.shape)

        stars_metric = celestial_to_ecef(star_data, time_str=obs_time.isot)

        all_obj_pts.append(stars_metric)
        all_img_pts.append(star_data[:, :2])  # x, y

    # Stack all observations from all images
    object_points = np.vstack(all_obj_pts)
    image_points = np.vstack(all_img_pts)
    print(f"NUMBER OF STARS = {len(object_points)}")

    gt_lat = np.deg2rad(43)
    gt_lon = np.deg2rad(-83)
    gt_t = lla_to_ecef(gt_lat, gt_lon)

    gt_quat = solve_rotation_svd(image_points, object_points, K, gt_t)
    gt_quat /= np.linalg.norm(gt_quat)

    plot_3d_pose(dataset, gt_quat, gt_t)


    cam_2_world = quaternion_to_transformation_matrix(gt_quat, gt_t)

    world_2_cam = np.linalg.inv(cam_2_world)

    star_pts_homo = np.vstack((object_points.T, np.ones(object_points.shape[0])))
    star_pts_cam = world_2_cam[:3, :] @ star_pts_homo

    proj_points = K @ star_pts_cam[:3, :]
    proj_points = (proj_points[:2, :] / proj_points[2, :]).T

    viz_projection(image_points, proj_points)

    N = 500

    pf = ParticleFilter(
        num_particles=N,
        image_points=image_points,
        star_points=object_points,
        camera_matrix=K,
        percent_visible=0.99
    )

    # Run particle filter for a fixed number of iterations
    initial_lat_std = 1
    initial_lon_std = 1
    num_iters = 200
    num_stars = 10

    for i in tqdm(range(num_iters)):
        lat_std = initial_lat_std * (0.99 ** i)
        lon_std = initial_lon_std * (0.99 ** i)
        pf.predict(np.deg2rad(lat_std), np.deg2rad(lon_std), image_points, object_points, K, percent_visible=0.99)

        #subsample a random amount of stars (num_stars). Stars are weighted by distance
        distances = np.linalg.norm(object_points, axis=1)
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()
        idx_stars = np.random.choice(len(object_points), num_stars, replace=False, p = weights)

        # pf.update_weights(image_points[idx_stars], object_points[idx_stars], K, scale=5)
        pf.update_weights(image_points, object_points, K, scale=5)
        
        #Print to see it change over time
        lat, lon = pf.estimate()
        # print("Estimated Position (Lat, Lon):", np.degrees(np.array([lat, lon])))
        # print(len(pf.particles))

        if i % 50 == 0:
            visualize_particles(pf.particles, step=i, estimate=(lat, lon))
            # frames = []
            # for p in pf.particles:
            #     t = lla_to_ecef(p.lat, p.lon)
            #     T = quaternion_to_transformation_matrix(p.quat, t/np.linalg.norm(t))
            #     pred_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
            #     pred_frame.transform(T)
            #     frames.append(pred_frame)
            # o3d.visualization.draw_geometries(frames + [axis, pcd_adj, earth])

        print(N*0.7)
        print(pf.N_eff())

        if pf.N_eff() < N*0.7:
            print("resampling!!!!")
            pf.resample()

    # Final pose estimate
    lat, lon = pf.estimate()
    t = lla_to_ecef(lat, lon)

    quat = solve_rotation_svd(image_points, object_points, K, t)
    quat = quat / np.linalg.norm(quat)

    print("Estimated Quaternion:", quat)
    print("Estimated Position (Lat, Lon):", np.degrees(np.array([lat, lon])))

    plot_3d_pose(dataset, quat, t)

    cam_2_world = quaternion_to_transformation_matrix(quat, t)

    world_2_cam = np.linalg.inv(cam_2_world)

    star_pts_homo = np.vstack((object_points.T, np.ones(object_points.shape[0])))
    star_pts_cam = world_2_cam[:3, :] @ star_pts_homo

    proj_points = K @ star_pts_cam[:3, :]
    proj_points = (proj_points[:2, :] / proj_points[2, :]).T

    viz_projection(image_points, proj_points)



if __name__ == "__main__":
    Gaia.TIMEOUT = 120  # Increase timeout to 2 minutes
    # Gaia.login(user="mrucker", password="Rob3dpass_")
    main()
