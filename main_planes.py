import numpy as np
import cv2
import os
import identify_planes as p
import image_filter as f
import shutil
import get_planes as gp
import reproject_planes as rp
from datetime import datetime, timezone
from scipy.interpolate import splprep, splev
import evaluate_planes as ep

#Intrisics
K = np.array([[1395.14165465379, 0, 980.742176449420],
            [0, 1387.40627391815, 528.268905594248],
            [0, 0, 1]])

#Distortion
D = np.array([-0.391719060689475, 0.133262225859808, 0, 0])

if __name__ == "__main__":
    cluster = 24
    cluster_name = f"cluster_{cluster}"
    # image_path = f"filtered_images/{cluster_name}/2025-04-03T162714947Z.jpeg"
    cluster_dir = f"filtered_images/{cluster_name}"
    image_path = f"filtered_images/{cluster_name}/2025-04-03T162839977Z.jpeg" # last image
    image_disp = cv2.imread(image_path)
    image_disp = cv2.undistort(image_disp, K, D)
    height, width = image_disp.shape[:2]
    image_disp = image_disp[60:, :width-30]
    height, width = image_disp.shape[:2]

    export_path = f"marked_planes/{cluster_name}"
    if os.path.exists(export_path):
        try:
            if not os.listdir(export_path):  # Check if directory is empty
                os.rmdir(export_path)
                print(f"Directory '{export_path}' removed (empty).")
            else:
                shutil.rmtree(export_path)
                print(f"Directory '{export_path}' and its contents removed.")
        except Exception as e:
             print(f"Error removing directory '{export_path}': {e}")
    else:
        print(f"Directory '{export_path}' does not exist.")
    
    os.makedirs(export_path) #, exist_ok=True)

    # edges, contrail_ends = p.identify_planes(image)
    # cv2.imshow("Edges", edges)
    # for e in contrail_ends:
    #     for c in e:
    #         cv2.circle(image, c, 5, (0, 0, 255), -1)

    planes_with_time = []
    image_points = []
    plane_pos = -1
    plane = []

    for root, _, files in os.walk(cluster_dir):
        for filename in files:
            # Get time in datetime format
            base = os.path.splitext(filename)[0]

            date_part, time_part = base.split("T")
            time_str = time_part.rstrip("Z")

            seconds = time_str[:6]
            milliseconds = time_str[6:]

            full_time_str = f"{date_part} {seconds}.{milliseconds}"

            dt = datetime.strptime(full_time_str, "%Y-%m-%d %H%M%S.%f")

            # preprocess image
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            # image = cv2.resize(image, (1920, 1080))
            image = cv2.undistort(image, K, D)
            image = image[60:, :]

            # define center
            height, width = image.shape[:2]
            image = image[60:, :width-30]
            center = np.array([width / 2, height / 2])

            # get contrails
            edges, contrail_ends = p.identify_planes(image)
            if not contrail_ends:
                continue
            else:
                contrail_ends = np.array(contrail_ends)
                if plane_pos == -1:
                    contrail_ends = np.reshape(contrail_ends, (np.shape(contrail_ends)[1], np.shape(contrail_ends)[2]))
                    dist = 99999
                    for i, cand in enumerate(contrail_ends):
                        cand_dist = np.linalg.norm(np.array(cand) - center)
                        if cand_dist < dist:
                            dist = cand_dist
                            plane = cand
                            plane_pos = i
                    planes_with_time.append((plane, dt))
                    
                else:
                    # if multiple contrails
                    if np.shape(contrail_ends)[0] > 1:

                        best_dist = 9999
                        for i, e in enumerate(contrail_ends):
                            if plane_pos == 0: # left
                                dist = np.linalg.norm(e[plane_pos][0] - 0)
                                if dist < best_dist:
                                    pl = e[plane_pos]
                                    best_dist = dist
                            elif plane_pos == 1: # right
                                dist = np.linalg.norm(e[plane_pos][0] - width)
                                if dist < best_dist:
                                    pl = e[plane_pos]
                                    best_dist = dist
                            
                            elif plane_pos == 2: # top
                                dist = np.linalg.norm(e[plane_pos][1] - 0)
                                if dist < best_dist:
                                    pl = e[plane_pos]
                                    best_dist = dist

                            else: # bottom
                                dist = np.linalg.norm(e[plane_pos][1] - height)
                                if dist < best_dist:
                                    pl = e[plane_pos]
                                    best_dist = dist

                        contrail_ends = np.reshape(e, (4, 2))
                        plane = pl
                    else:
                        contrail_ends = np.reshape(contrail_ends, (4, 2))
                        plane = contrail_ends[plane_pos]
                    planes_with_time.append((plane, dt))
                    image_points.append(plane)
                    cv2.circle(image, plane, 5, (0, 0, 255), -1)
                    export_img_path = os.path.join(export_path, filename)
                    print(f"Exporting to: {export_img_path}")
                    cv2.imwrite(export_img_path, image)

    # print(planes_with_time)
    end_str = "2025-04-03T162744958Z"
    end = datetime.strptime(end_str, "%Y-%m-%dT%H%M%S%fZ")

    planes_detected = []
    for (p, dt) in planes_with_time:
        if dt < end:
            cv2.circle(image_disp, p, 5, (0, 0, 255), -1)
            planes_detected.append(p)

    planes_detected = np.array(planes_detected)

    image_points = np.array(image_points)
    # print(np.shape(image_points))
    cam_bearing_deg = 205.0 - 40.0
    cam_pitch_deg = 38.0
    cam_lat = 42.56075    # North is positive
    cam_lon = -83.63903   # West is negative
    cam_alt = 270.0
    R_cam = rp.camera_rotation_matrix(cam_bearing_deg, cam_pitch_deg)
    
    json_dir = "./plane-info-dump"
    start_dt = planes_with_time[0][1]
    end_dt = planes_with_time[-1][1]
    gps_points = gp.get_gt_planes(start_dt, end_dt, json_dir)

    cam_ecef = rp.gps_to_ecef(cam_lat, cam_lon, cam_alt)
    reprojected_points = []
    for (lat, lon, alt) in gps_points:
        plane_ecef = rp.gps_to_ecef(lat, lon, alt)
        point_enu = rp.ecef_to_enu(plane_ecef[0], plane_ecef[1], plane_ecef[2],
                    cam_lat, cam_lon, cam_alt)
        point_3d = R_cam @ point_enu
        point_2d = rp.project_point(K, point_3d)
        if point_2d is not None:
            point_2d = np.array([int(point_2d[0]), int(point_2d[1])])
            reprojected_points.append(point_2d)
            cv2.circle(image_disp, point_2d, 5, (255, 0, 0), -1)

    reprojected_points = np.array(reprojected_points)

    planes_gt = []
    for (w, h) in reprojected_points:
        if 0 < h < height and 0 < w < width:
            planes_gt.append((w, h))

    planes_gt = np.array(planes_gt)

    # print(np.shape(planes_detected))
    # print(np.shape(planes_gt))

    ep.draw_spline(planes_detected, image_disp, (0, 0, 255))
    ep.draw_spline(planes_gt, image_disp, (255, 0))
                    
    cv2.imshow("Result", image_disp)
    export_img_path = os.path.join(export_path, "result.jpeg")
    cv2.imwrite(export_img_path, image_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()