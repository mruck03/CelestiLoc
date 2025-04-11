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


def read_rdls_file(file_path):
    hdul = fits.open(file_path)
    data = hdul[1].data  # Table is usually in HDU 1

    # Extract RA and Dec
    ra = data['RA']
    dec = data['DEC']

    hdul.close()

    return ra, dec

def read_xyls_file(file_path):
    hdul = fits.open(file_path)
    data = hdul[1].data  # Table is usually in HDU 1

    # Extract X and Y pixel coordinates
    x_pixels = data['X']
    y_pixels = data['Y']

    hdul.close()

    return x_pixels, y_pixels

def load_star_data_batch(image_names, fits_dir, txt_path):
    all_data = []

    for image_name in image_names:
        star_data = load_star_data(image_name, fits_dir)

        obs_time = get_image_timestamp(txt_path, image_name)

        all_data.append({
            "star_data": star_data,
            "obs_time": obs_time,
        })

    return all_data


def load_star_data(image_name, fits_dir):

    rdls_file = "{}_enhanced.rdls".format(image_name)
    rdls_path = os.path.join(fits_dir, image_name, rdls_file)
    xyls_file = "{}_enhanced-indx.xyls".format(image_name)
    xyls_path = os.path.join(fits_dir, image_name, xyls_file)
    image_file = "{}_enhanced.jpg".format(image_name)
    image_path = os.path.join(fits_dir, image_name,image_file)
    saved_files = "out"
    
    # Read in Image
    img = cv2.imread(image_path)

    if os.path.exists(f"{fits_dir}/{saved_files}/{image_name}/{image_name}_star_data.csv"):
        # Load the data from the CSV file
        star_data = np.loadtxt(f"{fits_dir}/{saved_files}/{image_name}/{image_name}_star_data.csv", delimiter=',')
    else:
        # Extract RA and Dec
        ra, dec = read_rdls_file(rdls_path)
        # print(ra)
        # print(dec)

        #Read pixels
        x_pixels, y_pixels = read_xyls_file(xyls_path)

        #Create Numpy array of star data
        # (x, y, ra, dec)
        ra_np = np.array(ra)
        dec_np = np.array(dec)
        x_pixels_np = np.array(x_pixels)
        y_pixels_np = np.array(y_pixels)

        star_data = np.vstack((x_pixels_np, y_pixels_np, ra_np, dec_np)).T
        
        #Plot data using opencv
        plot_data(img, star_data, f"{fits_dir}/{image_name}/{image_name}_stars_identified.jpg")

        #Get Star Distances:
        print("Processing stars for Distances")
        dists = np.array(get_star_distance(star_data))
        # dists = np.ones((1, star_data.shape[0]))*100000000000000000

        #Star data: (x, y, ra, dec, dist (m))
        star_data = np.vstack((star_data.T, dists)).T
        star_data = star_data[star_data[:, 4] != None]
        # star_data[:, 4] /= star_data[:, 4].max()
        # star_data[:, 4] *= 1000

        # print(star_data.shape)
        star_data = star_data.astype(float)

        np.savetxt(f"{fits_dir}/{image_name}/{image_name}_star_data.csv", star_data, delimiter=',', fmt='%.6f')

    return star_data

def get_image_timestamp(txt_path, image_filename, default_time="2025-04-01 04:00:00"):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    print(lines)
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            name = parts[0].strip()
            ts = parts[1].strip()
            print(name, image_filename)
            if name == image_filename:
                dt_utc = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                # dt_utc = dt_est + timedelta(hours=4)  #EST → UTC
                return Time(dt_utc)

    #default if its not in timestamp.txt
    dt_utc = datetime.strptime(default_time, "%Y-%m-%d %H:%M:%S")
    return Time(dt_utc)

def plot_data(img, star_data, save_path="out/stars_identified.jpg"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    out_image = np.array(img)
    for x, y, ra, dec in star_data:
        x = round(x)
        y = round(y)
        cv2.circle(out_image, (x,y), radius=4, color=(0, 0, 255), thickness=2)
        cv2.putText(out_image, f"Ra: {ra:.2f}, Dec: {dec:.2f}", (x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
            color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.imwrite(save_path, out_image)

def get_star_distance(star_data):

    dists = []

    for x, y, ra, dec in tqdm(star_data, desc="Querying Gaia", ncols=100, unit=" star"):
        # Set up the coordinates for RA/Dec (in degrees)
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
        
        radius_deg = 1.0 / 30.0  # 1 arcminute in degrees
        
        # Corrected ADQL query with CIRCLE and ORDER BY DISTANCE
        query = f"""
            SELECT TOP 1 source_id, parallax 
            FROM gaiaedr3.gaia_source 
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {coords.ra.deg}, {coords.dec.deg}, {radius_deg})
            ) = 1
            """
        

        try:
            # Launch the query asynchronously
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            
            if len(results) > 0:
                parallax_mas = results['parallax'][0]  # Gaia parallax is in milliarcseconds
                if parallax_mas > 0:
                    distance_pc = 1 / (parallax_mas / 1000)  # Convert mas to arcseconds
                    distance_m = distance_pc * 3.086e16  # Convert parsecs to meters
                    # dists.append((distance_pc, distance_m))
                    dists.append(distance_m)
                else:
                    # dists.append((None, None))
                    dists.append(None)

            else:
                # dists.append((None, None))
                dists.append(None)
        
        except Exception as e:
            print(f"Error querying Gaia: {e}")
            # dists.append((None, None))
            dists.append(None)

    return dists


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
        [np.cos(gst_rad), np.sin(gst_rad), 0],
        [-np.sin(gst_rad), np.cos(gst_rad), 0],
        [0, 0, 1]
    ])

    x, y, z = celestial_to_cartesian(star_data[:, 2], star_data[:, 3], star_data[:, 4])

    star_coords = np.vstack((x, y, z))

    star_coords_adj = rotation_matrix @ star_coords

    return star_coords_adj.T

def plot_3d_pose(star_data, scale=1):
    x, y, z = celestial_to_cartesian(star_data[:, 2], star_data[:, 3], star_data[:, 4])

    # Rescale for visualization
    pc = np.vstack((x, y, z)).T * scale

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # # Set colors for stars (white)
    # colors = np.ones_like(pc)  # All stars white
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create coordinate frame (origin)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # Visualize
    o3d.visualization.draw_geometries([pcd, axis])

def solve_pnp(object_points, image_points, camera_matrix, dist_coeffs=None, method=cv2.SOLVEPNP_ITERATIVE):
    """
    Solves the Perspective-n-Point (PnP) problem to estimate camera pose.

    Parameters:
    - object_points: (Nx3 array) 3D points in the world frame.
    - image_points: (Nx2 array) 2D points in the image frame.
    - camera_matrix: (3x3 array) Camera intrinsic matrix.
    - dist_coeffs: (optional, default=None) Distortion coefficients (set to None if no distortion).
    - method: (default=cv2.SOLVEPNP_ITERATIVE) Method for solving PnP (e.g., cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_AP3P).

    Returns:
    - success: Boolean indicating if solving was successful.
    - rvec: (3x1 array) Rotation vector.
    - tvec: (3x1 array) Translation vector.
    """
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    print(object_points.shape)
    
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4,1))  # Assuming no distortion if not provided
    
    success, rvec, tvec, _ = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs)
    # success, rvec, tvec, inliers = cv2.solvePnPRansac(
    # object_points, image_points, camera_matrix, dist_coeffs, 
    # rvec=None, tvec=None, useExtrinsicGuess=False, iterationsCount=100, 
    # reprojectionError=8.0, confidence=0.99, flags=method
    # )
    
    return success, rvec, tvec

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

def residual_quaternion(params, img_pts, star_pts, camera_matrix):
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
    quat = params[:4]
    quat = quat / np.linalg.norm(quat)
    lat, lon = params[4:]

    t = lla_to_ecef(lat, lon)
    cam_2_world = quaternion_to_transformation_matrix(quat, t)

    world_2_cam = np.linalg.inv(cam_2_world)

    star_pts_homo = np.vstack((star_pts.T, np.ones(star_pts.shape[0])))
    star_pts_cam = world_2_cam[:3, :] @ star_pts_homo

    in_front = star_pts_cam[2, :] > 0
    if np.sum(in_front) == 0:
        return 1e9  # Penalize bad poses heavily

    star_pts_cam = star_pts_cam[:, in_front]
    img_pts_filtered = img_pts[in_front]

    proj_points = camera_matrix @ star_pts_cam[:3, :]
    proj_points = (proj_points[:2, :] / proj_points[2, :]).T

    # Compute reprojection errors
    errors = np.linalg.norm(img_pts_filtered - proj_points, axis=1)

    # Compute the mean error
    sum_error = np.sum(errors)

    missing_penalty = 100000 * (star_pts.shape[0] - np.sum(in_front))
    # print(sum_error + missing_penalty)

    # Get normalized vectors from camera to world star points
    star_rays_cam = star_pts_cam.T / np.linalg.norm(star_pts_cam.T, axis=1, keepdims=True)

    # Back-project image points into camera rays
    img_pts_homo = np.vstack((img_pts_filtered.T, np.ones(img_pts_filtered.shape[0])))
    img_pts_3D = np.linalg.inv(camera_matrix) @ img_pts_homo

    img_rays = img_pts_3D.T / np.linalg.norm(img_pts_3D.T, axis=1, keepdims=True)

    # Compute angle between rays and star vectors
    dot_products = np.sum(star_rays_cam * img_rays, axis=1)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

    print(f"Current Lat: {np.degrees(lat)}, Lon: {np.degrees(lon)}, Residual: {np.sum(angles**2) + sum_error + missing_penalty}")

    return np.sum(angles**2) + sum_error + missing_penalty
    

def residual_quaternion_R(params, opt_t ,img_pts, star_pts, camera_matrix):
    """
    Computes the residuals between transformed points in frame A and points in frame B using a quaternion.

    Args:
        params (np.ndarray): Optimization parameters (first 4 values are quaternion, last 3 are translation).
        img_pts (np.ndarray): Array of shape (N, 2) representing indexes of image
        star_pts (np.ndarray): Array of shape (N, 3) representing points of stars in world frame

    Returns:
        np.ndarray: Residuals (differences) between transformed points_a and points_b.
    """
    # Extract quaternion and translation parameters
    quat = params[:4]
    quat = quat / np.linalg.norm(quat)
    lat, lon = opt_t


    t = lla_to_ecef(lat, lon)
    cam_2_world = quaternion_to_transformation_matrix(quat, t)

    world_2_cam = np.linalg.inv(cam_2_world)

    star_pts_homo = np.vstack((star_pts.T, np.ones(star_pts.shape[0])))
    star_pts_cam = world_2_cam[:3, :] @ star_pts_homo

    in_front = star_pts_cam[2, :] > 0
    if np.sum(in_front) == 0:
        return 1e9  # Penalize bad poses heavily

    star_pts_cam = star_pts_cam[:, in_front]
    img_pts_filtered = img_pts[in_front]

    proj_points = camera_matrix @ star_pts_cam[:3, :]
    proj_points = (proj_points[:2, :] / proj_points[2, :]).T

    # Compute reprojection errors
    errors = np.linalg.norm(img_pts_filtered - proj_points, axis=1)

    # Compute the mean error
    sum_error = np.sum(errors)

    missing_penalty = 100000 * (star_pts.shape[0] - np.sum(in_front))
    # print(sum_error + missing_penalty)

    return sum_error + missing_penalty

# def residual_quaternion_t(params, quat, img_pts, star_pts, camera_matrix):
#     """
#       OLD DOES NOT WORK!!!!!
#     Computes the residuals between transformed points in frame A and points in frame B using a quaternion.

#     Args:
#         params (np.ndarray): Optimization parameters (first 4 values are quaternion, last 3 are translation).
#         img_pts (np.ndarray): Array of shape (N, 2) representing indexes of image
#         points_b (np.ndarray): Array of shape (N, 3) representing points of stars in world frame

#     Returns:
#         np.ndarray: Residuals (differences) between transformed points_a and points_b.
#     """
#     # Extract quaternion and translation parameters
#     lat, lon = params


#     t = lla_to_ecef(lat, lon)
#     cam_2_world = quaternion_to_transformation_matrix(quat, t)

#     world_2_cam = np.linalg.inv(cam_2_world)

#     star_pts_homo = np.vstack((star_pts.T, np.ones(star_pts.shape[0])))
#     star_pts_cam = world_2_cam[:3, :] @ star_pts_homo

#     in_front = star_pts_cam[2, :] > 0
#     if np.sum(in_front) == 0:
#         return 1e9  # Penalize bad poses heavily

#     star_pts_cam = star_pts_cam[:, in_front]
#     img_pts_filtered = img_pts[in_front]

#     proj_points = camera_matrix @ star_pts_cam[:3, :]
#     proj_points = (proj_points[:2, :] / proj_points[2, :]).T

#     # Compute reprojection errors
#     errors = np.linalg.norm(img_pts_filtered - proj_points, axis=1)

#     # Compute the mean error
#     sum_error = np.sum(errors)

#     # mean_error, _ = compute_reprojection_error(star_pts, img_pts, rvec, t, camera_matrix)
#     # print(f"Current Lat: {np.degrees(lat)}, Lon: {np.degrees(lon)}, Residual: {mean_error}")
#     missing_penalty = 100000 * (star_pts.shape[0] - np.sum(in_front))
#     # print(np.sum(in_front))
#     # print(sum_error + missing_penalty)
#     print(f"Current Lat: {np.degrees(lat)}, Lon: {np.degrees(lon)}, Residual: {sum_error + missing_penalty}")

#     return (sum_error + missing_penalty)

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

    print(f"Current Lat: {np.degrees(lat)}, Lon: {np.degrees(lon)}, Residual: {np.sum(angles**2)}")

    return np.sum(angles**2)# or np.mean(angles) if needed

def solve_least_squares(object_points, image_points, camera_matrix, dist_coeffs=None):

    initial_quat = np.array([0.0, 0.0, 0.0, 1.0]) 
    initial_translation = np.deg2rad(np.array([42.0, -83.0])) # (latitude, longitude)
    initial_params = np.hstack((initial_quat, initial_translation))

    # Define bounds for quaternion and translation parameters
    quat_bounds = (-1, 1)  # Example bounds for quaternion values (to keep normalized)
    translation_bounds = ([-np.pi/2, -np.inf], [np.pi/2, np.inf])  # Set bounds for translation (adjust as needed)

    # Combine bounds for all parameters
    lower_bounds = np.hstack((np.full(4, quat_bounds[0]), translation_bounds[0]))
    upper_bounds = np.hstack((np.full(4, quat_bounds[1]), translation_bounds[1]))
    bounds = list(zip(lower_bounds, upper_bounds))

    options = {
        # 'xatol': 1e-10,  # Absolute tolerance on parameters (small values make parameters change less)
        # 'fatol': 1e-100,   # Absolute tolerance on objective function value
        'maxiter': 1000,  # Maximum number of iterations
        'disp': True,      # Display optimization process
        # 'line_search':'strong_wolfe'
    }

    # Optimizing both params at the same time
    # result = minimize(
    #     lambda x: residual_quaternion(x, image_points, object_points, camera_matrix),
    #     initial_params,
    #     options = options,
    #     bounds=bounds,
    #     method= "L-BFGS-B" #"BFGS"#'Nelder-Mead'#"L-BFGS-B" # or 'TNC', 'SLSQP', etc., #TODO: Mess around with these to see which is best
    # )
    # optimized_quat = result.x[:4] / np.linalg.norm(result.x[:4])
    # optimized_t = result.x[4:]
    
    # print("Optimizing for R")
    result = minimize(
        lambda x: residual_quaternion_R(x, initial_translation, image_points, object_points, camera_matrix),
        initial_quat,
        options = options,
        bounds=list(zip(np.full(4, quat_bounds[0]), np.full(4, quat_bounds[1]))),
        method= "L-BFGS-B" #"BFGS"#'Nelder-Mead'#"L-BFGS-B" # or 'TNC', 'SLSQP', etc., #TODO: Mess around with these to see which is best
    )
    # result = least_squares(residual_quaternion_R, initial_quat, args=(initial_translation, image_points, object_points, camera_matrix), bounds=(np.full(4, quat_bounds[0]), np.full(4, quat_bounds[1])), method="dogbox")
    optimized_quat = result.x / np.linalg.norm(result.x)

    print("Optimization for Lat/Lon")
    result = minimize(
        lambda x: residual_quaternion_t(x, optimized_quat, image_points, object_points, camera_matrix),
        initial_translation,
        options = options,
        bounds=list(zip(translation_bounds[0], translation_bounds[1])),
        method= "Nelder-Mead" #"BFGS"#'Nelder-Mead'#"L-BFGS-B" # or 'TNC', 'SLSQP', etc., #TODO: Mess around with these to see which is best
    )
    # result = least_squares(residual_quaternion_t, initial_translation, args=(optimized_quat, image_points, object_points, camera_matrix), bounds=(translation_bounds[0], translation_bounds[1]), method="dogbox")
    optimized_t = result.x

    #TODO: Maybe optimize rotation and then translation????

    return optimized_quat, optimized_t


def main():
    image_names = ["2025-04-07T055924556Z", "2025-04-01T100253631Z"]
    fits_dir = "fits_files"
    txt_path = os.path.join(fits_dir, "timestamp.txt")

    dataset = load_star_data_batch(image_names, fits_dir, txt_path)

    all_obj_pts = []
    all_img_pts = []

    for entry in dataset:
        star_data = entry["star_data"]
        obs_time = entry["obs_time"]

        stars_metric = celestial_to_ecef(star_data, time_str=obs_time.isot)

        all_obj_pts.append(stars_metric)
        all_img_pts.append(star_data[:, :2])  # x, y

    # Stack all observations from all images
    object_points = np.vstack(all_obj_pts)
    image_points = np.vstack(all_img_pts)

    # Solve
    quat, pos = solve_least_squares(object_points, image_points, K)
    t = lla_to_ecef(pos[0], pos[1])

    print("Estimated Quaternion:", quat)
    print("Estimated Position (Lat, Lon):", np.degrees(pos))

    T = quaternion_to_transformation_matrix(quat, t/np.linalg.norm(t))

    print("Transformation Matrix")
    print(T)

    #Vizualize
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
    frame.transform(T)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])

    # Visualize the transformed coordinate frame
    star_dist_norm = star_data[:, 4] / star_data[:, 4].max()
    star_dist_norm += 10

    x_norm, y_norm, z_norm = celestial_to_cartesian(star_data[:, 2], star_data[:, 3], star_dist_norm)
    star_data[:, 4] = star_dist_norm
    stars_metric_norm = celestial_to_ecef(star_data, obs_time.isot)


    pc = np.vstack((x_norm, y_norm, z_norm)).T
    pc_adj = stars_metric_norm
    # print(pc.shape)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pc), 3)) * [1, 0, 0])

    pcd_adj = o3d.geometry.PointCloud()
    pcd_adj.points = o3d.utility.Vector3dVector(pc_adj)
    pcd_adj.colors = o3d.utility.Vector3dVector(np.ones((len(pc_adj), 3)) * [0, 0, 1])

    earth = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    

    # Create coordinate frame (origin)
    o3d.visualization.draw_geometries([frame, axis, pcd, pcd_adj, earth])


if __name__ == "__main__":
    Gaia.TIMEOUT = 120  # Increase timeout to 2 minutes
    # Gaia.login(user="mrucker", password="Rob3dpass_")
    main()
