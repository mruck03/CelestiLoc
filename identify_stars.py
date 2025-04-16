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

def read_axy_file(file_path):
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
    axy_file = "{}_enhanced.axy".format(image_name)
    axy_path = os.path.join(fits_dir, image_name, axy_file)
    image_file = "{}_enhanced.jpg".format(image_name)
    image_path = os.path.join(fits_dir, image_name,image_file)
    saved_files = "out"
    
    # Read in Image
    img = cv2.imread(image_path)

    if os.path.exists(f"{fits_dir}/{image_name}/{image_name}_star_data.csv"):
        # Load the data from the CSV file
        star_data = np.loadtxt(f"{fits_dir}/{image_name}/{image_name}_star_data.csv", delimiter=',')
    else:
        # Extract RA and Dec
        ra, dec = read_rdls_file(rdls_path)
        # print(ra)
        # print(dec)

        #Read pixels
        x_star_pixels, y_star_pixels = read_xyls_file(xyls_path)
        # x_detect, y_detect = read_axy_file(axy_path)

        #Create Numpy array of star data
        # (x, y, ra, dec)
        ra_np = np.array(ra)
        dec_np = np.array(dec)
        x_pixels_np = np.array(x_star_pixels)
        y_pixels_np = np.array(y_star_pixels)

        xyd_star_data = np.vstack((x_pixels_np, y_pixels_np)).T
        # xyd_detected = np.vstack((x_detect, y_detect)).T

        # Build KD-tree for detected points
        # tree = cKDTree(xyd_detected)

        # # Query nearest detected star for each star_data point
        # dists, indices = tree.query(xyd_star_data, distance_upper_bound=3.0)  # 3 px tolerance

        # # Keep only those with valid matches
        # valid = dists != np.inf

        # # Filter star_data to only include matched rows
        # filtered_ra = ra_np[valid]
        # filtered_dec = dec_np[valid]
        # matched_detected = xyd_detected[indices[valid]]  # shape: (N, 2)
        # filtered_star_data = star_data[valid]

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


class Particle:
    def __init__(self, lat, lon, quat=np.array([0, 0, 0, 1]), weight=1.0):
        self.lat = lat
        self.lon = lon
        self.weight = weight
        self.quat = quat / np.linalg.norm(quat)

class ParticleFilter:
    def __init__(self, num_particles, image_points, star_points, camera_matrix, percent_visible=0.8):
        self.particles = []
        min_visible = max(int(percent_visible * len(star_points)), 3)
        while len(self.particles) < num_particles:
            #Sample latitude and lon uniformly over the globe
            lat = np.arcsin(np.random.uniform(-1, 1)) 
            lon = np.random.uniform(-np.pi, np.pi)

            visible_mask = filter_visible_stars(lat, lon, star_points)
            if np.sum(visible_mask) < min_visible:
                continue

            tvec = lla_to_ecef(lat, lon)
            quat = solve_rotation_svd(image_points, star_points, camera_matrix, tvec)
            quat = quat / np.linalg.norm(quat)

            if is_camera_facing_down(quat, lat, lon):
                continue
            
            self.particles.append(Particle(lat, lon, quat))

    def predict(self, lat_std, lon_std, image_points, star_points, camera_matrix, percent_visible=0.8):

        min_visible = max(int(percent_visible * len(star_points)), 3)
        for p in self.particles:
            lat = clamp_latitude(p.lat + np.random.normal(0, lat_std))
            lon = wrap_longitude(p.lon + np.random.normal(0, lon_std))

            visible_mask = filter_visible_stars(lat, lon, star_points)
            if np.sum(visible_mask) < min_visible:
                continue

            # Solve for rotation at proposed position
            tvec = lla_to_ecef(lat, lon)
            quat = solve_rotation_svd(image_points, star_points, camera_matrix, tvec)
            quat = quat / np.linalg.norm(quat)

            # Reject if camera would be facing into Earth
            if is_camera_facing_down(quat, lat, lon):
                continue

            # Accept
            p.lat = lat
            p.lon = lon
            p.quat = quat


    def update_weights(self, image_points, star_points, camera_matrix, scale=1):
        weights = []
        # for p in tqdm(self.particles, desc="Updating particles"):
        residuals = []

        for p in self.particles:            
            residual = residual_quaternion_t([p.lat, p.lon], p.quat, image_points, star_points, camera_matrix) #TODO: Change to Reprojection Error!
            residuals.append(residual)

        res_min = np.min(residuals)
        res_max = np.max(residuals)
        res_range = res_max - res_min if res_max != res_min else 1.0

        print(f"res_range: {res_range}")

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

def main():
    image_names = ["2025-04-07T055924556Z", "2025-04-08T085807477Z", "2025-04-07T045859245Z", "2025-04-15T015950238Z"]
    # image_names = ["2025-04-07T045859245Z"]
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
    num_stars = 1

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
