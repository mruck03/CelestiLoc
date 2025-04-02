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

def residual_quaternion_R(params, t, img_pts, star_pts, camera_matrix):
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
    rvec = params

    # Ensure quaternion is normalized
    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
            
    star_pts_homo = np.vstack((star_pts.T, np.ones((1, star_pts.shape[0]))))
    
    proj_points = camera_matrix @ T[:3, :] @ star_pts_homo
    proj_points = proj_points[:2, :] / proj_points[2, :]
    proj_points = proj_points.T

    return ((proj_points - img_pts)**2).flatten()

def residual_quaternion_t(params, rvec, img_pts, star_pts, camera_matrix):
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
    t = params

    # Ensure quaternion is normalized
    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
            
    star_pts_homo = np.vstack((star_pts.T, np.ones((1, star_pts.shape[0]))))
    
    proj_points = camera_matrix @ T[:3, :] @ star_pts_homo
    proj_points = proj_points[:2, :] / proj_points[2, :]
    proj_points = proj_points.T

    return ((proj_points - img_pts)**2).flatten()

def solve_least_squares(object_points, image_points, camera_matrix, iters=1000, dist_coeffs=None):

    initial_rvec = np.array([0.0, 0.0, 0.0]) 
    initial_translation = np.array([0.0, 0.0, 0.0])
    initial_params = np.hstack((initial_rvec, initial_translation))

    # Define bounds for quaternion and translation parameters
    rvec_bounds = (-1, 1)  # Example bounds for quaternion values (to keep normalized)
    translation_bounds = ([-2, -2, -2], [2, 2, 2])  # Set bounds for translation (adjust as needed)

    # Combine bounds for all parameters
    lower_bounds = np.hstack((np.full(3, rvec_bounds[0]), translation_bounds[0]))
    upper_bounds = np.hstack((np.full(3, rvec_bounds[1]), translation_bounds[1]))

    # Optimize the parameters to minimize residuals
    result = least_squares(residual_quaternion_R, initial_rvec, args=(initial_translation, image_points, object_points, camera_matrix), bounds=(np.full(3, rvec_bounds[0]), (np.full(3, rvec_bounds[1]))))
    optimized_rvec = result.x

    result = least_squares(residual_quaternion_t, initial_translation, args=(optimized_rvec, image_points, object_points, camera_matrix), bounds=(translation_bounds[0], translation_bounds[1]))
    optimized_t = result.x

    result = least_squares(residual_quaternion_R, optimized_rvec, args=(optimized_t, image_points, object_points, camera_matrix), bounds=(np.full(3, rvec_bounds[0]), (np.full(3, rvec_bounds[1]))))
    optimized_rvec = result.x

    mean, err = compute_reprojection_error(object_points, image_points, optimized_rvec, optimized_t, camera_matrix)
    
    # while mean > 10 and cur_iters < iters:
    #     initial_params = np.hstack((optimized_rvec, optimized_t))
    #     result = least_squares(residual_quaternion, initial_params, args=(image_points, object_points, camera_matrix), bounds=(lower_bounds, upper_bounds))
    #     optimized_rvec = result.x[:3]
    #     optimized_t = result.x[3:6]
    #     mean, err = compute_reprojection_error(object_points, image_points, optimized_rvec, optimized_t, camera_matrix)
    #     cur_iters += 1
    #     print(mean)

    return optimized_rvec, optimized_t

def compute_reprojection_error(object_points, image_points, rvec, tvec, K, dist_coeffs=None):
    """
    Computes the reprojection error for a given set of 3D points and their corresponding 2D image points.

    Parameters:
    - object_points: (N, 3) numpy array of 3D world (or ECI) points
    - image_points: (N, 2) numpy array of corresponding 2D image points
    - rvec: Rotation vector (3, 1) from solvePnP
    - tvec: Translation vector (3, 1) from solvePnP
    - K: (3, 3) Camera intrinsic matrix
    - dist_coeffs: (5, 1) Distortion coefficients (set to None if no distortion)

    Returns:
    - mean_error: The mean reprojection error across all points
    - errors: List of individual reprojection errors for each point
    """

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Project the 3D object points into the image plane
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)

    # Reshape projected points to (N, 2)
    projected_points = projected_points.reshape(-1, 2)

    # Compute the Euclidean distance between observed and projected points
    errors = np.linalg.norm(image_points - projected_points, axis=1)

    # Compute the mean error
    mean_error = np.mean(errors)

    return mean_error, errors

def main():

    image_name = "enhanced_iphone" #INPUT IMAGE NAME

    # Open the .rdls file
    fits_dir = "fits_files"
    rdls_file = "{}.rdls".format(image_name)
    rdls_path = os.path.join(fits_dir, rdls_file)
    xyls_file = "{}-indx.xyls".format(image_name)
    xyls_path = os.path.join(fits_dir, xyls_file)
    image_file = "{}.jpg".format(image_name)
    image_path = os.path.join(fits_dir, image_file)

    # Read in Image
    img = cv2.imread(image_path)

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
    plot_data(img, star_data)

    #Get Star Distances:
    print("Processing stars for Distances")
    dists = np.array(get_star_distance(star_data))
    # dists = np.ones((1, star_data.shape[0]))*10

    #Star data: (x, y, ra, dec, dist (m))
    star_data = np.vstack((star_data.T, dists)).T
    star_data = star_data[star_data[:, 4] != None]
    star_data[:, 4] /= star_data[:, 4].max()
    star_data[:, 4] *= 1000

    print(star_data.shape)
    star_data = star_data.astype(float)

    # plot_3d_pose(star_data)

    x, y, z = celestial_to_cartesian(star_data[:, 2], star_data[:, 3], star_data[:, 4])
    # success, rvec, t = solve_pnp(np.vstack((x, y, z)).T, star_data[:, :2], K)

    rvec, t = solve_least_squares(np.vstack((x, y, z)).T, star_data[:, :2], K)

    print(rvec)
    mean_err, errs = compute_reprojection_error(np.vstack((x, y, z)).T, star_data[:, :2], rvec, t, K)
    print(f"Reprojection Error: {mean_err}")

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    # T = quaternion_to_transformation_matrix(q, t)

    print("Transformation Matrix")
    print(T)

    #Vizualize
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    frame.transform(np.linalg.inv(T))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # Visualize the transformed coordinate frame

    pc = np.vstack((x, y, z)).T

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # # Set colors for stars (white)
    # colors = np.ones_like(pc)  # All stars white
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create coordinate frame (origin)
    o3d.visualization.draw_geometries([frame, axis, pcd])

if __name__ == "__main__":
    Gaia.TIMEOUT = 120  # Increase timeout to 2 minutes
    Gaia.login(user="mrucker", password="Rob3dpass_")
    main()
