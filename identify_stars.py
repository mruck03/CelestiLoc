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
    
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=method)
    
    return success, rvec, tvec


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
    dists = get_star_distance(star_data)

    #Star data: (x, y, ra, dec, dist (m))
    star_data = np.vstack((star_data.T, np.array(dists))).T
    star_data = star_data[star_data[:, 4] != None]

    print(star_data.shape)
    star_data = star_data.astype(float)

    plot_3d_pose(star_data)

    x, y, z = celestial_to_cartesian(star_data[:, 2], star_data[:, 3], star_data[:, 4])
    scale_factor = np.linalg.norm(np.vstack((x, y, z)), axis=0).max()
    
    print(f"Scale Factor: {scale_factor}")

    x /= scale_factor
    y /= scale_factor
    z /= scale_factor

    success, rvec, t = solve_pnp(np.vstack((x, y, z)).T, star_data[:, :2], K, method=cv2.SOLVEPNP_EPNP)
    if success:
        t *= scale_factor

    print("PnP output")
    print(success)
    print(rvec)
    print(t)

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    print("Transformation Matrix")
    print(T)

    #Vizualize
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    frame.transform(T)

    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # Visualize the transformed coordinate frame
    # o3d.visualization.draw_geometries([frame, axis], window_name="Transformed Frame")

if __name__ == "__main__":
    Gaia.TIMEOUT = 120  # Increase timeout to 2 minutes
    Gaia.login(user="mrucker", password="Rob3dpass_")
    main()
