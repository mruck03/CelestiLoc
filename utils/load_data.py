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