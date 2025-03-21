from astropy.io import fits
import os
import numpy as np
import cv2
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm


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
        
        radius_deg = 1.0 / 60.0  # 1 arcminute in degrees
        
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

def main():

    image_name = "test2" #INPUT IMAGE NAME

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

    star_data = np.vstack((star_data.T, np.array(dists))).T
    star_data = star_data[star_data[:, 4] != None]

    print(star_data.shape)

if __name__ == "__main__":
    Gaia.TIMEOUT = 120  # Increase timeout to 2 minutes
    Gaia.login(user="mrucker", password="Rob3dpass_")
    main()
