# CelestiLoc
Project for Rob 3D - Localization from Celestial Objects

## Requirements
- Astrometry.net
- numpy
- astropy
- astroquery
- opencv
- open3d

To install Astrometry.net things, follow Instructions here: https://astrometry.net/use.html
Astrometry.net is what provides fit files that can identify stars in an image.


For other dependencies for running code, easiest setup is to create a conda environment with:
```
conda create --name astro_env --file requirements.txt
```

## Use

### Process images with astrometry.net

To prepare a file for astrometry processing, place your image into the *fits_files* folder under a folder with the same name as the image. Once this is done, run the *image_filter.py* python script to remove distortion from the image and filter out noise from the image.

(NOTE: You need to change the camera intrinics and Distortion model at the top of image_fitler.py. Also, be sure to put the change the name of the file inside the script.)

This will create an enhanced version of the image with the stars more clear. To create fits file for use, run astrometry solver on the enhanced image:

(NOTE: You need to download indexs so it can solve for stars. I used these: https://data.astrometry.net/4100/)

```
solve-field --overwrite --downsample 2 --scale-low 10 fits_files/2025-04-12T085905112Z/2025-04-12T085905112Z_enhanced.jpg --sigma 0
```

(We found the parameters above are the best and fastest ways to solve the image.)

This will create the .rdls (viewing angle of stars) and .xyls (index of stars in image) files we need for processing stars.

### Running code


To run code, input the measured pitch, bearing, and roll into the file and change the file_name in the python script and run
```
python identify_stars.py
```

This runs a particle filter that samples latitudes and longitudes over earth in order to solve for the best match. It add a bit of noise to each particles lat, lon, pitch, bearing, and roll to get a guess of the postition and the true angle of the camera.