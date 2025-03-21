# CelestiLoc
Project for Rob 3D - Localization from Celestial Objects

## Requirements
- Astrometry.net
- numpy
- astropy
- astroquery

To install Astrometry.net things, follow Instructions here: https://astrometry.net/use.html
Astrometry.net is what provides fit files that can identify stars in an image.


For other dependencies for running code, easiest setup is to create a conda environment with:
```
conda create --name astro_env --file requirements.txt
```

## Use

### Process images with astrometry.net

To create fits file for use, run astrometry solver on image:

(NOTE: You need to download indexs so it can solve for stars. I used these: https://data.astrometry.net/4100/)

```
solve-field fits_files test2.jpg
```

This will create the .rdls (viewing angle of stars) and .xyls (index of stars in image) files we need for processing stars.

### Running code

To run code, simply change the file_name in the python script and run
```
python identify_stars.py
```

Currently, this solves for the distances of stars using the Gaia catalgo and outputs an image which outlines detected stars