import numpy as np
import cv2
from scipy.interpolate import splprep, splev

def draw_spline(image_points, image_disp, color):

    x = image_points[:, 0]
    y = image_points[:, 1]

    # parametric spline
    tck, u = splprep([x, y], s=1.0)  # s = smoothing factor
    u_new = np.linspace(0, 1, 500)
    x_new, y_new = splev(u_new, tck)

    # draw the spline
    for i in range(len(x_new) - 1):
        pt1 = (int(x_new[i]), int(y_new[i]))
        pt2 = (int(x_new[i+1]), int(y_new[i+1]))
        cv2.line(image_disp, pt1, pt2, color, 2)