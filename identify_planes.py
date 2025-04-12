import numpy as np
import cv2
import os
import datetime

daytime = True # need to figure out how to automate this
nighttime = False

# def filter_contrail_contours(contours, min_length=100, min_aspect_ratio=5):
#     valid_contours = []
    
#     for cont in contours:
#         x, y, w, h = cv2.boundingRect(cont)
#         # aspect_ratio = max(w, h) / min(w, h + 1e-5)
#         aspect_ratio = w / (h + 1e-5)
#         length = cv2.arcLength(cont, closed=False)
        
#         if length > min_length and aspect_ratio > min_aspect_ratio:
#             valid_contours.append(cont)

#     return valid_contours

def is_contrail_shape(contour, min_length=100, min_aspect_ratio=5, max_curvature=0.05):
    if len(contour) < 5:
        return False
    
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    length = cv2.arcLength(contour, closed=False)
    
    if length < min_length or aspect_ratio < min_aspect_ratio:
        return False

    # Optionally: fit a line and compute how well points stay close to it
    [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    unit_vector = np.array([vx, vy])
    
    deviations = []
    for pt in contour:
        pt_vec = np.array([pt[0][0] - x0, pt[0][1] - y0])
        projection = np.dot(pt_vec.T, unit_vector)
        perpendicular = pt_vec - projection * unit_vector
        deviations.append(np.linalg.norm(perpendicular))
    
    avg_deviation = np.mean(deviations)

    # Curvier lines will have higher deviation from the best-fit line
    return avg_deviation < max_curvature * length


def identify_planes(image):
    if daytime:
        # identify using the contrail
        image_blurred = cv2.GaussianBlur(image, (3,3), 0)
        edges = cv2.Canny(image_blurred, 50, 250)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contrail_contours = filter_contrail_contours(contours)
        contrail_contours = [cont for cont in contours if is_contrail_shape(cont)]
        cv2.drawContours(image, contrail_contours, -1, (0, 255, 0), 2)

        center_height = np.shape(image)[0] / 2
        center_width = np.shape(image)[1] / 2

        contrail_ends = []

        for contrail_contour in contrail_contours:
            contrail_contour = np.array(contrail_contour)
            # candidates = [
            #     tuple(contrail_contour[contrail_contour[:, :, 0].argmin()][0]), # leftmost
            #     tuple(contrail_contour[contrail_contour[:, :, 0].argmax()][0]), # rightmost
            #     tuple(contrail_contour[contrail_contour[:, :, 1].argmin()][0]), # topmost
            #     tuple(contrail_contour[contrail_contour[:, :, 1].argmax()][0]), # bottommost
            # ]
            contrail_end = [
                tuple(contrail_contour[contrail_contour[:, :, 0].argmin()][0]), # leftmost
                tuple(contrail_contour[contrail_contour[:, :, 0].argmax()][0]), # rightmost
                tuple(contrail_contour[contrail_contour[:, :, 1].argmin()][0]), # topmost
                tuple(contrail_contour[contrail_contour[:, :, 1].argmax()][0]), # bottommost
            ]
                        
            dist = np.shape(image)[1] / 2
            # for cand in candidates:
            #     cand_dist = np.linalg.norm(np.array(cand) - np.array([center_width, center_height]))
            #     if cand_dist < dist:
            #         dist = cand_dist
            #         contrail_end = cand
            # print('end', contrail_end)

            contrail_ends.append(contrail_end)

        return edges, contrail_ends

    else:
        # identify using speed
        return
