import numpy as np
import cv2

daytime = True # need to figure out how to automate this
nighttime = False

def filter_contrail_contours(contours, min_length=100, min_aspect_ratio=5):
    valid_contours = []
    
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        # aspect_ratio = max(w, h) / min(w, h + 1e-5)
        aspect_ratio = w / (h + 1e-5)
        length = cv2.arcLength(cont, closed=False)
        
        if length > min_length and aspect_ratio > min_aspect_ratio:
            valid_contours.append(cont)

    return valid_contours

def identify_planes(image):
    if daytime:
        print('shape', np.shape(image))
        # identify using the contrail
        image_blurred = cv2.GaussianBlur(image, (3,3), 0)
        edges = cv2.Canny(image_blurred, 10, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contrail_contours = filter_contrail_contours(contours)
        cv2.drawContours(image, contrail_contours, -1, (0, 255, 0), 2)

        center_height = np.shape(image)[0] / 2
        center_width = np.shape(image)[1] / 2

        contrail_ends = []

        for contrail_contour in contrail_contours:
            contrail_contour = np.array(contrail_contour)
            candidates = [
                tuple(contrail_contour[contrail_contour[:, :, 0].argmin()][0]), # leftmost
                tuple(contrail_contour[contrail_contour[:, :, 0].argmax()][0]), # rightmost
                tuple(contrail_contour[contrail_contour[:, :, 1].argmin()][0]), # topmost
                tuple(contrail_contour[contrail_contour[:, :, 1].argmax()][0]), # bottommost
            ]

            dist = np.shape(image)[1] / 2
            for cand in candidates:
                cand_dist = np.linalg.norm(np.array(cand) - np.array([center_width, center_height]))
                if cand_dist < dist:
                    dist = cand_dist
                    contrail_end = cand
            print('end', contrail_end)

            contrail_ends.append(contrail_end)

        return edges, contrail_ends

    else:
        # identify using speed
        return

if __name__ == "__main__":
    # image_path = "contrails/plane2.jpg"
    image_path = "contrails/2025-03-19T173627361Z.jpeg"
    # image_path = "contrails/2025-03-19T163904410Z.jpeg"
    # image_path = "contrails/2025-03-19T164523643Z.jpeg"
    image = cv2.imread(image_path)
    # crop out the timestamp
    image = image[50:, :]

    edges, contrail_ends = identify_planes(image)
    # cv2.imshow("Edges", edges)

    for end in contrail_ends:
        cv2.circle(image, end, 5, (0, 0, 255), -1)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
