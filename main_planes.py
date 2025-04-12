import numpy as np
import cv2
import os
import datetime
import identify_planes as p

if __name__ == "__main__":
    cluster = 24
    cluster_name = f"cluster_{cluster}"
    image_path = f"filtered_images/{cluster_name}/2025-04-03T162544933Z.jpeg" # first plane
    cluster_dir = f"filtered_images/{cluster_name}"
    image_path = f"filtered_images/{cluster_name}/2025-04-03T162805008Z.jpeg" # last image
    image = cv2.imread(image_path)
    image = image[60:, :]
    export_path = f"marked_planes/{cluster_name}"
    os.makedirs(export_path, exist_ok=True)

    # edges, contrail_ends = p.identify_planes(image)
    # cv2.imshow("Edges", edges)
    # for e in contrail_ends:
    #     for c in e:
    #         cv2.circle(image, c, 5, (0, 0, 255), -1)

    planes_with_time = []
    plane_pos = -1
    plane = []

    for root, _, files in os.walk(cluster_dir):
        for filename in files:
            # Get time in datetime format
            base = os.path.splitext(filename)[0]

            date_part, time_part = base.split("T")
            time_str = time_part.rstrip("Z")

            seconds = time_str[:6]
            milliseconds = time_str[6:]

            full_time_str = f"{date_part} {seconds}.{milliseconds}"

            dt = datetime.datetime.strptime(full_time_str, "%Y-%m-%d %H%M%S.%f")

            # preprocess image
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            image = image[60:, :width-30]
            center = np.array([width / 2, height / 2])

            # get contrails
            edges, contrail_ends = p.identify_planes(image)
            if not contrail_ends:
                continue
            else:
                contrail_ends = np.array(contrail_ends)
                if plane_pos == -1:
                    contrail_ends = np.reshape(contrail_ends, (np.shape(contrail_ends)[1], np.shape(contrail_ends)[2]))
                    dist = 99999
                    for i, cand in enumerate(contrail_ends):
                        cand_dist = np.linalg.norm(np.array(cand) - center)
                        if cand_dist < dist:
                            dist = cand_dist
                            plane = cand
                            plane_pos = i
                    planes_with_time.append((plane, dt))
                    
                else:
                    # if multiple contrails
                    if np.shape(contrail_ends)[0] > 1:

                        best_dist = 9999
                        for i, e in enumerate(contrail_ends):
                            if plane_pos == 0: # left
                                dist = np.linalg.norm(e[plane_pos][0] - 0)
                                if dist < best_dist:
                                    pl = e[plane_pos]
                                    best_dist = dist
                            elif plane_pos == 1: # right
                                dist = np.linalg.norm(e[plane_pos][0] - width)
                                if dist < best_dist:
                                    pl = e[plane_pos]
                                    best_dist = dist
                            
                            elif plane_pos == 2: # top
                                dist = np.linalg.norm(e[plane_pos][1] - 0)
                                if dist < best_dist:
                                    pl = e[plane_pos]
                                    best_dist = dist

                            else: # bottom
                                dist = np.linalg.norm(e[plane_pos][1] - height)
                                if dist < best_dist:
                                    pl = e[plane_pos]
                                    best_dist = dist

                        contrail_ends = np.reshape(e, (4, 2))
                        plane = pl
                    else:
                        contrail_ends = np.reshape(contrail_ends, (4, 2))
                        plane = contrail_ends[plane_pos]
                    planes_with_time.append((plane, dt))
                    cv2.circle(image, plane, 5, (0, 0, 255), -1)
                    export_img_path = os.path.join(export_path, filename)
                    print(f"Exporting to: {export_img_path}")
                    cv2.imwrite(export_img_path, image)

    # print(planes_with_time)
    for (p, _) in planes_with_time:
        cv2.circle(image, p, 5, (0, 0, 255), -1)

    #####TODO: calcualte pose of planes and compare to ground truth (in .json files)########
                    
    cv2.imshow("Result", image)
    export_img_path = os.path.join(export_path, "result.jpeg")
    cv2.imwrite(export_img_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()