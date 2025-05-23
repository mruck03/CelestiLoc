import cv2
import numpy as np
import os

#Camera Parameter - Cales Camera(2688x1512)
#Intrisics
# K = np.array([[1395.14165465379, 0, 980.742176449420],
#             [0, 1387.40627391815, 528.268905594248],
#             [0, 0, 1]])

# #Distortion
# D = np.array([-0.391719060689475, 0.133262225859808, 0, 0])

#Camera Parameter - Cales Camera (1920x1080)
#Intrisics
K = np.array([[1395.14165465379, 0, 980.742176449420],
            [0, 1387.40627391815, 528.268905594248],
            [0, 0, 1]])

#Distortion
D = np.array([-0.391719060689475, 0.133262225859808, 0, 0])

#Camera Parameter - Iphone Camera
#Intrisics
# K = np.array([[4110.67810701671, 0, 2866.48871930108],
#             [0, 4102.30392612357, 1586.15943599271],
#             [0, 0, 1]])

# #Distortion
# D = np.array([0.179096572759464, -0.465605122130898, 0, 0])

def enhance_stars(image):
    # Load the image
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Apply thresholding to extract stars
    _, thresholded = cv2.threshold(enhanced, 85, 255, cv2.THRESH_BINARY)
    thresholded[-200:, :] = 0
    # thresholded[:, -400:] = 0
    
    # Show the processed image
    # cv2.imshow("Original", img)
    # cv2.imwrite("Enhanced", enhanced)
    # cv2.imshow("Thresholded", thresholded)

    
    return thresholded



if __name__ == "__main__":
    # Example usage
    dir = "fits_files"
    image_name = "2025-04-17T035648703Z"  # Replace with your image file
    img_path = os.path.join(os.path.join(dir, image_name), f"{image_name}.jpeg")

    #Saved an enhanced image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1920, 1080))


    undistorted_image = cv2.undistort(img, K, D)
    result = enhance_stars(undistorted_image)
    # result = undistorted_image
    save_path = os.path.join(os.path.join(dir, image_name), f"{image_name}_enhanced.jpg")

    # Save the result
    if result is not None:
        cv2.imwrite(save_path, result)
