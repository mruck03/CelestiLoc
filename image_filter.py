import cv2
import numpy as np
import os

def enhance_stars(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image.")
        return
    
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Apply thresholding to extract stars
    _, thresholded = cv2.threshold(enhanced, 45, 255, cv2.THRESH_BINARY)
    
    # Show the processed image
    # cv2.imshow("Original", img)
    # cv2.imwrite("Enhanced", enhanced)
    # cv2.imshow("Thresholded", thresholded)

    
    return thresholded

# Example usage
dir = "fits_files"
image_name = "dark.jpg"  # Replace with your image file
img_path = os.path.join(dir, image_name)
result = enhance_stars(img_path)
save_path = os.path.join(dir, "enhanced_stars.jpg")

# Save the result
if result is not None:
    cv2.imwrite(save_path, result)
