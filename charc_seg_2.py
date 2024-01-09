import cv2
import numpy as np
import os

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    angle_sum = 0
    count = 0

    for line in lines:
        for rho, theta in line:
            angle_sum += theta
            count += 1

    average_angle = angle_sum / count

    rotated = rotate_image(image, average_angle * (180 / np.pi))
    return rotated

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    return result

def projection_analysis(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Vertical projection
    vertical_projection = np.sum(binary, axis=0)
    
    # Identify peaks and valleys
    peaks = np.where(vertical_projection > threshold)[0]
    valleys = np.where(vertical_projection < threshold)[0]

    return peaks, valleys

def save_characters(image, peaks, valleys, output_folder, min_char_width):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through the valleys to extract and save characters
    for i in range(len(valleys) - 1):
        start_col = valleys[i]
        end_col = valleys[i + 1]

        # Extract character
        character = image[:, start_col:end_col]

        # Check if the width is above a minimum threshold
        if end_col - start_col >= min_char_width:
            # Save character to the output folder
            char_filename = os.path.join(output_folder, f"char_{i}.png")
            cv2.imwrite(char_filename, character)

# Example usage
image_path = './a.jpg'
output_folder = 'output_characters'
original_image = cv2.imread(image_path)

# De-skew the image
deskewed_image = deskew(original_image)

# Set your threshold value based on experimental observations
threshold_value = 500  # Adjust as needed based on the size of your characters

# Set a minimum character width threshold
min_char_width = 50  # Adjust as needed based on the size of your characters

# Perform projection analysis
peaks, valleys = projection_analysis(deskewed_image, threshold_value)

# Save segmented characters to the output folder
save_characters(deskewed_image, peaks, valleys, output_folder, min_char_width)

# Display the images for visualization (optional)
cv2.imshow('Original Image', original_image)
cv2.imshow('Deskewed Image', deskewed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
