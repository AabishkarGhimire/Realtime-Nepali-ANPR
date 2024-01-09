import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path): 
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_license_plate_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 500
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    license_plate_contour = valid_contours[0]
    return license_plate_contour

def segment_characters(license_plate_contour, image):
    x, y, w, h = cv2.boundingRect(license_plate_contour)
    license_plate_roi = image[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(license_plate_roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)

    # Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)

    character_images = []
    for i in range(1, num_labels):  # Skip the background label (0)
        char_mask = (labels == i).astype(np.uint8)
        char_image = cv2.bitwise_and(license_plate_roi, license_plate_roi, mask=char_mask)

        # Filter out small components
        component_area = stats[i, cv2.CC_STAT_AREA]
        if component_area > 40:  # Adjust the threshold as needed
            character_images.append(char_image)

    return character_images

def save_characters(character_images, output_folder): 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    for i, char_image in enumerate(character_images):
        output_path = os.path.join(output_folder, f"char_{i+1}.jpg")
        cv2.imwrite(output_path, char_image)
        print(f"Character {i+1} saved at: {output_path}")

if __name__ == "__main__":
    image_path = "./seg.jpg"
    output_folder = "seg_chars"
    binary_image = preprocess_image(image_path)
    license_plate_contour = find_license_plate_contour(binary_image)
    original_image = cv2.imread(image_path)
    character_images = segment_characters(license_plate_contour, original_image)
    save_characters(character_images, output_folder)
