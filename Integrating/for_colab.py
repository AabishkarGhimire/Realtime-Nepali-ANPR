from ultralytics import YOLO
import cv2
import os
import numpy as np

def preprocess_image(img): 
    # img = cv2.imread(image_path)
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
    
    projection = np.sum(binary_roi, axis=0)
    peaks = np.where(projection > 0.9 * np.max(projection))[0]
    
    
    character_images = []
    for i in range(len(peaks) - 1):
        char_image = license_plate_roi[:, peaks[i]-2:peaks[i+1]+2]
        height, width, _ = char_image.shape

        if (width/height) > (250/1033):
            character_images.append(char_image)
    return character_images

def save_characters(character_images, output_folder, input_file_name): 
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder) 
    for i, char_image in enumerate(character_images):
        output_path = os.path.join(output_folder, f"{input_file_name.replace('.jpg', '')}_char_{i+1}.jpg")
        cv2.imwrite(output_path, char_image)
        print(f"Character {i+1} saved at: {output_path}")

license_plate_detector = YOLO('/content/best.pt')

input_folder = '/content/Dataset/train/images'
output_folder = '/content/SegmentedCharacters'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all images in the input folder
for filename in os.listdir(input_folder):
    image_path = os.path.join(input_folder, filename)
    # Read the image
    frame = cv2.imread(image_path)
    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    # Process each detected license plate
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        # Crop the license plate region
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        # Preprocess and segment characters
        binary_image = preprocess_image(license_plate_crop)
        license_plate_contour = find_license_plate_contour(binary_image)
        character_images = segment_characters(license_plate_contour, license_plate_crop)
        # Save segmented characters to a single output folder
        save_characters(character_images, output_folder, filename)
