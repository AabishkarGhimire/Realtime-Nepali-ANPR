from ultralytics import YOLO
import cv2
import os
from segment import save_characters, find_license_plate_contour, preprocess_image, segment_characters

license_plate_detector = YOLO('./models/best.pt')

input_folder = './input_images'
output_folder = './output_segmented_chars'

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
        save_characters(character_images, output_folder)
