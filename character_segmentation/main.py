import cv2
import numpy as np

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh, image

def find_and_filter_contours(thresh):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    min_contour_area = 100
    max_contour_area = 5000

    filtered_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

    return filtered_contours

def draw_contours(image, contours):
    # Draw contours on a copy of the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', contour_image)
    cv2.waitKey(0)

def extract_characters(image, contours):
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        character = image[y:y + h, x:x + w]
        cv2.imshow(f'Character {i+1}', character)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'your_image_path.jpg' with the actual path to your image
    image_path = './p37.jpg'

    # Preprocess the image
    thresh, original_image = preprocess_image(image_path)

    # Find and filter contours
    contours = find_and_filter_contours(thresh)

    # Draw contours for visualization
    draw_contours(original_image, contours)

    # Extract and display individual characters
    extract_characters(original_image, contours)
