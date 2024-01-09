import cv2
import numpy as np

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to obtain a binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def find_contours(binary_image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by their x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    return contours

def segment_characters(image_path):
    # Preprocess the image
    binary_image = preprocess_image(image_path)

    # Find contours in the binary image
    contours = find_contours(binary_image)

    # Extract individual characters
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out small contours (noise)
        if w > 10 and h > 10:
            character = binary_image[y:y+h, x:x+w]
            characters.append(character)

    return characters

def main():
    image_path = './a.jpg'
    characters = segment_characters(image_path)

    # Save or process each character as needed
    for i, char in enumerate(characters):
        cv2.imwrite(f'./segmented_characters/character_{i+1}.png', char)

if __name__ == "__main__":
    main()
