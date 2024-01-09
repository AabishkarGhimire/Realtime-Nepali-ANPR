import cv2
import numpy as np

def segment_characters(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to convert the image to binary
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and extract individual characters
    segmented_characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small contours (noise)
        if w > 10 and h > 10:
            # Extract the character region from the original image
            character = image[y:y+h, x:x+w]

            # Append the character to the list
            segmented_characters.append(character)

    return segmented_characters

# Example usage
image_path = './a.jpg'
characters = segment_characters(image_path)

# Display or save the segmented characters
for i, character in enumerate(characters):
    cv2.imshow(f'Character {i+1}', character)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
