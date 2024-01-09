import cv2
import numpy as np

def segment_characters(image_path):
    # Read the license plate image
    plate_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Display the original image
    cv2.imshow('Original Image', plate_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Apply thresholding or other preprocessing steps if needed

    # Compute vertical projection
    projection = np.sum(plate_image, axis=0)

    # Threshold the projection to find potential character boundaries
    threshold = 5  # Adjust this threshold based on your image
    peaks = np.where(projection > threshold)[0]

    # Iterate through peaks to identify character boundaries
    character_boundaries = []
    start = peaks[0]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] > 1:
            end = peaks[i-1]
            character_boundaries.append((start, end))
            start = peaks[i]

    # Extract and save individual characters
    for i, (start, end) in enumerate(character_boundaries):
        character = plate_image[:, start:end]
        cv2.imwrite(f'./chars/character_{i+1}.png', character)
        cv2.imshow('segmented Image', character)

# Example usage
segment_characters('./a.jpg')
