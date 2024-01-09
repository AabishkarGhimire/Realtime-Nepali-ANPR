import cv2
import numpy as np

# Load the image
image = cv2.imread('./seg_chars/char_2.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and find the largest contour (assuming it's the document)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
largest_contour = contours[0]

# Approximate the contour to a polygon
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

# Print the number of points in approx_polygon
print("Number of points in approx_polygon:", len(approx_polygon))

# Draw the contour on the original image
cv2.drawContours(image, [approx_polygon], -1, (0, 255, 0), 2)

# Rearrange corner points for perspective transformation
if len(approx_polygon) == 4:
    pts1 = np.float32(approx_polygon.reshape(4, 2))
    pts2 = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective transformation to flatten the image
    result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    # Display the original image with detected contour and the flattened image
    cv2.imshow('Original Image with Contour', image)
    cv2.imshow('Flattened Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not find a quadrilateral contour.")
