import cv2
import numpy as np

# Load your binary image
binary_image = cv2.imread('p0.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Grayscale', binary_image)

# Apply morphological operations to remove small patches and dots
kernel_size = 3  # Adjust the kernel size based on the size of the noise you want to remove
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Erosion to remove small patches
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Dilation to restore the size of characters
dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

# Display the original, eroded, and dilated images
cv2.imshow('Original Binary Image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
