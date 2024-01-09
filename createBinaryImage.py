import cv2

# Read a grayscale image
gray_image = cv2.imread('gov.png', cv2.IMREAD_GRAYSCALE)

# Apply thresholding
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Display the original and binary images
cv2.imshow('Original Image', gray_image)
cv2.imshow('Binary Image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
