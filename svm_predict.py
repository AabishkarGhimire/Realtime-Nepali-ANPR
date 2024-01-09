import joblib
import numpy as np
import os
import cv2
from skimage.feature import hog
from skimage import exposure

# Load the trained model
model = joblib.load('svm_model.pkl')  # Replace with the actual path to your trained model file

# Path to the unseen image


def prediction(image_path):
	# Load and preprocess the unseen image
	unseen_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	unseen_image = cv2.resize(unseen_image, (64, 64))

	# Extract HOG features
	features, hog_image = hog(unseen_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
							visualize=True)

	# Enhance the contrast of the HOG image
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

	# Flatten the features to match the format expected by the model
	unseen_features = features.flatten()

	# Standardize the features using the scaler from training
	scaler = joblib.load('scaler.pkl')  # Load the scaler used during training
	unseen_features_scaled = scaler.transform([unseen_features])

	# Make predictions
	prediction = model.predict(unseen_features_scaled)

	# Display the results
	print(f'Predicted class: {prediction[0]}')

	# original_image=cv2.imread(image_path)
	# cv2.imshow("original Image", original_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


	# Display the HOG image
	# cv2.imshow("HOG Image", hog_image_rescaled)
	# # cv2.imwrite("feature_image.png", hog_image_rescaled)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()



for i in range(1,6):
	image_path = f'./seg_chars/char_{i}.jpg'  # Replace with the path to your unseen image
	prediction(image_path)
