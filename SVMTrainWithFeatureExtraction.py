import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from skimage.feature import hog
from skimage import exposure
import joblib

# Define the path to your dataset
dataset_path = './characters for training'

# Initialize lists to store HOG features and corresponding labels
X = []
y = []

# Load images and labels
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read images as grayscale
            img = cv2.resize(img, (64, 64))  # Resize images if needed

            # Extract HOG features
            features, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                                      visualize=True)

            # Enhance the contrast of the HOG image
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            X.append(features)  # Append HOG features to the list
            y.append(folder_name)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but often beneficial)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Create an SVM classifier
clf = svm.SVC(kernel='linear', C=1.0)

# Train the classifier
clf.fit(X_train_scaled, y_train)

# Save the trained model for future use
joblib.dump(clf, 'svm_model.pkl')

# Make predictions on the test set
y_pred = clf.predict(X_test_scaled)

# Evaluate performance
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
