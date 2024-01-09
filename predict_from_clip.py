import joblib
import numpy as np
import cv2
import pyperclip
from PIL import Image
import io

# Load the trained model
model = joblib.load('./svm_model.pkl')  # Replace with the actual path to your trained model file

# Get image data from clipboard
image_data = pyperclip.paste()
cv2.imshow('fdsd', image_data)
if not image_data:
    print("No image found in the clipboard.")
else:
    try:
        # Convert the image data to a NumPy array using Pillow
        image_np = np.array(Image.open(io.BytesIO(image_data)))

        # Load and preprocess the unseen image
        unseen_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        unseen_image = cv2.resize(unseen_image, (64, 64))
        unseen_image = unseen_image.flatten()

        # Make predictions
        predictions = model.predict([unseen_image])

        # Display the results
        print(f'Predicted class: {predictions[0]}')
        cv2.imshow("License Plate Contour", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
