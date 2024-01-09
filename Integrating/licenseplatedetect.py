from ultralytics import YOLO
import cv2
from segment import find_license_plate_contour,preprocess_image,segment_characters, save_characters
from svm_predict import prediction
# import mysql.connector

# db_config = {
#     'host': 'your_host',
#     'user': 'your_username',
#     'password': 'your_password',
#     'database': 'your_database',
# }

# connection = mysql.connector.connect(**db_config)
# cursor = connection.cursor()
# query = f"SELECT COUNT(*) FROM {table} WHERE {column} = %s"
# cursor.execute(query, (value,))
# count = cursor.fetchone()[0]

#     # Close the cursor
# cursor.close()

license_plate_detector = YOLO('./models/best.pt')

image_path = './test.jpg'
frame = cv2.imread(image_path)
height, width = frame.shape[:2]

# Resize the image to have a width of 1000 pixels while maintaining the aspect ratio
frame = cv2.resize(frame, (1000, int(1000 * height / width)))

license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate

    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
    
    cv2.imwrite('plate.jpg', license_plate_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # output_folder = "segmented_chars"
    binary_image = preprocess_image(license_plate_crop)
    license_plate_contour = find_license_plate_contour(binary_image)
    
    character_images = segment_characters(license_plate_contour, license_plate_crop)
    save_characters(character_images, 'segments')
    for character_image in character_images:
        prediction(character_image)