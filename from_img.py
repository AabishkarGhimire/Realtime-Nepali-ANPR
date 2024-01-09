from ultralytics import YOLO
import cv2

license_plate_detector = YOLO('./models/best.pt')

# load image
image_path = './test1.jpg'
frame = cv2.imread(image_path)

# detect license plates
license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Display the frame
cv2.imshow('Plate', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
