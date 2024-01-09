import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

license_plate_crop_path = './image.jpg'
license_plate_crop = cv2.imread(license_plate_crop_path)
license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
# read license plate number
detections = reader.readtext(license_plate_crop)
for detection in detections:
        bbox, text, score = detection
        
print(text)		