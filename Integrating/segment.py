import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt

def preprocess_image(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_license_plate_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 500
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    license_plate_contour = valid_contours[0]
    return license_plate_contour

def segment_characters(license_plate_contour, image):
    x, y, w, h = cv2.boundingRect(license_plate_contour)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("rect roi",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    license_plate_roi = image[y:y+h, x:x+w]
    # cv2.imshow("License Plate Contour", license_plate_roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    gray_roi = cv2.cvtColor(license_plate_roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY_INV)

    # Invert the binary image before dilation
    inverted_roi = cv2.bitwise_not(binary_roi)

    # Dilate the black regions to connect broken parts of characters
    kernel = np.ones((2, 2), np.uint8)
    dilated_roi = cv2.dilate(inverted_roi, kernel, iterations=1)

    # Invert the dilated image back to get the final result
    binary_roi = cv2.bitwise_not(dilated_roi)
    
    # cv2.imshow("binary roi",binary_roi)
    # # cv2.imwrite("binary roi.jpg",binary_roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Projection profile along height to detect character boundaries
    height_projection = np.sum(binary_roi, axis=1)
    height_peaks = np.where(height_projection > 0.8 * np.max(height_projection))[0]

    top = 0
    prevdiff = 0
    for i in range(len(height_peaks)-1):
        newdiff = height_peaks[i+1]-height_peaks[i]
        if newdiff > prevdiff:
            prevdiff = newdiff
            top = i

    projection = np.sum(binary_roi[height_peaks[top]:height_peaks[top+1], :], axis=0)
    peaks = np.where(projection > 0.93 * np.max(projection))[0]
    






    # plt.plot(np.arange(w), projection)
    # plt.show()


    # print(height_peaks)
    # input("fsfds")
    debug = binary_roi[height_peaks[top]:height_peaks[top+1], :]
    cv2.imshow("debug", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

    # input('fdfsfds')

    # a= 3
    character_images = []
    for i in range(len(peaks) - 1):
        char_image = license_plate_roi[height_peaks[top]-10:height_peaks[top+1]+10, peaks[i]-2:peaks[i+1]+2]
        aspect_ratio = char_image.shape[1]/char_image.shape[0]
        if aspect_ratio > 0.3:
            character_images.append(char_image)
    return character_images

def save_characters(character_images, output_folder): 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    for i, char_image in enumerate(character_images):
        output_path = os.path.join(output_folder, f"char_{i+1}.jpg")
        cv2.imwrite(output_path, char_image)
        print(f"Character {i+1} saved at: {output_path}")

if __name__ == "__main__":
    image_path = "./plate.jpg"
    output_folder = "seg_chars"
    image = cv2.imread(image_path)
    binary_image = preprocess_image(image)
    license_plate_contour = find_license_plate_contour(binary_image)
    original_image = cv2.imread(image_path)
    character_images = segment_characters(license_plate_contour, original_image)
    save_characters(character_images, output_folder)
