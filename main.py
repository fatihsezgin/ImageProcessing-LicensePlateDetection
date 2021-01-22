from preprocess import PreProcess
import cv2 
import numpy as np
import pytesseract
import time

def build_tesseract_options(psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        # return the built options string
        return options

def license_plate_extract(plate_like_objects, preprocess):
    number_of_candidates = len(plate_like_objects)

    if number_of_candidates == 0:
        print("License plate could not be located in the image")
        return []
    else:
        license_plate = pre_process.validatePlate(plate_like_objects)
        return license_plate


img = cv2.imread("dataset1/040603/0003.jpg", cv2.IMREAD_GRAYSCALE)  # read as grayscale
cv2.imshow("original image", img)
cv2.waitKey(0)

pre_process = PreProcess(img)
threshold = pre_process.threshold(grayImage=img)


plate_like_objects = pre_process.getPlateLikeObjects()

if len(plate_like_objects) == 0:
    print("plate bulunamadı")
else: 
    license_plate = license_plate_extract(plate_like_objects, pre_process)

    cv2.imshow("license_plate", license_plate)
    cv2.waitKey(0) 
    if len(license_plate) == 0:
        print("Error")
    else:
        options = build_tesseract_options()
        license_plate_text = pytesseract.image_to_string(license_plate, config=options)
        print(license_plate_text)

