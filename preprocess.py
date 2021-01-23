import cv2
import numpy as np
import imutils
from skimage import measure
from skimage.measure import regionprops


class PreProcess():

    def __init__(self, image):
        super().__init__()
        self.rectKernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (13, 5))  # UNIVERSAL LICENSE PLATE KERNEL
        self.originalImage = image
        self.binaryImage = self.threshold(self.originalImage)

    def threshold(self, grayImage):
        gray = cv2.bilateralFilter(grayImage, 11, 17, 17)
        blackhat = self.blackhat(gray)
        #self.show_image_process("blackhat", blackhat)
        light = self.light(gray)
        #self.show_image_process("light", light)
        gradX = self.getGradX(blackhat)
        #self.show_image_process("gradX", gradX)
        # first erode and clear the unwanted areas
        thresh = cv2.erode(gradX, None, iterations=4)
        # after that enlarge the areas
        thresh = cv2.dilate(thresh, None, iterations=2)
        #self.show_image_process("thresh1", thresh)
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.erode(thresh, None, iterations=2)
        #self.show_image_process("thresh2", thresh)

        return thresh

    def getPlateLikeObjects(self):
        self.labelImage = measure.label(self.binaryImage)

        threshold = self.binaryImage
        plate_dimensions = (
            0.08*threshold.shape[0], 0.2*threshold.shape[0], 0.15*threshold.shape[1], 0.4*threshold.shape[1])
        minHeight, maxHeight, minWidth, maxWidth = plate_dimensions
        plate_like_objects = []
        for region in regionprops(self.labelImage):
            if region.area < 50:
                continue

            minimumRow, minimumCol, maximumRow, maximumCol = region.bbox
            regionHeight = maximumRow - minimumRow
            regionWidth = maximumCol - minimumCol
            if minHeight <= regionHeight <= maxHeight and minWidth <= regionWidth <= maxWidth and regionWidth > regionHeight:
                plate_like_objects.append(self.originalImage[minimumRow:maximumRow,
                                                            minimumCol:maximumCol])
        if len(plate_like_objects) == 0: # if plate_like_object does not found in the above process
            self.findContours(plate_like_objects)

        return plate_like_objects

    def validatePlate(self, candidates):
        for c in candidates:
            height, width = c.shape
            license_plate = []
            highest_average = 0
            total_white_pixels = 0
            for column in range(width):
                total_white_pixels += sum(c[:, column])

            average = float(total_white_pixels) / width
            if average >= highest_average:
                license_plate = c
        return license_plate

    def findContours(self, plate_like_objects):
        cnts = cv2.findContours(self.binaryImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            print(ar)
            if ar >= 4 and ar <= 10:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                img = cv2.drawContours(self.originalImage,[box],0,(0,0,225),2)
                plate_like_objects.append(self.originalImage[ y: y+h , x: x+w])
                break

    def getGradX(self, blackhat):
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")

        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_GRADIENT, self.rectKernel)
        return cv2.threshold(gradX, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def blackhat(self, gray):
        # to reveal dark regieons on light backgrounds a blackhat morphological
        # operation is used.
        return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rectKernel)

    def light(self, gray):
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        return cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def show_image_process(self, title, image):
        cv2.imshow(title, image)
        cv2.waitKey(0)
