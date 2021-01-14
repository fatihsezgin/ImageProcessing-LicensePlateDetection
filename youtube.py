import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

import pytesseract

#img = cv2.imread("license_plates/group2/001.jpg")
#img = cv2.imread("dataset/0024.jpeg")
img = cv2.imread("dataset1/040603/0006.jpg")
#img = cv2.imread("FiftyStatesLicensePlates/Alabama.jpg")

img = imutils.resize(img, width=500)
cv2.imshow("original image", img)
cv2.imwrite("images/original.png", img)
cv2.waitKey(0)
#img = cv2.imread('license_plates/group1/003.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray, 11, 17, 17)

#plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5)) # UNIVERSAL LICENSE PLATE KERNEL
# to reveal dark regieons on light backgrounds a blackhat morphological
# operation is used.
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,rectKern)
#cv2.imshow("blackhat",blackhat)
#cv2.waitKey(0)
cv2.imwrite("images/blackhat.png", blackhat)

squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv2.imshow("light",light)

cv2.imwrite("images/light.png", light)


gradX = cv2.Sobel(blackhat, ddepth= cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
#cv2.imshow("gradX",gradX)
#cv2.waitKey(0)
cv2.imwrite("images/gradX.png", gradX)


gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_GRADIENT, rectKern)
thresh = cv2.threshold(gradX, 127, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv2.imshow("thresh",thresh)
#cv2.waitKey(0)
cv2.imwrite("images/thresh.png", thresh)


thresh = cv2.erode(thresh, None, iterations=4) # first erode and clear the unwanted areas
thresh = cv2.dilate(thresh, None, iterations=2) # after that enlarge the areas.
#cv2.imshow("erode, dilate thresh",thresh)
#cv2.waitKey(0)
cv2.imwrite("images/erode-dilate-thresh.png", thresh) 


thresh = cv2.bitwise_and(thresh, thresh, mask=light)
thresh = cv2.dilate(thresh, None, iterations=4)
thresh = cv2.erode(thresh, None, iterations=2)
#cv2.imshow("bitwise thresh",thresh)
#cv2.waitKey(0)
cv2.imwrite("images/bitwisethresh.png", thresh)


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

# thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13,5)
# cnts, h = cv2.findContours(thresh,1,2)

img2 = img.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3) 
cv2.imshow("top 10 countours",img2) #top 5 contours
cv2.imwrite("images/top10.png", img2)
cv2.waitKey(0)


# c = max(cnts)

# epsilon = 0.18*cv2.arcLength(c,True)
# approx = cv2.approxPolyDP(c, epsilon, True)
# (x,y,w,h) = cv2.boundingRect(approx)
# ar = w / float(h)
# print(ar)

# rect = cv2.minAreaRect(c)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# img = cv2.drawContours(img,[box],0,(0,0,225),2)
# cv2.imshow("im",img)
# cv2.waitKey(0)


for c in cnts:
    epsilon = 0.18*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    (x,y,w,h) = cv2.boundingRect(approx)
    ar = w / float(h)
    print(ar)
    if ar >= 4 and ar <= 10:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(img,[box],0,(0,0,225),2)
        cv2.imshow("im",img)
        #cv2.imwrite("images/result.png", img)
        cv2.waitKey(0)
        break


    #licensePlate = gray[y: y+h, x:x+w]
    #roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # peri= cv2.arcLength(c,True)
    # approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # print(approx)
    # if len(approx) == 4:
    #     license = approx
    #     x,y,w,h = cv2.boundingRect(c)
    #     licensePlate = gray[y:y+h,x:x+w]
    #     cv2.imshow("license plate",licensePlate)
    #     cv2.waitKey(0)
    #     # cv2.imshow("roi",roi)
    #     # cv2.waitKey(0)

    #break
