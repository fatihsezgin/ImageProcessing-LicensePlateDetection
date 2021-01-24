import cv2
import numpy as np
import imutils
import pytesseract

class LicensePlateDetection():
    """
        A class to combine image preprocess methods for licence place
        detection
        ...

        Attributes
        ----------
        image : a grayscaled image

        Methods
        -------
        preprocess(grayImage):
            Processes the given image to the better detection for license plate
        
        blackhat(gray)
            To reveal the dark regions in the image
        
        getGradX(blackhat):
            To reveal the license plate characters in the image

        light(gray):
            To demonstrate the light regions in the image

        show_image_process(title, image)
            Shows the image with given title

        build_tesseract_options(psm=9 oem=3)
            Builds the tesseract options
        
        drawContours(contours, image)
            Draws the contours in the given image
        
        findContours(thresh)
            Finds the contours in the given binary image

    """
    def __init__(self,image):
        super().__init__()
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5)) # UNIVERSAL LICENSE PLATE KERNEL
        self.originalImage = image
        self.binaryImage = self.preprocess(self.originalImage)
    

    
    def preprocess(self, grayImage):
        '''
            Returns the processed binary image
                Parameters:
                    grayImage (ndarray): A grayscaled image
                
                Returns:
                    thresh (ndarray): A binary image
        '''
        gray = cv2.bilateralFilter(grayImage, 11, 17, 17) # blurring the image
        blackhat = self.blackhat(gray)
        #self.show_image_process("blackhat", blackhat)
        light = self.light(gray)
        #self.show_image_process("light", light)
        gradX = self.getGradX(blackhat)
        #self.show_image_process("gradX", gradX)

        # first erode and clear the unwanted areas
        thresh = cv2.erode(gradX, None, iterations=4) # thickness or the size of the fregorund object dcreases in white regions
        # after that enlarge the areas
        thresh = cv2.dilate(thresh, None, iterations=2) # Erosion removes white noises but also shrinks the object. 
        #self.show_image_process("thresh1", thresh)
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)

        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.erode(thresh, None, iterations=2)
        #self.show_image_process("thresh2", thresh)

        return thresh

    def getGradX(self, blackhat):
        '''
            Processes the image to revea light backgrounds where the license plate located.
                Parameters:
                    blackhat (ndarray): Blackhat morphological operation performed image
                Returns:
                    gradX(ndarray): Thresholded image
        '''
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1) # Compute the sobel edge detection
        gradX = np.absolute(gradX)
        # scaling the intensities back to the range [0-255]
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8") 

        # To reveal the white regions, where the license plate characters are located.
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_GRADIENT, self.rectKernel)
        return cv2.threshold(gradX, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def blackhat(self, gray):
        '''
            To reveal dark regions on light backgrounds a blackhat morphological operation is used.
                Parameters:
                    grayImage (ndarray): A grayscaled image
                
                Returns:
                    thresh (ndarray): Blackhat morphological operation performed image
        '''
        # 
        return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rectKernel)

    def light(self, gray):
        '''
            Reveal the light regions in the image that may contain license plate characters
                Parameters:
                    grayImage( ndarray): A grayscaled image
                
                Returns:
                    light(ndarray): A binary image that closing operation performed to image
                    to reveal the white surfaces in the image that may contain the license plate.
        '''
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        return cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def show_image_process(self, title, image):
        '''
            For demonstrating the image process
                Parameters:
                    title (str): Title of the image
                    image(ndarray): Image that will be shown
                Returns:
                    None
        '''
        cv2.imshow(title, image)
        cv2.waitKey(0)
    
    def build_tesseract_options(self, psm=8, oem=3):
        '''
            For building the tesseract options which is a OCR library.
                Parameters:
                    psm(int): Page Segmentation Method, default value is 7 which threads the image as a
                    single text line, but we are using the option 8 which is threats the image as single word
                    oem(int): OCR engine modes, 3 indicates that default, based on what is available.
                Returns:
                    options(str): the options for the tesseract
        '''
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        # set the OEM mode
        options += " --oem {}".format(oem)
        # return the built options string
        return options

    def drawContours(self, contours, image):
        '''
            Draws the contours the image
                Parameters:
                    contours(list): list of contours that have been found in the given image
                    image(ndarray): the target image to be drawn the contours
                Returns:
                    contourImage(ndarray): final image that includes the contour in the image
        '''
        contourImage = image.copy()
        cv2.drawContours(contourImage,contours,-1,(0,255,0),3)
        return contourImage

    def findContours(self, thresh):
        '''
            Finds contours in the image, and evaluates the options that may be suitable for the license plate.
                
            The plate dimensions were based on the following characteristics
            i.  They are rectangular in shape.
            ii. The width is more than the height
            iii. The ratio of the width to height is approximately 2:1
            iv. The proportion of the width of the license plate region to the 
            full image ranges between 15% to 40% depending on how the car image 
            was taken
            v.  The proportion of the height of the license plate region to the 
            full image is between 8% to 20%

                Parameters: 
                    thresh(ndarray): An image that will be used for the contour detection
                Returns:
                    roi(ndarray): License plate extracted binary image, which contains only the region in interest
                    in our case license plate. It's returned in binary for simplify the OCR.
        ''' 
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        contouredImage = self.drawContours(cnts, self.originalImage.copy())
        cv2.imshow("top 10 countours",contouredImage) #top 10 contours
        cv2.imwrite("images/top10.png", contouredImage)
        cv2.waitKey(0)

        for c in cnts:    
            (x,y,w,h) = cv2.boundingRect(c)
            ar = w / float(h)
            if ar >= 3 and ar <= 10: # to fit the regions of the license plate, the area should be between 3-10
                # a rectangle will be drawn into the image where the contour has found.
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                img = cv2.drawContours(self.originalImage.copy(),[box],0,(0,0,225),2)

                cv2.imshow("draw contour",img)
                cv2.imwrite("images/result.png", img)
                cv2.waitKey(0)

                # extracting the license plate regions from the image.
                licensePlate = self.originalImage[y:y + h, x:x + w]

                contrast = cv2.convertScaleAbs(licensePlate, alpha=1, beta=0) # increase the contrast for license plate to reveal the digits and characters clearly
                cv2.imshow("contrast", contrast)
                cv2.waitKey(0)
                roi = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # threshold the license plate to simplify the OCR process.
                cv2.imshow("licenseplate", roi)
                cv2.waitKey(0)
                return roi



img = cv2.imread("dataset/0002.jpg", cv2.IMREAD_GRAYSCALE) # read the image as grayscale
img = imutils.resize(img, width=500) # resize the image
detection = LicensePlateDetection(img) # initialize the LicensePlateDetection with given grayscale image.

threshold = detection.preprocess(grayImage=img) # preprocess the image
roi = detection.findContours(thresh=threshold) # find the contours and extract the license plate

options = detection.build_tesseract_options() # build the tesseract options
license_plate_text = pytesseract.image_to_string(roi, config=options) 
print(license_plate_text) # print the license plate
