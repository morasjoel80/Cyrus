import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Constants :

OFFSET = 20
#   OFFSETS the cropped image to capture the entire hand
IMG_SIZE = 300
#   Defines the size of imgwhite

# Variables :

counter = 0
folder = "Model_Data/Hello"
#   Path to save samples

while True:
    try:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        #   Finds Hands from img
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            #   returns the Bounding Box attributes of hand

            imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
            #   Initialize white background to put cropped image onto
            imgCrop = img[y-OFFSET: y + h + OFFSET, x-OFFSET: x + w + OFFSET]
            #   Crops Hand with Bone structure

            aspectRatio = h/w

            if aspectRatio > 1:
                k = IMG_SIZE/h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((IMG_SIZE - wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                #   Pastes imgCrop onto imgWhite

            else:
                k = IMG_SIZE / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((IMG_SIZE - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                #   Pastes imgCrop onto imgWhite

            cv2.imshow("ImageCrop", imgCrop)
            #   Display cropped hand
            cv2.imshow("ImageWhite", imgWhite)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

#   Collecting samples  :
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            #   Saves samples in the specified folder
            print(counter)
    except cv2.error:
        # Prevents crash when hand is present outside the frame
        print("Cannot detect (Out of Bounds)")

