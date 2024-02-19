import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)

# Constants

OFFSET = 20
IMG_SIZE = 600
# Variables

counter = 0
folder = "Data/Hello"
#   Path to save samples

while True:
    try:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        #   Finds Hands from img
        if hands:
            hand1 = hands[0]
            x, y, w, h = hand1['bbox']
            #   returns the Bounding Box attributes of hand

            imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
            #   Initialize white background to put cropped image onto

            if len(hands) == 1:     #   When one hand is detected
                imgCrop = img[y - OFFSET: y + h + OFFSET, x - OFFSET: x + w + OFFSET]
                #   Crops Hand with Bone structure

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = IMG_SIZE / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((IMG_SIZE - wCal) / 2)
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

            if len(hands) == 2:     #   When two hands are detected
                hand2 = hands[1]
                if hand1["type"] == "Right":    #   Checks whether the first hand to be detected is right
                    x, y, w, h = hand1['bbox']
                    x1, y1, w1, h1 = hand2['bbox']
                else:
                    x, y, w, h = hand2['bbox']
                    x1, y1, w1, h1 = hand1['bbox']

                imgCrop = img[y - OFFSET: y1 + h1 + (OFFSET + 20), x - OFFSET: x1 + w1 + (OFFSET + 20)]

                aspectRatio = ((h+h1)/2) / ((w+w1)/2)

                if aspectRatio > 1:
                    k = IMG_SIZE / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((IMG_SIZE - wCal) / 2)
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
            #   Display cropped hand/s
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
        print("\nCannot Detect(Out of bounds)")
        # Prevents crash when hand is present outside the frame



