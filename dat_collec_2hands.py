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
                    centerpoint1 = hand1["center"]
                    x1, y1, w1, h1 = hand2['bbox']
                    centerpoint2 = hand2["center"]
                else:
                    x, y, w, h = hand2['bbox']
                    centerpoint1 = hand2["center"]
                    x1, y1, w1, h1 = hand1['bbox']
                    centerpoint2 = hand1["center"]

                length, info, img = detector.findDistance(centerpoint1, centerpoint2, img)
                print("\n", info)
                #   for testing

                if y < y1:
                    #   Crops with respect to the left hand (if left hand is higher than the right)
                    imgCrop = img[y - OFFSET : info[3] +h1 + OFFSET, x - OFFSET: info[2] + w1 + (OFFSET + 50)]
                else:
                    #   Crops with respect to the right hand (if right hand is higher than the left)
                    imgCrop = img[y1 - OFFSET : info[1] + h + OFFSET, x - OFFSET: info[2] + w1 + (OFFSET + 50)]

                Havg = (info[1]+info[3])+(y+y1)/2
                Wavg = (info[0]+info[2])+(x+x1)/2
                aspectRatio = Havg / Wavg

                if aspectRatio > 1:
                    k = IMG_SIZE / Havg
                    wCal = math.ceil(k * Wavg)
                    imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((IMG_SIZE - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    #   Pastes imgCrop onto imgWhite

                else:
                    k = IMG_SIZE / Wavg
                    hCal = math.ceil(k * Havg)
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



