import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from playsound import playsound
import threading
import numpy as np
import math
import time

MODEL_PATH = "../Model/Old_Model/keras_model.h5"
LABEL_PATH = "../Model/Old_Model/labels.txt"
SPEECH_PATH = "../Speech"

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)
classifier = Classifier(MODEL_PATH, LABEL_PATH)

# Constants

OFFSET = 20
IMG_SIZE = 300
VOICE = 0   # 0 Male    1 Female

# Variables

folder = open(LABEL_PATH, "r")
f = folder.read().splitlines()
Labels = f
print(Labels)
folder.close()

# Init text to speech


def speech(audio):
    print(audio)
    done = False
    while not done:
        try:
            playsound(f'{SPEECH_PATH}/{audio}.mp3')
            done = True
        except:
            continue
    time.sleep(1)


def capture():
    text = ""

    while True:
        try:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                imgCrop = img[y-OFFSET: y + h + OFFSET, x-OFFSET: x + w + OFFSET]

                imgCropShape = imgCrop.shape

                aspectRatio = h/w

                if aspectRatio > 1:
                    k = IMG_SIZE/h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((IMG_SIZE - wCal)/2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)
                    print(prediction, index)

                else:
                    k = IMG_SIZE / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((IMG_SIZE - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)

        # Output Box

                cv2.putText(imgOutput, Labels[index], (x, y-OFFSET), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        # Text To Speech

                if text != Labels[index]:
                    text = str(Labels[index])
                    threading.Thread(
                        target=speech, args=(text,)     # Passing text as tuple to prevent errors
                    ).start()

                # cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Image", imgOutput)
            cv2.waitKey(1)
        except cv2.error:
            print("\nOut of bounds")


if __name__ == "__main__":
    capture()

