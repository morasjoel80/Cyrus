import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import threading
import numpy as np
import pyttsx3
import math


MODEL_PATH = "../Model/Old_Model/keras_model.h5"
LABEL_PATH = "../Model/Old_Model/labels.txt"

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.5)
classifier = Classifier(MODEL_PATH, LABEL_PATH)

# Constants

OFFSET = 20
IMG_SIZE = 300
VOICE = 0   # 0 Male    1 Female

# Variables

folder = open(LABEL_PATH, "r")
f = folder.read().splitlines()
Labels = f
folder.close()

# Init text to speech


def speech(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[VOICE].id)
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        print("\nSpeech error (Multiple inputs)")


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
                    text = Labels[index]
                    threading.Thread(
                        target=speech, args=text, daemon=True
                    ).start()

                # cv2.imshow("ImageCrop", imgCrop)
                # cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Image", imgOutput)
            cv2.waitKey(1)
        except cv2.error:
            print("\nCannot Detect (Out of Bounds)")


if __name__ == "__main__":
    capture()

