import numpy as np
import matplotlib.pyplot as plt
import cv2

nadia = cv2.imread('../DATA/Nadia_Murad.jpg', 0)
denis = cv2.imread('../DATA/Denis_Mukwege.jpg', 0)
solvay = cv2.imread('../DATA/solvay_conference.jpg', 0)

faceCascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')

def detectFace(img):
    faceImg = img.copy()
    faceRects = faceCascade.detectMultiScale(faceImg)
    for (x,y,w,h) in faceRects:
        cv2.rectangle(faceImg, (x,y), (x+w,y+h), (255,255,255), 10)

    return faceImg

def adjustedDetectFace(img):
    faceImg = img.copy()
    # scalefactor and minneighbors adjust the precision
    faceRects = faceCascade.detectMultiScale(faceImg,scaleFactor=1.2, minNeighbors=5)
    for (x,y,w,h) in faceRects:
        cv2.rectangle(faceImg, (x,y), (x+w,y+h), (255,255,255), 10)

    return faceImg

def detectEyes(img):
    eyeImg = img.copy()
    eyeRects = eyeCascade.detectMultiScale(eyeImg, scaleFactor=1.2, minNeighbors=5)
    for (x,y,w,h) in eyeRects:
        cv2.rectangle(eyeImg, (x,y), (x+w,y+h), (255,255,255), 10)

    return eyeImg

# result = adjustedDetectFace(solvay)
# plt.imshow(result, cmap='gray')
# plt.show()

"""
This works really well for Nadia, but doesnt work for Dennis due to the dark whites of his eyes 
this is an issue from the photo editing
"""
# eyeCascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')
# result = detectEyes(denis)
# plt.imshow(result, cmap='gray')
# plt.show()

cap=cv2.VideoCapture(0)

while True:

    ret, frame = cap.read(0)

    frame = detectFace(frame)
    cv2.imshow('Video Face Detect', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()