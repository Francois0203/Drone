import cv2
import numpy as np
import sys, os, time

# Current working directory
sys.path.append(os.getcwd())

# Global variables
fbRangeFace = [6200, 6800]

def findFace(img):
    cascade_path = os.path.join("Resources", "haarcascades", "haarcascade_frontalface_default.xml")
    faceCascade = cv2.CascadeClassifier(cascade_path)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) # Image, Start point, End point, Color, Line width

        cx = (x + w)//2
        cy = (y + h)//2
        area = w * h

        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED) # Image, Center coordinates, Radius, Colour, Thickness
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    # Track closest face
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def trackObject(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - w//2
    speed = pid[0]*error + pid[1]*(error - pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbRangeFace[0] and area < fbRangeFace[1]:
        fb = 0
    elif area > fbRangeFace[1]:
        fb = -20
    elif area < fbRangeFace[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    #me.send_rc_control(0, fb, 0, speed)
    return error

def __main__():
    # Variables
    w, h = 360, 240
    pid = [0.4, 0.4, 0] # Proportional, Integral, Derivative
    pError = 0
    
    # Capture device
    cap = cv2.VideoCapture(0) # 0 if only 1 webcam, 1 if you have multiple webcams

    while True:
        _, img = cap.read()
        img = cv2.resize(img, (w, h))
        img, info = findFace(img)
        pError = trackObject(info, w, pid, pError)
        print("Center", info[0], "Area", info[1]) # Move drone based on the 2nd value (area)
        cv2.imshow("Output", img)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    __main__()
