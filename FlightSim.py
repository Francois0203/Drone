import numpy as np
import cv2
import sys
import os
import time
import math

# Current working directory
sys.path.append(os.getcwd())

# Import custom python scripts
import KeyPress

# Global variables
global img

fSpeed = 117/10 # Forward speed in cm/s
aSpeed = 360/10 # Angular speed in degrees/s
interval = 0.25

dInterval = fSpeed * interval
aInterval = aSpeed * interval

x, y = 500, 500
angle = 0
yaw = 0
points = [(0, 0), (0, 0)]

# Check key presses to control drone manually
def get_keyboard_input():
    lr, fb, ud, yv = 0, 0, 0, 0 # Left/Right, Forward/Backward, Up/Down, Yaw Velocity
    speed = 15
    aSpeed = 50
    distance = 0
    global x, y, yaw, angle

    # Initialize KeyPress
    KeyPress.init()

    # Move left and right
    if KeyPress.get_key("LEFT"): 
        lr = -speed
        distance = dInterval
        angle = -180
    elif KeyPress.get_key("RIGHT"): 
        lr = speed
        distance = -dInterval
        angle = 180

    # Move forward and backward
    if KeyPress.get_key("UP"): 
        fb = speed
        distance = dInterval
        angle = 270
    elif KeyPress.get_key("DOWN"): 
        fb = -speed
        distance = -dInterval
        angle = -90

    # Move up and down
    if KeyPress.get_key("w"): 
        ud = speed
    elif KeyPress.get_key("s"): 
        ud = -speed

    # Yaw left and right
    if KeyPress.get_key("a"): 
        yv = aSpeed
        yaw -= aInterval
    elif KeyPress.get_key("d"): 
        yv = -aSpeed
        yaw += aInterval

    # Land drone
    if KeyPress.get_key("q"): 
        time.sleep(3)
        exit()

    # Capture Image
    if KeyPress.get_key("z"): 
        cv2.imwrite(f"Images/{time.time()}.png", img) # Save image to "Images" folder
        time.sleep(0.3) # Add delay so that multiple images are not taken if button is pressed once

    # Cartesian coordinates to map drone environment
    time.sleep(interval)
    angle += yaw
    x += int(distance * math.cos(math.radians(angle)))
    y += int(distance * math.sin(math.radians(angle)))

    return [lr, fb, ud, yv, x, y]

# Show drone path
def drawPoints(img, points):
    for point in points:
        cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED) #Blue Green Red
        cv2.circle(img, points[-1], 8, (0, 255, 0), cv2.FILLED) #Blue Green Red

    cv2.putText(img, f"({(points[-1][0] - 500) / 100}, {(points[-1][1] - 500) / 100})m", 
                (points[-1][0] + 10, points[-1][1] + 30), cv2.FONT_HERSHEY_PLAIN, 1, 
                (255, 0, 255), 1)

while True:
    values = get_keyboard_input()

    if (points[-1][0] != values[4]) or (points[-1][1] != values[5]):
        points.append((values[4], values[5]))

    img = np.zeros((1000, 1000, 3), np.uint8) # Matrix size - 1000 x 1000, 3 - RGB, uint8 - 0 to 255 values
    drawPoints(img, points)
    cv2.imshow("Output", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break