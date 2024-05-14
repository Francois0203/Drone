from djitellopy import tello
import cv2
import sys
import os
import time

# Current working directory
sys.path.append(os.getcwd())

# Import custom python scripts
import KeyPress

# Global variables
global img

# Create drone object and connect to the drone
drone = tello.Tello()
drone.connect()

# Start running camera in video mode
drone.streamon()

# Check key presses to control drone manually
def get_keyboard_input():
    lr, fb, ud, yv = 0, 0, 0, 0 # Left/Right, Forward/Backward, Up/Down, Yaw Velocity
    speed = 50

    # Move left and right
    if KeyPress.get_key("LEFT"): lr = -speed
    elif KeyPress.get_key("RIGHT"): lr = speed

    # Move up and down
    if KeyPress.get_key("UP"): fb = speed
    elif KeyPress.get_key("DOWN"): fb = -speed

    # Move forward and backward
    if KeyPress.get_key("w"): ud = speed
    elif KeyPress.get_key("s"): ud = -speed

    # Yaw left and right
    if KeyPress.get_key("a"): yv = speed
    elif KeyPress.get_key("d"): yv = -speed

    # Land drone
    if KeyPress.get_key("q"): 
        time.sleep(3)
        drone.land()

    # Takeoff drone
    if KeyPress.get_key("e"): drone.takeoff()

    # Capture Image
    if KeyPress.get_key("z"): 
        cv2.imwrite(f"Images/{time.time()}.png", img) # Save image to "Images" folder
        time.sleep(0.3) # Add delay so that multiple images are not taken if button is pressed once

    return [lr, fb, ud, yv]

while True:
    values = get_keyboard_input()
    drone.send_rc_control(values[0], values[1], values[2], values[3])

    img = drone.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Image", img)
    cv2.waitKey(1) # Wait before shutting down the frame (Milliseconds)