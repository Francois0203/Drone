
# ITRI 616  -   Cloud Crusaders
This is the ITRI 616 (AI) assignment for semester 1 finall submitted date: 28 May 2024.






## Developers
The Cloud Crusaders Team consist of the following developers which created the application.
*	Francois Meiring (Team Leader)  - 38276909
*	Michael Wakeford  -  37569368
*	Etienne Berg  -  37443445

## Installation

System Requirements
*	A computer with a webcam
*	Python 3.12
*	A compatible drone (optional)

Dependencies
The following listed dependencies are used to allow the operation ability of the program. Ensure you have the following Python packages installed. These dependencies are listed in the requiremtnts.txt file for automatic installation using this file.
*	opencv-python
*	numpy
*	pygame
*	matplotlib
*	pillow
*	keras
*	scikit-learn
*	tensorflow

Installation Steps
*	Clone the repository from GitHub: 
```bash
  git clone https://github.com/Francois0203/Drone.git
```
    

*	Create a virtual environment and activate it.
*	Install the required dependencies.
*	Run the FullDrone.py application or its various sub-programs found in the folder.


    
## Features
This application utilises a live webcam feed to detect the user's face, determine its position relative to the camera, and send output commands to a drone to adjust its flight path accordingly. Additionally, it captures gestures and processes the images to identify the perceived gestures using a Convolutional Neural Network (CNN) developed from scratch. The output states the gesture and accuracy of the model to detect the gesture. 

The prediction with future development can be used for controlling the drone or other interactive tasks.


The Repo cosists of the following Python program scripts:

*   FaceTracking.py
*   FlightSim.py
*   FullDrone.py
*   IdentifyGestures.py
*   ImageProcessing.py
*   Keypress.py
*   Preprocess Training.py
*   TakeImages.py
*   TrainGestures.py
*   Train_AdamOptimizer.py


