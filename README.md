
# ITRI 616  -   Cloud Crusaders
This is the ITRI 616 (AI) assignment for semester 1 finall submitted date: 28 May 2024.






## Developers
The Cloud Crusaders Team consist of the following developers which created the application.
*	Francois Meiring (Team Leader)  -   38276909
*	Michael Wakeford    -    37569368
*	Etienne Berg    -    37443445

## Installation

System Requirements
*	A computer with a webcam
*	Python 3.12

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
This application utilises a live webcam feed to detect the user's face. Additionally, it captures gestures and processes the images to identify the perceived gestures using a Convolutional Neural Network (CNN) developed from scratch. The output states the gesture and accuracy of the model to detect the gesture. 

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

The **TrainGestures.py**, **Preprocess training.py**, and the **Train_AdamOptimizer.py** scripts build and develop the CNN on various frameworks with different approaches.

The **ImageProcessing.py** script utilises image processing techniques and functions utilised in the various gesture detection functions on the various scripts, mainly in **IdentifyGestures.py** and utilised in the **FullDrone.py** script.


The Repo includes the **Resources** folder containing the dataset utilised for the final CNN along with the **Trainin (Kaggle Dataset)** utilised in development of the CNN.

The **lookup.pkl** and **reverselookup.pkl** files are used to store mappings between gesture category names and their corresponding numeric labels.

## Contact Information

For any questions or support, please contact:
*	Cloud Crusaders: 38276909@mynwu.ac.za
*	GitHub: https://github.com/Francois0203/Drone

## Acknowledgements

This project was developed as part of a group project at North-West University, under the guidance of Prof Absalom as the Cloud Crusaders. We thank him for his support and contributions.

