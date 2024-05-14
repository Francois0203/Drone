import sys, os, time, datetime # Standard system libraries
import cv2 # Image processing library
import numpy as np # Transform data with this library
from tensorflow import keras
from keras.models import load_model
import pickle # Used to retrieve model dictionaries in this case   

sys.path.append(os.getcwd()) # Current working directory

# Custom libraries
import ImageProcessing as IP

# Access dictionaries from model
with open('lookup.pkl', 'rb') as f:
    lookup = pickle.load(f)
with open('reverselookup.pkl', 'rb') as f:
    reverselookup = pickle.load(f)

# Load pre-trained model
model = load_model('Resources/Models/gesture_recognition_Edges.h5')
print("Successfully loaded model")

# Predict the gesture name of the image using the model
def predict_gesture(image_data):
    predictions = model.predict(image_data)
    predicted_indx = np.argmax(predictions[0])
    predicted_gesture = reverselookup[predicted_indx]
    score = float("%0.2f" % (max(predictions[0]) * 100))

    return predicted_gesture, score

def __main__():
    cap = cv2.VideoCapture(0) # 0 usually indicates default webcam

    while True:
        ret, frame = cap.read() # Read frame from camera object

        current_time = time.time() # Get current time

        # Exit program when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save frame to folder when 'space' is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '): # Save image
            save_location = IP.save_image(frame)
            print(f"Successfully saved image to:  {save_location}")

        if cv2.waitKey(1) & 0xFF == ord('e'):
            print("Now identifying gesture...") # Show button has been pressed

            # Predict image
            img = cv2.imread(save_location)
            no_bg, method_used = IP.remove_background(img, 1)
            denoised, methods_used = IP.remove_noise(no_bg, 1)
            gray = IP.gray_scale(denoised)
            edges = IP.canny_edge_detection(gray)
            image_data = IP.preprocess_image(gray)

            # Display images
            IP.display_image(img, 'Original')
            IP.display_image(no_bg, 'No Background')
            IP.display_image(denoised, 'Denoised')
            IP.display_image(edges, 'Edges')

            # Make prediction
            predicted_gesture, score = predict_gesture(image_data)
            print(f"Predicted gesture: {predicted_gesture}, Confidence: {score}%") 

        cv2.imshow('Webcam', frame) # Display webcam feed
        #cv2.imshow('Edges', IP.canny_edge_detection(frame))

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def test():
    image_path = "Resources/TestImages/test4.png"
    img = cv2.imread(image_path)

    # Predict image
    no_bg, method_used = IP.remove_background(img, 1)
    denoised, methods_used = IP.remove_noise(no_bg, 1)
    gray = IP.gray_scale(denoised)
    edges = IP.canny_edge_detection(gray)
    image_data = IP.preprocess_image(edges)

    # Display images
    IP.display_image(img, 'Original')
    IP.display_image(no_bg, 'No Background')
    IP.display_image(denoised, 'Denoised')
    IP.display_image(edges, 'Edges')

    # Make prediction with the model
    predicted_gesture, score = predict_gesture(image_data)

    # Print output
    print(f"Predicted gesture: {predicted_gesture}, Confidence: {score}%") 

__main__() 
#test()