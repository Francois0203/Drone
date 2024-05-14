import sys, os, time # Standard system libraries
import cv2 # Image processing library

# Current working directory
sys.path.append(os.getcwd())

def save_image(img, counter):
    try:
        counter += 1
        filename = f"Resources/Images/image{counter}.png"
        cv2.imwrite(filename, img)

        return filename
    except Exception as e:
        print(f"Error saving image: {e}")

        return None

def __main__():
    # Variables
    count = 0

    cap = cv2.VideoCapture(0) # 0 usually indicates default webcam

    last_capture_time = 0 # Setup for capturing every two seconds
    capture_time = 1 # After every few seconds, the frame should be used to identify gesture

    while True:
        ret, frame = cap.read() # Read frame from camera object

        # Check if 1 second has passed or the spacebar has been pressed
        current_time = time.time()

        # Exit program when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if (current_time - last_capture_time >= capture_time):
            last_capture_time = current_time

            # Save image
            count = count + 1
            save_location = save_image(frame, count)
            print(f"Successfully saved image to:  {save_location}")  

        # Display webcam feed
        cv2.imshow('Webcam', frame)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

__main__()