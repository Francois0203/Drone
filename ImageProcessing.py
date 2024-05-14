import numpy as np
import time, datetime, sys, os
import cv2

# Current working directory
sys.path.append(os.getcwd())

# Grayscale image
def gray_scale(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    return image

# Function to preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (320, 120))  # Resize

    arr = np.array(img, dtype = 'float32')
    arr = arr.reshape((1, 120, 320, 1))  # Add batch dimension
    arr /= 255.0  # Normalize

    return arr

# Save the image to Resources/Images
def save_image(img):
    try:
        now = datetime.datetime.now() # Get current date and time
        filename = "Resources/Images/" + now.strftime("%Y-%m-%d_%H-%M-%S.png")
        cv2.imwrite(filename, img)

        return filename
    except Exception as e:
        print(f"Error saving image: {e}")

        return None

# Removes noise from an image using various OpenCV techniques
def remove_noise(img, method):
    methods = []  # List to store the methods used

    if method == 1: # Method 1: Gaussian Blurring (Good for general noise)
        denoised_img = cv2.GaussianBlur(img, (5, 5), 0)
        methods.append("Gaussian Blurring")
    elif method == 2: # Method 2: Median Blurring (Good for salt-and-pepper noise)
        denoised_img = cv2.medianBlur(img, 5)
        methods.append("Median Blurring")
    elif method == 3: # Method 3: Fast Non-Local Means Denoising (More advanced)
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        methods.append("Fast Non-Local Means Denoising")

    return denoised_img, methods 

def remove_background(img, method):
    # Method 1: GrabCut (Useful for more complex backgrounds)
    if method == 1:
        mask = np.zeros(img.shape[:2], np.uint8) 
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10) # Rough estimate of object area
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = img * mask2[:, :, np.newaxis]
        method = "GrabCut" 
    else:
        # Method 2: Simple Thresholding (Good for simple backgrounds)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(img, img, mask = thresh)
        result = mask.copy()
        method = "Simple Thresholding"

    return result, method

# Outline edges
def canny_edge_detection(frame): 	
	# Apply Gaussian blur to reduce noise and smoothen edges 
	blurred = cv2.GaussianBlur(src = frame, ksize = (3, 5), sigmaX = 0.5) 
	
	# Perform Canny edge detection 
	edges = cv2.Canny(blurred, 70, 135) 
	
	return edges

# Show the image and pause processing until image is closed
def display_image(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

# Main program for testing
def __main__():
    image_path = "Resources/TestImages/test11.png"
    image = cv2.imread(image_path)

    # Prepare image
    denoised_image, methods_used = remove_noise(image, 1)
    image_without_bg, method_used = remove_background(denoised_image, 0)
    print("Method Used", method_used)
    image_data = preprocess_image(image_without_bg)

    # Display images
    display_image(image, "Original")
    display_image(denoised_image, "Removed Noise")
    display_image(image_without_bg, "Removed Background")

#__main__()