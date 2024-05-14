import os, sys # For handling directories
from random import randint

import numpy as np # We'll be storing our data as numpy arrays
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras.utils import to_categorical
from keras import layers
from keras import models

# Current working directory
sys.path.append(os.getcwd())

# Check number of GPU's available to train model
print("Num GPU's available: ", len(tf.config.list_physical_devices('GPU')))

lookup = dict() 
reverselookup = dict()
count = 0

for j in os.listdir('Resources/leapGestRecog/00/'):
    if not j.startswith('.'): # Ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1

x_data = []
y_data = []
datacount = 0 # Count the amount of images in the dataset

# Loop over the ten top-level folders
for i in range(0, 10): 
    print("i-value: ", i)

    for j in os.listdir('Resources/leapGestRecog/0' + str(i) + '/'):
        print("j-value: ", j)

        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # Count how many images are there per gesture type

            for k in os.listdir('Resources/leapGestRecog/0' + str(i) + '/' + j + '/'): 
                print('Resources/leapGestRecog/0' + str(i) + '/' + j + '/')

                img = Image.open('Resources/leapGestRecog/0' + str(i) + '/' + j + '/' + k).convert('L') # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1

            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count

print("Number of Images: ", datacount)
x_data = np.array(x_data, dtype = 'float32')

y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size

# Display different types of images
def display_image_types():
    for i in range(0, 10):
        plt.imshow(x_data[i*200, :, :])
        plt.title(reverselookup[y_data[i*200, 0]])
        plt.show()

#display_image_types()

y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255

x_train, x_further, y_train, y_further = train_test_split(x_data, y_data, test_size = 0.2)
x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size = 0.5)

# Build neural network and classification system
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides = (2, 2), activation = 'relu', input_shape = (120, 320, 1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10, batch_size = 64, verbose = 1, validation_data = (x_validate, y_validate))

# Save the model for use in other scripts
model.save('Resources/Models/gesture_recognition_LG.h5')
print("Model saved successfully")

# Save dictionaries in pickle files
with open('lookup.pkl', 'wb') as f:
    pickle.dump(lookup, f)

with open('reverselookup.pkl', 'wb') as f:
    pickle.dump(reverselookup, f)

[loss, acc] = model.evaluate(x_test, y_test, verbose = 1)
print("Accuracy:" + str(acc))