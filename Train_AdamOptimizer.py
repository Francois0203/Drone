import os, sys
from random import randint
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical
from keras import layers, models
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Current working directory
sys.path.append(os.getcwd())

# Custom libraries
import ImageProcessing as IP

lookup = dict()
reverselookup = dict()
count = 0

for j in os.listdir('Resources/NiceHands/00/'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count += 1

x_data = []
y_data = []
datacount = 0

# Loop over the 5 top-level folders
for i in range(0, 5):
    print("i-value: ", i)
    for j in os.listdir('Resources/NiceHands/0' + str(i) + '/'):
        print("j-value: ", j)
        if not j.startswith('.'):
            count = 0
            for k in os.listdir('Resources/NiceHands/0' + str(i) + '/' + j + '/'):
                print('Resources/NiceHands/0' + str(i) + '/' + j + '/')
                img = Image.open('Resources/NiceHands/0' + str(i) + '/' + j + '/' + k).convert('L')
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr)
                count += 1
                y_values = np.full((count, 1), lookup[j])
                y_values = y_values.reshape(-1)
                y_data.append(lookup[j])
            datacount += count

print("Number of Images: ", datacount)
x_data = np.array(x_data, dtype = 'float32')

y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1)

# Display different types of images
def display_image_types():
    for i in range(0, 3):
        plt.imshow(x_data[i*200, :, :], cmap = 'gray')
        plt.title(reverselookup[y_data[i*200, 0]])
        plt.show()

display_image_types()

num_classes = len(np.unique(y_data))
print("Number of classes: ", num_classes)

y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255

x_train, x_further, y_train, y_further = train_test_split(x_data, y_data, test_size = 0.2)
x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size = 0.5)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)
datagen.fit(x_train)

# Check number of GPUs available to train model
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPU's available: ", len(tf.config.list_physical_devices('GPU')))
print("Devices: ", physical_devices)

# Train model using GPU
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Build neural network and classification system
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (120, 320, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(num_classes, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

# Early stopping and model checkpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
model_checkpoint = ModelCheckpoint('Resources/Models/best_model_Adam.keras', save_best_only = True)

# Plot Training History
def plot_history(history):
    plt.figure(figsize = (12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label = 'Training Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

history = model.fit(datagen.flow(x_train, y_train, batch_size = 32), epochs = 50, validation_data = (x_validate, y_validate), callbacks = [early_stopping, model_checkpoint])

plot_history(history)

# Save the model for use in other scripts
model.save('Resources/Models/gesture_recognition_Adam.h5')
print("Model saved successfully")

# Save dictionaries in pickle files
with open('lookup.pkl', 'wb') as f:
    pickle.dump(lookup, f)

with open('reverselookup.pkl', 'wb') as f:
    pickle.dump(reverselookup, f)

# Evaluate the best model
model.load_weights('Resources/Models/best_model_Adam.keras')
loss, acc = model.evaluate(x_test, y_test, verbose = 1)
print("Accuracy: " + str(acc))



