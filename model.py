# Import required libraries
import os
import csv
import cv2
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
#from skimage import io, color, exposure, filters, img_as_ubyte
#from skimage.transform import resize
#from skimage.util import random_noise

# Functions
def PreprocessingLayer():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20),(0,0))))
    return model


def generator(samples, batch_size=32):

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
        
        
def DrivingLogLines(dataPath, skipHeader = True):
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def combineImages(center, left, right, measurement, correction):
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return(imagePaths, measurements)


def findImages(dataPath):
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerFinal = []
    leftFinal = []
    rightFinal = []
    measurementsFinal = []
    
    for directory in dataDirectories:
        lines = DrivingLogLines(directory)
        center = []
        left = []
        right = []
        measurements = []
        
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[0].strip())
        
        centerFinal.extend(center)
        leftFinal.extend(left)
        rightFinal.extend(right)
        measurementsFinal.extend(measurements)  

    return (centerFinal, leftFinal, rightFinal, measurementsFinal)


def nVidiaModel():
    model = PreprocessingLayer()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(160,320,3), activation='relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same", activation='relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same", activation='relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same", activation='relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model    
          
# Image Loading
centerPaths, leftPaths, rightPaths, measurements = findImages('data')
imagePaths, measurements = combineImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
print('Total Images: {}'.format(len(imagePaths)))

# Splitting samples
samples = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Training samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

# plot the distribution
plt.hist(measurements, bins = 100)
plt.show()

generated_trainingdata = generator(train_samples, batch_size = 32)
generated_validationdata = generator(validation_samples, batch_size = 32)

# Model Creation
model = nVidiaModel()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(generated_trainingdata,samples_per_epoch=len(train_samples), validation_data=generated_validationdata, nb_val_samples=len(validation_samples),nb_epoch=3,verbose=1)

model.save('model.h5')
print('Done! Model Saved!')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

# keras method to print the model summary
model.summary()

