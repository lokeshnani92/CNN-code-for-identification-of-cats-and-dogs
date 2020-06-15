# CNN-code-for-identification-of-cats-and-dogs
# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
import keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding second Convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step 3 - Flattering
classifier.add(Flatten())

# step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Cpmpiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
    
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\nani\Desktop\Machine Learning A-Z\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\dataset\training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(r'C:\Users\nani\Desktop\Machine Learning A-Z\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
from IPython.display import display
from PIL import Image
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)
