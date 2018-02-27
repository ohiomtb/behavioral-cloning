import os
import csv
import cv2
import math
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import matplotlib.pyplot as plt
import sklearn
from random import shuffle
from keras.layers.advanced_activations import ELU

samples = []
with open('./udacity_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

del lines[0]


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.15)


def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(0, 3):

                    name = './udacity_data/IMG/'+batch_sample[i].split('/')[-1]
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)

                    if(i == 0):
                        angles.append(center_angle)
                    elif(i == 1):
                        angles.append(center_angle+0.2)
                    elif(i == 2):
                        angles.append(center_angle-0.2)

                    images.append(cv2.flip(center_image, 1))
                    if(i == 0):
                        angles.append(center_angle*-1)
                    elif(i == 1):
                        angles.append((center_angle+0.2)*-1)
                    elif(i == 2):
                        angles.append((center_angle-0.2)*-1)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('elu'))

# layer 7- fully connected layer 1
model.add(Dense(50))
model.add(Activation('elu'))

# layer 8- fully connected layer 1
model.add(Dense(10))
model.add(Activation('elu'))

# layer 9- fully connected layer 1
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(
    train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')
