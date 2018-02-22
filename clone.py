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
from sklearn.utils import shuffle
from random import shuffle
from sklearn.model_selection import train_test_split

samples = []
with open('./1_lap/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    if abs(angle) >= 0:
                        correction = 0.2
                        if i == 1:
                            angle = angle + correction
                        if i == 2:
                            angle = angle - correction
                        images.append(image)
                        angles.append(angle)
                        image_flipped = np.fliplr(image)
                        angle_flipped = -angle
                        images.append(image_flipped)
                        angles.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

row, col, ch = 160, 320, 3
# images = []
# measurements = []
# for line in lines:
#     for i in range(3):
#         source_path = line[0]
#         filename = source_path.split('/')[-1]
#         current_path = './data/IMG/' + filename
#         image = cv2.imread(current_path)
#         images.append(image)
#     correction = 0.2
#     measurement = float(line[3])
#     measurements.append(measurement)
#     measurements.append(measurement+correction)
#     measurements.append(measurement-correction)
#
# augmented_images = []
# augmented_measurements = []
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     flipped_image = cv2.flip(image, 1)
#     flipped_measurement = float(measurement) * -1.0
#     augmented_images.append(flipped_image)
#     augmented_measurements.append(flipped_measurement)
#
# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples) // 32, validation_data=validation_generator, validation_steps=len(validation_samples) // 32, max_queue_size=10,
                    callbacks=callbacks_list, epochs=2, verbose = 1)


model.save('model.h5')

    # visualize_model(h)
