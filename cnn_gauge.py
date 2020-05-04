import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import *

from keras.utils import to_categorical
import keras.layers as layers
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



gauge_state_file = open(r'data/Ising_state_gauge.pkl', 'rb')
gauge_states = pickle.load(gauge_state_file)
gauge_state_file.close()

def process_pickle(states):
    L = states.shape[1]
    y = np.zeros(states.shape[0])
    y[100:] = 1

    return states, y




X, y = process_pickle(gauge_states)

datagen = ImageDataGenerator(
            width_shift_range=.2,
            height_shift_range=.2,
            horizontal_flip=True
          )




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
L = X.shape[1]


model = Sequential([
          layers.Reshape(target_shape=(L,L,1), input_shape=(L,L)),
          layers.Conv2D(128, (4,4), padding='valid', activation='relu'),
          layers.Flatten(), 
          layers.Dense(128, activation='relu'),
          layers.Dropout(.5),
          layers.Dense(1, activation='sigmoid')
        ])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data = (X_test, y_test))




