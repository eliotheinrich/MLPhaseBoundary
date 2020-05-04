import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import *

from keras.utils import to_categorical
import keras.layers as layers
from keras import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ice_state_file = open(r'data/Ising_state_sq_ice.pkl', 'rb')
ice_states = pickle.load(ice_state_file)
ice_state_file.close()

def process_pickle(states):
    L = states.shape[1]
    y = np.zeros(states.shape[0])
    y[100:] = 1
    y_1hot = to_categorical(y)

    return states, y_1hot




X, y = process_pickle(ice_states)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
L = X.shape[1]


model = Sequential([
          layers.Reshape(target_shape=(L,L,1), input_shape=(L,L)),
          layers.Conv2D(20, (6,6), padding='valid'),
          layers.MaxPooling2D(pool_size=2),
          layers.Flatten(), 
          layers.Dense(100, activation='relu'),
          layers.Dense(2, activation='softmax')
        ])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data = (X_test, y_test))

