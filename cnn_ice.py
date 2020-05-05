import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import *
from plot_lattice import plot_checkerboard

from keras.utils import to_categorical
import keras.layers as layers
from keras import Sequential

from sklearn.model_selection import train_test_split

ice_state_file = open(r'Ising_state_ice.pkl', 'rb')
ice_states = pickle.load(ice_state_file)
ice_state_file.close()

def process_pickle(states):
    L = states.shape[1]
    y = np.zeros(states.shape[0])
    y[50:] = 1
    y_1hot = to_categorical(y)

    return states, y_1hot




X, y = process_pickle(ice_states)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
L = X.shape[1]


model = Sequential([
          layers.Reshape(target_shape=(L,L,1), input_shape=(L,L)),
          layers.Conv2D(128, (3,3), padding='valid', activation='relu'),
          layers.Flatten(), 
          layers.Dense(128, activation='relu'),
          layers.Dropout(.5),
          layers.Dense(2, activation='softmax')
        ])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])


hist =  model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data = (X_test, y_test))
with open('ice_hist.pkl', 'wb') as f: pickle.dump(hist.history, f)

