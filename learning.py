import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import *

from keras.utils import to_categorical
import keras.layers as layers
from keras import Sequential

from sklearn.model_selection import train_test_split

state_file = open(r'data/Ising_state_hex.pkl', 'rb')
states = pickle.load(state_file)
state_file.close()

L = states.shape[1]
M = np.array([np.abs(np.sum(i))/L**2 for i in states])
y = np.round(M)
y_1hot = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(states, y_1hot, test_size=.2)

model = Sequential([
          layers.Flatten(input_shape=(L,L)),
          layers.Dense(100, activation='relu'),
          layers.Dense(2, activation='softmax'),
        ])


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data = (X_test, y_test))








