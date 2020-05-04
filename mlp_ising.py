import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import *

from keras.utils import to_categorical
import keras.layers as layers
from keras import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

hex_state_file = open(r'data/Ising_state_hex.pkl', 'rb')
hex_states = pickle.load(hex_state_file)
hex_state_file.close()

tri_state_file = open(r'data/Ising_state_tri.pkl', 'rb')
tri_states = pickle.load(tri_state_file)
tri_state_file.close()

sq_state_file = open(r'data/Ising_state_tri.pkl', 'rb')
sq_states = pickle.load(sq_state_file)
sq_state_file.close()

def process_pickle(states):
    L = states.shape[1]
    M = np.array([np.abs(np.sum(i))/L**2 for i in states])
    y_1hot = to_categorical(np.round(M))

    return states, y_1hot


X_hex, y_hex = process_pickle(hex_states)
X_tri, y_tri = process_pickle(tri_states)
X_sq, y_sq = process_pickle(sq_states)


X_hex_train, X_hex_test, y_hex_train, y_hex_test = train_test_split(X_hex, y_hex, test_size=.2)
L_hex = X_hex.shape[1]

model = Sequential([
          layers.Flatten(input_shape=(L_hex,L_hex)),
          layers.Dense(100, activation='relu'),
          layers.Dense(2, activation='softmax'),
        ])


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_hex_train, y_hex_train, epochs=20, batch_size=32, validation_data = (X_hex_test, y_hex_test))


y_hex_pred = np.argmax(model.predict(X_hex_test), axis=1)
y_tri_pred = np.argmax(model.predict(X_tri), axis=1)
y_sq_pred = np.argmax(model.predict(X_sq), axis=1)


print("Hexagonal classification: ")
print(classification_report(np.argmax(y_hex_test,axis=1), y_hex_pred))

print("Triangular classification: ")
print(classification_report(np.argmax(y_tri,axis=1), y_tri_pred))

print("Square classification: ")
print(classification_report(np.argmax(y_sq,axis=1), y_sq_pred))






