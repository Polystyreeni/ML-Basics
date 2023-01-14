"""
(Very) Basic Neural Network for Cifar-10 dataset

A full connected Neural Network using Keras


Cifar-10 is a dataset that contains 60000 32x32 images in 10 classes
The dataset was presented by Alex Krizhevsky in 2009.
More info, download, and the tech report can be found at
http://www.cs.toronto.edu/~kriz/cifar.html

"""

import pickle
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import os
import sys
sys.path.append(os.getcwd())


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Calculate the class accuracy
def class_acc(pred, gt, return_value=False):
    correct = 0
    total = len(gt)

    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct += 1

    print(f"Accuracy: {(correct / total) * 100} %")

    # Additional print if we're using training data
    if correct == total:
        print("Test was conducted with training data!")
    if return_value:
        return (correct / total) * 100


def one_hot_code_classes(Y):
    coded = np.zeros([Y.shape[0], 10])
    for i in range(0, 10):
        coded[np.where(Y == i), i] = 1

    return coded


# Load all training batches to one dict
datadict = {"data": np.array([]), "labels": np.array([])}
for i in range(1, 6):
    print(f"Load training data for batch {i}")
    batch = unpickle('cifar-10-data/data_batch_' + str(i))
    datadict["data"] = np.append(datadict["data"], batch["data"])
    datadict["labels"] = np.append(datadict["labels"], batch["labels"])

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('cifar-10-data/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint32")
Y = np.array(Y)

testdict = unpickle('cifar-10-data/test_batch')
testX = testdict["data"]
realY = testdict["labels"]
realY = np.array(realY)

testX = testX.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint32")

# Convert class numbers to one-hot vectors
y_tr_2 = one_hot_code_classes(Y)

# Create Neural Network
neurons = 20
epochs = 100
learning_rate = 0.1

model = Sequential()
model.add(Dense(neurons, input_dim=3072, activation='sigmoid'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(Dense(neurons, activation='sigmoid'))
# model.add(Dense(neurons, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))

opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

# Normalizing data for better accuracy
X_1dim = X.reshape(50000, 3072)
X_1dim = X_1dim / 255

# Train neural network with training data
print(f"Training model: neurons={neurons} epochs={epochs} lr={learning_rate}")
model.fit(X_1dim, y_tr_2, epochs=epochs, verbose=1)

# Test with cifar10 test data
testX_1dim = testX.reshape(10000, 3072)
pred = model.predict(testX_1dim)

predY = np.empty((10000, 1))
for i in range(pred.shape[0]):
    label = 0
    max_val = 0
    for c in range(0, 10):
        if pred[i][c] > max_val:
            label = c
            max_val = pred[i][c]
    predY[i] = label

class_acc(predY, realY)


