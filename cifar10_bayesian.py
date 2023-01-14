"""
Bayesian classifier for Cifar-10 dataset
Classifying Cifar-10 dataset images using Bayesian method.
Predicts the labels for each image, and prints the accuracy of the predictions

The accuracy of this model is around 42,8 % when image dimensions are 8 x 8

This exercise contains 2 different methods
1. Naive methods, which assumes the R, G and B channels are all independent (simpler)
2. Non-naive method, which does not assume that color channels are independent (better)

Cifar-10 is a dataset that contains 60000 32x32 images in 10 classes
The dataset was presented by Alex Krizhevsky in 2009.
More info, download, and the tech report can be found at
http://www.cs.toronto.edu/~kriz/cifar.html

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk
import math as math
from scipy.stats import norm
from scipy.stats import multivariate_normal

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


# Resize the image to one pixel
def cifar10_color(X):
    Xp = np.empty((X.shape[0], 3))
    for i in range(X.shape[0]):
        resized = sk.resize(X[i], (1, 1, 3), False)
        Xp[i][0] = resized[0][0][0]
        Xp[i][1] = resized[0][0][1]
        Xp[i][2] = resized[0][0][2]

    return Xp


# Resize image to given dimension
def cifar10_color_nxn(X, dim):
    Xp = np.empty((X.shape[0], dim, dim, 3))
    for i in range(X.shape[0]):
        resized = sk.resize(X[i], (dim, dim, 3), False)
        Xp[i] = resized

    return Xp


# Computes the normal distribution parameters for all ten classes
def cifar10_naivebayes_learn(Xp, Y):
    # Initialize mu, sigma and p
    mu = np.empty((10, 3))
    sigma = np.empty((10, 3))
    p = np.empty((10, 1))

    for label in range(0, 10):
        label_r = []
        label_g = []
        label_b = []
        class_elements = 0
        for i in range(Xp.shape[0]):
            if Y[i] == label:
                label_r.append(Xp[i][0])    # Red channel
                label_g.append(Xp[i][1])    # Green chanel
                label_b.append(Xp[i][2])    # Blue channel
                class_elements += 1
        mu[label] = [np.mean(label_r), np.mean(label_g), np.mean(label_b)]
        sigma[label] = [np.var(label_r), np.var(label_g), np.var(label_b)]
        p[label] = class_elements / Xp.shape[0]

    return mu, sigma, p


# Naive classifier function
def cifar10_classifier_naivebayes(testXp, mu, sigma, p):
    pred_label = 0
    probability = 0

    for i in range(mu.shape[0]):
        prob = calculate_probability(testXp[0], mu[i][0], sigma[i][0]) \
        * calculate_probability(testXp[1], mu[i][1], sigma[i][1]) * calculate_probability(testXp[2], mu[i][2], sigma[i][2])
        if prob > probability:
            pred_label = i
            probability = prob

    return pred_label


# Calculate the Gaussian probability distribution function for x
# This was done before I realised scipy had a proper norm function...
def calculate_probability(x, mean, sigma):
    exponent = math.exp(-((x - mean) ** 2 / (2 * sigma)))
    return (1 / (math.sqrt(2 * math.pi) * math.sqrt(sigma))) * exponent


# Learn function for NxN sized images
def cifar10_bayes_learn(Xf, Y):
    num_of_elements = Xf[0].shape[0] * Xf[0].shape[0] * 3
    mu = np.empty((10, num_of_elements))
    sigma = np.empty((10, num_of_elements, num_of_elements))
    p = np.empty((10, 1))

    for label in range(0, 10):
        label_images = []
        for i in range(Xf.shape[0]):
            if Y[i] == label:
                label_images.append(Xf[i])

        # Transform images to a two dimensional array
        label_images = np.array(label_images)
        label_images = label_images.reshape(label_images.shape[0], num_of_elements)

        # Calculate the mu array
        for i in range(num_of_elements):
            mu[label, i] = np.mean(label_images[:, i])

        # Calculate the covariance matrix
        covariance = np.cov(np.transpose(label_images))
        sigma[label] = covariance
        p[label] = label_images.shape[0] / Xf.shape[0]

    return mu, sigma, p


# Classify function for NxN sized images
def cifar10_classifier_bayes(x, mu, sigma, p):
    pred_label = 0
    probability = 0
    for i in range(mu.shape[0]):
        prob = multivariate_normal.pdf(x, mu[i], sigma[i])
        if prob > probability:
            pred_label = i
            probability = prob

    return pred_label


# Classify images using multivariate normal and given mu & sigma variables
# logpdf is used here to prevent 0.0 values from pdf function
def cifar10_classifier_bayes_nxn(x, mu, sigma, p):
    pred_label = 0
    probability = -math.inf

    for elem in range(mu.shape[0]):
        prob = multivariate_normal.logpdf(x, mu[elem], sigma[elem])
        if prob > probability:
            pred_label = elem
            probability = prob

    return pred_label


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

# This is where the fun starts!
print()
print("Begin test for naive Bayesian method:")
Xp = cifar10_color(X)
testXp = cifar10_color(testX)
muN, sigmaN, pN = cifar10_naivebayes_learn(Xp, Y)
print("Naive Bayes learning complete!")

print("Classifying test data with Naive Bayesian classifier...")
naiveY = np.empty((10000, 1))
for test_elem in range(testX.shape[0]):
    naiveY[test_elem] = cifar10_classifier_naivebayes(testXp[test_elem], muN, sigmaN, pN)

print()
print("Naive bayesian classifiying completed!")
class_acc(naiveY, realY)

print()
print("Begin test for Non-naive method:")
accuracies = []
dimensions = [1, 2, 4, 8]   # The image sizes to calculate, default: 1, 2, 4, 8, 16, 32
for dim in dimensions:
    Xp = cifar10_color_nxn(X, dim)
    print(f"Learning started for {dim} x {dim} images...")
    mu, sigma, p = cifar10_bayes_learn(Xp, Y)
    print("Learning complete!")

    testXp = cifar10_color_nxn(testX, dim)

    print(f"Classifying test data {dim} x {dim} with Bayesian classifier...")
    bayesY = np.empty((10000, 1))
    tested = 0

    # Reshape all test images
    testXp = testXp.reshape(testXp.shape[0], 1, mu.shape[1])

    # Classify test images
    for test_elem in range(testX.shape[0]):
        bayesY[test_elem] = cifar10_classifier_bayes_nxn(testXp[test_elem], mu, sigma, p)
        tested += 1
        if tested % 1000 == 0:
            print(f"Tested {tested} of {testX.shape[0]} samples")

    print(f"Classifying for {dim} x {dim} images done!")
    print()
    accuracies.append(class_acc(bayesY, realY, True))

# Plot the accuracy values
plt.xlim(min(dimensions), 32)
plt.ylim(0, 100)
plt.xlabel("Image dimensions (N x N)")
plt.ylabel("Accuracy (%)")

plt.plot(dimensions, accuracies, 'ro')
plt.draw()
plt.show()
