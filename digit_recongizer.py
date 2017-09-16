#Load lib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

#Settings 
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 2500

DROPOUT = 0.5
BATCH_SIZE = 50

VALIDATION_SIZE = 2000

IMAGE_TO_DISPLAY = 10

#Load train and test data
train = pd.read_csv('E:/GitHub/Digit-Recognizer/data/train.csv')
test = pd.read_csv('E:/GitHub/Digit-Recognizer/data/test.csv')

#Separate images and labels
images = train.iloc[:, 1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0/255.0)
labels = train.iloc[:, 0].values

#Image size
image_size = images.shape[1]
width = height = np.sqrt(image_size).astype(np.uint8)

#Display image
def display(img):
	image = img.reshape(width, height)
	plt.axis('off')
	plt.imshow(image, cmap = cm.binary)

#display(images[IMAGE_TO_DISPLAY])

#Convert class labels from scalars to one-hot vectors
def one_hot(labels):
    index = np.arange(labels.size) * 10 + labels
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels.flat[index] = 1
    return one_hot_labels

labels = one_hot(labels)

#TensorFlow graph
