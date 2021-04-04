import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist # Gets mnist data set
(x_train, y_train), (x_test, y_test) = mnist.load_data() # Loads data into training and test sets

# Normalize training features from 256-bit color to binary
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential() # Defines type of neural network
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # Add input layer (Take images as an array of 28,28 datapoints)
# Add hidden layer
model.add(tf.keras.layers.Dense(units=192, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation = tf.nn.softmax)) # Add output layer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs=5) # Declares what data the model will use

accuracy, loss = model.evaluate(x_test, y_test) # Declares what data the model will use to test
print(accuracy)
print(loss)

