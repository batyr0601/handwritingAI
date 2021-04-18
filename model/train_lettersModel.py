import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from extra_keras_datasets import emnist

if ((os.getcwd()).split(os.sep)[-1] == 'handwritingAI'):
    pass
else:
    os.chdir(f'{os.getcwd()}//handwritingAI')

# Get data
(x_train, y_train), (x_test, y_test) = emnist.load_data(type='letters') # Loads data into training and test sets
'Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373'

# Reshape data to  fit the model
x_train = x_train.reshape(124800,28,28,1)
x_test = x_test.reshape(20800,28,28,1)

# Define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, 3, padding="same", strides= (2,2), activation=tf.nn.relu, input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Conv2D(128, 3, padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.35))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.35))
model.add(tf.keras.layers.Dense(27, activation = tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Compile model
history = model.fit(x_train,y_train, epochs=15, validation_data=(x_test,y_test)) # Train model


# Plot loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Crossentropy loss")

# Get loss history & plot it
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label ='val loss')

plt.legend()
plt.show()

model.save("model.emnistLetters")