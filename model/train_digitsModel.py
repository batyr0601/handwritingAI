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
(x_train, y_train), (x_test, y_test) = emnist.load_data(type='digits') # Loads data into training and test sets
'Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373'

# Reshape data to  fit the model
x_train = x_train.reshape(240000,28,28,1)
x_test = x_test.reshape(40000,28,28,1)

# Categorize labels by one-hot encoding (Only used for digit classification)
y_test = tf.keras.utils.to_categorical(y_test)
y_train = tf.keras.utils.to_categorical(y_train)

# Define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, 3, strides=(2,2), activation=tf.nn.relu, input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.35))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.35))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile model for digit classification
history = model.fit(x_train,y_train, epochs=11, validation_data=(x_test,y_test)) # Train model

# Plot loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Crossentropy loss")

# Get loss history & plot it
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label ='val loss')

plt.legend()
plt.show()

model.save("model.emnistDigits")