import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle

mnist = tf.keras.datasets.mnist # Gets mnist data set
(x_train, y_train), (x_test, y_test) = mnist.load_data() # Loads data into training and test sets

# Reshape data to  fit the model
x_test = x_test.reshape(10000,28,28,1)
x_train = x_train.reshape(60000,28,28,1)

# Categorize labels by one-hot encoding
y_test = tf.keras.utils.to_categorical(y_test)
y_train = tf.keras.utils.to_categorical(y_train)

model = tf.keras.models.Sequential() # Defines type of neural network
model.add(tf.keras.layers.Conv2D(32, 3, strides=(2,2), activation=tf.nn.relu, input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

try:
    model = tf.keras.models.load_model('model.tflearn')
except:
    model.fit(x_train,y_train, epochs=10, validation_data=(x_test,y_test))
    model.save("model.tflearn")


# Plot loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Crossentropy loss")

# Get loss history & plot it
'''plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label ='val loss')

plt.legend()
plt.show()'''


while(True): # Input loop for numbers
    path = input("Absolute path of 28x28 number 0-9 (EXIT to exit): ")
    if(path == 'EXIT'):
        break
    else:
        path = path.replace(os.sep,'/') # Fix bug with file paths
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = np.array([img])
        img = img.reshape(1,28,28,1)
        prediction = model.predict(img)
        print(prediction)
        plt.imshow(img[0])
        plt.show()

