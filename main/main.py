import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

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

history = model.fit(x_train,y_train, epochs=5) # Declares what data the model will use

accuracy, loss = model.evaluate(x_test, y_test) # Declares what data the model will use to test
print(accuracy)
print(loss)

# Plot loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Crossentropy loss")

plt.plot(history.history['loss']) # Get loss history & plot it

plt.show()


while(True): # Input loop for numbers
    path = input("Absolute path of 28x28 number 0-9 (EXIT to exit): ")
    if(path == 'EXIT'):
        break
    else:
        path = path.replace(os.sep,'/') # Fix bug with file paths
        img = cv.imread(path)
        ret, img = cv.threshold(img,254,255,cv.THRESH_BINARY) # Black and white the image
        img = np.invert(np.array([img])) # Invert to be white on black
        plt.imshow(img[0])
        plt.show()
        prediction = model.predict(img[0])
        count = 0
        for x in prediction:
            print(f'{count}: {(x*100):.2f}')
            count += 1
        plt.imshow(img[0])
        plt.show()


