import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model_v1 = tf.keras.models.load_model('model-v1')

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
        prediction = model-v1.predict(img)
        print(prediction)
        plt.imshow(img[0])
        plt.show()