import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

if ((os.getcwd()).split(os.sep)[-1] == 'models'):
    pass
elif((os.getcwd()).split(os.sep)[-1] == 'handwritingAI'):
    os.chdir(f'{os.getcwd()}//models')
else:
    os.chdir(f'{os.getcwd()}//handwritingAI//models')

model = tf.keras.models.load_model('model.emnistLetters')

while(True): # Input loop for numbers
    path = input("Absolute path of 28x28 number 0-9 (EXIT to exit): ")
    if(path == 'EXIT'):
        break
    else:
        path = path.replace(os.sep,'/') # Fix bug with file paths
        img = cv.imread(path)
        img = cv.resize(img,(28,28))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cnt, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        x,y,w,h = cv.boundingRect(cnt[0])
        img = img[y:y+h, x:x+w]
        img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value = (255,255,255))
        img = cv.resize(img,(28,28))
        img = np.invert(np.array([img]))
        img = img.reshape(1,28,28,1)
        prediction = model.predict(img)
        predictionDict = {}

        for i in range(3):
            predictionDict["ans"+str(i)] = [chr(int(f"{np.argmax(prediction)+96}")),str(f"{((prediction[0][np.argmax(prediction)])*100):.2f}%")]
            prediction[0][np.argmax(prediction)] = 0

        for x in predictionDict.keys():
            print(f"{(predictionDict[x])[0]} {(predictionDict[x])[1]}")

        plt.imshow(img[0])
        plt.show()