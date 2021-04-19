import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math

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
        bwImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Remove color channels
        bwImg = cv.adaptiveThreshold(bwImg,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,115,1) # Make image B&W

        cnts, _ = cv.findContours(bwImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 1:
            cnt = cnts[0]

        else:
            cnt = cnts[1]

        x,y,w,h = cv.boundingRect(cnt)

        if w > h:
            h1 = h
            h = w
            diff = math.floor((h-h1)/2)
            y = y-diff

        else:
            w1 = h
            w = h
            diff = math.floor((w-w1)/2)
            x = x-diff

        cntImg = bwImg[y:y+h, x:x+w]
        cntImg1 = cv.copyMakeBorder(cntImg, round(h/28), round(h/28), round(w/28), round(w/28), cv.BORDER_CONSTANT, value = (255,255,255))

        invImg = cv.resize(cntImg1,(28,28))
        invImg = np.invert([invImg])
        finalImg = invImg.reshape(1,28,28,1)
        prediction = model.predict(finalImg)
        predictionDict = {}

        for i in range(3):
            predictionDict["ans"+str(i)] = [chr(int(f"{np.argmax(prediction)+96}")),str(f"{((prediction[0][np.argmax(prediction)])*100):.2f}%")]
            prediction[0][np.argmax(prediction)] = 0

        for z in predictionDict.keys():
            print(f"{(predictionDict[z])[0]} {(predictionDict[z])[1]}")

        plt.imshow(finalImg[0])
        plt.show()