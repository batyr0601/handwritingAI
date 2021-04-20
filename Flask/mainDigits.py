from flask import Flask, redirect, url_for, render_template, request
import os
import cv2 as cv
from PIL import Image
import base64
import matplotlib.pyplot as plt
import io
import numpy as np
import re
import tensorflow as tf
import math

if ((os.getcwd()).split(os.sep)[-1] == 'models'):
    pass
elif((os.getcwd()).split(os.sep)[-1] == 'handwritingAI'):
    os.chdir(f'{os.getcwd()}//models')
else:
    os.chdir(f'{os.getcwd()}//handwritingAI//models')

model = tf.keras.models.load_model('model.emnistDigits')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    file = request.form['my_hidden']

    imgstr = re.search(r'base64,(.*)', file).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)
    img = Image.new("RGB", im.size, "WHITE")
    img.paste(im, (0, 0), im)
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

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
    cntImg1 = cv.copyMakeBorder(cntImg, round(h/14), round(h/14), round(w/14), round(w/14), cv.BORDER_CONSTANT, value = (255,255,255))

    invImg = cv.resize(cntImg1,(28,28))
    invImg = np.invert([invImg])
    finalImg = invImg.reshape(1,28,28,1)
    prediction = model.predict(finalImg)
    predictionDict = {}

    for i in range(3):
        predictionDict["ans"+str(i)] = [str(f"{np.argmax(prediction)}"),str(f"{((prediction[0][np.argmax(prediction)])*100):.2f}%")]
        prediction[0][int(predictionDict["ans"+str(i)][0])] = 0

    return render_template('index.html',
                            ans1=f"{predictionDict['ans0'][0]} {predictionDict['ans0'][1]}",
                            ans2=f"{predictionDict['ans1'][0]} {predictionDict['ans1'][1]}",
                            ans3=f"{predictionDict['ans2'][0]} {predictionDict['ans2'][1]}")

if __name__ == '__main__':
    app.run(debug=True)
