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

if ((os.getcwd()).split(os.sep)[-1] == 'models'):
    pass
elif((os.getcwd()).split(os.sep)[-1] == 'handwritingAI'):
    os.chdir(f'{os.getcwd()}//models')
else:
    os.chdir(f'{os.getcwd()}//handwritingAI//models')

model = tf.keras.models.load_model('model.emnistLetters')

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

    return render_template('index.html',
                        ans1=f"{predictionDict['ans0'][0]} {predictionDict['ans0'][1]}",
                        ans2=f"{predictionDict['ans1'][0]} {predictionDict['ans1'][1]}",
                        ans3=f"{predictionDict['ans2'][0]} {predictionDict['ans2'][1]}")

if __name__ == '__main__':
    app.run(debug=True)
