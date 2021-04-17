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

if ((os.getcwd()).split(os.sep)[-1] == 'handwritingAI'):
    pass
else:
    os.chdir(f'{os.getcwd()}//handwritingAI')

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
    img = np.invert(np.array([img]))
    img = img.reshape(1,28,28,1)
    prediction = model.predict(img)

    return render_template('index.html',content=str(chr(np.argmax(prediction)+96)))

if __name__ == '__main__':
    app.run(debug=True)
