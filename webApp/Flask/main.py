from flask import Flask, redirect, url_for, render_template, request
import os
import cv2 as cv
from PIL import Image
import base64
import matplotlib.pyplot as plt
import io
import numpy as np
'''
import tensorflow as tf

if ((os.getcwd()).split(os.sep)[-1] == 'handwritingAI'):
    pass
else:
    os.chdir(f'{os.getcwd()}//handwritingAI')

model = tf.keras.models.load_model('model.tflearn')
'''

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    file = request.form['my_hidden']
    print(file)

    return "gg"

if __name__ == '__main__':
    app.run(debug=True)


