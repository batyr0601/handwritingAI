from flask import Flask, redirect, url_for, render_template, request
import os
import tensorflow as tf
import cv2 as cv
import numpy as np


if ((os.getcwd()).split(os.sep)[-1] == 'handwritingAI'):
    pass
else:
    os.chdir(f'{os.getcwd()}//handwritingAI')

model = tf.keras.models.load_model('model.tflearn')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(file.filename)
    print(file)
    return "gg"

if __name__ == '__main__':
    app.run(debug=True)


