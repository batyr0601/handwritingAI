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
    file = request.files['file'].stream.read()
    img = np.array(file)
    return 


'''
    img = cv.imread(file)
    img = cv.resize(img,(28,28))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.invert(np.array([img]))
    img = img.reshape(1,28,28,1)
    prediction = model.predict(img)
    print(np.argmax(prediction))

    return np.argmax(prediction)
    #return redirect(url_for('home'))
'''
if __name__ == '__main__':
    app.run(debug=True)


