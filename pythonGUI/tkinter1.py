import tkinter as tk
from PIL import ImageGrab
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

if ((os.getcwd()).split(os.sep)[-1] == 'handwritingAI'):
    pass
else:
    os.chdir(f'{os.getcwd()}//handwritingAI')

model = tf.keras.models.load_model('model.tflearn')

lastx, lasty = 0, 0

def xy(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y # Gets and sets x,y position of mouse

def addLine(event):
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y), width = 50)

    lastx, lasty = event.x, event.y

def save(event):
    x=root.winfo_rootx()+canvas.winfo_x()
    y=root.winfo_rooty()+canvas.winfo_y()
    x1=x+canvas.winfo_width()
    y1=y+canvas.winfo_height()
    im = ImageGrab.grab((x, y, x1, y1))
    img = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
    img = cv.resize(img,(28,28))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.invert(np.array([img]))
    img = img.reshape(1,28,28,1)
    prediction = model.predict(img)
    print(np.argmax(prediction))
    plt.imshow(img[0])
    plt.show()

# Creates tkinter frame
root = tk.Tk()
root.geometry("800x800")
root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)

canvas = tk.Canvas(root, bg='white') # Create canvas object
canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S)) # Assign canvas object to tkinter frame

# Event listeners
canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", addLine)
root.bind("<Control-s>", save)

root.mainloop()