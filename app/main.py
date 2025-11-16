import cv2
import pickle
import numpy as np
import os

# Obtener el directorio base de la aplicación
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

estacionamientos = []
with open(os.path.join(BASE_DIR, 'espacios.pkl'), 'rb') as file:
    estacionamientos = pickle.load(file)

# leer el video 
video = cv2.VideoCapture(os.path.join(BASE_DIR, 'video.mp4'))

while True:
    check, img = video.read()
    # escala de grises
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # más info en https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    # más info en https://theailearner.com/tag/cv2-medianblur/
    # filtro pasa baja
    imgMedian = cv2.medianBlur(imgTH, 5)
    kernel = np.ones((5,5), np.int8)
    # dilatar las áreas o regiones de la imagen
    imgDil = cv2.dilate(imgMedian, kernel)

    for x, y, w, h in estacionamientos:
        espacio = imgDil[y:y+h, x:x+w]
        count = cv2.countNonZero(espacio)
        cv2.putText(img, str(count), (x,y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        if count < 900:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('video', img)
    # cv2.imshow('video TH', imgTH)
    # cv2.imshow('video Median', imgMedian)
    # cv2.imshow('video Dilatada', imgDil)
    cv2.waitKey(10)
