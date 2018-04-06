from keras.models import load_model
model = load_model('cnn.h5')

import cv2
import numpy as np

def image_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    ret2,th2 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    print(th2.shape[1]*th2.shape[0]/2)
    print(cv2.countNonZero(th2))
    if (cv2.countNonZero(th2)) >  (th2.shape[1]*th2.shape[0]/2):
        ret2,th2 = cv2.threshold(th2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    resized = cv2.resize(th2, (28,28))
    final_image = cv2.GaussianBlur(resized, (3,3), 0)

    cv2.imshow('gray', gray)
    cv2.imshow('blur', blurred)
    
    cv2.imshow('resized', resized)
    cv2.imshow('threshold', th2)
    cv2.imshow('Final Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    reshaped_image = final_image.reshape((1,28,28,1))
    return (final_image, reshaped_image)
    

image = cv2.imread('') #Enter directory here
(final_image, reshaped_image) = image_preprocessing(image)
model.predict_classes(reshaped_image/255)
