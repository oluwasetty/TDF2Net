# -*- coding: utf-8 -*-
# import the necessary packages
import numpy as np
import cv2

class Preprocessor:
    @staticmethod
    
    def addNoise(img):
    
        # Getting the dimensions of the image
        row , col = img.shape

        salt_vs_pepper = 0.5
        amount = 0.005
        num_salt = np.ceil(amount * img.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))
        
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        img[coords[0], coords[1]] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        img[coords[0], coords[1]] = 0
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
