# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:58:41 2019

@author: Lenovo
"""

#import matplotlib.pyplot as plt
#import numpy as np
import cv2
#import argparse

#parser = argparse.ArgumentParser(description='Code')
#parser.add_argument('--input1', help='Path to the first input image.', default='1original.png')
#args = parser.parse_args()   args.input1

class Binarizador:
    """
    Esto binariza una imagen
    """
    def __init__(self,imagen):
        """
        Acá esta la clave: on_trackbar es un método de la clase y por lo
        tanto tiene acceso a sus atributos (size y offset) sin necesidad
        de convertirlos en variables globales.
        """
        self.imagen = imagen
        default_size = imagen.shape[0]%2 - 1 + imagen.shape[0]
        print(default_size)
        self.size = default_size
        self.offset = 0
        self.title_window = 'Ventana'
        cv2.namedWindow(self.title_window, cv2.WINDOW_KEEPRATIO)
        
        cv2.createTrackbar('Size', self.title_window, self.size, default_size, self.on_trackbar_size)
        cv2.createTrackbar('Offset', self.title_window, self.offset, 20, self.on_trackbar_offset)
        self.update()

    def on_trackbar_size(self, val):
        """
        Trackbar for size
        """
        self.size = (1 - val%2) + val + 2 # Para que siempre sea impar y mayor a 2
        self.update()

    def on_trackbar_offset(self, val):
        """
        Trackbar for offset
        """
        self.offset = val
        self.update()

    def update(self):
        """
        Esto updetea
        """
        print("Size\t %i \t Offset \t %i \n"%(self.size, self.offset))
        binarizada = cv2.adaptiveThreshold(self.imagen, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, self.size, self.offset)
        cv2.imshow(self.title_window, binarizada)

#imagen = cv2.imread('1original.png', cv2.IMREAD_GRAYSCALE)
#Binarizador(imagen)

# Show some stuff
#b.update()
# Wait until user press some key
#cv2.waitKey()
#cv2.destroyAllWindows()
