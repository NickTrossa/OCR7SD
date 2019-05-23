# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:16:30 2019

@author: Lenovo
"""

#import matplotlib.pyplot as plt
#import numpy as np
import cv2
#import argparse

#parser = argparse.ArgumentParser(description='Code')
#parser.add_argument('--input1', help='Path to the first input image.', default='1original.png')
#args = parser.parse_args()   args.input1



class BrightContr:
    """
    Esto binariza una imagen
    """
    def __init__(self, image):
        """
        Acá esta la clave: on_trackbar es un método de la clase y por lo
        tanto tiene acceso a sus atributos (size y offset) sin necesidad
        de convertirlos en variables globales.
        """
        import numpy as np
        self.imagen = image
        self.alpha = 1
        self.beta = 0
        self.title_window = 'Ventana Brightness and Contrast'
        cv2.namedWindow(self.title_window, cv2.WINDOW_KEEPRATIO)
        
        cv2.createTrackbar('Alpha', self.title_window, 50, 100, self.on_trackbar_alpha)
        cv2.createTrackbar('Beta', self.title_window, 50, 100, self.on_trackbar_beta)
        self.update()

    def on_trackbar_alpha(self, val):
        """
        Trackbar for size
        """
        import numpy as np
        self.alpha = np.interp(val, [0,50,100], [0,1,10]) # Map to valid interval
        self.update()

    def on_trackbar_beta(self, val):
        """
        Trackbar for offset
        """
        import numpy as np
        self.beta = np.interp(val, [0,100], [-255,255])  # Map to valid interval
        self.update()

    def update(self):
        """
        Esto updetea
        """
        import numpy as np
#        print("Alpha\t %i \t Beta \t %i \n"%(self.alpha, self.beta))
#        new_image = np.zeros(self.imagen.shape, self.imagen.dtype)
        
        transformada = np.clip(self.alpha*self.imagen.astype('float32') + self.beta, 0, 255)

        cv2.imshow(self.title_window, transformada.astype('uint8'))

#foto = cv2.imread('./img/1original.png', cv2.IMREAD_GRAYSCALE)
#bc = BrightContr(foto)

# Show some stuff
#b.update()
# Wait until user press some key
#cv2.waitKey()
#cv2.destroyAllWindows()
