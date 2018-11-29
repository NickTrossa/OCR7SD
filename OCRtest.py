# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:31:38 2018

@author: nicot

OCR - Main
 _
|_|
|_|

--- Cosas para mejorar ---
* Thresholding para binarization
https://imagej.nih.gov/ij/docs/guide/146-29.html
https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
IsoData paper: http://imagej.net/Auto_Threshold#IsoData
* Agregar método para corregir metodo segmentos en caso de no encontrar coincidencias.
* Cuantificar metodo segmentos (En función de distancia mínima al umbral, por ejemplo)
*agregar segunda opcion (cambiando el digito mas cercano al umbral)

--- Recursos ---
* Tutorial Matplotlib Images
https://matplotlib.org/users/image_tutorial.html
* Tutorial OpenCV
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html#display-image
* Transformaciones geometricas
https://docs.opencv.org/3.4.3/da/d6e/tutorial_py_geometric_transformations.html
* Video capture
https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
* Interacción con imagen
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html#mouse-handling

"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from OCRauxiliar import convGris, mat2img, elegirCoord, setupROI, binarizar,\
     suavizarImagen, CargarBaseReescalar, comparar, posibilidadesPorcentaje, \
     fragmentDigitos, metodoSegmentos, mostrarWebcam

#%% MAIN BODY
"""
plt.close('all')
plt.ion()
print("--- JediCapture 1.0 -- Seven Segment Optical Character Recognition ---")
cap = cv2.VideoCapture(0) # Objeto Video Capture
print("[*] Objeto de cámara creado")
# - - - - - - - - - - - - - - - - - - - - - - - -
input("--> Montar display apagado ... <enter> ...")
mostrarWebcam(cap)
ret, fondo = cap.read()
imgApagado = convGris(fondo) #imagen float 32 #mpimg.imread(fondo)
print("Fondo capturado.")
# - - - - - - - - - - - - - - - - - - - - - - - -
input("--> Encender display... <enter> ...")
mostrarWebcam(cap)
# Primeros pasos: eljo ROI y numero de digitos y cargo base
ret, imagen = cap.read()
imgPrendido = convGris(imagen) #imagen float 32 #mpimg.imread(imagen)
print("Foto con dígitos capturada.")
"""
#%% - - - - - - - - - - - - - - - - - - - - - - - -
imgApagado = cv2.imread("fotoB.png",cv2.IMREAD_GRAYSCALE)
imgPrendido = cv2.imread("fotoA.png",cv2.IMREAD_GRAYSCALE)

fotoDif = mat2img(np.abs(imgApagado.astype("float32") - imgPrendido.astype("float32"))) #imagen uint8
# Recortar región de interés
#c_t = elegirCoord(fotoDif)
#N = int(input("Inserte número de dígitos..."))
N = 3
# Recortar y segmentar
#digitos = setupROI(fotoDif, N, c_t)
# Cargar base
#num_base0, num_base = CargarBaseReescalar("numeros_base.png", digitos, mostrar=True)

#%% Loop: capturo, resto, agarro ROI, determino resultado

loop = True
n = 0
#t0 = time.time()
#minutos = 3
while loop:
#    n += 1
#    ret, imagen = cap.read()
#    ret, imagen = cap.read()
    imagen = imgPrendido
    print("Foto capturada")
    # Cargar imagenes
#    imgPrendido = convGris(imagen) # Out: imagen float 32
    # Restarlas para marcar diferencias
    fotoDif = mat2img(np.abs(imgApagado.astype("float32") - imgPrendido.astype("float32"))) # Out: imagen uint8    
    # Segmentación de dígiqtos    
    digitos = setupROI(fotoDif, N, c_t)
    # Para mostrar dígitos y para binarización
    digitos_bin = binarizar(digitos, mostrar=True)
    # Para suavizar la imagen binarizada
    digitos_bin_suav = suavizarImagen(digitos_bin, pix=4, mostrar=True)
    print("---- Método segmentos ----")
    metodoSegmentos(digitos_bin_suav)
    print("---- Método comparar con base ----")
    res_posibles, confianzas = comparar(digitos_bin_suav, num_base, mostrar=True)
#    if time.time()-t0 > minutos*60:
#        loop = False
#    if input("<Enter> para seguir o 'q' para finalizar >> ") == "q":
    loop = False
    cv2.destroyAllWindows()
    plt.close('all')
#cap.release()
cv2.destroyAllWindows()
plt.close('all')