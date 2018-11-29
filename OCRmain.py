# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 00:37:40 2018

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
plt.close('all')

print("--- JediCapture 1.0 -- Seven Segment Optical Character Recognition ---")
print("\n IMPORTANTE: cuando aparezca una figura, cerrarla presionando 'q' para que continúe el programa.")
cap = cv2.VideoCapture(0) # Objeto Video Capture
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Turn off autofocus
print("[*] Objeto de cámara creado")
# - - - - - - - - - - - - - - - - - - - - - - - -
print("\n --> Montar display apagado...")
mostrarWebcam(cap)
ret, fondo = cap.read()
imgApagado = convGris(fondo) #imagen float 32
print("Fondo capturado.")
# - - - - - - - - - - - - - - - - - - - - - - - -
print("\n --> Encender display y presionar 'q' para adquirir primera imagen...")
mostrarWebcam(cap)
# Primeros pasos: eljo ROI y numero de digitos y cargo base
ret, imagen = cap.read()
imgPrendido = convGris(imagen) #imagen float 32
print("Foto con dígitos capturada.")

#%% - - - - - - - - - - - - - - - - - - - - - - - -

fotoDif = mat2img(np.abs(imgApagado - imgPrendido)) #imagen uint8
# Recortar región de interés
c_t = elegirCoord(fotoDif) # Coordenadas de referencia
N = int(input("\n --> Inserte número de dígitos >> "))
# Recortar y segmentar
digitos = setupROI(fotoDif, N, c_t)
# Cargar base
num_base0, num_base = CargarBaseReescalar("numeros_base.png", digitos, mostrar=True)

#%% Loop: capturo, resto, agarro ROI, determino resultado

loop = True
n = 0
t0 = time.time()
minutos = 1
espera = 0.3 # En segundos
out_file = open("out.txt", mode='w')
out_file.write("Tiempo [s], Metodo 1, Metodo 2, Metodo 1 (2da op),  Metodo 2 (2da op) \n")
while loop:
#    n += 1
    ret, imagen = cap.read()
    ret, imagen = cap.read()
    print("Foto capturada")
    # Cargar imagenes
    imgPrendido = convGris(imagen) # Out: imagen float 32
    # Restarlas para marcar diferencias
    fotoDif = mat2img(np.abs(imgApagado - imgPrendido)) # Out: imagen uint8    
    # Segmentación de dígitos    
    digitos = setupROI(fotoDif, N, c_t, mostrar=False)
    # Para mostrar dígitos y para binarización
    digitos_bin = binarizar(digitos, mostrar=False)
    # Para suavizar la imagen binarizada
    digitos_bin_suav = suavizarImagen(digitos_bin, pix=4, mostrar=False)
    print("---- Método comparar con base ----")
    res_posibles, confianzas = comparar(digitos_bin_suav, num_base, mostrar=False)
    print("---- Método segmentos ----")
    res_posibles_MS, distancias_MS = metodoSegmentos(digitos_bin_suav)
    
    out_file.write("%.2f, \t"%(time.time()-t0))
    out_file.write(str([i for i in res_posibles_MS[:,0]])[1:-1]+", \t")
    out_file.write(str([i for i in res_posibles[:,-1]])[1:-1]+", \t")
    
    out_file.write(str([i for i in res_posibles_MS[:,1]])[1:-1]+", \t")
    out_file.write(str([i for i in res_posibles[:,-2]])[1:-1]+"\n")
    time.sleep(espera)
    if time.time()-t0 > minutos*60:
        loop = False
#    if input("Enter para seguir o q para finalizar") == "q":
#        loop = False
#    cv2.destroyAllWindows()
#    plt.close('all')
cap.release()
cv2.destroyAllWindows()
out_file.close()
plt.close('all')

#%%

# This plots the alternatives only for 2 digits in the display.
plt.figure(1), plt.clf()
datos = np.genfromtxt("out.txt",delimiter=',', skip_header=1)

r_c_1 = 0
r_c_2 = 0
tiempo = datos[:,0]
for i in range(N):
    r_c_1 += (10**((N-1)-i))*datos[:,i+1]
    r_c_2 += (10**((N-1)-i))*datos[:,i+1+N]

plt.plot(tiempo, r_c_1, label="Método 1")
plt.plot(tiempo, r_c_2, label="Método 2")

#    plt.plot(datos[:,0],datos[:,1]*10+datos[:,2], label="Digitos 1")
#plt.plot(datos[:,0],datos[:,3]*10+datos[:,4],label="Digitos 2")
#plt.plot(datos[:,0],datos[:,1]*10+datos[:,4],label="Digitos Unidades cambiadas")
plt.legend()
plt.xlabel("Tiempo [s]")
plt.ylabel("Medición")
plt.show()
