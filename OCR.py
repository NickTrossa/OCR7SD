#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:38:21 2018

@author: nicolas

Tiene 4 funciones.
1) configImagen
2) configCamara
3) adquirirImagen
4) adquirirNumero
--> 2 y 3 son para usar con camara web
--> 1 y 4 se pueden usar con imagenes en el disco

Para crear el paquete: https://packaging.python.org/tutorials/distributing-packages/
"""

import numpy as np
import cv2

import OCRauxiliar as ocraux
import brightnessContrast as brco

def configImagen(img):
    """
    Función para elegir ROI de una imagen del disco.
    Devuelve: coordenadas, cantidad de dígitos y los números de la base escalados.
    """
    fotoDif = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # Recortar región de interés
    c_t = ocraux.elegirCoord(fotoDif) # Coordenadas de referencia
    N = int(input("\n --> Inserte número de dígitos >> "))
    
    # Recortar ROI
    imROI = ocraux.cutROI(fotoDif, c_t)
    
    # Ajustar brillo y contraste
    bc = brco.BrightContr(imROI)
    cv2.waitKey()
    cv2.destroyAllWindows()
    alpha, beta = bc.alpha, bc.beta

    #Binarizar
    imROI_bin = ocraux.binarizarUnaImagen(imROI, mostrar=True)
    # Recortar y segmentar
    digitos = ocraux.setupROI(imROI_bin, N, c_t)
#    # Binarización de prueba
#    binarizar(digitos, mostrar=True)
    # Cargar base
    num_base = ocraux.CargarBaseReescalar("./img/numeros_base.png", digitos, mostrar=True)

    return c_t, N, num_base, alpha, beta

def configCamara(cap):
    """
    Función para setear ROI usando cámara web.
    cap es una instancia VideoCapture del módulo cv2 (la cámara)
    """
    print("--- JediCapture 1.0 -- Seven Segment Optical Character Recognition ---")
    print("\n IMPORTANTE: cuando aparezca una figura, cerrarla presionando 'q' para que continúe el programa.")
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Turn off autofocus
    print("[*] Objeto de cámara creado")
    # - - - - - - - - - - - - - - - - - - - - - - - -
    print("\n --> Montar display apagado...")
    ocraux.mostrarWebcam(cap)
    ret, fondo = cap.read()
    imgApagado = ocraux.convGris(fondo) #imagen float 32
    print("Fondo capturado.")
    # - - - - - - - - - - - - - - - - - - - - - - - -
    print("\n --> Encender display y presionar 'q' para adquirir primera imagen...")
    ocraux.mostrarWebcam(cap)
    # Primeros pasos: eljo ROI y numero de digitos y cargo base
    ret, imagen = cap.read()
    imgPrendido = ocraux.convGris(imagen) #imagen float 32
    print("Foto con dígitos capturada.")
    
    #%% - - - - - - - - - - - - - - - - - - - - - - - -
    
    fotoDif = ocraux.mat2img(np.abs(imgApagado - imgPrendido)) #imagen uint8
    # Recortar región de interés
    c_t = ocraux.elegirCoord(fotoDif) # Coordenadas de referencia
    N = int(input("\n --> Inserte número de dígitos >> "))
    # Recortar y segmentar
    digitos = ocraux.setupROI(fotoDif, N, c_t)
    # Binarización de prueba
    ocraux.binarizar(digitos, mostrar=True)
    # Cargar base
    num_base = ocraux.CargarBaseReescalar("./img/numeros_base.png", digitos, mostrar=True)

    return imgApagado, c_t, N, num_base

def adquirirImagen(cap, imgApagado):
    """
    Función para capturar imagen de la cámara web
    """
    # Clear the buffer before reading
    for i in range(5):
        ret, imagen = cap.read()
    if not(ret):
        print("Error de adquisición...")
    imgPrendido = ocraux.convGris(imagen) # Out: imagen float 32
    # Restarlas para marcar diferencias
    fotoDif = ocraux.mat2img(np.abs(imgApagado - imgPrendido)) # Out: imagen uint8
    return fotoDif

def adquirirNumero(fotoDif, c_t, N, num_base, alpha, beta, size, ver=False):
    """
    Función que devuelve los resultados de dígitos posibles de la imagen fotoDif.
    """
    # Tomo el ROI de la foto en base a configImagen: c_t, N y num_base
    imROI = ocraux.cutROI(fotoDif,c_t,mostrar=ver)
    
    # Ajusto brillo y contraste según setup
    imROIbc = np.clip(alpha* imROI.astype('float32') + beta, 0, 255)
    imROIbc = imROIbc.astype('uint8')
    
    # Binarizo el ROI copmleto, con un método adaptativo
    imROI_bin = ocraux.binarizarUnaImagen(imROIbc, size=size, mostrar=ver)
    # Segmentación de dígitos
    digitos_bin = ocraux.setupROI(imROI_bin, N, c_t, mostrar=ver)

    res_posibles, confianzas = ocraux.comparar(digitos_bin, num_base, mostrar=ver)
    return res_posibles, confianzas