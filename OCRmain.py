# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 00:37:40 2018

@author: nicot

OCR - Main

--- Cosas para mejorar ---
* Thresholding para binarization
https://imagej.nih.gov/ij/docs/guide/146-29.html
https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
IsoData paper: http://imagej.net/Auto_Threshold#IsoData
* Agrandar imagen para seleccionar bordes con más precisión
* Quitar manchitas aisladas de la binarización
* Marcar separación entre dígitos para encuadrarlos mejor


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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time

def convGris(img):
    """
    img: tipo 'uint8' o 'float32', de 3 canales.
    """
    red, green, blue = img[:,:,0], img[:,:,1],img[:,:,2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return gray

def mat2img(mat):
    mat = mat.astype("float32")
    return (mat/mat.max()*255).astype("uint8")

def elegirROI(fotoDif):
    def dameCoordenadas(event,x,y,flags,param):
        print("Entró")
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            coord.append(x)
            coord.append(y)
    coord = []
    
    cv2.namedWindow('fotoDif')
    cv2.setMouseCallback('fotoDif',dameCoordenadas)
    print("Seleccione borde inferior izquierdo primero...")
    print("Seleccione borde superior derecho después...")
    i = 0
    while(len(coord) < 4):
        print(i)
        i += 1
        cv2.imshow('fotoDif',fotoDif)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        plt.pause(1)
    cv2.destroyAllWindows()
    
    return coord

def segmentDigitos(recorte, N=None):
    if N == None:
        N = int(input("Inserte el número de dígitos."))
    dx = int(recorte.shape[1]/N) # ancho de digito
    dy = recorte.shape[0] # Alto de dígito
    digitos = np.zeros((dy,dx,N)) # Matriz base
    for i in range(N):
        digito = recorte[:,i*dx:(i+1)*dx]
        digitos[:,:,i] = digito
    return digitos.astype("uint8")

def binarizar(digitos, mostrar=True):
    N = digitos.shape[2]
    digitos_bin = np.zeros(digitos.shape, dtype='uint8')
    for i in range(N):
        dig = digitos[:,:,i]
        digitos_bin[:,:,i] = cv2.threshold(dig,int(np.mean(dig)),255,cv2.THRESH_BINARY_INV)[1] 
        #    digitos_bin.append(cv2.adaptiveThreshold(digitos[:,:,i],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #            cv2.THRESH_BINARY,25,2))
    if mostrar:
        fig_digitos, ax_digitos = plt.subplots(2,N)
        fig_digitos.suptitle("Digitos segmentados y binarización.")
        for i in range(N):
            plt.subplot(2,N,i+1)
            plt.imshow(digitos[:,:,i], cmap='Greys')
            plt.subplot(2,N,N+i+1)
            plt.imshow(digitos_bin[:,:,i], cmap='Greys_r')
        plt.waitforbuttonpress()
    return digitos_bin

def suavizarImagen(digitos_bin, pix=4, mostrar=True):
    """
    img: uint8 o float32
    pix: tamaño de la cuadrícula donde "promediar"
    """
    digitos_bin_suav = digitos_bin # Lo voy a modificar
    for n in range(digitos_bin.shape[2]):
        r_x = pix
        r_y = pix
        img = digitos_bin[:,:,n].astype("float32")
        img_2 = (img[:-r_x,:-r_y]+img[r_x:,:-r_y]+img[:-r_x,r_y:]+img[r_x:,r_y:])/4
        for i in range(img_2.shape[0]):
           for j in range(img_2.shape[1]):
               if img_2[i,j] > (128 + 2):
                   img_2[i,j] = 255
               else:
                   img_2[i,j] = 0
        erosion_size = 1
        erosion_type = 0
        element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
        dil = cv2.dilate(img_2.astype("uint8"),element)
        wwwwwww # COMPLETAR ACA
        dil = cv2.resize(dil, dsize=img.shape[::-1], interpolation = cv2.INTER_CUBIC)
        digitos_bin_suav[:,:,n] = dil
#    if mostrar:
#        cv2.imshow('SuavImagen',np.vstack((img, suav_re)))
            cv2.imshow('SuavImagen',np.vstack((img, suav_re)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return digitos_bin_suav

def CargarBaseReescalar(file_base, digitos_bin, mostrar=True):
    # Cargar base 
    base = mat2img(convGris(mpimg.imread(file_base)))
    dx = int(base.shape[1]/10) # Width0
    dy = base.shape[0] # Height0
    num_base0 = np.zeros((dy,dx,10)) # Matriz base
    for i in range(10):
        num_base0[:,:,i] = base[:,i*dx:(i+1)*dx]
    
    # Re-escalar
#    width_factor = digitos_bin.shape[1]/dx
#    height_factor = digitos_bin.shape[0]/dy
    num_base = np.zeros((digitos_bin.shape[0],digitos_bin.shape[1],10)) # Matriz base para reescalar
    for i in range(10):
        base0_i = mat2img(num_base0[:,:,i])
        num_base[:,:,i] = cv2.resize(base0_i,dsize=digitos_bin.shape[:-1][::-1], interpolation = cv2.INTER_CUBIC)    
    
    if mostrar:
        fig_digitos, ax_digitos = plt.subplots(2,10)
        fig_digitos.suptitle("Base original y escalada.")
        for i in range(10):
            plt.subplot(2,10,i+1)
            plt.imshow(num_base0[:,:,i], cmap='Greys_r')
            plt.xticks([]), plt.yticks([])
            plt.subplot(2,10,10+i+1)
            plt.imshow(num_base[:,:,i], cmap='Greys_r')
            plt.xticks([]), plt.yticks([])
        plt.waitforbuttonpress()
    return num_base0, num_base

#%% Comparo con la base de datos
def comparar(digitos_bin, num_base, mostrar=True):
    resultado = []
    analisis = [] # Acá pongo cuánto coincide con cada dígito
    for n in range(digitos_bin.shape[2]):
        pesos = []
        for num in range(10):
            peso_i = 0
            for i in range(digitos_bin.shape[0]):
                for j in range(digitos_bin.shape[1]):
                    if digitos_bin[i,j,n] == num_base[i,j,num]:
                        peso_i += 1
            pesos.append(peso_i)
        resultado.append(pesos.index(max(pesos)))
        analisis.append(pesos)
    
    # Cuantificacion
    analisis = np.array(analisis)
    # Los numeros posibles orenados de menos probable a más probable
    res_posibles = np.argsort(analisis, axis=1)
    # Los valores de coincidencias correspondientes
    ordenado = np.sort(analisis, axis=1)
    # Armo los pesos con las distancias al siguiente valor y 
    intervalos = (np.max(analisis, axis=1)-np.min(analisis, axis=1)).reshape(analisis.shape[0],1)
    confianza = (ordenado[:,1:]-ordenado[:,:-1])*(1/intervalos)
    
    if mostrar:
        plt.figure(), plt.title("Coincidencias por dígito")
        i = 0
        for pesos in analisis:
            plt.plot(pesos, label=str(i))
            i += 1
        plt.legend()
        plt.pause(0.1)
    return res_posibles, confianza

#res_posibles, confianza = comparar(digitos_bin, num_base)
#
#print(res_posibles[:,-1],confianza[:,-1])
#print(res_posibles[:,-2],confianza[:,-2])

#%% Por fracción ocupada de negro
def posibilidadesPorcentaje(digitos_bin, num_base):
    def porcentajeSegmentos(digitos_bin):
        porcentajes = []
        for i in range(digitos_bin.shape[2]):
            fraccionBlanca = np.count_nonzero(digitos_bin[:,:,i])/digitos_bin[:,:,i].size
            porcentajes.append(100*(1-fraccionBlanca))
        return np.array(porcentajes)
    # Porcentaje de los numeros de la base
    porcentaje_base = porcentajeSegmentos(num_base)
    # Porcentajes de cada uno de los digitos de la foto
    porcentaje_dig = porcentajeSegmentos(digitos_bin)
    
    
    posibles_tot = []
    for i in range(digitos_bin.shape[2]):
        posibles = []
        distancias = np.abs(porcentaje_dig[i]-porcentaje_base)
        for j in range(len(distancias)):
            if distancias[j] < 7:
                posibles.append(j)
        posibles_tot.append(posibles)
    
    return posibles_tot
    
#print(posibilidadesPorcentaje(digitos_bin, num_base))

#%% Sub fragmentacion de digito
def fragmentDigitos(digito):
#    N = int(input("Inserte el número de dígitos."))
    dx = int(digito.shape[1]/3) # Divisiones de ancho
    dy = int(digito.shape[0]/5) # Divisiones de altura
    
    s1 = digito[(1*dy):(2*dy),:dx] # 2do 5to izquierda
    s2 = digito[:dy,:] # 1er 5to arriba
    s3 = digito[(1*dy):(2*dy),(2*dx):(3*dx)] # Segundo quinto derecha
    s4 = digito[(3*dy):(4*dy),(2*dx):(3*dx)] # 4to 5to derecha
    s5 = digito[(4*dy):(5*dy),:] # 5to 5to abajo
    s6 = digito[(3*dy):(4*dy),:dx] # 
    s7 = digito[(2*dy):(3*dy),:] # 3er 5to centro
    
    lista = [s1,s2,s3,s4,s5,s6,s7]
    secuencia = [0,0,0,0,0,0,0]
    porcentajes = []
    for i in range(7):
        fraccionBlanca = np.count_nonzero(lista[i])/lista[i].size
        porcentajes.append(round(fraccionBlanca*100))
        if fraccionBlanca < 0.9:
            secuencia[i] = 1
        
    return secuencia, porcentajes

def metodoSegmentos(digitos):
    # Secuencias de cada dígito
    DIGITS_LOOKUP = [
        [1, 1, 1, 1, 1, 1, 0], # 0
        [0, 0, 1, 1, 0, 0, 0], # 1
        [0, 1, 1, 0, 1, 1, 1], # 2
        [0, 1, 1, 1, 1, 0, 1], # 3
        [1, 0, 1, 1, 0, 0, 1], # 4
        [1, 1, 0, 1, 1, 0, 1], # 5
        [1, 1, 0, 1, 1, 1, 1], # 6
        [0, 1, 1, 1, 0, 0, 0], # 7
        [1, 1, 1, 1, 1, 1, 1], # 8
        [1, 1, 1, 1, 1, 0, 1], # 9
        [0, 0, 0, 0, 0, 0, 1]] # -
    numeros = []
    for i in range(digitos.shape[2]):
        secuencia_i, porcentajes_i = fragmentDigitos(digitos[:,:,i])
        print("Numero base %i dio secuencia %s \n con porcentajes %s"%(i,secuencia_i,porcentajes_i))
        try:
            print("Eso es un:", DIGITS_LOOKUP.index(secuencia_i))
            numeros.append(DIGITS_LOOKUP.index(secuencia_i))
#            print("    Ahí está! \n")
        except ValueError:
            print("No hubo coincidencia \n")
    return numeros

#print(metodoSegmentos(num_base))
#print(metodoSegmentos(digitos_bin))
#%% Cargar imagen de cámara web

#cap = cv2.VideoCapture(0) # Objeto Video Capture
def mostrarWebcam(cap):
    print("Mostrando imagen. Salir con 'q'...")
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Our operations on the frame come here
    #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
    #    cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print("Dejo de mostrar")
# When everything done, release the capture
#cap.release()
#

#%% MAIN BODY
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
# - - - - - - - - - - - - - - - - - - - - - - - -
fotoDif = mat2img(np.abs(imgApagado - imgPrendido)) #imagen uint8
coord = elegirROI(fotoDif) #Necesita uint8

N = int(input("Inserte número de dígitos..."))
recorte = mat2img(fotoDif[coord[3]:coord[1],coord[0]:coord[2]]) # Para equalizar histograma de grises. uint 8    
digitos = segmentDigitos(recorte, N=N) #Necesita uint8

# - - - - - - - - - - - - - - - - - - - - - - - -
# Para mostrar dígitos y para binarización
digitos_bin = binarizar(digitos, mostrar=True)
num_base0, num_base = CargarBaseReescalar("numeros_base.png", digitos_bin, mostrar=True)
input("Enter para continuar y cerrar ventanas activas.")
plt.close('all')
#%%
# Loop: capturo, resto, agarro ROI, determino resultado
loop = True
n = 0
#t0 = time.time()
#minutos = 3
while loop:
#    n += 1
    ret, imagen = cap.read()
    ret, imagen = cap.read()
    print("ret: ", ret)
    print("Foto capturada")
    # Cargar imagenes
    imgPrendido = convGris(imagen) # Out: imagen float 32
    # Restarlas para marcar diferencias
    fotoDif = mat2img(np.abs(imgApagado - imgPrendido)) # Out: imagen uint8    
    # Recortar región de interés
    recorte = mat2img(fotoDif[coord[3]:coord[1],coord[0]:coord[2]]) # Para equalizar histograma de grises. uint 8    
    # Segmentación de dígitos    
    digitos = segmentDigitos(recorte, N=N)
    # Para mostrar dígitos y para binarización
    digitos_bin = binarizar(digitos, mostrar=True)
#    plt.pause(1)
    print("---- Método segmentos ----")
    print(metodoSegmentos(digitos_bin))
    print("---- Método comparar con base ----")
    res_posibles, confianzas = comparar(digitos_bin, num_base, mostrar=True)
    print(res_posibles[:,-1])
    print(confianzas[:,-1])
#    if time.time()-t0 > minutos*60:
#        loop = False
    if input("Enter para seguir o q para finalizar") == "q":
        loop = False
    cv2.destroyAllWindows()
    plt.close('all')
cap.release()
cv2.destroyAllWindows()
plt.close('all')
#%%
   

"""
 _
|_|
|_|
"""

#%%

#cv2.imshow('Num0',digitos_bin[:,:,0])
#cv2.imshow('Base0',num_base[:,:,0].astype("uint8"))
