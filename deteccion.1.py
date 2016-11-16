import cv2
import numpy as np
import math


x, y, h, w = (493, 305, 125, 90)
cX, cY = (0,0)
kernel = np.ones((5,5),np.float32)/15     
cap = cv2.VideoCapture(0)                                #se realiza la captura del video
ret, frame1 = cap.read()  
ret, frame = cap.read()                                #leemos la captura y lo pasamos al frame1
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)        #aplicamos una mascara blanco y negro para facilidad en la comparacion y procesamiento
frame1 = cv2.GaussianBlur(frame1, (5,5), 0)              # aplicamos blur y eliminamos ruido

# frame1 queda como nuestro valor de comparacion contra los siguientes frames de la camara

while True:
        
    ret, frame = cap.read()                                #proximo frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        #pasamos a gris
    frame = cv2.GaussianBlur(frame, (5,5), 0)              #aplicamos blue
    framef = cv2.absdiff(frame, frame1)             ############### realizamos la DIFERENCIA pixel a pixel
    byn = framef
    retf, framef = cv2.threshold(framef,40,255,cv2.THRESH_BINARY) #aplicamos un treshold para binarizar el resultado
    framef = cv2.dilate(framef, None , iterations=2)              #dilatamos para eliminar error
    
    #framef = cv2.morphologyEx(framef, cv2.MORPH_CLOSE, kernel)
    
    _, contours, _ = cv2.findContours(framef, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    #buscamos contornos
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue   
        x,y,w,h = rect
        M = cv2.moments(c)      
        Xant = cX
        Yant = cY                            # buscamos el centro de masa
        cX = int(M["m10"] / M["m00"])              
        cY = int((M["m01"] / M["m00"]))
        px= int(cX+(cX-Xant)*0.5)
        py= int(cY+(cY-Yant)*0.5)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),4)    # dibujamos un rectangulo al rededor del contorno
        cv2.circle(frame, (cX, cY), 7, (255, 255, 255),2)
        #if np.square([(np.power(cX-Xant),2)+np.power((cY-Yant),2), 2]) >10:
        px= cX
        py= cY
        cv2.line(frame,(cX,cY),(px,py),(255,255,255),5)    
        cv2.circle(frame, (px, py), 7, (255, 255, 255),4)  # colocamos un circulo en el centro de masa
    
    
    # de aqui en adelante es solo codigo para acondicionar las imagenes y mostrarlas en pantalla todas juntas
    
    bothframes = np.hstack((frame,frame1)) 
    bothframes = cv2.resize(bothframes, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('1',bothframes)
    byn = cv2.resize(byn, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('2',byn)
    
    k = cv2.waitKey(30) & 0xff                  #con el boton "Q" salimos del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# eliminamos las capturas al finalizar
cap.release()
cv2.destroyAllWindows()
