import cv2
import imutils
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Ruta de tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carga la imagen a traves de cv2 y la convierte a matriz numerica
image = cv2.imread('cars/placa2.jpg')

# Cambia el tamaño de la matriz a 1000 de ancho, manteniendo proporcion
image = imutils.resize(image, width=1000)

# Implementacion de la convolucion de una matriz con un kernel
def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output_row = image_row - kernel_row + 1
    output_col = image_col - kernel_col + 1

    output = np.zeros((output_row, output_col))

    # Aplica la convolución recorriendo la imagen
    for row in range(output_row):
        for col in range(output_col):
            output[row, col] = np.sum(image[row:row + kernel_row, col:col + kernel_col] * kernel)

    return output

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplica la convolución con el kernel Sobel en X y Y
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

sobel_y_kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

# Aplica convolución en X y en Y para detección de bordes horizontales y verticales
edged_x = convolution(gray, sobel_x_kernel)
edged_y = convolution(gray, sobel_y_kernel)

# Combina los bordes detectados en X y Y utilizando la magnitud del gradiente
edged_combined = np.sqrt(edged_x**2 + edged_y**2)
edged_combined = np.uint8(edged_combined)

# Detección de contornos
cnts, _ = cv2.findContours(edged_combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Ordena los contornos por área en orden descendente y selecciona los 30 más grandes
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
lic_num = None

# Busca el contorno que tenga exactamente 4 vértices, que corresponde a la forma de una placa
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        lic_num = approx
        break

# Si se detecta un contorno con 4 vértices, procede a extraer la placa
if lic_num is not None:
    # Dibuja el contorno detectado en la imagen original
    cv2.drawContours(image, [lic_num], -1, (0, 255, 0), 3)

    # Calcula el área delimitada por el contorno y la recorta
    x, y, w, h = cv2.boundingRect(lic_num)
    nplate = gray[y:y + h, x:x + w]
    
    # Si se pudo recortar la región de interés (la placa), se guarda y procesa
    if nplate.size > 0:
        # Guarda la imagen recortada de la placa
        cv2.imwrite('Cropped Image.png', nplate)
        nplate = imutils.resize(nplate, width=300)
        
        # Utiliza Tesseract para extraer el texto de la placa
        print(pytesseract.image_to_string(nplate, config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

        # Muestra la imagen recortada de la placa
        cv2.imshow('Region de Interes', nplate)
        cv2.waitKey(0)
    else:
        print("No se pudo recortar la imagen de la placa.")
else:
    print("No se detectó ningún contorno de placa.")
