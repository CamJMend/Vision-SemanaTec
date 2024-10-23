import cv2
import imutils
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

image = cv2.imread('cars/Placa.jpg')
image = imutils.resize(image, width=1000)

def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output_row = image_row - kernel_row + 1
    output_col = image_col - kernel_col + 1

    output = np.zeros((output_row, output_col))

    for row in range(output_row):
        for col in range(output_col):
            output[row, col] = np.sum(image[row:row + kernel_row, col:col + kernel_col] * kernel)

    return output

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Usa la convolución con un kernel de Sobel en lugar de Canny
sobel_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])

edged = convolution(gray, sobel_kernel)

# Continua con la detección de contornos como lo tienes
cnts, _ = cv2.findContours(edged.astype("uint8").copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
lic_num = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        lic_num = approx
        break

if lic_num is not None:
    cv2.drawContours(image, [lic_num], -1, (0, 255, 0), 3)
    cv2.imshow('contornos', image)
    cv2.waitKey(0)

    x, y, w, h = cv2.boundingRect(lic_num)
    nplate = gray[y:y + h, x:x + w]

    if nplate.size > 0:
        cv2.imwrite('Cropped Image.png', nplate)
        nplate = imutils.resize(nplate, width=300)

        print(pytesseract.image_to_string(nplate, config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
        cv2.imshow('Region de Interes', nplate)
        cv2.waitKey(0)
    else:
        print("No se pudo recortar la imagen de la placa.")
else:
    print("No se detectó ningún contorno de placa.")
