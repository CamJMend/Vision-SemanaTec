import cv2
import imutils
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

image = cv2.imread('cars/car_image2.jpg')
image = imutils.resize(image, width=800)

def adjust_gamma(image, gamma):
  invGamma = 1.0 / gamma
  table = [((i / 255.0) ** invGamma) * 255 for i in range(256)]
  table = np.array(table).astype("uint8")
  return cv2.LUT(image, table)

image = adjust_gamma(image, gamma=2.4)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 20, 20)
edged = cv2.Canny(gray, 100, 200)
#cv2.imshow('Canny', edged)
#cv2.moveWindow('Image', 45, 300)
#cv2.waitKey(0)

cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
  cv2.imshow('contornos',image)
  cv2.moveWindow('Image',45,10)
  cv2.waitKey(0)  

  x, y, w, h = cv2.boundingRect(lic_num)
  nplate = gray[y:y + h, x:x + w]

  if nplate.size > 0:
    cv2.imwrite('Cropped Image.png', nplate)
    nplate = imutils.resize(nplate, width=300)

    print(pytesseract.image_to_string(nplate, config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    cv2.imshow('Region de Interes', nplate)
    cv2.waitKey(0)
  else:
    print("No se pudo recortar la imagen de la placa.")
else:
    print("No se detectó ningún contorno de placa.")


#cv2.imshow("Final Image With Number Plate Detected", image)
#cv2.waitKey(0)