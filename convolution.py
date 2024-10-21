import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("La imagen no se pudo cargar. Verificar ruta")
    
    print("Imagen cargada correctamente. Tamaño:", image.shape)
    
    return image

def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output_row = image_row - kernel_row + 1
    output_col = image_col - kernel_col + 1

    output = np.zeros((output_row, output_col))

    for row in range(output_row):
        for col in range(output_col):
            output[row, col] = np.sum(image[row:row + kernel_row, col:col + kernel_col] * kernel)
    
    plt.imshow(output, cmap='gray')
    plt.title("Output Image".format(output_row, output_col, kernel_row, kernel_col))
    plt.show()
    
    return output

if __name__ == "__main__":
    image_path = "C:/Users/PC/Desktop/Placa.jpg"

    kernel = np.array([[1, 0, -1],
                       [2,0, -2],
                       [1, 0, -1]])

    image = load_image(image_path)
    print("Tamaño de la imagen:", image.shape)

    result = convolution(image, kernel)
    