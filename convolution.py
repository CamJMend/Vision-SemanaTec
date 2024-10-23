import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

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

# Paso 1: Aplicar un filtro Gaussiano para suavizar la imagen
def gaussian_blur(image):
    gaussian_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16.0
    return convolution(image, gaussian_kernel)

# Paso 2: Calcular los gradientes usando Sobel
def sobel_operator(image):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    grad_x = convolution(image, sobel_x)
    grad_y = convolution(image, sobel_y)

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = np.arctan2(grad_y, grad_x)

    return gradient_magnitude, gradient_direction

# Paso 3: Aplicar supresión no máxima (básica)
def non_max_suppression(gradient_magnitude, gradient_direction):
    output = np.zeros(gradient_magnitude.shape)
    angle = gradient_direction * 180.0 / np.pi  # Convertir a grados
    angle[angle < 0] += 180

    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            try:
                q = 255
                r = 255

                # 0 grados
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                # 45 grados
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                # 90 grados
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                # 135 grados
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    output[i, j] = gradient_magnitude[i, j]
                else:
                    output[i, j] = 0

            except IndexError as e:
                pass

    return output

# Paso 4: Aplicar umbral (thresholding)
def apply_threshold(image, low_threshold, high_threshold):
    strong = 255
    weak = 75
    result = np.zeros_like(image)
    
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    return result

# Función completa de Canny
def custom_canny(image, low_threshold, high_threshold):
    blurred_image = gaussian_blur(image)
    gradient_magnitude, gradient_direction = sobel_operator(blurred_image)
    suppressed_image = non_max_suppression(gradient_magnitude, gradient_direction)
    canny_image = apply_threshold(suppressed_image, low_threshold, high_threshold)
    
    return canny_image

# Cargar la imagen
image = cv2.imread('Placa.jpg', cv2.IMREAD_GRAYSCALE)
image = imutils.resize(image, width=1000)

# Ejecutar el Canny personalizado
canny_output = custom_canny(image, low_threshold=50, high_threshold=150)

# Mostrar el resultado
plt.imshow(canny_output, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()
