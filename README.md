# Detección de Matrículas de Vehículos

### Descripción del Proyecto

El proyecto **"Detección de Matrículas de Vehículos"** utiliza técnicas de procesamiento de imágenes y reconocimiento óptico de caracteres (OCR) para detectar y extraer matrículas de vehículos a partir de imágenes. Emplea las bibliotecas OpenCV para el procesamiento de imágenes, NumPy para operaciones matemáticas y Pytesseract para la extracción de texto. El flujo del programa incluye la carga de una imagen, preprocesamiento, detección de bordes, identificación de contornos de la matrícula y, finalmente, la extracción del texto de la matrícula.

### Estructura del Proyecto

La estructura del proyecto debe ser la siguiente:

```
/tu-proyecto
|-- deteccion.py      # Archivo principal que ejecuta la detección de matrículas
|-- /cars             # Carpeta que contiene imágenes de prueba
    |-- placa2.jpg    # Ejemplo de imagen de matrícula para la detección
```

### Instalación de Dependencias

Para ejecutar el código, necesitarás instalar algunas bibliotecas. Puedes hacerlo utilizando `pip`. Abre tu terminal y ejecuta los siguientes comandos:

```bash
pip install opencv-python numpy imutils pytesseract matplotlib
```

### Instalación de Tesseract OCR

1. **Descargar Tesseract**: Visita el siguiente enlace y descarga el instalador para tu sistema operativo: [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).

2. **Instalación**: Sigue las instrucciones para instalar Tesseract en tu sistema.

3. **Configuración de la Ruta de Instalación**:
   - Por defecto, la ruta de instalación suele ser:
     ```
     C:\Program Files\Tesseract-OCR\tesseract.exe
     ```
   - Abre el archivo `deteccion.py` y actualiza la línea que establece la ruta de Tesseract para que coincida con tu instalación:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Actualiza esta línea
     ```

### Ejecución del Código

1. Asegúrate de que tienes una imagen de matrícula de prueba en la carpeta `cars` y que se llama `placa2.jpg`.

2. Ejecuta el script `deteccion.py` con el siguiente comando en tu terminal:

   ```bash
   python deteccion.py
   ```

Si todo está configurado correctamente, deberías ver la imagen recortada de la matrícula y el texto extraído impreso en la consola.
