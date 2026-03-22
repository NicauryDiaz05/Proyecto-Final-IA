import pytesseract
import cv2
import numpy as np
import constante
import re

pytesseract.pytesseract.tesseract_cmd = constante.TESSERACT_PATH

# Convierte PIL a formato OpenCV para procesarla 
# y aumenta el contraste para mejorar la lectura del OCRNicaury Diaz 23-SISN-2-028 
def preprocesar_imagen(imagen_pil):
    imagen_np = np.array(imagen_pil)
    imagen_gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    imagen_gris = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return imagen_gris

 # Elimina caracteres especiales, espacios dobles, lineas vacias y los espacios al inicial 
 # y finalizar las lineas Nicaury Diaz 23-SISN-2-028
def limpiar_texto(texto):
    texto = re.sub(r'[^\w\s\.,;:¿?¡!áéíóúÁÉÍÓÚüÜñÑ\-\'\"]', '', texto)
    texto = re.sub(r' +', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = '\n'.join(line.strip() for line in texto.splitlines())
    return texto.strip()

# Preprocesa la imagen antes de pasarla al OCR Nicaury Diaz 23-SISN-2-028
def extraer_texto(imagen_pil):
    imagen_procesada = preprocesar_imagen(imagen_pil)
    texto = pytesseract.image_to_string(imagen_procesada, lang='spa+eng')
    texto_limpio = limpiar_texto(texto)
    return texto_limpio

