import sys
import os

sys.path.insert(0, os.getcwd())

from core.imagej_bridge import show_imagej_gui, pil_to_imageplus, get_current_imagej_image_as_pil
from PIL import Image

# Ruta a tu imagen de prueba
carpeta_proyecto = os.getcwd()
imagen_entrada = os.path.join(
    carpeta_proyecto, "Samples Hologram", "telecentric", "T_Probiotics_20x_632_3.75.bmp"
)

print("Abriendo imagen de prueba con PIL...")
pil_img = Image.open(imagen_entrada)
print(f"Imagen cargada: {pil_img.size}")

print("Mostrando ventana de ImageJ...")
show_imagej_gui()

print("Enviando imagen a ImageJ...")
pil_to_imageplus(pil_img, title="Prueba desde mi app")

print("\nAhora ve a la ventana de ImageJ, edita la imagen con cualquier herramienta.")
input("Cuando termines, presiona Enter aquí para recuperarla...")

print("Recuperando imagen editada desde ImageJ...")
pil_resultado, titulo = get_current_imagej_image_as_pil()

if pil_resultado is None:
    print("No se encontró ninguna imagen activa en ImageJ.")
else:
    print(f"Imagen recuperada: '{titulo}', tamaño: {pil_resultado.size}")
    ruta_guardado = os.path.join(carpeta_proyecto, "Samples Hologram", "resultado_desde_imagej.png")
    pil_resultado.save(ruta_guardado)
    print(f"Guardada en: {ruta_guardado}")

print("Prueba completada.")