import scyjava

scyjava.config.set_manage_deps(False)
scyjava.config.add_classpath(r"C:\Users\gaezn\Fiji\jars\*")
scyjava.config.add_classpath(r"C:\Users\gaezn\Fiji\plugins\*")

from scyjava import jimport
import numpy as np
import os

IJ = jimport('ij.IJ')
ImageJMain = jimport('ij.ImageJ')
WindowManager = jimport('ij.WindowManager')

# Abrimos la ventana principal de ImageJ
ventana_imagej = ImageJMain()
print("Ventana de ImageJ abierta")

# Abrimos una imagen y la mostramos EN la ventana de ImageJ (editable ahí)
carpeta_proyecto = os.getcwd()
imagen_entrada = os.path.join(
    carpeta_proyecto, "Samples Hologram", "telecentric", "T_Probiotics_20x_632_3.75.bmp"
)
imp = IJ.openImage(imagen_entrada)
imp.show()  # <-- Esto la muestra en la ventana de ImageJ, lista para editar

print("Imagen mostrada en ImageJ. Edítala con las herramientas del menú.")
input("Cuando termines de editar en ImageJ, presiona Enter aquí para recuperarla...")

# Recuperamos la imagen actualmente activa en ImageJ (con las ediciones del usuario)
imagen_activa = WindowManager.getCurrentImage()

if imagen_activa is None:
    print("No hay ninguna imagen activa en ImageJ.")
else:
    procesador = imagen_activa.getProcessor()
    array_java = procesador.getFloatArray()
    array_numpy = np.array(array_java)
    print("Imagen recuperada de ImageJ, forma:", array_numpy.shape)
    print("Esta es la versión editada, lista para usar en tu app")