"""Bridge module to communicate with ImageJ using local jars (no Maven required)."""
import io
import numpy as np
from PIL import Image

_ij_initialized = False
IJ = None
WindowManager = None
ImagePlus = None

FIJI_JARS_PATH = r"C:\Users\gaezn\Fiji\jars\*"
FIJI_PLUGINS_PATH = r"C:\Users\gaezn\Fiji\plugins\*"

_imagenes_activas = {}
_ultimo_hash = {}


def _ensure_imagej_started():
    """Starts the JVM and loads ImageJ classes only once per process."""
    global _ij_initialized, IJ, WindowManager, ImagePlus

    if _ij_initialized:
        return

    import scyjava
    scyjava.config.set_manage_deps(False)
    scyjava.config.add_classpath(FIJI_JARS_PATH)
    scyjava.config.add_classpath(FIJI_PLUGINS_PATH)

    from scyjava import jimport
    IJ = jimport('ij.IJ')
    WindowManager = jimport('ij.WindowManager')
    ImagePlus = jimport('ij.ImagePlus')

    _ij_initialized = True


def show_imagej_gui():
    """Opens the main ImageJ window (only once)."""
    _ensure_imagej_started()
    from scyjava import jimport
    ImageJMain = jimport('ij.ImageJ')
    if WindowManager.getCurrentImage() is None and IJ.getInstance() is None:
        ImageJMain()


def pil_to_imageplus(pil_img: Image.Image, title="FromApp"):
    """Converts a PIL image to an ImageJ ImagePlus and shows it in ImageJ."""
    _ensure_imagej_started()
    import jpype
    from scyjava import jimport

    array = np.array(pil_img.convert("L"), dtype=np.uint8)
    height, width = array.shape

    ByteProcessor = jimport('ij.process.ByteProcessor')
    processor = ByteProcessor(width, height)

    flat = array.flatten().astype(np.int8)
    java_byte_array = jpype.JArray(jpype.JByte)(flat.tolist())
    processor.setPixels(java_byte_array)

    imp = ImagePlus(title, processor)
    imp.show()

    _imagenes_activas[title] = imp
    _ultimo_hash[title] = None
    return imp


def get_current_imagej_image_as_pil():
    """Retrieves the currently active ImageJ image and converts it to PIL."""
    _ensure_imagej_started()

    imagen_activa = WindowManager.getCurrentImage()
    if imagen_activa is None:
        return None, None

    pil_img = _imageplus_a_pil(imagen_activa)
    titulo = str(imagen_activa.getTitle())
    return pil_img, titulo


def _imageplus_a_pil(imp):
    """Converts any ImagePlus (grayscale, LUT, RGB, overlays, scale bars, etc.) to PIL."""
    from scyjava import jimport
    ImageIO = jimport('javax.imageio.ImageIO')
    ByteArrayOutputStream = jimport('java.io.ByteArrayOutputStream')

    imagen_aplanada = imp.flatten()
    buffered_image = imagen_aplanada.getBufferedImage()

    baos = ByteArrayOutputStream()
    ImageIO.write(buffered_image, "png", baos)
    png_bytes = bytes(baos.toByteArray())

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def check_for_imagej_updates():
    """Checks all tracked ImageJ images for ANY change (pixels, color, LUT, overlays, etc.)."""
    _ensure_imagej_started()

    cambios = []
    titulos_a_quitar = []

    for titulo, imp in _imagenes_activas.items():
        if WindowManager.getImage(imp.getID()) is None:
            titulos_a_quitar.append(titulo)
            continue

        try:
            pil_img = _imageplus_a_pil(imp)
            hash_actual = hash(pil_img.tobytes())

            if _ultimo_hash.get(titulo) != hash_actual:
                _ultimo_hash[titulo] = hash_actual
                cambios.append((titulo, pil_img))
        except Exception as e:
            print(f"Error revisando '{titulo}': {e}")
            continue

    for titulo in titulos_a_quitar:
        del _imagenes_activas[titulo]
        _ultimo_hash.pop(titulo, None)

    return cambios