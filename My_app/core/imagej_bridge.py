import io
import json
import os
import numpy as np
from PIL import Image
import scyjava
from scyjava import jimport
import jpype

_ij_initialized = False
_ij_frame = None
IJ = None
WindowManager = None
ImagePlus = None

_active_images = {}
_last_hash = {}

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".my_app_fiji_config.json")

COMMON_LOCATIONS = [
    r"C:\Fiji.app",
    r"C:\Program Files\Fiji.app",
    r"C:\Fiji",
    os.path.join(os.path.expanduser("~"), "Fiji"),
    os.path.join(os.path.expanduser("~"), "Fiji.app"),
    os.path.join(os.path.expanduser("~"), "Desktop", "Fiji.app"),
    os.path.join(os.path.expanduser("~"), "Downloads", "Fiji.app"),
]


def _is_valid_fiji_folder(path):
    """Checks if a folder looks like a valid Fiji/ImageJ installation."""
    return os.path.isdir(os.path.join(path, "jars")) and os.path.isdir(os.path.join(path, "plugins"))


def _load_saved_path():
    """Loads a previously saved Fiji path from the config file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                path = data.get("fiji_path")
                if path and _is_valid_fiji_folder(path):
                    return path
        except Exception:
            pass
    return None

def is_imagej_started():
    """Returns True only if ImageJ has already been successfully initialized."""
    return _ij_initialized

def _save_path(path):
    """Saves the Fiji path for future runs."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"fiji_path": path}, f)
    except Exception:
        pass


def _autodetect_fiji():
    """Tries common installation locations."""
    for path in COMMON_LOCATIONS:
        if _is_valid_fiji_folder(path):
            return path
    return None


def get_fiji_path(parent_window=None):
    """Gets the Fiji installation path: saved config -> autodetect -> ask user."""
    path = _load_saved_path()
    if path:
        return path

    path = _autodetect_fiji()
    if path:
        _save_path(path)
        return path

    # Not found automatically: ask the user
    import wx
    with wx.DirDialog(
        parent_window,
        "Select your Fiji installation folder (the one containing 'jars' and 'plugins')",
        style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
    ) as dlg:
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            if _is_valid_fiji_folder(path):
                _save_path(path)
                return path
            else:
                wx.MessageBox(
                    "The selected folder doesn't look like a valid Fiji installation "
                    "(missing 'jars' or 'plugins' subfolders).",
                    "Invalid Folder",
                    wx.ICON_ERROR
                )
    return None


def _ensure_imagej_started(parent_window=None):
    """Starts the JVM and loads ImageJ classes only once per process."""
    global _ij_initialized, IJ, WindowManager, ImagePlus

    if _ij_initialized:
        return

    fiji_path = get_fiji_path(parent_window)
    if fiji_path is None:
        raise RuntimeError("Fiji installation path was not provided.")


    scyjava.config.set_manage_deps(False)
    scyjava.config.add_classpath(os.path.join(fiji_path, "jars", "*"))
    scyjava.config.add_classpath(os.path.join(fiji_path, "plugins", "*"))


    IJ = jimport('ij.IJ')
    WindowManager = jimport('ij.WindowManager')
    ImagePlus = jimport('ij.ImagePlus')

    _ij_initialized = True


def _close_all_imagej_images():
    for image_id in list(WindowManager.getIDList() or []):
        imp = WindowManager.getImage(image_id)
        if imp is not None:
            imp.changes = False  # prevent a blocking "save changes?" dialog
            imp.close()
    _active_images.clear()
    _last_hash.clear()


def show_imagej_gui(parent_window=None):
    global _ij_frame
    _ensure_imagej_started(parent_window)



    if _ij_frame is not None:
        if not _ij_frame.isVisible():
            _ij_frame.setVisible(True)
        return

    ImageJMain = jimport('ij.ImageJ')
    _ij_frame = ImageJMain()

    for listener in list(_ij_frame.getWindowListeners()):
        _ij_frame.removeWindowListener(listener)

    @jpype.JImplements('java.awt.event.WindowListener')
    class SafeCloseListener:
        @jpype.JOverride
        def windowOpened(self, event):
            pass

        @jpype.JOverride
        def windowClosing(self, event):
            _close_all_imagej_images()
            _ij_frame.setVisible(False)

        @jpype.JOverride
        def windowClosed(self, event):
            pass

        @jpype.JOverride
        def windowIconified(self, event):
            pass

        @jpype.JOverride
        def windowDeiconified(self, event):
            pass

        @jpype.JOverride
        def windowActivated(self, event):
            pass

        @jpype.JOverride
        def windowDeactivated(self, event):
            pass

    _ij_frame.addWindowListener(SafeCloseListener())


def pil_to_imageplus(pil_img: Image.Image, title="FromApp", parent_window=None):

    _ensure_imagej_started(parent_window)


    array = np.array(pil_img.convert("L"), dtype=np.uint8)
    height, width = array.shape

    flat = array.flatten().astype(np.int8)
    java_byte_array = jpype.JArray(jpype.JByte)(flat.tolist())

    existing = _active_images.get(title)
    if existing is not None and WindowManager.getImage(existing.getID()) is not None:
        existing.getProcessor().setPixels(java_byte_array)
        existing.updateAndDraw()
        existing.getWindow().toFront()
        return existing

    ByteProcessor = jimport('ij.process.ByteProcessor')
    processor = ByteProcessor(width, height)
    processor.setPixels(java_byte_array)

    imp = ImagePlus(title, processor)
    imp.show()

    _active_images[title] = imp
    _last_hash[title] = None
    return imp


def get_current_imagej_image_as_pil():
    """Retrieves the currently active ImageJ image and converts it to PIL."""
    _ensure_imagej_started()

    active_image = WindowManager.getCurrentImage()
    if active_image is None:
        return None, None

    pil_img = _imageplus_to_pil(active_image)
    title = str(active_image.getTitle())
    return pil_img, title


def _imageplus_to_pil(imp):
    """Converts any ImagePlus (grayscale, LUT, RGB, overlays, scale bars, etc.) to PIL."""

    ImageIO = jimport('javax.imageio.ImageIO')
    ByteArrayOutputStream = jimport('java.io.ByteArrayOutputStream')

    flattened_image = imp.flatten()
    buffered_image = flattened_image.getBufferedImage()

    baos = ByteArrayOutputStream()
    ImageIO.write(buffered_image, "png", baos)
    png_bytes = bytes(baos.toByteArray())

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def check_for_imagej_updates():
    """Checks all tracked ImageJ images for ANY change (pixels, color, LUT, overlays, etc.)."""
    if not _ij_initialized:
        return []

    changes = []
    titles_to_remove = []

    for title, imp in _active_images.items():
        if WindowManager.getImage(imp.getID()) is None:
            titles_to_remove.append(title)
            continue

        try:
            pil_img = _imageplus_to_pil(imp)
            current_hash = hash(pil_img.tobytes())

            if _last_hash.get(title) != current_hash:
                _last_hash[title] = current_hash
                changes.append((title, pil_img))
        except Exception as e:
            print(f"Error checking '{title}': {e}")
            continue

    for title in titles_to_remove:
        del _active_images[title]
        _last_hash.pop(title, None)

    return changes