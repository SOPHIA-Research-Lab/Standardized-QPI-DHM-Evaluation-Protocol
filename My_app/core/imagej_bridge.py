import io
import json
import os
import sys
import numpy as np
from PIL import Image


def _ensure_java_home():
    """Point JPype at the JVM shipped inside this Python environment.

    conda installs one under <env>/Library/lib/jvm on Windows, but JPype only
    consults JAVA_HOME and the registry. Without this the bridge fails with
    "No JVM shared library file (jvm.dll) found" on a machine that has a
    perfectly good JVM sitting beside the interpreter."""
    if os.environ.get("JAVA_HOME"):
        return
    candidates = (
        os.path.join(sys.prefix, "Library", "lib", "jvm"),   # conda, Windows
        os.path.join(sys.prefix, "lib", "jvm"),              # conda, POSIX
    )
    for candidate in candidates:
        if (os.path.isfile(os.path.join(candidate, "bin", "server", "jvm.dll"))
                or os.path.isfile(os.path.join(candidate, "lib", "server",
                                               "libjvm.so"))):
            os.environ["JAVA_HOME"] = candidate
            return


_ensure_java_home()

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
#: Whether each title was sent as 32-bit phase or as an 8-bit picture.
_sent_as_phase = {}

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
            with open(CONFIG_FILE, "r") as configFile:
                savedData = json.load(configFile)
                fijiPath = savedData.get("fiji_path")
                if fijiPath and _is_valid_fiji_folder(fijiPath):
                    return fijiPath
        except Exception:
            pass
    return None


def _save_path(path):
    """Saves the Fiji path for future runs."""
    try:
        with open(CONFIG_FILE, "w") as configFile:
            json.dump({"fiji_path": path}, configFile)
    except Exception:
        pass


def _autodetect_fiji():
    """Tries common installation locations."""
    for fijiPath in COMMON_LOCATIONS:
        if _is_valid_fiji_folder(fijiPath):
            return fijiPath
    return None


def get_fiji_path(parent_window=None):
    """Gets the Fiji installation path: saved config -> autodetect -> ask user."""
    fijiPath = _load_saved_path()
    if fijiPath:
        return fijiPath

    fijiPath = _autodetect_fiji()
    if fijiPath:
        _save_path(fijiPath)
        return fijiPath

    import wx
    with wx.DirDialog(
        parent_window,
        "Select your Fiji installation folder (the one containing 'jars' and 'plugins')",
        style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
    ) as dialog:
        if dialog.ShowModal() == wx.ID_OK:
            fijiPath = dialog.GetPath()
            if _is_valid_fiji_folder(fijiPath):
                _save_path(fijiPath)
                return fijiPath
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

    fijiPath = get_fiji_path(parent_window)
    if fijiPath is None:
        raise RuntimeError("Fiji installation path was not provided.")


    scyjava.config.set_manage_deps(False)
    scyjava.config.add_classpath(os.path.join(fijiPath, "jars", "*"))
    scyjava.config.add_classpath(os.path.join(fijiPath, "plugins", "*"))


    IJ = jimport('ij.IJ')
    WindowManager = jimport('ij.WindowManager')
    ImagePlus = jimport('ij.ImagePlus')

    _ij_initialized = True


def _close_all_imagej_images():
    for imageId in list(WindowManager.getIDList() or []):
        imagePlus = WindowManager.getImage(imageId)
        if imagePlus is not None:
            imagePlus.changes = False  # prevent a blocking "save changes?" dialog
            imagePlus.close()
    _active_images.clear()
    _last_hash.clear()
    _sent_as_phase.clear()


def _bring_to_front(window):
    """Raise a Java window above the application that launched it.

    toFront() on its own is usually ignored on Windows, which does not let a
    window steal the foreground from the one that has it; flicking
    always-on-top is what actually raises it."""
    if window is None:
        return
    try:
        window.setVisible(True)
        window.setAlwaysOnTop(True)
        window.toFront()
        window.requestFocus()
        window.setAlwaysOnTop(False)
    except Exception:
        # Raising is a courtesy: never let it cost the image itself.
        pass


def show_imagej_gui(parent_window=None):
    global _ij_frame
    _ensure_imagej_started(parent_window)


    if _ij_frame is not None:
        if not _ij_frame.isVisible():
            _ij_frame.setVisible(True)
        _bring_to_front(_ij_frame)
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
    _bring_to_front(_ij_frame)


def _show_or_replace(title, processor):
    """Show `processor` in ImageJ under `title`, reusing that window if open."""
    existingImage = _active_images.get(title)
    if existingImage is not None and WindowManager.getImage(existingImage.getID()) is not None:
        existingImage.setProcessor(processor)
        existingImage.updateAndDraw()
        _bring_to_front(existingImage.getWindow())
        return existingImage

    imagePlus = ImagePlus(title, processor)
    imagePlus.show()
    # A window that has just been created is not raised either: show() leaves
    # it wherever the window manager put it, which is behind this application.
    _bring_to_front(imagePlus.getWindow())

    _active_images[title] = imagePlus
    _last_hash[title] = None
    return imagePlus


def phase_to_imageplus(phase, title="FromApp", parent_window=None):
    """Send a phase map to ImageJ as a 32-bit float image, values in radians.

    ImageJ draws it in gray over its own min..max, so it looks as it always
    did, but the pixel values under the cursor are the phase itself and
    filters and arithmetic run on it in float. Brightness/Contrast and LUTs
    change only how it is drawn, unless Apply is pressed."""
    _ensure_imagej_started(parent_window)

    phaseArray = np.ascontiguousarray(phase, dtype=np.float32)
    height, width = phaseArray.shape

    FloatProcessor = jimport('ij.process.FloatProcessor')
    processor = FloatProcessor(width, height,
                               jpype.JArray(jpype.JFloat)(phaseArray.ravel()))
    processor.resetMinAndMax()

    imagePlus = _show_or_replace(title, processor)
    _sent_as_phase[title] = True
    return imagePlus


def pil_to_imageplus(pil_img: Image.Image, title="FromApp", parent_window=None):
    """Send an 8-bit picture: for a tab that holds no phase (a hologram)."""
    _ensure_imagej_started(parent_window)

    pixelArray = np.array(pil_img.convert("L"), dtype=np.uint8)
    height, width = pixelArray.shape

    flatPixels = pixelArray.flatten().astype(np.int8)
    javaByteArray = jpype.JArray(jpype.JByte)(flatPixels.tolist())

    ByteProcessor = jimport('ij.process.ByteProcessor')
    processor = ByteProcessor(width, height)
    processor.setPixels(javaByteArray)

    imagePlus = _show_or_replace(title, processor)
    _sent_as_phase[title] = False
    return imagePlus


def _imageplus_to_pil(imp):
    """What the image LOOKS like in ImageJ: LUT, contrast and overlays included."""

    ImageIO = jimport('javax.imageio.ImageIO')
    ByteArrayOutputStream = jimport('java.io.ByteArrayOutputStream')

    flattenedImage = imp.flatten()
    bufferedImage = flattenedImage.getBufferedImage()

    outputStream = ByteArrayOutputStream()
    ImageIO.write(bufferedImage, "png", outputStream)
    pngBytes = bytes(outputStream.toByteArray())

    return Image.open(io.BytesIO(pngBytes)).convert("RGB")


def _imageplus_to_phase(imp):
    """The pixel VALUES of a 32-bit image as a float array, or None.

    Only a 32-bit image still holds phase in radians: one converted to 8-bit,
    16-bit or RGB in ImageJ holds display levels, and reading those as phase
    would be wrong without saying so. A stack gives its current slice."""
    if imp.getBitDepth() != 32:
        return None
    processor = imp.getProcessor()
    width, height = processor.getWidth(), processor.getHeight()
    values = np.array(processor.getPixels(), dtype=np.float32)
    return values.reshape(height, width).astype(float)


def _read(title, imagePlus):
    """One reading of an ImageJ image: what it looks like and what it holds."""
    pilImage = _imageplus_to_pil(imagePlus)
    phase = _imageplus_to_phase(imagePlus)
    # Either side changing is news: new data, or only a new look.
    currentHash = hash((pilImage.tobytes(),
                        None if phase is None else phase.tobytes()))
    return {
        'title': title,
        'image': pilImage,
        'phase': phase,
        # Sent as phase but no longer 32-bit: converted inside ImageJ.
        'phase_lost': phase is None and _sent_as_phase.get(title, False),
        'hash': currentHash,
    }


def snapshot(title):
    """Read the ImageJ image sent under `title` now, and count it as seen."""
    imagePlus = _active_images.get(title)
    if imagePlus is None:
        return None
    reading = _read(title, imagePlus)
    _last_hash[title] = reading['hash']
    return reading


def check_for_imagej_updates():
    """Every tracked ImageJ image that changed, in its pixels or in its look.

    Each change is a dict: 'title', 'image' (the picture as drawn in ImageJ),
    'phase' (the 32-bit values in radians, or None) and 'phase_lost' (True
    when it was sent as phase and has since been converted away from 32-bit)."""
    if not _ij_initialized:
        return []

    changes = []
    titlesToRemove = []

    for title, imagePlus in _active_images.items():
        if WindowManager.getImage(imagePlus.getID()) is None:
            titlesToRemove.append(title)
            continue

        try:
            reading = _read(title, imagePlus)
            if _last_hash.get(title) != reading['hash']:
                _last_hash[title] = reading['hash']
                changes.append(reading)
        except Exception as error:
            print(f"Error checking '{title}': {error}")
            continue

    for title in titlesToRemove:
        del _active_images[title]
        _last_hash.pop(title, None)
        _sent_as_phase.pop(title, None)

    return changes
