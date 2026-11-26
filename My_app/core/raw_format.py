"""Headerless .raw / .bin phase maps: the bare numbers, nothing else.

Such a file says nothing about its own width, height or data type, so they
travel in its NAME when this application writes one (fase_640x480_float32.raw)
and are asked for, prefilled from that name, when one is opened."""
import math
import os
import re

import numpy as np

#: The types offered when opening, and how each is read. Floating point is
#: phase in radians, used as it stands; 8-bit is gray levels, mapped onto
#: [-pi, pi] like an 8-bit image file.
DTYPES = {
    'float32': np.float32,
    'float64': np.float64,
    'uint8': np.uint8,
}

#: What this application writes, and proposes first when opening: the common
#: layout of NumPy, MATLAB and Windows, and ImageJ's "32-bit Real" import.
DEFAULT_DTYPE = 'float32'
DEFAULT_LITTLE_ENDIAN = True

RAW_EXTENSIONS = ('.raw', '.bin')


def raw_file_name(base_name, phase, extension='.raw'):
    """A file name that carries what the file itself cannot: size and type."""
    height, width = np.shape(phase)
    return "%s_%dx%d_%s%s" % (base_name, width, height, DEFAULT_DTYPE, extension)


def save_raw(path, phase):
    """Write `phase` as float32, little-endian, row by row, no header."""
    np.ascontiguousarray(phase, dtype='<f4').tofile(path)


def guess_layout(path):
    """Width, height, dtype and byte order suggested by a file's name and size.

    Reads WxH and a type word (float32, f64, uint8, ...) from the name, and
    'be'/'big' for big-endian. Without a size in the name, a file holding a
    square number of values is taken to be square. Anything not found falls
    back to the defaults; the user confirms it either way."""
    name = os.path.basename(path).lower()
    fileSize = os.path.getsize(path)

    dtypeName = DEFAULT_DTYPE
    for candidate, pattern in (('float64', r'float64|f64|double'),
                               ('float32', r'float32|f32|single|float'),
                               ('uint8', r'uint8|u8|8bit|8-bit')):
        if re.search(pattern, name):
            dtypeName = candidate
            break

    littleEndian = not re.search(r'(^|[_\-.])(be|big|bigendian)([_\-.]|$)', name)

    width = height = 0
    sizeMatch = re.search(r'(\d+)\s*[x×]\s*(\d+)', name)
    if sizeMatch:
        width, height = int(sizeMatch.group(1)), int(sizeMatch.group(2))
    else:
        count = fileSize // np.dtype(DTYPES[dtypeName]).itemsize
        side = math.isqrt(count)
        if side > 0 and side * side == count:
            width = height = side

    return {'width': width, 'height': height, 'dtype': dtypeName,
            'little_endian': littleEndian, 'offset': 0}


def expected_size(width, height, dtype_name, offset=0):
    """Bytes a file of that layout must hold."""
    return offset + width * height * np.dtype(DTYPES[dtype_name]).itemsize


def load_raw(path, width, height, dtype_name, little_endian=True, offset=0):
    """Read a headerless file into a (height, width) float array.

    Raises ValueError when the file size does not match the layout, rather
    than reading a wrong shape into a plausible-looking but scrambled image."""
    needed = expected_size(width, height, dtype_name, offset)
    actual = os.path.getsize(path)
    if width <= 0 or height <= 0 or actual != needed:
        raise ValueError(
            "the file holds %d bytes, but %d x %d values of %s%s need %d."
            % (actual, width, height, dtype_name,
               "" if offset == 0 else " after a %d-byte header" % offset, needed))

    dtype = np.dtype(DTYPES[dtype_name]).newbyteorder('<' if little_endian else '>')
    values = np.fromfile(path, dtype=dtype, count=width * height, offset=offset)
    return values.reshape(height, width).astype(float)
