"""Background segmentation for Module I.

Everything here decides WHICH pixels are background. None of it touches the
values the metrics read: the filters run on a normalized copy used only to
choose the mask, and the metrics are always given the original phase. Keeping
that separation is what lets the segmentation be tuned without the measured
numbers moving.

Only what earned its place is here. Measured against a known mask on a field
with residual curvature, forty objects and speckle:

    otsu, no filter ............ IoU 0.213
    median 2 + otsu ............ IoU 0.371
    triangle, no filter ........ IoU 0.650
    top-hat 30 + otsu .......... IoU 1.000

A Gaussian blur (IoU 0.153) and Sauvola local thresholding (IoU 0.03 at every
k tried) both made the mask worse and are deliberately absent: Sauvola is built
for text on paper, and on a near-uniform background its local deviation tends
to zero, which drops the threshold below the mean and turns noise into sample.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np
from skimage import filters, morphology

from .utilitiesRBPV import separate_touching_samples

#: Thresholding methods offered, in the order the dialog lists them.
THRESHOLD_METHODS = ("otsu", "triangle", "manual")

THRESHOLD_LABELS = {
    "otsu": "Otsu",
    "triangle": "Triangle",
    "manual": "Manual",
}

THRESHOLD_HINTS = {
    "otsu": "Splits the histogram at its widest separation. The usual choice.",
    "triangle": "Better when the background takes up most of the field.",
    "manual": "You set the level yourself with the slider.",
}


@dataclass
class Recipe:
    """How the background mask is obtained. Shared by every image of a batch,
    except for `manual_threshold`, which is the one value tuned per image."""

    # Preprocessing, applied to the normalized copy only. 0 means "off".
    median: int = 0              # radius, px
    tophat: int = 0              # radius, px

    # Thresholding.
    method: str = "otsu"
    manual_threshold: float = 0.5

    # The background is taken to be the LARGER of the two regions, which in
    # QPI-DHM it almost always is. This flips that, for the rare field where
    # the sample covers more than half of it.
    invert: bool = False

    # Mask cleanup.
    min_area: int = 100          # px, objects smaller than this are discarded
    hole_area: int = 50          # px, holes smaller than this are filled
    guard: int = 0               # px, the sample is grown by this before inverting
    border: int = 0              # px, margin excluded from the background

    def scaled(self, step: int) -> "Recipe":
        """The same recipe on an image subsampled by `step`.

        Lengths shrink with the step and areas with its square. Without this
        the preview would not be showing what the full mask does."""
        if step <= 1:
            return self
        return replace(
            self,
            median=self.median // step,
            tophat=self.tophat // step,
            min_area=max(1, self.min_area // (step * step)),
            hole_area=max(1, self.hole_area // (step * step)),
            guard=self.guard // step,
            border=self.border // step,
        )

    def invisible_in_preview(self, step: int) -> list:
        """Filters that round away at this subsampling, so the preview omits
        them. Saying so beats showing a preview that quietly differs."""
        if step <= 1:
            return []
        lost = []
        if self.median and self.median // step < 1:
            lost.append("median")
        if self.tophat and self.tophat // step < 1:
            lost.append("top-hat")
        if self.guard and self.guard // step < 1:
            lost.append("guard band")
        return lost

    def same_as(self, other) -> bool:
        """Whether two recipes are the same recipe.

        `manual_threshold` is left out unless the method is Manual: with an
        automatic method it holds the level that one image happened to get,
        and that differs from image to image by design. Comparing it would
        mark every row after the first as a departure from the batch."""
        if other is None:
            return False
        mine = {name: value for name, value in vars(self).items()
                if name != "manual_threshold"}
        theirs = {name: value for name, value in vars(other).items()
                  if name != "manual_threshold"}
        if mine != theirs:
            return False
        if self.method == "manual":
            return abs(self.manual_threshold - other.manual_threshold) < 1e-9
        return True

    def describe(self, level: float = None) -> str:
        """One line naming every choice, for the results table.

        A segmentation nobody can reproduce is not a measurement, so this
        string travels with the numbers into the exported file. `level` is the
        value the method actually landed on for one image; passing it makes
        the line self-sufficient, since an automatic method would otherwise
        name itself without saying where it ended up."""
        preprocessing = []
        if self.median:
            preprocessing.append("median%d" % self.median)
        if self.tophat:
            preprocessing.append("tophat%d" % self.tophat)

        if self.method == "manual":
            thresholding = "manual %.4f" % self.manual_threshold
        elif level is None:
            thresholding = self.method
        else:
            thresholding = "%s %.4f" % (self.method, level)

        cleanup = ["min%d" % self.min_area, "hole%d" % self.hole_area]
        if self.guard:
            cleanup.append("guard%d" % self.guard)
        if self.border:
            cleanup.append("border%d" % self.border)
        if self.invert:
            cleanup.append("inverted")

        return " | ".join(["+".join(preprocessing) if preprocessing else "no filter",
                           thresholding,
                           " ".join(cleanup)])


def _odd(value) -> int:
    """The nearest odd number at or above `value`; window sizes require it."""
    value = int(value)
    return value if value % 2 == 1 else value + 1


def normalize(image: np.ndarray) -> np.ndarray:
    """The phase mapped onto [0, 1] by its own extremes, for thresholding only."""
    image = np.asarray(image, dtype=np.float32)
    low, high = float(np.min(image)), float(np.max(image))
    if high - low < 1e-12:
        return np.zeros_like(image, dtype=np.float32)
    return (image - low) / (high - low)


def bright_sample(normalized: np.ndarray) -> bool:
    """Whether the sample sits above the background, from the histogram's skew.

    The sample is the minority, so it lives in a tail: a right tail means it is
    brighter than the background, a left tail that it is darker. Reading the
    skew needs no threshold, which is why it survives a field with residual
    curvature -- it was right on 10 of 10 synthetic cases where deciding from
    an Otsu split was wrong on one."""
    data = np.asarray(normalized, dtype=np.float64).ravel()
    centerd = data - data.mean()
    spread = centerd.std()
    if spread < 1e-12:
        return True
    return float((centerd ** 3).mean() / (spread ** 3)) > 0


def preprocess(normalized: np.ndarray, recipe: Recipe,
               sample_is_bright: bool = True) -> np.ndarray:
    """Apply the recipe's filters, in the order that makes them worth having.

    The median first, so the top-hat is not asked to tell speckle from
    structure; the top-hat last, because it is what reshapes the histogram."""
    filtered = np.asarray(normalized, dtype=np.float32)

    if recipe.median >= 1:
        # Isolated phase outliers vanish without the edges moving.
        size = _odd(2 * int(recipe.median) + 1)
        filtered = (cv2.medianBlur(filtered, 3) if size <= 3
                    else _median_large(filtered, size))

    if recipe.tophat >= 1:
        # Removes the slowly varying background. Residual phase curvature is
        # exactly that, and it is what defeats a single global threshold.
        size = _odd(2 * int(recipe.tophat) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        # Which tail the sample lives in decides which top-hat keeps it.
        operation = cv2.MORPH_TOPHAT if sample_is_bright else cv2.MORPH_BLACKHAT
        filtered = cv2.morphologyEx(filtered, operation, kernel)
        # The residual lives on a different scale; renormalizing keeps the
        # threshold slider meaning the same thing from one recipe to the next.
        filtered = normalize(filtered)

    return np.clip(filtered, 0.0, 1.0)


def _median_large(image: np.ndarray, size: int) -> np.ndarray:
    """Median with a window OpenCV will not take on float data."""
    # cv2.medianBlur only accepts ksize > 5 on 8-bit input, and the 8-bit round
    # trip is harmless here: the result only picks the mask, never the values.
    asBytes = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.medianBlur(asBytes, size).astype(np.float32) / 255.0


def threshold(filtered: np.ndarray, recipe: Recipe):
    """Binarize the filtered copy. Returns (above_threshold, level)."""
    if recipe.method == "manual":
        level = float(np.clip(recipe.manual_threshold, 0.0, 1.0))
        return filtered > level, level

    finite = filtered[np.isfinite(filtered)]
    if finite.size == 0 or float(np.ptp(finite)) < 1e-12:
        return np.zeros(filtered.shape, dtype=bool), 0.0

    if recipe.method == "triangle":
        # Suits a histogram one class dominates, which is the usual case here:
        # the background is most of the field.
        level = float(filters.threshold_triangle(finite))
    else:
        level = float(filters.threshold_otsu(finite))

    return filtered > level, level


#: How close to an even split before the majority rule stops being safe.
AMBIGUOUS_MARGIN = 0.10


def build_mask(image: np.ndarray, recipe: Recipe) -> dict:
    """Run the whole recipe. Returns every stage, so a preview can show them.

    Keys: normalized, filtered, sample, background, level, bright_sample,
    ambiguous. `image` is the phase and is never modified."""
    normalized = normalize(image)
    brightSample = bright_sample(normalized)
    filtered = preprocess(normalized, recipe, brightSample)
    above, level = threshold(filtered, recipe)

    # The background is the LARGER region, so the sample is whichever side of
    # the threshold holds fewer pixels. One rule, no question asked, and its
    # single failure mode -- a sample covering more than half the field -- is
    # the one `ambiguous` reports and `invert` fixes.
    aboveFraction = float(np.mean(above))
    sample = above if aboveFraction <= 0.5 else ~above
    if recipe.invert:
        sample = ~sample
    sampleIsForeground = bool(np.mean(sample) <= 0.5)

    try:
        if np.any(sample):
            sample = separate_touching_samples(sample)
    except Exception:
        # Watershed is a refinement: a failure must not cost the whole mask.
        pass

    # remove_small_objects works on the foreground, so a majority sample inverts.
    if sampleIsForeground:
        sample = morphology.remove_small_objects(sample, min_size=max(1, recipe.min_area))
    else:
        sample = ~morphology.remove_small_objects(~sample, min_size=max(1, recipe.min_area))

    sample = morphology.remove_small_holes(sample,
                                           area_threshold=max(1, recipe.hole_area))

    if recipe.guard >= 1:
        # The halo around the sample is where compensation error concentrates.
        # Growing the sample keeps it out of the background instead of letting
        # it inflate STD and PV.
        sample = morphology.binary_dilation(sample, morphology.disk(int(recipe.guard)))

    if recipe.border >= 1:
        # Applied last, so the cleanup cannot undo it. Reconstruction leaves
        # its worst artefacts at the edges of the field -- unwrapping seams,
        # filter ringing, the corners of the spectrum -- and they are not
        # background however flat the middle of the image looks.
        margin = int(recipe.border)
        sample[:margin, :] = True
        sample[-margin:, :] = True
        sample[:, :margin] = True
        sample[:, -margin:] = True

    return {
        "normalized": normalized,
        "filtered": filtered,
        "sample": sample,
        "background": ~sample,
        "level": level,
        "bright_sample": brightSample,
        "ambiguous": abs(aboveFraction - 0.5) < AMBIGUOUS_MARGIN,
    }
