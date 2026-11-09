"""Intermediate results reused across metrics and images."""


class AnalysisCache:
    """Per-image masks, zone selections, unwrapped phase and TSM thresholds."""

    #: Unwrapped copies kept in memory; each is as large as the phase itself.
    UNWRAPPED_LIMIT = 2

    def __init__(self):
        self.masks = {}           # image name -> {fingerprint, mask, threshold}
        self.unwrapped = {}       # image name -> {fingerprint, unwrapped}
        self.zones = {}           # cache key   -> (zones, num_zones)
        self.shared_zones = None  # (zones, num_zones) reused across images
        #: (image name, unwrapped) -> (fingerprint, threshold) for TSM.
        self.tsm_thresholds = {}
        #: The Module I segmentation recipe, settled on the first image and
        #: reused for the rest so that every row of the table is comparable.
        self.segmentation_recipe = None

    # ------------------------------------------------------------------ zones
    def reset_zones(self, shared=False):
        """Start a fresh batch; `shared` also drops the zones reused across images."""
        self.zones.clear()
        if shared:
            self.shared_zones = None

    # -------------------------------------------------------------- unwrapped
    def get_unwrapped(self, image_name, fingerprint):
        """Cached unwrapped phase, or None if absent or stale."""
        entry = self.unwrapped.get(image_name)
        if entry is not None and entry['fingerprint'] == fingerprint:
            return entry['unwrapped']
        return None

    def store_unwrapped(self, image_name, fingerprint, array):
        """Keep the unwrapped phase, evicting the oldest past UNWRAPPED_LIMIT."""
        self.unwrapped[image_name] = {'fingerprint': fingerprint, 'unwrapped': array}
        while len(self.unwrapped) > self.UNWRAPPED_LIMIT:
            self.unwrapped.pop(next(iter(self.unwrapped)))
        return array

    # ------------------------------------------------------------- life cycle
    def forget_image(self, image_name):
        """Drop everything tied to one image, when its tab is closed."""
        self.masks.pop(image_name, None)
        self.unwrapped.pop(image_name, None)
        for key in [cachedKey for cachedKey in self.tsm_thresholds if cachedKey[0] == image_name]:
            del self.tsm_thresholds[key]
        for key in [cachedKey for cachedKey in self.zones if cachedKey.startswith("%s_" % image_name)]:
            del self.zones[key]

    def clear(self):
        """Drop every cached result."""
        self.segmentation_recipe = None
        self.masks.clear()
        self.unwrapped.clear()
        self.zones.clear()
        self.tsm_thresholds.clear()
        self.shared_zones = None
