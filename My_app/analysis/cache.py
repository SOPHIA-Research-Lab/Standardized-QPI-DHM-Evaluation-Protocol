"""
Intermediate results reused across metrics and images.

These four stores used to be four loose attributes on MainFrame, cleared and
purged by hand from three different places. Their lifecycles are identical --
everything keyed to an image dies with that image, and everything dies on
Clear -- so keeping them together is what makes that correct by construction
instead of by remembering.

None of this touches wx: it is storage and eviction only. Asking the user for
a threshold or for zones stays in the UI layer; what comes back is handed here.
"""


class AnalysisCache:
    """Per-image masks, zone selections and unwrapped phase."""

    #: Most recently unwrapped images kept in memory. An unwrapped copy is as
    #: large as the phase itself (134 MB for 4096x4096), so one per open image
    #: would cost gigabytes. Batches run every metric of an image consecutively,
    #: so a couple of entries preserve the whole speed-up.
    UNWRAPPED_LIMIT = 2

    def __init__(self):
        self.masks = {}          # image name -> {fingerprint, mask, threshold}
        self.unwrapped = {}      # image name -> {fingerprint, unwrapped}
        self.zones = {}          # cache key   -> (zones, num_zones)
        self.shared_zones = None  # (zones, num_zones) reused across images
        #: (image name, unwrapped) -> (fingerprint, threshold) for TSM. The
        #: wrapped and unwrapped phase of one image are different pictures,
        #: so they get their own threshold.
        self.tsm_thresholds = {}

    # ------------------------------------------------------------------ zonas
    def zone_key(self, image_name, metric_id=None):
        """Key for a zone selection: per image, or per image and metric."""
        return "%s_%s" % (image_name, metric_id if metric_id else "zones")

    def reset_zones(self, shared=False):
        """Start a fresh batch. `shared` also drops the zones reused across images."""
        self.zones.clear()
        if shared:
            self.shared_zones = None

    # -------------------------------------------------------------- desenvuelto
    def get_unwrapped(self, image_name, fingerprint):
        """Cached unwrapped phase, or None if absent or stale."""
        entrada = self.unwrapped.get(image_name)
        if entrada is not None and entrada['fingerprint'] == fingerprint:
            return entrada['unwrapped']
        return None

    def store_unwrapped(self, image_name, fingerprint, array):
        self.unwrapped[image_name] = {'fingerprint': fingerprint, 'unwrapped': array}
        while len(self.unwrapped) > self.UNWRAPPED_LIMIT:
            self.unwrapped.pop(next(iter(self.unwrapped)))
        return array

    # ------------------------------------------------------------------ ciclo
    def forget_image(self, image_name):
        """Drop everything tied to one image, when its tab is closed."""
        self.masks.pop(image_name, None)
        self.unwrapped.pop(image_name, None)
        for k in [k for k in self.tsm_thresholds if k[0] == image_name]:
            del self.tsm_thresholds[k]
        for k in [k for k in self.zones if k.startswith("%s_" % image_name)]:
            del self.zones[k]

    def clear(self):
        self.masks.clear()
        self.unwrapped.clear()
        self.zones.clear()
        self.tsm_thresholds.clear()
        self.shared_zones = None
