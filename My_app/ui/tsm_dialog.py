"""Choosing the binarisation threshold for the TSM metric."""
# The definition fixes all but where to binarise, so the user chooses it.
import numpy as np
import wx
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analysis.module2.global_phase import histogram_valley, normalised_phase
from ui import hidpi
from ui.threshold_dialog import numpy_to_wx_bitmap

#: Same palette as the Legendre chart and the toolbar icons.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e5e4e0"
ACCENT = "#2a78d6"
MARK = "#e34948"


class TsmThresholdDialog(wx.Dialog):
    """Binarised preview, histogram, and a slider over the threshold."""

    #: Idle time before the preview is recomputed, in milliseconds.
    PREVIEW_DELAY_MS = 120

    #: What the histogram's shape allowed; see histogram_valley().
    SHAPE_NOTE = {
        'valley': "",
        'triangle': "   (one hump only: the level comes from the triangle "
                    "construction, not from a valley between two peaks)",
        'flat': "   (no hump found: the level is the mean of the image)",
    }

    def __init__(self, parent, phase, image_name=""):
        super().__init__(parent, title="TSM - binarisation threshold",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.phi_nor = normalised_phase(phase)
        self.info = histogram_valley(self.phi_nor)
        self.suggested = self.info['threshold']
        self.threshold = self.suggested
        self.confirmed = False

        # Shown small, so binarising the full array for it is wasted work.
        step = max(1, max(self.phi_nor.shape) // 512)
        self.preview = self.phi_nor[::step, ::step] if step > 1 else self.phi_nor

        self._build_ui(image_name)
        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda evt: self._refresh(), self._timer)
        self._refresh()
        # The size its contents ask for: a fixed one fits only one screen.
        self.GetSizer().Fit(self)
        self.CentreOnParent()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, image_name):
        """Lay out the two previews, the reading, the slider and the buttons."""
        mainSizer = wx.BoxSizer(wx.VERTICAL)

        if image_name:
            title = wx.StaticText(self, label=image_name)
            title.SetFont(title.GetFont().Bold())
            mainSizer.Add(title, 0, wx.LEFT | wx.TOP, 10)

        previewRow = wx.BoxSizer(wx.HORIZONTAL)

        leftSizer = wx.BoxSizer(wx.VERTICAL)
        leftSizer.Add(wx.StaticText(self, label="Binarised image"), 0,
                      wx.ALIGN_CENTER | wx.BOTTOM, 4)
        self.bmp_binary = wx.StaticBitmap(self)
        self.bmp_binary.SetMinSize(self.FromDIP(wx.Size(360, 360)))
        leftSizer.Add(self.bmp_binary, 1, wx.EXPAND)
        previewRow.Add(leftSizer, 0, wx.ALL, 10)

        rightSizer = wx.BoxSizer(wx.VERTICAL)
        rightSizer.Add(wx.StaticText(self, label="Histogram of the normalised phase"),
                       0, wx.ALIGN_CENTER | wx.BOTTOM, 4)
        self.bmp_hist = wx.StaticBitmap(self)
        self.bmp_hist.SetMinSize(self.FromDIP(wx.Size(460, 360)))
        rightSizer.Add(self.bmp_hist, 1, wx.EXPAND)
        previewRow.Add(rightSizer, 1, wx.ALL | wx.EXPAND, 10)

        mainSizer.Add(previewRow, 1, wx.EXPAND)

        self.readout = wx.StaticText(self, label="")
        mainSizer.Add(self.readout, 0, wx.LEFT | wx.RIGHT, 12)

        self.slider = wx.Slider(self, value=int(self.threshold * 1000),
                                minValue=0, maxValue=1000, style=wx.SL_HORIZONTAL)
        mainSizer.Add(self.slider, 0, wx.EXPAND | wx.ALL, 10)

        buttonRow = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_reset = wx.Button(
            self, label="Back to the suggestion (%.4f)" % self.suggested)
        buttonRow.Add(self.btn_reset, 0, wx.RIGHT, 10)
        buttonRow.AddStretchSpacer()
        self.btn_ok = wx.Button(self, wx.ID_OK, "OK")
        self.btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        buttonRow.Add(self.btn_ok, 0, wx.RIGHT, 6)
        buttonRow.Add(self.btn_cancel, 0)
        mainSizer.Add(buttonRow, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(mainSizer)

        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        self.btn_reset.Bind(wx.EVT_BUTTON, self._on_reset)
        self.btn_ok.Bind(wx.EVT_BUTTON, self._on_ok)

    # -------------------------------------------------------------- events
    def _on_slider(self, event):
        """Follow the hand with the number; make the heavy part wait."""
        self.threshold = self.slider.GetValue() / 1000.0
        self._update_label()
        self._timer.Start(self.PREVIEW_DELAY_MS, oneShot=True)

    def _on_reset(self, event):
        """Put the threshold back on the suggested valley."""
        self.threshold = self.suggested
        self.slider.SetValue(int(self.threshold * 1000))
        self._update_label()
        self._refresh()

    def _on_ok(self, event):
        """Accept the current threshold."""
        self._timer.Stop()
        self.confirmed = True
        self.EndModal(wx.ID_OK)

    # --------------------------------------------------------------- paint
    def _update_label(self):
        """Show the threshold, and what the histogram's shape allowed.

        The resulting metric is deliberately not previewed here: this window
        computed it on its own, and the two would have to be kept in step for
        the reading to mean anything."""
        self.readout.SetLabel(
            "Threshold %.4f%s" % (self.threshold,
                                  self.SHAPE_NOTE[self.info['mode']]))

    def _refresh(self):
        """Redraw both previews for the current threshold."""
        self._update_label()

        binarised = (self.preview > self.threshold).astype(np.uint8) * 255
        self.bmp_binary.SetBitmap(
            numpy_to_wx_bitmap(binarised, self.FromDIP(wx.Size(360, 360)), self))
        self.bmp_hist.SetBitmap(self._histogram_bitmap())
        self.Layout()

    def _histogram_bitmap(self):
        """The histogram, the two peaks, the chord, and the current threshold."""
        centres = self.info['centres']
        smoothed = self.info['smoothed']

        # At the screen's real density, and without pyplot, which doubles it.
        fig = Figure(figsize=(4.6, 3.6), dpi=100 * hidpi.scale(self))
        FigureCanvasAgg(fig)
        fig.patch.set_facecolor(SURFACE)
        axes = fig.add_subplot(111)
        axes.set_facecolor(SURFACE)

        axes.fill_between(centres, smoothed, color=ACCENT, alpha=0.25, linewidth=0)
        axes.plot(centres, smoothed, color=ACCENT, linewidth=1.6)

        if self.info['mode'] != 'flat':
            # The chord and the drop: where the suggestion comes from. Drawn
            # for the triangle construction too, since the level is read off
            # the chord there in exactly the same way.
            chordLo, chordHi = self.info['chord_lo'], self.info['chord_hi']
            axes.plot([centres[chordLo], centres[chordHi]],
                      [smoothed[chordLo], smoothed[chordHi]],
                      color=INK_SOFT, linewidth=1.0, linestyle='-')
            axes.plot([centres[chordLo], centres[chordHi]],
                      [smoothed[chordLo], smoothed[chordHi]], 'o',
                      color=INK_SOFT, markersize=5)
            valleyIndex = self.info['valley']
            chordHeight = np.interp(centres[valleyIndex],
                                    [centres[chordLo], centres[chordHi]],
                                    [smoothed[chordLo], smoothed[chordHi]])
            axes.plot([centres[valleyIndex], centres[valleyIndex]],
                      [smoothed[valleyIndex], chordHeight],
                      color=INK_SOFT, linewidth=1.0, linestyle=':')

        axes.axvline(self.threshold, color=MARK, linewidth=2.0)
        axes.annotate("%.4f" % self.threshold,
                      (self.threshold, axes.get_ylim()[1]),
                      xytext=(4, -12), textcoords="offset points",
                      color=MARK, fontsize=9)

        axes.set_xlim(0, 1)
        axes.set_xlabel("Normalised phase", color=INK_SOFT, fontsize=9)
        axes.set_ylabel("Pixels", color=INK_SOFT, fontsize=9)
        axes.tick_params(colors=INK_SOFT, labelsize=8, length=0)
        axes.grid(True, axis="y", color=GRID, linewidth=1.0)
        axes.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.spines["bottom"].set_color(GRID)
        fig.tight_layout()

        return hidpi.figure_bitmap(fig, self)

    # -------------------------------------------------------------- result
    def get_threshold(self):
        """Show the dialog; return the chosen threshold, or None if cancelled."""
        if self.ShowModal() != wx.ID_OK or not self.confirmed:
            return None
        return self.threshold


def ask_tsm_threshold(parent, phase, image_name=""):
    """Ask the user where to binarise. Returns the threshold, or None."""
    dialog = TsmThresholdDialog(parent, phase, image_name)
    try:
        return dialog.get_threshold()
    finally:
        dialog.Destroy()
