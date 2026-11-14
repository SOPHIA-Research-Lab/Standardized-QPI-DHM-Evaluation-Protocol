"""Choosing the binarization threshold for the TSM metric."""
# The definition fixes all but where to binarize, so the user chooses it.
import numpy as np
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure

from analysis.module1.segmentation import bright_sample
from analysis.module2.global_phase import histogram_valley, normalized_phase
from ui.threshold_dialog import numpy_to_wx_bitmap

#: Same palette as the Legendre chart and the toolbar icons.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e5e4e0"
ACCENT = "#2a78d6"
MARK = "#e34948"

#: The sample side in the preview, and what is left of the field around it.
SAMPLE_TINT = (227, 73, 72)
FIELD_SHADE = (232, 234, 237)


class TsmThresholdDialog(wx.Dialog):
    """Binarized preview, histogram, and a slider over the threshold."""

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
        super().__init__(parent, title="TSM - binarization threshold",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.phi_nor = normalized_phase(phase)
        self.info = histogram_valley(self.phi_nor)
        self.suggested = self.info['threshold']
        self.threshold = self.suggested
        # Which side holds the sample is the other half of the definition, and
        # the skew that suggests it is read from the whole field -- a residual
        # surface can outweigh the sample's tail and flip it. Suggested here
        # like the threshold, and changed here like the threshold.
        hump = self.info.get('peak')
        self.suggested_bright = (bright_sample(self.phi_nor) if hump is None
                                 else bool(hump <= self.suggested))
        self.sample_is_bright = self.suggested_bright
        self.confirmed = False
        self._dragging = False

        # Shown small, so binarizing the full array for it is wasted work.
        step = max(1, max(self.phi_nor.shape) // 512)
        self.preview = self.phi_nor[::step, ::step] if step > 1 else self.phi_nor

        self._build_ui(image_name)
        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda evt: self._refresh(), self._timer)
        self._refresh()
        # The size its contents ask for: a fixed one fits only one screen.
        self.GetSizer().Fit(self)
        self.CenterOnParent()

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
        leftSizer.Add(wx.StaticText(
            self, label="Binarized image  (the sample side is tinted)"), 0,
            wx.ALIGN_CENTER | wx.BOTTOM, 4)
        self.bmp_binary = wx.StaticBitmap(self)
        self.bmp_binary.SetMinSize(self.FromDIP(wx.Size(360, 360)))
        leftSizer.Add(self.bmp_binary, 1, wx.EXPAND)
        previewRow.Add(leftSizer, 0, wx.ALL, 10)

        rightSizer = wx.BoxSizer(wx.VERTICAL)
        rightSizer.Add(wx.StaticText(
            self, label="Histogram of the normalized phase  "
                        "(drag the red line to move the threshold)"),
            0, wx.ALIGN_CENTER | wx.BOTTOM, 4)
        # A live canvas rather than a rendered bitmap, so the line can be
        # grabbed where it is drawn instead of only through the slider.
        self.figure = Figure(figsize=(4.6, 3.6))
        self.canvas = FigureCanvasWxAgg(self, wx.ID_ANY, self.figure)
        self.canvas.SetMinSize(self.FromDIP(wx.Size(460, 360)))
        self.canvas.SetCursor(wx.Cursor(wx.CURSOR_SIZEWE))
        self.axes = self.figure.add_subplot(111)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)
        rightSizer.Add(self.canvas, 1, wx.EXPAND)
        previewRow.Add(rightSizer, 1, wx.ALL | wx.EXPAND, 10)

        mainSizer.Add(previewRow, 1, wx.EXPAND)

        self.readout = wx.StaticText(self, label="")
        mainSizer.Add(self.readout, 0, wx.LEFT | wx.RIGHT, 12)

        # What the percentage will actually be, before it reaches the table.
        self.readout_side = wx.StaticText(self, label="")
        mainSizer.Add(self.readout_side, 0, wx.LEFT | wx.RIGHT | wx.TOP, 12)

        self.slider = wx.Slider(self, value=int(self.threshold * 1000),
                                minValue=0, maxValue=1000, style=wx.SL_HORIZONTAL)
        mainSizer.Add(self.slider, 0, wx.EXPAND | wx.ALL, 10)

        self.radio_side = wx.RadioBox(
            self, label="Which side is the sample",
            choices=["Bright - above the threshold",
                     "Dark - below the threshold"],
            majorDimension=2, style=wx.RA_SPECIFY_COLS)
        self.radio_side.SetSelection(0 if self.sample_is_bright else 1)
        mainSizer.Add(self.radio_side, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        buttonRow = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_reset = wx.Button(
            self, label="Back to the suggestion (%.4f)" % self.suggested)
        buttonRow.Add(self.btn_reset, 0, wx.RIGHT, 10)
        buttonRow.AddStretchSpacer()
        self.btn_ok = wx.Button(self, wx.ID_OK, "OK")
        self.btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        # Enter accepts the threshold on screen.
        self.btn_ok.SetDefault()
        buttonRow.Add(self.btn_ok, 0, wx.RIGHT, 6)
        buttonRow.Add(self.btn_cancel, 0)
        mainSizer.Add(buttonRow, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(mainSizer)

        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        self.radio_side.Bind(wx.EVT_RADIOBOX, self._on_side)
        self.btn_reset.Bind(wx.EVT_BUTTON, self._on_reset)
        self.btn_ok.Bind(wx.EVT_BUTTON, self._on_ok)

    # -------------------------------------------------------------- events
    def _on_slider(self, event):
        """The slider moved: one road into _set_threshold, like the line."""
        self._set_threshold(self.slider.GetValue() / 1000.0)

    def _set_threshold(self, value):
        """Put the threshold at `value`, wherever the gesture came from.

        Rounded to the slider's own step, so dragging the line and dragging
        the slider cannot end up disagreeing in the fourth decimal."""
        value = round(float(np.clip(value, 0.0, 1.0)) * 1000) / 1000.0
        if value == self.threshold:
            return
        self.threshold = value
        self.slider.SetValue(int(round(value * 1000)))
        self._update_label()
        # The histogram follows the hand; binarizing the image is the slow
        # part, so that one waits for the gesture to settle.
        self._draw_histogram()
        self._timer.Start(self.PREVIEW_DELAY_MS, oneShot=True)

    def _on_canvas_press(self, event):
        """Grab the threshold wherever the histogram was clicked."""
        if event.inaxes is not self.axes or event.xdata is None:
            return
        self._dragging = True
        self._set_threshold(event.xdata)

    def _on_canvas_motion(self, event):
        """Carry the line along with the pointer while the button is down."""
        if not self._dragging or event.xdata is None:
            return
        self._set_threshold(event.xdata)

    def _on_canvas_release(self, event):
        """Let go."""
        self._dragging = False

    def _on_side(self, event):
        """The operator states which side the sample is on."""
        self.sample_is_bright = self.radio_side.GetSelection() == 0
        # Both panels say which side is counted, so both have to follow.
        self._refresh()

    def _on_reset(self, event):
        """Put the threshold and the side back where the histogram put them."""
        self.threshold = self.suggested
        self.sample_is_bright = self.suggested_bright
        self.slider.SetValue(int(self.threshold * 1000))
        self.radio_side.SetSelection(0 if self.sample_is_bright else 1)
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

        share = self.sample_share()
        self.readout_side.SetLabel(
            "TSM will read %.2f %% - the %s side of the threshold%s"
            % (share, "bright" if self.sample_is_bright else "dark",
               "   (more than half the field: a sample is the minority, so "
               "this is most likely the background)" if share > 50.0 else ""))
        self.readout_side.SetForegroundColour(MARK if share > 50.0 else INK_SOFT)
        self.Layout()

    def sample_share(self):
        """The percentage TSM will report for the side now chosen."""
        above = float(np.mean(self.phi_nor > self.threshold))
        return 100.0 * (above if self.sample_is_bright else 1.0 - above)

    def _refresh(self):
        """Redraw both panels for the threshold and the side now chosen."""
        self._update_label()
        self.bmp_binary.SetBitmap(
            numpy_to_wx_bitmap(self._preview_rgb(),
                               self.FromDIP(wx.Size(360, 360)), self))
        self._draw_histogram()
        self.Layout()

    def _preview_rgb(self):
        """The binarized preview, with the side being counted tinted.

        Plain black and white said which pixels were above the threshold,
        which is not the question: what the metric reports is the side
        declared to be the sample, and that can be either one."""
        sample = (self.preview > self.threshold if self.sample_is_bright
                  else self.preview <= self.threshold)
        rgb = np.empty(sample.shape + (3,), dtype=np.uint8)
        rgb[...] = np.array(FIELD_SHADE, dtype=np.uint8)
        rgb[sample] = np.array(SAMPLE_TINT, dtype=np.uint8)
        return rgb

    def _draw_histogram(self):
        """The histogram, the construction behind the level, and the split.

        The shaded part is the side that will be counted: that is the whole
        question the dialog exists to settle, and a plain curve with a line
        across it does not answer it."""
        axes = self.axes
        axes.clear()

        centers = self.info['centers']
        smoothed = self.info['smoothed']

        self.figure.patch.set_facecolor(SURFACE)
        axes.set_facecolor(SURFACE)

        # The sample side filled, the rest left pale: which is which follows
        # the radio button, not the side of the threshold.
        isSample = (centers > self.threshold if self.sample_is_bright
                    else centers <= self.threshold)
        axes.fill_between(centers, smoothed, where=isSample, interpolate=True,
                          color=MARK, alpha=0.32, linewidth=0)
        axes.fill_between(centers, smoothed, where=~isSample, interpolate=True,
                          color=ACCENT, alpha=0.20, linewidth=0)
        axes.plot(centers, smoothed, color=ACCENT, linewidth=1.6)

        if self.info['mode'] != 'flat':
            # The chord and the drop: where the suggestion comes from. Drawn
            # for the triangle construction too, since the level is read off
            # the chord there in exactly the same way.
            chordLo, chordHi = self.info['chord_lo'], self.info['chord_hi']
            axes.plot([centers[chordLo], centers[chordHi]],
                      [smoothed[chordLo], smoothed[chordHi]],
                      color=INK_SOFT, linewidth=1.0, linestyle='-')
            axes.plot([centers[chordLo], centers[chordHi]],
                      [smoothed[chordLo], smoothed[chordHi]], 'o',
                      color=INK_SOFT, markersize=5)
            valleyIndex = self.info['valley']
            chordHeight = np.interp(centers[valleyIndex],
                                    [centers[chordLo], centers[chordHi]],
                                    [smoothed[chordLo], smoothed[chordHi]])
            axes.plot([centers[valleyIndex], centers[valleyIndex]],
                      [smoothed[valleyIndex], chordHeight],
                      color=INK_SOFT, linewidth=1.0, linestyle=':')

        axes.axvline(self.threshold, color=MARK, linewidth=2.0)

        axes.set_xlim(0, 1)
        axes.set_ylim(bottom=0)
        top = axes.get_ylim()[1]
        axes.annotate("%.4f" % self.threshold, (self.threshold, top),
                      xytext=(4, -12), textcoords="offset points",
                      color=MARK, fontsize=9)

        # Says which filled area is which, where the areas themselves are.
        share = self.sample_share()
        sampleSide = 0.5 * (1.0 + self.threshold) if self.sample_is_bright             else 0.5 * self.threshold
        axes.annotate("sample\n%.2f %%" % share, (sampleSide, top * 0.86),
                      ha="center", va="top", color=MARK, fontsize=9)
        axes.annotate("background\n%.2f %%" % (100.0 - share),
                      (0.5 * self.threshold if self.sample_is_bright
                       else 0.5 * (1.0 + self.threshold), top * 0.86),
                      ha="center", va="top", color=INK_SOFT, fontsize=9)

        axes.set_xlabel("Normalized phase", color=INK_SOFT, fontsize=9)
        axes.set_ylabel("Pixels", color=INK_SOFT, fontsize=9)
        axes.tick_params(colors=INK_SOFT, labelsize=8, length=0)
        axes.grid(True, axis="y", color=GRID, linewidth=1.0)
        axes.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.spines["bottom"].set_color(GRID)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # -------------------------------------------------------------- result
    def get_threshold(self):
        """Show the dialog; return (threshold, sample_is_bright), or None."""
        if self.ShowModal() != wx.ID_OK or not self.confirmed:
            return None
        return self.threshold, self.sample_is_bright


def ask_tsm_threshold(parent, phase, image_name=""):
    """Ask where to binarize and which side is the sample.

    Returns (threshold, sample_is_bright), or None if it was cancelled."""
    dialog = TsmThresholdDialog(parent, phase, image_name)
    try:
        return dialog.get_threshold()
    finally:
        dialog.Destroy()
