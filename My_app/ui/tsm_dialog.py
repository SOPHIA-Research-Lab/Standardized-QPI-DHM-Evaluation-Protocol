"""
Choosing the binarisation threshold for the TSM metric.

The published definition,

    TSM = [ Nx*Ny - SUM SUM binarization(phi_comp_nor) ] / (Nx*Ny)

says what to do with the binarised image but not where to put the threshold,
beyond "an adjusted threshold". Left to a local or global mean, the answer
collapses: the mean splits any symmetric distribution in half, so TSM sat near
50% whatever the noise amplitude.

So the choice is handed to the user, with the histogram in front of them: the
suggested threshold is the valley between the two humps (see
global_phase.histogram_valley), the binarised image shows what that choice
means, and the resulting TSM updates as the slider moves.
"""
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

    #: Idle time before the preview is recomputed, in milliseconds. Binarising
    #: a 4096x4096 image on every one of the slider's 1000 steps would queue
    #: far more work than the hand can consume.
    PREVIEW_DELAY_MS = 120

    def __init__(self, parent, phase, image_name=""):
        super().__init__(parent, title="TSM - binarisation threshold",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.phi_nor = normalised_phase(phase)
        self.info = histogram_valley(self.phi_nor)
        self.suggested = self.info['threshold']
        self.threshold = self.suggested
        self.confirmed = False

        # The preview is shown small; binarising the full array for it is
        # wasted work on large images.
        paso = max(1, max(self.phi_nor.shape) // 512)
        self.vista = self.phi_nor[::paso, ::paso] if paso > 1 else self.phi_nor

        self._build_ui(image_name)
        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda evt: self._refresh(), self._timer)
        self._refresh()
        # The window takes the size its contents ask for: a fixed one was
        # right on the screen it was written on and nowhere else.
        self.GetSizer().Fit(self)
        self.CentreOnParent()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, image_name):
        principal = wx.BoxSizer(wx.VERTICAL)

        if image_name:
            titulo = wx.StaticText(self, label=image_name)
            titulo.SetFont(titulo.GetFont().Bold())
            principal.Add(titulo, 0, wx.LEFT | wx.TOP, 10)

        fila = wx.BoxSizer(wx.HORIZONTAL)

        izq = wx.BoxSizer(wx.VERTICAL)
        izq.Add(wx.StaticText(self, label="Binarised image"), 0,
                wx.ALIGN_CENTER | wx.BOTTOM, 4)
        self.bmp_binary = wx.StaticBitmap(self)
        self.bmp_binary.SetMinSize(self.FromDIP(wx.Size(360, 360)))
        izq.Add(self.bmp_binary, 1, wx.EXPAND)
        fila.Add(izq, 0, wx.ALL, 10)

        der = wx.BoxSizer(wx.VERTICAL)
        der.Add(wx.StaticText(self, label="Histogram of the normalised phase"),
                0, wx.ALIGN_CENTER | wx.BOTTOM, 4)
        self.bmp_hist = wx.StaticBitmap(self)
        self.bmp_hist.SetMinSize(self.FromDIP(wx.Size(460, 360)))
        der.Add(self.bmp_hist, 1, wx.EXPAND)
        fila.Add(der, 1, wx.ALL | wx.EXPAND, 10)

        principal.Add(fila, 1, wx.EXPAND)

        self.etiqueta = wx.StaticText(self, label="")
        principal.Add(self.etiqueta, 0, wx.LEFT | wx.RIGHT, 12)

        self.slider = wx.Slider(self, value=int(self.threshold * 1000),
                                minValue=0, maxValue=1000, style=wx.SL_HORIZONTAL)
        principal.Add(self.slider, 0, wx.EXPAND | wx.ALL, 10)

        botones = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_reset = wx.Button(
            self, label="Back to the valley (%.4f)" % self.suggested)
        botones.Add(self.btn_reset, 0, wx.RIGHT, 10)
        botones.AddStretchSpacer()
        self.btn_ok = wx.Button(self, wx.ID_OK, "OK")
        self.btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        botones.Add(self.btn_ok, 0, wx.RIGHT, 6)
        botones.Add(self.btn_cancel, 0)
        principal.Add(botones, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(principal)

        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        self.btn_reset.Bind(wx.EVT_BUTTON, self._on_reset)
        self.btn_ok.Bind(wx.EVT_BUTTON, self._on_ok)

    # -------------------------------------------------------------- eventos
    def _on_slider(self, event):
        self.threshold = self.slider.GetValue() / 1000.0
        self._update_label()
        self._timer.Start(self.PREVIEW_DELAY_MS, oneShot=True)

    def _on_reset(self, event):
        self.threshold = self.suggested
        self.slider.SetValue(int(self.threshold * 1000))
        self._update_label()
        self._refresh()

    def _on_ok(self, event):
        self._timer.Stop()
        self.confirmed = True
        self.EndModal(wx.ID_OK)

    # --------------------------------------------------------------- pintar
    def _tsm(self):
        """TSM for the current threshold, straight from the definition."""
        binarizacion = self.phi_nor > self.threshold
        return 100.0 * (1.0 - binarizacion.mean())

    def _update_label(self):
        aviso = "" if self.info['ok'] else \
            "   (no second peak: the histogram is not bimodal)"
        self.etiqueta.SetLabel(
            "Threshold %.4f        TSM %.2f %%%s"
            % (self.threshold, self._tsm(), aviso))

    def _refresh(self):
        self._update_label()

        binaria = (self.vista > self.threshold).astype(np.uint8) * 255
        self.bmp_binary.SetBitmap(
            numpy_to_wx_bitmap(binaria, self.FromDIP(wx.Size(360, 360)), self))
        self.bmp_hist.SetBitmap(self._histogram_bitmap())
        self.Layout()

    def _histogram_bitmap(self):
        """The histogram, the two peaks, the chord, and the current threshold."""
        c = self.info['centres']
        suave = self.info['smoothed']

        # Drawn at the screen's real pixel density, and built without pyplot:
        # pyplot hands the figure to the active interactive backend, and the wx
        # one silently doubles the dpi on a scaled screen. See
        # legendre_viewer.coefficient_figure.
        fig = Figure(figsize=(4.6, 3.6), dpi=100 * hidpi.scale(self))
        FigureCanvasAgg(fig)
        fig.patch.set_facecolor(SURFACE)
        ax = fig.add_subplot(111)
        ax.set_facecolor(SURFACE)

        ax.fill_between(c, suave, color=ACCENT, alpha=0.25, linewidth=0)
        ax.plot(c, suave, color=ACCENT, linewidth=1.6)

        if self.info['ok']:
            lo, hi = self.info['peak_lo'], self.info['peak_hi']
            # The chord between the peaks, and the drop to the valley: the
            # construction the suggestion comes from.
            ax.plot([c[lo], c[hi]], [suave[lo], suave[hi]],
                    color=INK_SOFT, linewidth=1.0, linestyle='-')
            ax.plot([c[lo], c[hi]], [suave[lo], suave[hi]], 'o',
                    color=INK_SOFT, markersize=5)
            v = self.info['valley']
            altura_acorde = np.interp(c[v], [c[lo], c[hi]], [suave[lo], suave[hi]])
            ax.plot([c[v], c[v]], [suave[v], altura_acorde],
                    color=INK_SOFT, linewidth=1.0, linestyle=':')

        ax.axvline(self.threshold, color=MARK, linewidth=2.0)
        ax.annotate("%.4f" % self.threshold,
                    (self.threshold, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points",
                    color=MARK, fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalised phase", color=INK_SOFT, fontsize=9)
        ax.set_ylabel("Pixels", color=INK_SOFT, fontsize=9)
        ax.tick_params(colors=INK_SOFT, labelsize=8, length=0)
        ax.grid(True, axis="y", color=GRID, linewidth=1.0)
        ax.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        fig.tight_layout()

        return hidpi.figure_bitmap(fig, self)

    # --------------------------------------------------------------- salida
    def get_threshold(self):
        """Show the dialog; return the chosen threshold, or None if cancelled."""
        if self.ShowModal() != wx.ID_OK or not self.confirmed:
            return None
        return self.threshold


def ask_tsm_threshold(parent, phase, image_name=""):
    """Ask the user where to binarise. Returns the threshold, or None."""
    dlg = TsmThresholdDialog(parent, phase, image_name)
    try:
        return dlg.get_threshold()
    finally:
        dlg.Destroy()
