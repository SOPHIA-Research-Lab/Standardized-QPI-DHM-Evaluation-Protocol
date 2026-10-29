import wx


def create(parent):
    help_menu = wx.Menu()

    help_id = wx.NewIdRef()
    parent.Bind(wx.EVT_MENU, lambda e: show_help(parent), id=help_id)
    help_menu.Append(help_id, "&View Help\tF1", "View application help")

    return help_menu


def show_help(parent):
    help_text = (
        "Standardized QPI-DHM Evaluation Protocol - Help\n\n"
        "This application evaluates quantitative phase imaging (QPI) in Digital "
        "Holographic Microscopy (DHM). It expects a COMPENSATED PHASE MAP as "
        "input, not a raw hologram: the background segmentation thresholds phase "
        "values, and on a raw interference pattern it segments the fringes.\n\n"

        "Input\n"
        "   -  PNG, JPG, BMP, TIFF: converted to phase by mapping the grey level "
        "linearly onto [-pi, pi].\n"
        "   -  MATLAB .mat holding a complex field: the true phase is kept "
        "(np.angle) and the amplitude is shown for viewing only. This is the more "
        "faithful route, since an 8-bit file carries only 256 phase levels.\n\n"

        "How the background is selected\n"
        "   Metrics in Module I are measured on BACKGROUND pixels only. A dialog "
        "asks you to set a threshold; everything on the chosen side of it is the "
        "sample, the rest is background. Two cleanup sizes then apply: objects "
        "smaller than the first are discarded, holes smaller than the second are "
        "filled. Raise the first one and small samples disappear.\n"
        "   The segmentation is asked ONCE PER IMAGE and reused by every metric "
        "of that image. A different image gets its own.\n\n"

        "Phase unwrapping\n"
        "   Variants marked 'Unwrapped' remove 2*pi discontinuities first. You are "
        "asked once per batch whether to allow it; answering No skips the metrics "
        "that require it rather than computing them on wrapped phase. Unwrapping "
        "can introduce errors of its own, which is why it is a question.\n\n"

        "MODULE I - Residual background phase variance\n"
        "   Each metric offers three variants: Background (wrapped phase), "
        "Unwrapped Background, and Background Zones (rectangles you draw).\n"
        "   -  STD: dispersion of the background phase. Low means a uniform "
        "background; high means noise or residual structure.\n"
        "   -  MAD: mean absolute deviation. Same idea as STD but less sensitive "
        "to outliers.\n"
        "   -  RMS: root mean square of the values themselves, sqrt(mean(x^2)). "
        "Note this includes the mean, so RMS^2 = mean^2 + STD^2. The background "
        "mean is reported alongside it as Mean_B, because that difference is the "
        "only thing RMS adds over STD.\n"
        "   -  PV: peak-to-valley, max minus min. Based on two single pixels, so "
        "it is sensitive to outliers.\n"
        "   -  FWHM: width at half maximum of the HISTOGRAM of background phase "
        "values. It measures how spread out the noise is, NOT image sharpness. A "
        "low FWHM means a uniform background. It returns NaN when the histogram "
        "has no peak with a crossing on both sides, which is common on wrapped "
        "phase.\n"
        "   -  Entropy: Shannon entropy of the background values, in bits.\n"
        "   -  Legendre: fits the background with Legendre polynomials and "
        "reports the coefficients, one column each. A window shows the "
        "reconstructed surface next to the coefficient chart.\n\n"

        "MODULE II - Global phase distortion\n"
        "   The same measures applied to the WHOLE image instead of the masked "
        "background. Each offers a Global and an Unwrapped variant.\n"
        "   -  PV: max minus min over the whole field.\n"
        "   -  GPG - Global Phase Gradient: mean local gradient magnitude from "
        "Prewitt kernels, per Yang et al., Opt. Express 31(22), 36188-36201 "
        "(2023), Eqs. (8)-(10). Low means a flat field; it rises with both tilt "
        "and background noise.\n"
        "   -  TSM: percentage of pixels below a local threshold. Be aware that "
        "the threshold is a local average, so the result sits near 50% for any "
        "symmetric distribution and barely responds to noise amplitude.\n"
        "   -  LE - Laplacian Energy: mean squared Laplacian. High values mean "
        "many edges and sharp detail, which over a background that should be "
        "uniform means noise.\n"
        "   -  SF - Spatial Frequency: RMS of the row and column differences. The "
        "phase is normalised to [0,1] first, so the value is dimensionless and "
        "does not see the phase amplitude.\n"
        "   -  Entropy: Shannon entropy over the whole field.\n"
        "   -  GSM - Generalized Sharpness Metric, after Fienup & Miller, JOSA A "
        "20, 609-620 (2003). Mean of the normalised intensity raised to beta.\n"
        "   -  FWHM: the Module I histogram estimator applied to the whole image.\n\n"

        "MODULE III - Ground-truth comparison\n"
        "   Requires a reference image of the SAME dimensions, loaded beforehand. "
        "Neither side is unwrapped: unwrapping an 8-bit reference invents "
        "discontinuities and stops it being a reference.\n"
        "   -  SSIM: 1 means structurally identical, 0 means unrelated.\n"
        "   -  MSE: 0 means identical; higher means larger differences.\n"
        "   -  PSNR, in dB: higher is better. Infinite when MSE is 0.\n\n"

        "MODULE IV - Computational complexity\n"
        "   Opens a separate tool that measures CPU instructions, cycles and "
        "execution time against input size, and fits complexity curves. The "
        "profiling relies on the Linux 'perf' utility, so measurements require "
        "Linux or WSL2. The window itself opens anywhere.\n\n"

        "Results\n"
        "   -  One row per image, one column per metric. Tables show 4 decimals; "
        "exports carry the full precision.\n"
        "   -  Right-click a table to export it alone or with the others, as "
        "Excel or CSV. Ctrl+S exports everything to Excel.\n"
        "   -  A metric that could not be computed says why instead of failing "
        "silently.\n"
        "   -  A long batch can be cancelled; whatever was computed stays.\n\n"

        "Shortcuts\n"
        "   Ctrl+O  Open image        Ctrl+S  Save results\n"
        "   Ctrl+L  Clear all         Ctrl+Q  Exit\n"
        "   F1      This help         Alt+key Menu access\n\n"

        "Setup\n"
        "   pip install -r requirements.txt\n"
        "   python app.py\n"
    )

    dialog = wx.Dialog(parent, title="Help", size=(720, 600))
    dialog.SetBackgroundColour(wx.Colour(255, 255, 255))

    panel = wx.Panel(dialog)

    sizer = wx.BoxSizer(wx.VERTICAL)

    text_ctrl = wx.TextCtrl(panel,
                            value=help_text,
                            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.VSCROLL)
    sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 10)

    close_btn = wx.Button(panel, label="Close")
    close_btn.Bind(wx.EVT_BUTTON, lambda evt: dialog.Close())
    sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

    panel.SetSizer(sizer)

    dialog.ShowModal()
    dialog.Destroy()
