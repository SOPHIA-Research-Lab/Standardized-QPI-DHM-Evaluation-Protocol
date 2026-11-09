import wx


def create(parent):
    helpMenu = wx.Menu()

    helpId = wx.NewIdRef()
    parent.Bind(wx.EVT_MENU, lambda e: show_help(parent), id=helpId)
    helpMenu.Append(helpId, "&View Help\tF1", "View application help")

    return helpMenu


def show_help(parent):
    helpText = (
        "Standardized QPI-DHM Evaluation Protocol - Help\n\n"
        "This application evaluates quantitative phase imaging (QPI) in Digital "
        "Holographic Microscopy (DHM). It expects a COMPENSATED PHASE MAP as "
        "input, not a hologram: the background segmentation thresholds phase "
        "values, and on an interference pattern it would segment the fringes.\n\n"

        "Input\n"
        "   -  PNG, JPG, BMP, TIFF: converted to phase by mapping the grey level "
        "linearly onto [-pi, pi].\n"
        "   -  MATLAB .mat holding a complex field: the true phase is kept "
        "(np.angle) and the amplitude is shown for viewing only. This is the more "
        "faithful route, since an 8-bit file carries only 256 phase levels.\n"
        "   -  MATLAB .mat holding a real array: taken to be the phase in "
        "radians and used as it stands, with no conversion.\n\n"

        "Hologram Samples menu\n"
        "   The files in this menu are HOLOGRAMS, not phase maps. They are "
        "included so that a reconstruction and compensation algorithm can be "
        "run on them and the quality of its result evaluated here; their "
        "filenames record the magnification, the wavelength, the pixel pitch "
        "and, for the non-telecentric ones, the defocus distance.\n"
        "   While the active tab holds one of them, Modules I to III are "
        "disabled: on an interference pattern they would measure the fringes. "
        "Open the compensated phase map with File > Open and they come back. "
        "Module IV is unaffected, since it does not read the workspace.\n\n"

        "How the background is selected\n"
        "   Metrics in Module I are measured on BACKGROUND pixels only, so a "
        "dialog asks how to tell background from sample. Everything in it "
        "chooses WHICH pixels are background. Each metric is then computed on "
        "the ORIGINAL phase values of those pixels: the filtered copy only "
        "places the threshold and is discarded, so the settings pick the "
        "region and never alter the quantity measured.\n"
        "   You are not asked which side is the sample. The background is "
        "taken to be the LARGER of the two regions, which in QPI-DHM it almost "
        "always is, and whether the sample is brighter or darker is read from "
        "the skew of the histogram. The Measured background preview tints the "
        "sample over the phase, so a wrong choice is visible at a glance; a "
        "warning appears when the two regions are of similar size, which is "
        "the one case the rule cannot settle, and an Invert box flips it.\n"
        "   The panel works top to bottom:\n"
        "   -  Remove background curvature (top-hat radius). The control that "
        "matters most: residual curvature is a slowly varying background, and "
        "it is what stops one threshold from fitting the whole field. Set the "
        "radius a little larger than the biggest sample feature.\n"
        "   -  Smooth isolated noise (median radius), for lone outlying pixels.\n"
        "   -  The threshold: Otsu, Triangle or Manual. Triangle suits a field "
        "the background dominates. The automatic ones recompute the level for "
        "each image.\n"
        "   -  Cleanup: discard objects under N px, fill holes under N px, a "
        "guard band that grows the sample before the background is taken, and "
        "an excluded border.\n"
        "      The guard band keeps the halo around the sample out of the "
        "measurement; that halo is where compensation error concentrates and "
        "it inflates STD and PV. The border drops a margin all the way round, "
        "because reconstruction leaves its worst artefacts at the edges of the "
        "field -- unwrapping seams, filter ringing, the corners of the "
        "spectrum -- and those are not background however flat the middle of "
        "the image looks.\n"
        "   The recipe is settled on the FIRST image and reused for the rest of "
        "the run, so that every row of the table was segmented the same way and "
        "the rows can be compared. Only the threshold is per image. A tick box "
        "lets one image depart from the recipe; its row is then marked with an "
        "asterisk in the Segmentation column of the Mask Data table, which "
        "records the full recipe so the result can be reproduced.\n\n"

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
        "   -  MAD: MEAN absolute deviation, mean(|x - mean(x)|). Same idea as "
        "STD, and less sensitive to outliers because the deviations are not "
        "squared, but it is built on the mean and is NOT the robust "
        "median-based MAD that the abbreviation usually denotes.\n"
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
        "reconstructed surface next to the coefficient chart. The fit first "
        "low-passes the phase to keep the smooth part, and a low-pass averages "
        "across a 2*pi jump as if it were real, so a wrapped map returns NaN "
        "rather than a smeared answer: use the Unwrapped variant for it. A "
        "hard edge around the sample does not trigger this, because only the "
        "background is examined.\n\n"

        "MODULE II - Global phase distortion\n"
        "   The same measures applied to the WHOLE image instead of the masked "
        "background. Each offers a Global and an Unwrapped variant.\n"
        "   -  PV: max minus min over the whole field.\n"
        "   -  PGM - Phase Gradient Metric: mean local gradient magnitude from "
        "normalised Prewitt kernels, after Yang et al., Opt. Express 31(22), "
        "36188-36201 (2023), Eqs. (8)-(10). Low means a flat field, but it "
        "rises with tilt, curvature and noise alike, so read it as roughness "
        "rather than as alignment. The kernels are normalised here, so the "
        "value is six times smaller than the figures in that paper.\n"
        "   -  GPG - Global Phase Gradient: the tilt alone. A plane is fitted "
        "to the whole map by least squares and the result is the magnitude of "
        "its slope. Fitting averages noise away, so unlike PGM this one stays "
        "near zero on a noisy but level field, and on defocus too.\n"
        "   -  TSM - Thresholding and Intensity Summation Metric: the phase is "
        "binarised, the pixels above the threshold are summed, and that sum is "
        "subtracted from the total, so the reading is the percentage below a "
        "threshold on the phase normalised to [0,1]. The published definition "
        "fixes everything except where to binarise, and any mean splits a "
        "symmetric distribution in half, so the choice is handed to you: a "
        "dialog shows the binarised image beside the histogram, suggests the "
        "valley between the two humps, and updates the reading as you move the "
        "slider. The answer is cached per image and per wrapped/unwrapped "
        "variant, so one question covers every TSM variant of that image.\n"
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

    dialog = wx.Dialog(parent, title="Help",
                       size=wx.Window.FromDIP(wx.Size(720, 600), parent))
    dialog.SetBackgroundColour(wx.Colour(255, 255, 255))

    panel = wx.Panel(dialog)

    mainSizer = wx.BoxSizer(wx.VERTICAL)

    textArea = wx.TextCtrl(panel, value=helpText,
                           style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.VSCROLL)
    mainSizer.Add(textArea, 1, wx.EXPAND | wx.ALL, 10)

    closeButton = wx.Button(panel, label="Close")
    closeButton.Bind(wx.EVT_BUTTON, lambda evt: dialog.Close())
    mainSizer.Add(closeButton, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

    panel.SetSizer(mainSizer)

    dialog.ShowModal()
    dialog.Destroy()
