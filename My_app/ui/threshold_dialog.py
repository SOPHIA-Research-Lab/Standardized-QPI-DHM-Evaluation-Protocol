"""Interactive background segmentation for Module I.

The previews sit on the left, the controls on the right, in the order they are
used: what is measured, how the image is prepared, where the threshold goes,
how the mask is cleaned. The recipe is chosen once for a batch; only the
threshold is tuned per image.
"""
# Split off from utilitiesRBPV, which pulled wx into the analysis layer.
import math

import numpy as np
import wx

from analysis.module1.segmentation import (Recipe, THRESHOLD_METHODS,
                                           THRESHOLD_HINTS, THRESHOLD_LABELS,
                                           build_mask)
from ui import hidpi

#: The notice is shown once per session, not once per image.
_WELCOME_SHOWN = False


class _Swatch(wx.Panel):
    """A rounded color chip. wx has no rounded static control, and a plain
    square panel reads as an unfinished widget rather than a key."""

    def __init__(self, parent, color, size):
        super().__init__(parent, size=size)
        self._color = wx.Colour(*color)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(self.GetParent().GetBackgroundColour()))
        dc.Clear()
        context = wx.GraphicsContext.Create(dc)
        if context is None:
            return
        width, height = self.GetSize()
        context.SetBrush(wx.Brush(self._color))
        # A hairline a shade darker, so a pale chip still has an edge.
        context.SetPen(wx.Pen(wx.Colour(max(0, self._color.Red() - 45),
                                        max(0, self._color.Green() - 45),
                                        max(0, self._color.Blue() - 45)), 1))
        context.DrawRoundedRectangle(0.5, 0.5, width - 1, height - 1,
                                     min(width, height) / 3.0)


def numpy_to_wx_bitmap(arr: np.ndarray, target_size=(400, 400), window=None):
    """Convert a numpy array to a wx.Bitmap fitted inside target_size."""
    # `target_size` is in physical pixels; `window` gives the screen's scale.
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)

    if arr.ndim == 2:
        sourceHeight, sourceWidth = arr.shape
        rgbPixels = np.dstack([arr, arr, arr])
    else:
        sourceHeight, sourceWidth, _ = arr.shape
        rgbPixels = arr

    image = wx.Image(sourceWidth, sourceHeight)
    image.SetData(rgbPixels.tobytes())

    # Fitted inside the box, keeping the aspect ratio.
    targetWidth, targetHeight = target_size
    fitFactor = min(targetWidth / sourceWidth, targetHeight / sourceHeight)

    if fitFactor != 1.0:
        image = image.Scale(int(sourceWidth * fitFactor),
                            int(sourceHeight * fitFactor), wx.IMAGE_QUALITY_HIGH)

    return hidpi.bitmap(image, window)


class WxThresholdDialog(wx.Dialog):
    """Interactive background segmentation using wxPython."""

    #: Idle time before the preview is recomputed, in milliseconds.
    PREVIEW_DELAY_MS = 200

    #: Whether the batch left this recipe free to change; see _sync_controls.
    _recipe_editable = True

    #: One preview. A sample mask is the exact negative of a background one,
    #: and the filtered copy only differs from the phase when a filter is on;
    #: what has to be checked is which pixels end up being measured, and that
    #: is only legible drawn over the phase itself.
    #: The sample is tinted; what stays gray is what the metrics will read.
    SAMPLE_TINT = (220, 70, 70)
    BACKGROUND_SWATCH = (150, 150, 150)

    def __init__(self, parent, image: np.ndarray, recipe: Recipe = None,
                 recipe_locked: bool = False, image_name: str = "",
                 title="Background Segmentation"):
        super().__init__(parent, title=title,
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.image = np.array(image, dtype=float)
        self.image_name = image_name
        self.recipe = Recipe(**vars(recipe)) if recipe is not None else Recipe()

        # Second image onward: the batch keeps its recipe so that every row of
        # the results table was segmented the same way. Only the threshold is
        # per image -- and the automatic methods recompute it on their own.
        self.recipe_locked = recipe_locked
        self.override = False

        self.confirmed = False

        # Only the PREVIEW is subsampled: 10.7 s per update against 0.38 s.
        self.preview_step = max(1, max(self.image.shape) // 1024)

        # A share of the screen, capped in logical units so the cap travels.
        # One preview instead of four, so it gets the room they shared.
        desktop = wx.GetClientDisplayRect()
        self.preview_w = min(self.FromDIP(640), int(desktop.width * 0.42))
        # The box follows the image's own shape, so a landscape field does not
        # sit in a portrait frame with a blank band under it.
        rows, columns = self.image.shape[:2]
        self.preview_h = min(int(self.preview_w * rows / max(1, columns)),
                             int(desktop.height * 0.62))
        self.preview_size = (self.preview_w, self.preview_h)

        self._last_result = None
        # Guards the slider/box pair against echoing each other's updates.
        self._syncing = False
        self._build_ui()
        self._sync_controls()
        self._update_previews()

        self.GetSizer().Fit(self)
        self.SetSize(wx.Size(min(self.GetSize().width, desktop.width),
                             min(self.GetSize().height, desktop.height)))
        self.Center()
        wx.CallAfter(self._show_welcome_message)

    # ------------------------------------------------------------------- help
    def _show_welcome_message(self):
        """State what the mask is for, once per session."""
        global _WELCOME_SHOWN
        if _WELCOME_SHOWN:
            return
        _WELCOME_SHOWN = True

        dialog = wx.MessageDialog(
            self,
            "The metrics of this module are measured on background pixels "
            "only.\n\n"
            "Select a mask that excludes the sample. The metrics then read the "
            "original phase values of the pixels the mask leaves as background; "
            "the filters below choose that region and nothing else.",
            "Background Segmentation",
            wx.OK | wx.ICON_INFORMATION
        )
        dialog.ShowModal()
        dialog.Destroy()

    # --------------------------------------------------------------------- ui
    def _build_ui(self):
        outer = wx.BoxSizer(wx.HORIZONTAL)
        outer.Add(self._build_previews(), 1, wx.EXPAND | wx.ALL, 10)
        outer.Add(self._build_controls(), 0, wx.EXPAND | wx.TOP | wx.BOTTOM | wx.RIGHT, 10)
        self.SetSizer(outer)

        # ~0.3 s per update over 1000 slider steps, so events are coalesced.
        self._preview_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda evt: self._update_previews(),
                  self._preview_timer)

    def _build_previews(self):
        column = wx.BoxSizer(wx.VERTICAL)

        caption = wx.StaticText(self, label="Measured background")
        font = caption.GetFont()
        font.SetPointSize(font.GetPointSize() + 3)
        caption.SetFont(font.Bold())
        caption.SetForegroundColour(wx.Colour(45, 45, 45))
        column.Add(caption, 0, wx.ALIGN_CENTER | wx.TOP, 18)

        column.Add(self._build_legend(), 0, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 10)

        self._bitmap = wx.StaticBitmap(self)
        self._bitmap.SetMinSize((self.preview_w, self.preview_h))
        column.Add(self._bitmap, 1, wx.ALIGN_CENTER)
        return column

    def _build_legend(self):
        """Says what the two colors mean, where they are being looked at."""
        row = wx.BoxSizer(wx.HORIZONTAL)
        chip = self.FromDIP(wx.Size(13, 13))
        entries = ((self.SAMPLE_TINT, "Sample", "excluded"),
                   (self.BACKGROUND_SWATCH, "Background", "measured"))
        for index, (color, name, role) in enumerate(entries):
            if index:
                row.AddSpacer(self.FromDIP(26))
            row.Add(_Swatch(self, color, chip), 0,
                    wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 7)

            label = wx.StaticText(self, label=name)
            label.SetForegroundColour(wx.Colour(45, 45, 45))
            row.Add(label, 0, wx.ALIGN_CENTER_VERTICAL)

            # The role in lighter gray: it is the gloss, not the key.
            note = wx.StaticText(self, label="  " + role)
            note.SetForegroundColour(wx.Colour(130, 130, 130))
            row.Add(note, 0, wx.ALIGN_CENTER_VERTICAL)
        return row

    def _build_controls(self):
        column = wx.BoxSizer(wx.VERTICAL)
        width = self.FromDIP(300)

        if self.image_name:
            heading = wx.StaticText(self, label=self.image_name)
            headingFont = heading.GetFont()
            headingFont.SetPointSize(headingFont.GetPointSize() + 2)
            heading.SetFont(headingFont.Bold())
            heading.SetForegroundColour(wx.Colour(45, 45, 45))
            column.Add(heading, 0, wx.BOTTOM, 10)

        column.Add(self._build_batch_box(width), 0, wx.EXPAND | wx.BOTTOM, 8)
        column.Add(self._build_prepare_box(width), 0, wx.EXPAND | wx.BOTTOM, 8)
        column.Add(self._build_threshold_box(width), 0, wx.EXPAND | wx.BOTTOM, 8)
        column.Add(self._build_cleanup_box(width), 0, wx.EXPAND | wx.BOTTOM, 8)
        column.Add(self._build_readout(width), 0, wx.EXPAND | wx.BOTTOM, 12)

        # Right under the result they confirm, not pinned to the bottom of a
        # column whose height depends on which boxes happen to be shown.
        buttonRow = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_cancel = wx.Button(self, label="Cancel")
        self.btn_ok = wx.Button(self, label="OK")
        self.btn_ok.SetDefault()
        for button in (self.btn_cancel, self.btn_ok):
            button.SetMinSize(self.FromDIP(wx.Size(112, 38)))
            buttonFont = button.GetFont()
            buttonFont.SetPointSize(buttonFont.GetPointSize() + 1)
            button.SetFont(buttonFont)
        buttonRow.AddStretchSpacer()
        buttonRow.Add(self.btn_cancel, 0, wx.RIGHT, 10)
        buttonRow.Add(self.btn_ok, 0)
        column.Add(buttonRow, 0, wx.EXPAND)

        column.AddStretchSpacer()

        self.btn_ok.Bind(wx.EVT_BUTTON, self._on_ok)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        # The buttons carry their own ids, so SetDefault alone leaves Enter and
        # Escape with nothing to aim at while a spin box holds the focus.
        self.SetAffirmativeId(self.btn_ok.GetId())
        self.SetEscapeId(self.btn_cancel.GetId())
        return column

    def _box(self, label, width):
        box = wx.StaticBox(self, label=label)
        box.SetMinSize(wx.Size(width, -1))
        return box, wx.StaticBoxSizer(box, orient=wx.VERTICAL)

    def _build_batch_box(self, width):
        """Only shown from the second image on, when a recipe already exists."""
        box, sizer = self._box("Batch recipe", width)
        if not self.recipe_locked:
            box.Hide()
            return sizer

        summary = wx.StaticText(self, label=self.recipe.describe())
        summary.Wrap(width - 20)
        sizer.Add(summary, 0, wx.ALL, 5)

        self.check_override = wx.CheckBox(
            self, label="Change the recipe for this image")
        self.check_override.SetToolTip(
            "The rows of the table stop being directly comparable. The "
            "deviation is recorded with the result.")
        sizer.Add(self.check_override, 0, wx.ALL, 5)
        self.check_override.Bind(wx.EVT_CHECKBOX, self._on_override)
        return sizer

    def _build_prepare_box(self, width):
        box, sizer = self._box("1 - Prepare the image", width)

        self.spin_tophat = self._spin_row(
            sizer, "Remove background curvature", 0, 200,
            "Top-hat radius, in pixels. Set it a little larger than the "
            "biggest sample feature. This is the control that matters when a "
            "single threshold cannot fit the whole field.")
        self.spin_median = self._spin_row(
            sizer, "Smooth isolated noise", 0, 20,
            "Median radius, in pixels. Removes lone outlying pixels without "
            "moving the edges.")
        return sizer

    def _build_threshold_box(self, width):
        box, sizer = self._box("2 - Place the threshold", width)

        self.choice_method = wx.Choice(
            self, choices=[THRESHOLD_LABELS[name] for name in THRESHOLD_METHODS])
        sizer.Add(self.choice_method, 0, wx.EXPAND | wx.ALL, 4)
        # The hint each method carries lives in its tooltip, not on a line of
        # its own that changes height every time the choice changes.
        self.choice_method.SetToolTip(
            "\n".join("%s - %s" % (THRESHOLD_LABELS[name], THRESHOLD_HINTS[name])
                      for name in THRESHOLD_METHODS))
        self.choice_method.Bind(wx.EVT_CHOICE, self._on_method)

        header = wx.BoxSizer(wx.HORIZONTAL)
        self.slider_label = wx.StaticText(self, label="")
        header.Add(self.slider_label, 1, wx.ALIGN_CENTER_VERTICAL)
        # The level is a fraction, so the box steps in thousandths.
        self.threshold_entry = wx.SpinCtrlDouble(
            self, min=0.0, max=1.0, inc=0.001,
            size=self.FromDIP(wx.Size(96, -1)))
        self.threshold_entry.SetDigits(4)
        header.Add(self.threshold_entry, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        sizer.Add(header, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 4)

        # 1000 steps, so the threshold moves in increments of 0.001.
        self.slider = wx.Slider(self, value=500, minValue=0, maxValue=1000,
                                style=wx.SL_HORIZONTAL)
        sizer.Add(self.slider, 0, wx.EXPAND | wx.ALL, 4)
        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        self.threshold_entry.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_threshold_typed)
        self.threshold_entry.Bind(wx.EVT_KILL_FOCUS, self._on_threshold_typed)
        return sizer

    def _on_threshold_typed(self, event):
        """The level typed or stepped in the box: move the slider under it."""
        event.Skip()
        if self._syncing:
            return
        typed = float(self.threshold_entry.GetValue())

        automatic = self._automatic_level()
        if automatic is not None and abs(typed - automatic) < 5e-4:
            # This also fires on KILL_FOCUS, which happens whenever the box is
            # tabbed past: only a level that actually moved takes over from
            # the method. Half a step of the box is the smallest real move.
            return
        self._take_over_threshold()

        self._syncing = True
        self.recipe.manual_threshold = typed
        self.slider.SetValue(int(round(typed * 1000)))
        self.slider_label.SetLabel("Threshold")
        self._syncing = False
        self._schedule_preview()

    def _build_cleanup_box(self, width):
        box, sizer = self._box("3 - Clean the mask", width)
        self.spin_min_area = self._spin_row(
            sizer, "Discard objects under (px)", 1, 100000,
            "Specks smaller than this are not treated as sample.", log=True)
        self.spin_hole_area = self._spin_row(
            sizer, "Fill holes under (px)", 1, 100000,
            "Gaps inside a sample smaller than this are filled.", log=True)
        self.spin_guard = self._spin_row(
            sizer, "Guard band around sample (px)", 0, 50,
            "Grows the sample before the background is taken, so the halo "
            "around it stays out of the measurement. That halo is where "
            "compensation error concentrates, and it inflates STD and PV.")
        self.spin_border = self._spin_row(
            sizer, "Exclude border (px)", 0, 300,
            "Drops a margin all the way round the field. Reconstruction "
            "leaves its worst artefacts at the edges -- unwrapping seams, "
            "filter ringing, the corners of the spectrum -- and they are not "
            "background however flat the middle of the image looks.")

        self.check_invert = wx.CheckBox(
            self, label="Invert (background is the smaller region)")
        self.check_invert.SetToolTip(
            "The background is normally the larger of the two regions. Tick "
            "this for a field where the sample covers more than half of it.")
        sizer.Add(self.check_invert, 0, wx.ALL, 4)
        self.check_invert.Bind(wx.EVT_CHECKBOX, self._on_change)
        return sizer

    def _spin_row(self, sizer, label, low, high, tip, log=False):
        """A labeled slider with a numeric box beside the name.

        The box is the value: it can be typed into or stepped with the arrows,
        and the slider follows. The slider is for sweeping, the box for saying
        exactly what you meant.

        `log` spreads a range of several decades evenly along the track; a
        linear slider over 1..100000 px would spend its first half below 500."""
        header = wx.BoxSizer(wx.HORIZONTAL)
        caption = wx.StaticText(self, label=label)
        caption.SetToolTip(tip)
        entry = wx.SpinCtrl(self, min=low, max=high,
                            size=self.FromDIP(wx.Size(82, -1)))
        entry.SetToolTip(tip)
        header.Add(caption, 1, wx.ALIGN_CENTER_VERTICAL)
        header.Add(entry, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        sizer.Add(header, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 4)

        slider = wx.Slider(self, minValue=0, maxValue=1000 if log else high,
                           style=wx.SL_HORIZONTAL)
        slider.SetToolTip(tip)
        sizer.Add(slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        slider.range = (low, high)
        slider.log = log
        slider.entry = entry
        entry.slider = slider

        slider.Bind(wx.EVT_SLIDER, self._on_slider_moved)
        entry.Bind(wx.EVT_SPINCTRL, self._on_entry_typed)
        # A value typed without pressing the arrows only lands on focus loss.
        entry.Bind(wx.EVT_KILL_FOCUS, self._on_entry_typed)
        return slider

    def _on_slider_moved(self, event):
        """Slider dragged: carry the value into its box."""
        slider = event.GetEventObject()
        if not self._syncing:
            self._syncing = True
            slider.entry.SetValue(self._slider_value(slider))
            self._syncing = False
        self._on_change(event)

    def _on_entry_typed(self, event):
        """Box typed into or stepped: move the slider under it."""
        event.Skip()
        entry = event.GetEventObject()
        if not self._syncing:
            self._syncing = True
            self._place_slider(entry.slider, entry.GetValue())
            self._syncing = False
        self._on_change(event)

    @staticmethod
    def _slider_value(slider):
        """The slider's position as the value it stands for."""
        if not slider.log:
            return slider.GetValue()
        low, high = slider.range
        span = math.log10(high) - math.log10(max(1, low))
        return int(round(10 ** (math.log10(max(1, low))
                                + span * slider.GetValue() / 1000.0)))

    @staticmethod
    def _place_slider(slider, value):
        """Move the slider to where `value` sits, without touching the box."""
        low, high = slider.range
        value = int(max(low, min(high, value)))
        if slider.log:
            span = math.log10(high) - math.log10(max(1, low))
            position = 1000.0 * (math.log10(max(1, value))
                                 - math.log10(max(1, low))) / span
            slider.SetValue(int(round(position)))
        else:
            slider.SetValue(value)

    def _set_slider(self, slider, value):
        """Put `value` on both the slider and its box."""
        low, high = slider.range
        value = int(max(low, min(high, value)))
        self._syncing = True
        self._place_slider(slider, value)
        slider.entry.SetValue(value)
        self._syncing = False

    def _build_readout(self, width):
        box, sizer = self._box("Result", width)
        self.text_background = wx.StaticText(self, label="")
        self.text_sample = wx.StaticText(self, label="")
        self.text_notice = wx.StaticText(self, label="")
        self.text_notice.SetForegroundColour(wx.Colour(170, 110, 0))
        for widget in (self.text_background, self.text_sample, self.text_notice):
            sizer.Add(widget, 0, wx.ALL, 3)
        return sizer

    # ---------------------------------------------------------------- binding
    def _sync_controls(self):
        """Push the recipe into the widgets, and lock what the batch fixed."""
        self.check_invert.SetValue(self.recipe.invert)
        self._set_slider(self.spin_tophat, self.recipe.tophat)
        self._set_slider(self.spin_median, self.recipe.median)
        self._set_slider(self.spin_min_area, self.recipe.min_area)
        self._set_slider(self.spin_hole_area, self.recipe.hole_area)
        self._set_slider(self.spin_guard, self.recipe.guard)
        self._set_slider(self.spin_border, self.recipe.border)
        self.choice_method.SetSelection(THRESHOLD_METHODS.index(self.recipe.method))

        # Everything but the threshold belongs to the batch.
        editable = (not self.recipe_locked) or self.override
        self._recipe_editable = editable
        for widget in (self.spin_tophat, self.spin_median, self.spin_min_area,
                       self.spin_hole_area, self.spin_guard,
                       self.choice_method, self.check_invert):
            widget.Enable(editable)

        self._sync_slider()
        self.Layout()

    def _sync_slider(self):
        """The slider and the box carry whichever level is in force.

        With Manual that is the level set here. With Otsu or Triangle it is the
        one the method found for this image, shown rather than hidden -- and
        movable, because an automatic level is a starting point: the method
        reads one histogram, the operator looks at the mask it produces.
        Moving it hands the level over to Manual, see _take_over_threshold."""
        manual = self.recipe.method == "manual"
        # Manual stays live even under a locked batch recipe: the threshold is
        # the one thing meant to be tuned per image. Correcting an automatic
        # level changes the METHOD, which is the batch's to decide, so that
        # needs the recipe unlocked.
        live = manual or self._recipe_editable
        self.slider.Enable(live)
        self.threshold_entry.Enable(live)

        if manual:
            level = self.recipe.manual_threshold
            self.slider_label.SetLabel("Threshold")
        else:
            level = None if self._last_result is None else self._last_result["level"]
            self.slider_label.SetLabel(
                "Threshold  (%s, per image)" % THRESHOLD_LABELS[self.recipe.method])

        self._syncing = True
        if level is not None and np.isfinite(level):
            level = float(np.clip(level, 0, 1))
            self.slider.SetValue(int(round(level * 1000)))
            self.threshold_entry.SetValue(level)
        self._syncing = False

    def _take_over_threshold(self):
        """A level moved by hand is Manual, and the method box must say so.

        Otsu and Triangle each compute one number from the histogram. Leaving
        their name on a level the operator moved would write a recipe into the
        results that re-running the method would not reproduce."""
        if self.recipe.method == "manual" or not self._recipe_editable:
            return
        self.recipe.method = "manual"
        self.choice_method.SetSelection(THRESHOLD_METHODS.index("manual"))

    def _automatic_level(self):
        """The level the current automatic method landed on, if there is one."""
        if self.recipe.method == "manual" or self._last_result is None:
            return None
        level = self._last_result["level"]
        if level is None or not np.isfinite(level):
            return None
        return float(np.clip(level, 0, 1))

    def _collect(self):
        """Read the widgets back into the recipe."""
        self.recipe.invert = self.check_invert.GetValue()
        # The box, not the slider: on a logarithmic track the position rounds,
        # so a typed 813 would come back as 812.
        for slider, field in ((self.spin_tophat, "tophat"),
                              (self.spin_median, "median"),
                              (self.spin_min_area, "min_area"),
                              (self.spin_hole_area, "hole_area"),
                              (self.spin_guard, "guard"),
                              (self.spin_border, "border")):
            setattr(self.recipe, field, slider.entry.GetValue())
        self.recipe.method = THRESHOLD_METHODS[self.choice_method.GetSelection()]

    def _schedule_preview(self):
        """Ask for a refresh, coalescing bursts of events into one."""
        self._preview_timer.Start(self.PREVIEW_DELAY_MS, oneShot=True)

    def _on_change(self, event):
        self._collect()
        self._schedule_preview()

    def _on_method(self, event):
        previousMethod = self.recipe.method
        self._collect()
        if self.recipe.method == "manual" and previousMethod != "manual":
            # The slider was reporting the level Otsu or Triangle landed on,
            # and Manual is usually picked to nudge exactly that level. Start
            # from it rather than jumping back to the level set before, which
            # would mean reading the number off the screen and typing it in.
            self._seed_manual_from_last_level()
        self._sync_slider()
        self.Layout()
        self._schedule_preview()

    def _seed_manual_from_last_level(self):
        """Carry the automatic level into the manual one, when there is one."""
        if self._last_result is None:
            return
        level = self._last_result["level"]
        if level is None or not np.isfinite(level):
            return
        self.recipe.manual_threshold = float(np.clip(level, 0.0, 1.0))

    def _on_slider(self, event):
        """Follow the hand with the number; make the heavy part wait."""
        self._take_over_threshold()
        self.recipe.manual_threshold = float(self.slider.GetValue()) / 1000.0
        if not self._syncing:
            self._syncing = True
            self.threshold_entry.SetValue(self.recipe.manual_threshold)
            self.slider_label.SetLabel("Threshold")
            self._syncing = False
        self._schedule_preview()

    def _on_override(self, event):
        self.override = self.check_override.GetValue()
        self._sync_controls()

    def _on_ok(self, event):
        self._preview_timer.Stop()
        self._collect()
        self.confirmed = True
        self.EndModal(wx.ID_OK)

    def _on_cancel(self, event):
        """Close without a mask, which skips the metric."""
        self._preview_timer.Stop()
        self.confirmed = False
        self.EndModal(wx.ID_CANCEL)

    # ----------------------------------------------------------------- render
    def _overlay(self, normalized, sample):
        """The phase with the sample tinted: what stays gray gets measured.

        A bare binary mask cannot tell you whether the right side was kept.
        Drawn on the phase, picking the wrong region is immediately visible."""
        gray = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
        rgb = np.dstack([gray, gray, gray]).astype(np.uint16)
        for channel, tint in enumerate(self.SAMPLE_TINT):
            # Half the tint, half the phase, so texture shows through.
            rgb[..., channel][sample] = (rgb[..., channel][sample] + tint) // 2
        return rgb.astype(np.uint8)

    def _update_previews(self):
        """Recompute the previews and the counts beside them."""
        step = self.preview_step
        previewImage = self.image[::step, ::step] if step > 1 else self.image
        result = build_mask(previewImage, self.recipe.scaled(step))
        self._last_result = result

        self._bitmap.SetBitmap(numpy_to_wx_bitmap(
            self._overlay(result["normalized"], result["sample"]),
            self.preview_size, self))

        backgroundFraction = float(np.mean(result["background"]))
        sampleFraction = 1.0 - backgroundFraction
        self.text_background.SetLabel(
            "Background: %d px  (%.1f %%)"
            % (round(backgroundFraction * self.image.size), 100 * backgroundFraction))
        self.text_sample.SetLabel(
            "Sample:     %d px  (%.1f %%)"
            % (round(sampleFraction * self.image.size), 100 * sampleFraction))

        notices = []
        # The one case where "the background is the larger region" stops being
        # safe, and the only one that needs the Invert box.
        if result["ambiguous"]:
            notices.append("the two regions are of similar size - check that "
                           "the tinted one is the sample")
        if step > 1:
            notices.append("preview at 1/%d scale" % step)
        lost = self.recipe.invisible_in_preview(step)
        if lost:
            notices.append("too fine to show here: " + ", ".join(lost))
        if backgroundFraction < 0.01:
            notices.append("almost nothing left as background")
        elif sampleFraction < 0.0005:
            notices.append("almost nothing found as sample")
        self.text_notice.SetLabel("\n".join(notices))
        self.text_notice.Wrap(self.FromDIP(290))

        if self.recipe.method != "manual":
            self._sync_slider()
        self.Layout()

    def get_result(self):
        """Show the dialog; return the Recipe, or None if it was cancelled."""
        self.ShowModal()
        return self.recipe if self.confirmed else None


def create_background_mask(image: np.ndarray, recipe: Recipe = None,
                           recipe_locked: bool = False, image_name: str = "",
                           parent=None):
    """Ask for a segmentation recipe and build the background mask from it.

    Returns (background_mask, threshold, recipe); all three are None when the
    dialog is cancelled. The recipe comes back so the batch can keep it."""
    image = np.array(image, dtype=float)

    createdApp = False
    app = wx.GetApp()
    if app is None:
        app = wx.App(False)
        createdApp = True

    dialog = WxThresholdDialog(parent, image, recipe=recipe,
                               recipe_locked=recipe_locked,
                               image_name=image_name)
    chosen = dialog.get_result()
    dialog.Destroy()

    if createdApp:
        try:
            app.Destroy()
        except Exception:
            pass

    if chosen is None:
        return None, None, None

    # Rebuilt at full resolution: the preview was subsampled.
    result = build_mask(image, chosen)
    level = result["level"]
    if level is not None:
        # So the table reports the level actually used, not the slider's.
        chosen.manual_threshold = float(level)
    return result["background"], level, chosen
