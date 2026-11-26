"""Ask the layout of a headerless .raw / .bin file before reading it."""
import os

import wx

from core import raw_format

_DTYPE_LABELS = [
    ('float32', "32-bit float  (phase in radians)"),
    ('float64', "64-bit float  (phase in radians)"),
    ('uint8', "8-bit unsigned  (gray levels, mapped onto [-pi, pi])"),
]

OK_INK = wx.Colour(30, 120, 60)
BAD_INK = wx.Colour(190, 40, 40)


class RawLayoutDialog(wx.Dialog):
    """Width, height, type, byte order and header: what the file does not say.

    Prefilled from the file name, and OK stays disabled until the layout
    accounts for every byte of the file, so a wrong guess cannot be read into
    a scrambled image that still looks like one."""

    def __init__(self, parent, path):
        super().__init__(parent, title="Open RAW / BIN - %s" % os.path.basename(path))
        self.path = path
        self.file_size = os.path.getsize(path)
        guess = raw_format.guess_layout(path)

        panel = self
        grid = wx.FlexGridSizer(cols=2, vgap=8, hgap=10)
        grid.AddGrowableCol(1)

        def spin(value, maximum=1 << 20):
            control = wx.SpinCtrl(panel, min=0, max=maximum, initial=value,
                                  size=self.FromDIP(wx.Size(110, -1)))
            control.Bind(wx.EVT_SPINCTRL, self._on_change)
            control.Bind(wx.EVT_TEXT, self._on_change)
            return control

        self.width = spin(guess['width'])
        self.height = spin(guess['height'])
        self.width.Bind(wx.EVT_SPINCTRL, self._on_width)
        self.width.Bind(wx.EVT_TEXT, self._on_width)

        self.dtype = wx.Choice(panel, choices=[label for _, label in _DTYPE_LABELS])
        self.dtype.SetSelection([name for name, _ in _DTYPE_LABELS].index(guess['dtype']))
        self.dtype.Bind(wx.EVT_CHOICE, self._on_change)

        self.byte_order = wx.Choice(panel, choices=["Little-endian (Windows, NumPy, MATLAB)",
                                                    "Big-endian"])
        self.byte_order.SetSelection(0 if guess['little_endian'] else 1)
        self.byte_order.Bind(wx.EVT_CHOICE, self._on_change)

        self.offset = spin(guess['offset'], maximum=1 << 30)

        for label, control in (("Width (px)", self.width),
                               ("Height (px)", self.height),
                               ("Data type", self.dtype),
                               ("Byte order", self.byte_order),
                               ("Header to skip (bytes)", self.offset)):
            grid.Add(wx.StaticText(panel, label=label), 0, wx.ALIGN_CENTER_VERTICAL)
            grid.Add(control, 1, wx.EXPAND)

        self.status = wx.StaticText(panel, label="")

        buttons = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.ok_button = self.FindWindowById(wx.ID_OK)

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(wx.StaticText(
            panel, label="A .raw/.bin file holds only the numbers. Say how they are laid out:"),
            0, wx.ALL, 12)
        mainSizer.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 12)
        mainSizer.Add(self.status, 0, wx.ALL, 12)
        mainSizer.Add(buttons, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        self.SetSizerAndFit(mainSizer)
        self.CenterOnParent()
        self._refresh()

    # ------------------------------------------------------------- values
    def layout(self):
        """The chosen layout, in the keywords raw_format.load_raw takes."""
        return {
            'width': self.width.GetValue(),
            'height': self.height.GetValue(),
            'dtype_name': _DTYPE_LABELS[self.dtype.GetSelection()][0],
            'little_endian': self.byte_order.GetSelection() == 0,
            'offset': self.offset.GetValue(),
        }

    # ------------------------------------------------------------- events
    def _on_width(self, event):
        """Fill the height the width implies, when the bytes divide evenly."""
        layout = self.layout()
        itemSize = raw_format.expected_size(1, 1, layout['dtype_name'])
        payload = self.file_size - layout['offset']
        width = layout['width']
        if width > 0 and payload > 0 and payload % (width * itemSize) == 0:
            self.height.SetValue(payload // (width * itemSize))
        self._refresh()
        event.Skip()

    def _on_change(self, event):
        self._refresh()
        event.Skip()

    def _refresh(self):
        layout = self.layout()
        needed = raw_format.expected_size(layout['width'], layout['height'],
                                          layout['dtype_name'], layout['offset'])
        matches = (layout['width'] > 0 and layout['height'] > 0
                   and needed == self.file_size)
        if matches:
            self.status.SetLabel("File: %d bytes - layout accounts for all of them." % self.file_size)
            self.status.SetForegroundColour(OK_INK)
        else:
            self.status.SetLabel("File: %d bytes - this layout needs %d." % (self.file_size, needed))
            self.status.SetForegroundColour(BAD_INK)
        self.ok_button.Enable(matches)
        self.Layout()


def ask_raw_layout(parent, path):
    """The layout chosen for `path`, or None if the dialog was cancelled."""
    dialog = RawLayoutDialog(parent, path)
    try:
        if dialog.ShowModal() != wx.ID_OK:
            return None
        return dialog.layout()
    finally:
        dialog.Destroy()
