import ctypes
import sys


def _enable_dpi_awareness():
    """
    Tell Windows this process draws at the real screen resolution.

    Without it the process is DPI-UNAWARE: on a 2880x1620 screen at 200% the
    app is drawn at 1440x810 and the system stretches the result, so text,
    icons and -- what matters here -- the phase images are shown at half the
    detail the monitor can display.

    Must run before the first window exists, hence before wx is imported.
    """
    if sys.platform != 'win32':
        return
    try:                                  # Windows 10 1703+
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:                              # Windows 8.1+
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass                          # older Windows: nothing to do


_enable_dpi_awareness()

import wx
from ui.main_frame import MainFrame

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MainFrame(None, title="Standardized Evaluation Protocol", size=wx.Window.FromDIP(wx.Size(1024, 768), None))
        self.frame.Show()
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
