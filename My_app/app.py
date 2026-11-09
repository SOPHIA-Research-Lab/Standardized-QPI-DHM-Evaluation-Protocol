import ctypes
import sys


def _enable_dpi_awareness():
    """Tell Windows this process draws at the real screen resolution."""
    # Unaware, the app is drawn at half the pixels and stretched back up.
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
        self.frame = MainFrame(None, title="Standardized Evaluation Protocol",
                               size=wx.Window.FromDIP(wx.Size(1024, 768), None))
        self.frame.Show()
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
