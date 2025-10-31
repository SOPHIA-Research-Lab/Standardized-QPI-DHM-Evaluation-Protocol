import wx
from ui.main_frame import MainFrame

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MainFrame(None, title="Standardized Evaluation Protocol", size=(1024, 768))
        self.frame.Show()
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
