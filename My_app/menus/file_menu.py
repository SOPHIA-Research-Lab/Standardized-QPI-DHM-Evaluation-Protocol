import wx

def create(parent_frame, notebook):
    file_menu = wx.Menu()
    file_menu.Append(wx.ID_OPEN, "&Open\tCtrl+O", "Open Image")
    file_menu.Append(wx.ID_SAVE, "&Save\tCtrl+S", "Save Current Image")
    file_menu.AppendSeparator()

    # Nueva opción "Clear"
    clear_id = wx.NewIdRef()
    file_menu.Append(clear_id, "&Clear\tCtrl+L", "Close all images and metric table")
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_clear, id=clear_id)

    file_menu.AppendSeparator()
    file_menu.Append(wx.ID_CLOSE_ALL, "&Exit\tCtrl+Q", "Exit Application")

    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_open, id=wx.ID_OPEN)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_save, id=wx.ID_SAVE)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_exit, id=wx.ID_CLOSE_ALL)

    return file_menu
