import wx

def create(parent_frame):
    fileMenu = wx.Menu()
    fileMenu.Append(wx.ID_OPEN, "&Open\tCtrl+O", "Open Image")

    # A bundled sample lives in the application's own folder, so this is how a
    # copy of one gets out to a reconstruction program.
    parent_frame.save_image_id = wx.NewIdRef()
    fileMenu.Append(parent_frame.save_image_id, "Save &Image As...\tCtrl+Shift+S",
                    "Save a copy of the file shown in the current tab")
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_save_image,
                      id=parent_frame.save_image_id)

    # Ctrl+S saves the RESULTS, not the image on screen.
    fileMenu.Append(wx.ID_SAVE, "&Save Results\tCtrl+S", "Export the metric and mask tables to an Excel file")
    fileMenu.AppendSeparator()

    clearId = wx.NewIdRef()
    fileMenu.Append(clearId, "&Clear\tCtrl+L", "Close all images and metric table")
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_clear, id=clearId)

    fileMenu.AppendSeparator()
    fileMenu.Append(wx.ID_CLOSE_ALL, "&Exit\tCtrl+Q", "Exit Application")

    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_open, id=wx.ID_OPEN)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_save, id=wx.ID_SAVE)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_exit, id=wx.ID_CLOSE_ALL)

    return fileMenu
