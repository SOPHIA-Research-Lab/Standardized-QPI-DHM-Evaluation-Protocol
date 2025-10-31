import wx

def create(parent_frame, notebook=None):
    edit_menu = wx.Menu()

    undo_item = edit_menu.Append(wx.ID_UNDO, "Undo\tCtrl+Z")
    copy_item = edit_menu.Append(wx.ID_COPY, "Copy\tCtrl+C")
    paste_item = edit_menu.Append(wx.ID_PASTE, "Paste\tCtrl+V")
    edit_menu.AppendSeparator()
    clear_item = edit_menu.Append(wx.ID_ANY, "Clear")

    # Bind con lambdas inline
    parent_frame.Bind(wx.EVT_MENU, lambda evt: wx.MessageBox("Undo"), undo_item)
    parent_frame.Bind(wx.EVT_MENU, lambda evt: wx.MessageBox("Copy"), copy_item)
    parent_frame.Bind(wx.EVT_MENU, lambda evt: wx.MessageBox("Paste"), paste_item)
    parent_frame.Bind(wx.EVT_MENU, lambda evt: wx.MessageBox("Clear"), clear_item)

    return edit_menu