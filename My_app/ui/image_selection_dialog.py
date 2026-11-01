import wx
class ImageSelectionDialog(wx.Dialog):
    """Dialog for selecting multiple images from notebook tabs."""
    
    def __init__(self, parent, image_names, title="Select Images for Analysis"):
        super().__init__(parent, title=title,
                         size=wx.Window.FromDIP(wx.Size(450, 400), parent))
        
        self.image_names = image_names
        self.selected_indices = []
        
        self._create_ui()
        self.CenterOnParent()
    
    def _create_ui(self):
        """Create the dialog UI."""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Instruction label
        instruction = wx.StaticText(
            panel, 
            label="Select the images you want to analyze:"
        )
        instruction_font = instruction.GetFont()
        instruction_font.PointSize += 1
        instruction_font = instruction_font.Bold()
        instruction.SetFont(instruction_font)
        main_sizer.Add(instruction, 0, wx.ALL | wx.EXPAND, 10)
        
        # Scrolled window for checkboxes
        scroll = wx.ScrolledWindow(panel, style=wx.VSCROLL)
        scroll.SetScrollRate(0, 20)
        scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create checkboxes
        self.checkboxes = []
        for i, name in enumerate(self.image_names):
            cb = wx.CheckBox(scroll, label=f"{i+1}. {name}")
            cb.SetValue(True)  # Default: all selected
            self.checkboxes.append(cb)
            scroll_sizer.Add(cb, 0, wx.ALL | wx.EXPAND, 5)
        
        scroll.SetSizer(scroll_sizer)
        main_sizer.Add(scroll, 1, wx.ALL | wx.EXPAND, 10)
        
        # Selection buttons
        selection_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        select_all_btn = wx.Button(panel, label="Select All")
        deselect_all_btn = wx.Button(panel, label="Deselect All")
        
        select_all_btn.Bind(wx.EVT_BUTTON, self.on_select_all)
        deselect_all_btn.Bind(wx.EVT_BUTTON, self.on_deselect_all)
        
        selection_sizer.Add(select_all_btn, 0, wx.ALL, 5)
        selection_sizer.Add(deselect_all_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(selection_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        # Info label
        self.info_label = wx.StaticText(panel, label="")
        self.update_info_label()
        main_sizer.Add(self.info_label, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        
        # Bind checkbox events to update info
        for cb in self.checkboxes:
            cb.Bind(wx.EVT_CHECKBOX, lambda e: self.update_info_label())
        
        # OK/Cancel buttons
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(panel, wx.ID_OK)
        cancel_btn = wx.Button(panel, wx.ID_CANCEL)
        
        ok_btn.Bind(wx.EVT_BUTTON, self.on_ok)
        cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)
        
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(cancel_btn)
        btn_sizer.Realize()
        
        main_sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        panel.SetSizer(main_sizer)
    
    def update_info_label(self):
        """Update the information label with selection count."""
        selected_count = sum(1 for cb in self.checkboxes if cb.GetValue())
        total_count = len(self.checkboxes)
        self.info_label.SetLabel(
            f"Selected: {selected_count} of {total_count} images"
        )
    
    def on_select_all(self, event):
        """Select all checkboxes."""
        for cb in self.checkboxes:
            cb.SetValue(True)
        self.update_info_label()
    
    def on_deselect_all(self, event):
        """Deselect all checkboxes."""
        for cb in self.checkboxes:
            cb.SetValue(False)
        self.update_info_label()
    
    def on_ok(self, event):
        """Handle OK button - validate and store selected indices."""
        self.selected_indices = [
            i for i, cb in enumerate(self.checkboxes) if cb.GetValue()
        ]
        
        if not self.selected_indices:
            wx.MessageBox(
                "Please select at least one image to analyze.",
                "No Selection",
                wx.ICON_WARNING
            )
            return
        
        self.EndModal(wx.ID_OK)
    
    def on_cancel(self, event):
        """Handle Cancel button."""
        self.selected_indices = []
        self.EndModal(wx.ID_CANCEL)
    
    def get_selected_indices(self):
        """Return list of selected image indices."""
        return self.selected_indices


def show_image_selection_dialog(parent, notebook):
    """
    Show image selection dialog and return selected indices.
    
    Args:
        parent: Parent window
        notebook: ImageNotebook instance
    
    Returns:
        list: Indices of selected images, or None if cancelled
    """
    count = notebook.GetPageCount()
    
    if count == 0:
        wx.MessageBox("No images opened", "Error", wx.ICON_ERROR)
        return None
    
    if count == 1:
        # Only one image, no need for selection dialog
        return [0]
    
    # Get all image names
    image_names = [notebook.GetPageText(i) for i in range(count)]
    
    # Show dialog
    dialog = ImageSelectionDialog(parent, image_names)
    
    if dialog.ShowModal() == wx.ID_OK:
        selected = dialog.get_selected_indices()
        dialog.Destroy()
        return selected
    else:
        dialog.Destroy()
        return None
