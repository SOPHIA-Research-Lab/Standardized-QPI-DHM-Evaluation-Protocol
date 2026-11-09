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
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        
        instructionText = wx.StaticText(
            panel, 
            label="Select the images you want to analyze:"
        )
        instructionFont = instructionText.GetFont()
        instructionFont.PointSize += 1
        instructionFont = instructionFont.Bold()
        instructionText.SetFont(instructionFont)
        mainSizer.Add(instructionText, 0, wx.ALL | wx.EXPAND, 10)
        
        scrollWindow = wx.ScrolledWindow(panel, style=wx.VSCROLL)
        scrollWindow.SetScrollRate(0, 20)
        scrollSizer = wx.BoxSizer(wx.VERTICAL)
        
        self.checkboxes = []
        for index, imageName in enumerate(self.image_names):
            checkbox = wx.CheckBox(scrollWindow, label=f"{index+1}. {imageName}")
            checkbox.SetValue(True)
            self.checkboxes.append(checkbox)
            scrollSizer.Add(checkbox, 0, wx.ALL | wx.EXPAND, 5)
        
        scrollWindow.SetSizer(scrollSizer)
        mainSizer.Add(scrollWindow, 1, wx.ALL | wx.EXPAND, 10)
        
        selectionRow = wx.BoxSizer(wx.HORIZONTAL)
        
        selectAllButton = wx.Button(panel, label="Select All")
        deselectAllButton = wx.Button(panel, label="Deselect All")
        
        selectAllButton.Bind(wx.EVT_BUTTON, self.on_select_all)
        deselectAllButton.Bind(wx.EVT_BUTTON, self.on_deselect_all)
        
        selectionRow.Add(selectAllButton, 0, wx.ALL, 5)
        selectionRow.Add(deselectAllButton, 0, wx.ALL, 5)
        
        mainSizer.Add(selectionRow, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        self.info_label = wx.StaticText(panel, label="")
        self.update_info_label()
        mainSizer.Add(self.info_label, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        
        for checkbox in self.checkboxes:
            checkbox.Bind(wx.EVT_CHECKBOX, lambda e: self.update_info_label())
        
        buttonSizer = wx.StdDialogButtonSizer()
        okButton = wx.Button(panel, wx.ID_OK)
        cancelButton = wx.Button(panel, wx.ID_CANCEL)
        
        okButton.Bind(wx.EVT_BUTTON, self.on_ok)
        cancelButton.Bind(wx.EVT_BUTTON, self.on_cancel)
        
        buttonSizer.AddButton(okButton)
        buttonSizer.AddButton(cancelButton)
        buttonSizer.Realize()
        
        mainSizer.Add(buttonSizer, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        panel.SetSizer(mainSizer)
    
    def update_info_label(self):
        """Update the information label with selection count."""
        selectedCount = sum(1 for checkbox in self.checkboxes if checkbox.GetValue())
        totalCount = len(self.checkboxes)
        self.info_label.SetLabel(
            f"Selected: {selectedCount} of {totalCount} images"
        )
    
    def on_select_all(self, event):
        """Select all checkboxes."""
        for checkbox in self.checkboxes:
            checkbox.SetValue(True)
        self.update_info_label()
    
    def on_deselect_all(self, event):
        """Deselect all checkboxes."""
        for checkbox in self.checkboxes:
            checkbox.SetValue(False)
        self.update_info_label()
    
    def on_ok(self, event):
        """Store the ticked indices, refusing an empty selection."""
        self.selected_indices = [
            index for index, checkbox in enumerate(self.checkboxes) if checkbox.GetValue()
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
        """Drop the selection and close."""
        self.selected_indices = []
        self.EndModal(wx.ID_CANCEL)
    
    def get_selected_indices(self):
        """Return list of selected image indices."""
        return self.selected_indices


def show_image_selection_dialog(parent, notebook):
    """Ask which open images to analyse; return their indices, or None."""
    pageCount = notebook.GetPageCount()
    
    if pageCount == 0:
        wx.MessageBox("No images opened", "Error", wx.ICON_ERROR)
        return None
    
    # One image: nothing to choose.
    if pageCount == 1:
        return [0]

    imageNames = [notebook.GetPageText(index) for index in range(pageCount)]
    dialog = ImageSelectionDialog(parent, imageNames)
    
    if dialog.ShowModal() == wx.ID_OK:
        selectedIndices = dialog.get_selected_indices()
        dialog.Destroy()
        return selectedIndices
    else:
        dialog.Destroy()
        return None
