import wx

class MetricSelectionDialog(wx.Dialog):
    """Dialog for selecting multiple metrics with checkboxes."""
    
    #: The list never grows past this, so a module with twenty metrics
    #: scrolls instead of running off the screen.
    MAX_LIST_HEIGHT = 430

    def __init__(self, parent, title="Select Metrics", metrics_config=None):
        # No fixed size: it is worked out from the number of metrics, so the
        # three-metric module does not open a window sized for twenty-one.
        super().__init__(parent, title=title)

        self.metrics_config = metrics_config or {}
        self.checkboxes = {}

        self._create_ui()
        self.Centre()
    
    def _create_ui(self):
        """Create the dialog UI with checkboxes."""
        panel = wx.Panel(self)
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        
        title = wx.StaticText(panel, label="Select the metrics you want to calculate:")
        titleFont = title.GetFont()
        titleFont.PointSize += 2
        titleFont = titleFont.Bold()
        title.SetFont(titleFont)
        mainSizer.Add(title, 0, wx.ALL, 10)
        
        scrollWindow = wx.ScrolledWindow(panel)
        scrollWindow.SetScrollRate(5, 5)
        scrollSizer = wx.BoxSizer(wx.VERTICAL)
        
        for metricId, metricInfo in self.metrics_config.items():
            checkbox = wx.CheckBox(scrollWindow, label=metricInfo['label'])
            self.checkboxes[metricId] = checkbox
            scrollSizer.Add(checkbox, 0, wx.ALL | wx.EXPAND, 5)
        
        scrollWindow.SetSizer(scrollSizer)
        # The list asks for the height its own metrics need, up to the cap.
        wanted = scrollSizer.GetMinSize().height
        # Also bounded by the screen, so the buttons cannot end up below the
        # bottom edge on a short display.
        cap = min(self.FromDIP(self.MAX_LIST_HEIGHT),
                  int(wx.GetClientDisplayRect().height * 0.55))
        scrollWindow.SetMinSize(wx.Size(self.FromDIP(410), min(wanted, cap)))
        mainSizer.Add(scrollWindow, 1, wx.ALL | wx.EXPAND, 10)
        
        buttonRow = wx.BoxSizer(wx.HORIZONTAL)
        
        selectAllButton = wx.Button(panel, label="Select All")
        deselectAllButton = wx.Button(panel, label="Deselect All")
        
        selectAllButton.Bind(wx.EVT_BUTTON, self.on_select_all)
        deselectAllButton.Bind(wx.EVT_BUTTON, self.on_deselect_all)
        
        buttonRow.Add(selectAllButton, 0, wx.ALL, 5)
        buttonRow.Add(deselectAllButton, 0, wx.ALL, 5)
        
        mainSizer.Add(buttonRow, 0, wx.ALIGN_CENTER)
        
        okCancelSizer = wx.StdDialogButtonSizer()
        okButton = wx.Button(panel, wx.ID_OK, "Apply")
        cancelButton = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        okCancelSizer.AddButton(okButton)
        okCancelSizer.AddButton(cancelButton)
        okCancelSizer.Realize()
        
        mainSizer.Add(okCancelSizer, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        panel.SetSizerAndFit(mainSizer)
        self.SetClientSize(panel.GetBestSize())
    
    def on_select_all(self, event):
        """Select all checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.SetValue(True)
    
    def on_deselect_all(self, event):
        """Deselect all checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.SetValue(False)
    
    def get_selected_metrics(self):
        """Return dictionary of selected metrics {metric_id: label}."""
        selectedMetrics = {}
        for metricId, checkbox in self.checkboxes.items():
            if checkbox.GetValue():
                selectedMetrics[metricId] = self.metrics_config[metricId]['label']
        return selectedMetrics


def show_metric_dialog(parent, title, metrics_config):
    """Show the metric checklist; return {metric_id: label}, or None if cancelled."""
    dialog = MetricSelectionDialog(parent, title, metrics_config)
    
    if dialog.ShowModal() == wx.ID_OK:
        selectedMetrics = dialog.get_selected_metrics()
        dialog.Destroy()
        return selectedMetrics
    
    dialog.Destroy()
    return None