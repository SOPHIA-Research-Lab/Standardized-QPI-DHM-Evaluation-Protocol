import wx

class MetricSelectionDialog(wx.Dialog):
    """Dialog for selecting multiple metrics with checkboxes."""
    
    def __init__(self, parent, title="Select Metrics", metrics_config=None):
        super().__init__(parent, title=title, size=(450, 500))
        
        self.metrics_config = metrics_config or {}
        self.checkboxes = {}
        
        self._create_ui()
        self.Centre()
    
    def _create_ui(self):
        """Create the dialog UI with checkboxes."""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="Select the metrics you want to calculate:")
        title_font = title.GetFont()
        title_font.PointSize += 2
        title_font = title_font.Bold()
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL, 10)
        
        # Scrolled window for checkboxes
        scroll = wx.ScrolledWindow(panel)
        scroll.SetScrollRate(5, 5)
        scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create checkboxes for each metric
        for metric_id, metric_info in self.metrics_config.items():
            cb = wx.CheckBox(scroll, label=metric_info['label'])
            self.checkboxes[metric_id] = cb
            scroll_sizer.Add(cb, 0, wx.ALL | wx.EXPAND, 5)
        
        scroll.SetSizer(scroll_sizer)
        main_sizer.Add(scroll, 1, wx.ALL | wx.EXPAND, 10)
        
        # Select All / Deselect All buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        select_all_btn = wx.Button(panel, label="Select All")
        deselect_all_btn = wx.Button(panel, label="Deselect All")
        
        select_all_btn.Bind(wx.EVT_BUTTON, self.on_select_all)
        deselect_all_btn.Bind(wx.EVT_BUTTON, self.on_deselect_all)
        
        button_sizer.Add(select_all_btn, 0, wx.ALL, 5)
        button_sizer.Add(deselect_all_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER)
        
        # OK and Cancel buttons
        ok_cancel_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(panel, wx.ID_OK, "Apply")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        ok_cancel_sizer.AddButton(ok_btn)
        ok_cancel_sizer.AddButton(cancel_btn)
        ok_cancel_sizer.Realize()
        
        main_sizer.Add(ok_cancel_sizer, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        panel.SetSizer(main_sizer)
    
    def on_select_all(self, event):
        """Select all checkboxes."""
        for cb in self.checkboxes.values():
            cb.SetValue(True)
    
    def on_deselect_all(self, event):
        """Deselect all checkboxes."""
        for cb in self.checkboxes.values():
            cb.SetValue(False)
    
    def get_selected_metrics(self):
        """Return dictionary of selected metrics {metric_id: label}."""
        selected = {}
        for metric_id, cb in self.checkboxes.items():
            if cb.GetValue():
                selected[metric_id] = self.metrics_config[metric_id]['label']
        return selected


def show_metric_dialog(parent, title, metrics_config):
    """
    Show metric selection dialog and return selected metrics.
    
    Args:
        parent: Parent window
        title: Dialog title
        metrics_config: Dict with format {metric_id: {'label': 'Display Name', 'handler': function}}
    
    Returns:
        Dictionary of selected metrics or None if cancelled
    """
    dialog = MetricSelectionDialog(parent, title, metrics_config)
    
    if dialog.ShowModal() == wx.ID_OK:
        selected = dialog.get_selected_metrics()
        dialog.Destroy()
        return selected
    
    dialog.Destroy()
    return None