import wx
import numpy as np
class ImageSelectionDialog(wx.Dialog):
    """Dialog for selecting multiple images from notebook tabs."""
    
    def __init__(self, parent, image_names, title="Select Images for Analysis"):
        super().__init__(parent, title=title, size=(450, 400))
        
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


# ============================================================================
# Integration helper functions for MainFrame
# ============================================================================

def process_metric_for_selected_images(
    main_frame, 
    metric_func, 
    metric_name, 
    process_type="simple",
    **kwargs
):
    """
    Process a metric for user-selected images.
    
    Args:
        main_frame: MainFrame instance
        metric_func: Function to calculate the metric
        metric_name: Name of the metric for display
        process_type: Type of processing ("simple", "unwrap", "zones", "legendre")
        **kwargs: Additional arguments for the metric function
    
    Returns:
        dict: Results for each processed image
    """
    from analysis.module1 import utilitiesRBPV as utRBPV
    from analysis.module1 import residual_background as rb
    
    # Get selected image indices
    selected_indices = show_image_selection_dialog(main_frame, main_frame.notebook)
    
    if selected_indices is None:
        return None
    
    results = {}
    
    # Progress dialog
    progress = wx.ProgressDialog(
        "Processing Images",
        f"Calculating {metric_name}...",
        maximum=len(selected_indices),
        parent=main_frame,
        style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
    )
    
    try:
        for idx, img_idx in enumerate(selected_indices):
            pil_img = main_frame.notebook.images[img_idx] if hasattr(main_frame.notebook, 'images') else None
            
            if pil_img is None:
                panel = main_frame.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
            
            if pil_img is None:
                continue
            
            name = main_frame.notebook.GetPageText(img_idx)
            page = main_frame.notebook.GetPage(img_idx)
            
            progress.Update(idx, f"Processing {name}...")
            
            # Convert to phase
            if hasattr(page, 'is_mat_complex') and page.is_mat_complex and hasattr(page, 'phase'):
                phase_img = page.phase
            else:
                grayscale = pil_img.convert("L")
                img_array = np.array(grayscale, dtype=float)
                phase_img = utRBPV.grayscaleToPhase(img_array)
            
            # Process based on type
            if process_type == "unwrap":
                unwrapped = rb.unwrap_with_scikit(phase_img)
                background_mask, background_values, threshold = utRBPV.create_background_mask(
                    unwrapped, method='otsu', parent=main_frame
                )
                value = metric_func(phase_img, background_mask, manual=False, num_zones=2)
                
                main_frame._update_tables_with_mask_data(
                    name, phase_img, background_mask, background_values, threshold
                )
                main_frame.update_table_with_zones(name, f"{metric_name}_Unwrapped_Background", value)
                
            elif process_type == "simple":
                background_mask, background_values, threshold = utRBPV.create_background_mask(
                    phase_img, method='otsu', parent=main_frame
                )
                value = metric_func(phase_img, background_mask, manual=False, num_zones=2)
                
                main_frame._update_tables_with_mask_data(
                    name, phase_img, background_mask, background_values, threshold
                )
                main_frame.update_table_with_zones(name, f"{metric_name}_Background", value)
                
            elif process_type == "zones":
                num_zones = kwargs.get('num_zones', 2)
                result = metric_func(phase_img, mask=None, manual=True, num_zones=num_zones)
                main_frame._handle_zone_analysis_result(name, f"{metric_name}_Background_Zones", result, num_zones)
            
            results[name] = value if 'value' in locals() else result
        
        progress.Update(len(selected_indices))
        
    finally:
        progress.Destroy()
    
    # Show summary
    summary = f"Processed {len(results)} images:\n\n"
    for img_name, val in results.items():
        if isinstance(val, tuple):
            val = val[0]
        summary += f"{img_name}: {val:.6f}\n"
    
    wx.MessageBox(summary, f"{metric_name} - Analysis Complete", wx.ICON_INFORMATION)
    
    return results


