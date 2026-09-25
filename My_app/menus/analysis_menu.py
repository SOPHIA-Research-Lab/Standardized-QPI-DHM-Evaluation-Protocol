import wx
from ui.metric_selection_dialog import show_metric_dialog
from ui.image_selection_dialog import show_image_selection_dialog

def create(parent_frame, notebook):
    analysis_menu = wx.Menu()
    
    # Create menu items for each module
    parent_frame.module1_metrics_id = wx.NewIdRef()
    parent_frame.module2_metrics_id = wx.NewIdRef()
    parent_frame.module3_metrics_id = wx.NewIdRef()
    parent_frame.computational_complexity_id = wx.NewIdRef()
    
    analysis_menu.Append(parent_frame.module1_metrics_id, "Background metrics...")
    analysis_menu.Append(parent_frame.module2_metrics_id, "Global metrics...")
    analysis_menu.Append(parent_frame.module3_metrics_id, "Ground-truth comparison...")
    analysis_menu.AppendSeparator()
    analysis_menu.Append(parent_frame.computational_complexity_id, "Computational complexity")
    
    # Bind events
    parent_frame.Bind(wx.EVT_MENU, lambda evt: _show_module1_dialog(parent_frame), id=parent_frame.module1_metrics_id)
    parent_frame.Bind(wx.EVT_MENU, lambda evt: _show_module2_dialog(parent_frame), id=parent_frame.module2_metrics_id)
    parent_frame.Bind(wx.EVT_MENU, lambda evt: _show_module3_dialog(parent_frame), id=parent_frame.module3_metrics_id)
    parent_frame.Bind(wx.EVT_MENU, parent_frame.on_computational_complexity, id=parent_frame.computational_complexity_id)
    
    return analysis_menu


# ==========================
# Module 1: Residual Background Phase Variance
# ==========================

def _show_module1_dialog(parent_frame):
    """Show dialog with ALL Module 1 metrics."""
    config = {
        # STD metrics
        'std_unwrapped': {'label': 'STD - Unwrapped Background'},
        'std_background': {'label': 'STD - Background'},
        'std_zones': {'label': 'STD - Background Zones'},
        
        # MAD metrics
        'mad_unwrapped': {'label': 'MAD - Unwrapped Background'},
        'mad_background': {'label': 'MAD - Background'},
        'mad_zones': {'label': 'MAD - Background Zones'},
        
        # RMS metrics
        'rms_unwrapped': {'label': 'RMS - Unwrapped Background'},
        'rms_background': {'label': 'RMS - Background'},
        'rms_zones': {'label': 'RMS - Background Zones'},
        
        # PV metrics
        'pv_unwrapped': {'label': 'PV - Unwrapped Background'},
        'pv_background': {'label': 'PV - Background'},
        'pv_zones': {'label': 'PV - Background Zones'},
        
        # FWHM metrics
        'fwhm_unwrapped': {'label': 'FWHM - Unwrapped Background'},
        'fwhm_background': {'label': 'FWHM - Background'},
        'fwhm_zones': {'label': 'FWHM - Background Zones'},
        
        # Entropy metrics
        'entropy_unwrapped': {'label': 'Entropy - Unwrapped Background'},
        'entropy_background': {'label': 'Entropy - Background'},
        'entropy_zones': {'label': 'Entropy - Background Zones'},
        
        # Legendre metrics
        'legendre_unwrapped': {'label': 'Legendre - Unwrapped Background'},
        'legendre_background': {'label': 'Legendre - Background'},
        'legendre_zones': {'label': 'Legendre - Background Zones'},
    }
    
    selected = show_metric_dialog(parent_frame, "Residual Background Phase Variance - Select Metrics", config)
    
    if selected:
        _process_selected_metrics(parent_frame, selected)


# ==========================
# Module 2: Global Phase Distortion Metrics
# ==========================

def _show_module2_dialog(parent_frame):
    """Show dialog with ALL Module 2 metrics."""
    config = {
        # Maximum-Minus-Minimum
        'mmm_global': {'label': 'PV - Global'},
        'mmm_unwrapped': {'label': 'PV - Unwrapped'},
        
        # Global Phase Gradient
        'gradient_global': {'label': 'GPG - Global'},
        'gradient_unwrapped': {'label': 'GPG - Unwrapped'},
        
        # TSM
        'tsm_global': {'label': 'TSM - Global'},
        'tsm_unwrapped': {'label': 'TSM - Unwrapped'},
        
        # # Phase Curvature
        # 'curvature_global': {'label': 'Phase Curvature'},
        # 'curvature_unwrapped': {'label': 'Phase Curvature - Unwrapped'},
        
        # Laplacian Energy
        'laplacian_global': {'label': 'LE - Global'},
        'laplacian_unwrapped': {'label': 'LE - Unwrapped'},
        
        # Spatial Frequency
        'spatial_freq_global': {'label': 'SF - Global'},
        'spatial_freq_unwrapped': {'label': 'SF - Unwrapped'},
        
        # Global Entropy
        'global_entropy': {'label': 'Entropy - Global'},
        'global_entropy_unwrapped': {'label': 'Entropy - Unwrapped'},
        
        # Sharpness/Contrast
        'sharpness_global': {'label': 'GSM - Global'},
        'sharpness_unwrapped': {'label': 'GSM - Unwrapped'},

        # FWHM/Resolution
        'fwhm_global': {'label': 'FWHM - Global'},
        'fwhm_global_unwrapped': {'label': 'FWHM - Unwrapped'},
    }
    
    selected = show_metric_dialog(parent_frame, "Global Phase Distortion Metrics - Select Metrics", config)
    
    if selected:
        _process_selected_metrics(parent_frame, selected)


# ==========================
# Module 3: Ground-Truth Comparisons
# ==========================

def _show_module3_dialog(parent_frame):
    """Show dialog with ALL Module 3 metrics."""
    # First check if ground truth is loaded
    if not hasattr(parent_frame, 'ground_truth_data') or parent_frame.ground_truth_data is None:
        response = wx.MessageBox(
            "No ground-truth image loaded.\n\nDo you want to load one now?",
            "Ground-Truth Required",
            wx.YES_NO | wx.ICON_QUESTION
        )
        
        if response == wx.YES:
            parent_frame.on_load_ground_truth(None)
            
            # Check again if it was loaded
            if not hasattr(parent_frame, 'ground_truth_data') or parent_frame.ground_truth_data is None:
                return  # User cancelled or failed to load
        else:
            return  # User chose not to load
    
    config = {
        # SSIM
        'ssim': {'label': 'Structural Similarity (SSIM)'},
        
        # MSE
        'mse': {'label': 'Mean Squared Error (MSE)'},
        
        # PSNR
        'psnr': {'label': 'Peak Signal-to-Noise Ratio (PSNR)'},
    }
    
    selected = show_metric_dialog(parent_frame, "Ground-Truth Comparisons - Select Metrics", config)
    
    if selected:
        _process_selected_metrics(parent_frame, selected)


# ==========================
# Helper function
# ==========================

def _process_selected_metrics(parent_frame, selected_metrics):
    """
    Process selected metrics using the cached multi-metric system.

    Args:
        parent_frame: MainFrame instance
        selected_metrics: Dict of {metric_id: label_string}
    """
    if not selected_metrics:
        wx.MessageBox("No metrics selected", "Info", wx.ICON_INFORMATION)
        return
    
    # Show image selection dialog ONCE
    selected_indices = show_image_selection_dialog(parent_frame, parent_frame.notebook)
    
    if selected_indices is None or len(selected_indices) == 0:
        return  # User cancelled
    
    # Process ALL metrics together (including zones) using the cache system
    # The cache system will handle zones intelligently (shared or per-image)
    parent_frame._calculate_multiple_metrics(selected_metrics, selected_indices)
