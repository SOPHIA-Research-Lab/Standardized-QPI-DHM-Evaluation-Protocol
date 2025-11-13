import wx
from ui.metric_selection_dialog import show_metric_dialog, MetricSelectionDialog
from ui.image_selection_dialog import show_image_selection_dialog

def create(parent_frame, notebook):
    analysis_menu = wx.Menu()
    
    # Create menu items for each module
    parent_frame.module1_metrics_id = wx.NewIdRef()
    parent_frame.module2_metrics_id = wx.NewIdRef()
    parent_frame.module3_metrics_id = wx.NewIdRef()
    parent_frame.computational_complexity_id = wx.NewIdRef()
    
    analysis_menu.Append(parent_frame.module1_metrics_id, "Residual Background Phase Variance...")
    analysis_menu.Append(parent_frame.module2_metrics_id, "Global Phase Distortion Metrics...")
    analysis_menu.Append(parent_frame.module3_metrics_id, "Ground-Truth Comparisons...")
    analysis_menu.AppendSeparator()
    analysis_menu.Append(parent_frame.computational_complexity_id, "Computational Complexity")
    
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
        handlers = {
            # STD
            'std_unwrapped': parent_frame.on_std_background_unwrapped,
            'std_background': parent_frame.on_std_background,
            'std_zones': parent_frame.on_std_background_zones,
            
            # MAD
            'mad_unwrapped': parent_frame.on_mad_background_unwrapped,
            'mad_background': parent_frame.on_mad_background,
            'mad_zones': parent_frame.on_mad_background_zones,
            
            # RMS
            'rms_unwrapped': parent_frame.on_rms_background_unwrapped,
            'rms_background': parent_frame.on_rms_background,
            'rms_zones': parent_frame.on_rms_background_zones,
            
            # PV
            'pv_unwrapped': parent_frame.on_pv_background_unwrapped,
            'pv_background': parent_frame.on_pv_background,
            'pv_zones': parent_frame.on_pv_background_zones,
            
            # FWHM
            'fwhm_unwrapped': parent_frame.on_fwhm_background_unwrapped,
            'fwhm_background': parent_frame.on_fwhm_background,
            'fwhm_zones': parent_frame.on_fwhm_background_zones,
            
            # Entropy
            'entropy_unwrapped': parent_frame.on_entropy_background_unwrapped,
            'entropy_background': parent_frame.on_entropy_background,
            'entropy_zones': parent_frame.on_entropy_background_zones,
            
            # Legendre
            'legendre_unwrapped': parent_frame.on_legendre_background_unwrapped,
            'legendre_background': parent_frame.on_legendre_background,
            'legendre_zones': parent_frame.on_legendre_background_zones,
        }
        
        _process_selected_metrics(parent_frame, selected, handlers)


# ==========================
# Module 2: Global Phase Distortion Metrics
# ==========================

def _show_module2_dialog(parent_frame):
    """Show dialog with ALL Module 2 metrics."""
    config = {
        # Maximum-Minus-Minimum
        'mmm_global': {'label': 'Maximum-Minus-Minimum'},
        'mmm_unwrapped': {'label': 'Maximum-Minus-Minimum - Unwrapped'},
        
        # Global Phase Gradient
        'gradient_global': {'label': 'Phase Gradient'},
        'gradient_unwrapped': {'label': 'Phase Gradient - Unwrapped'},
        
        # TSM
        'tsm_global': {'label': 'TSM'},
        'tsm_unwrapped': {'label': 'TSM - Unwrapped'},
        
        # Phase Curvature
        'curvature_global': {'label': 'Phase Curvature'},
        'curvature_unwrapped': {'label': 'Phase Curvature - Unwrapped'},
        
        # Laplacian Energy
        'laplacian_global': {'label': 'Laplacian Energy'},
        'laplacian_unwrapped': {'label': 'Laplacian Energy - Unwrapped'},
        
        # Spatial Frequency
        'spatial_freq_global': {'label': 'Spatial Frequency'},
        'spatial_freq_unwrapped': {'label': 'Spatial Frequency - Unwrapped'},
        
        # Global Entropy
        'global_entropy': {'label': 'Global Entropy'},
        'global_entropy_unwrapped': {'label': 'Global Entropy - Unwrapped'},
        
        # Sharpness/Contrast
        'sharpness_global': {'label': 'GSM (Generalized Sharpness Metric)'},
        'sharpness_unwrapped': {'label': 'GSM (Generalized Sharpness Metric) - Unwrapped'},
    }
    
    selected = show_metric_dialog(parent_frame, "Global Phase Distortion Metrics - Select Metrics", config)
    
    if selected:
        handlers = {
            # MMM
            'mmm_global': parent_frame.on_mmm_global,
            'mmm_unwrapped': parent_frame.on_mmm_global_unwrapped,
            
            # Gradient
            'gradient_global': parent_frame.on_gradient_global,
            'gradient_unwrapped': parent_frame.on_gradient_global_unwrapped,
            
            # TSM
            'tsm_global': parent_frame.on_tsm_global,
            'tsm_unwrapped': parent_frame.on_tsm_global_unwrapped,
            
            # Curvature
            'curvature_global': parent_frame.on_curvature_global,
            'curvature_unwrapped': parent_frame.on_curvature_global_unwrapped,
            
            # Laplacian
            'laplacian_global': parent_frame.on_laplacian_global,
            'laplacian_unwrapped': parent_frame.on_laplacian_global_unwrapped,
            
            # Spatial Frequency
            'spatial_freq_global': parent_frame.on_spatial_freq_global,
            'spatial_freq_unwrapped': parent_frame.on_spatial_freq_global_unwrapped,
            
            # Global Entropy
            'global_entropy': parent_frame.on_global_entropy_global,
            'global_entropy_unwrapped': parent_frame.on_global_entropy_global_unwrapped,
            
            # Sharpness
            'sharpness_global': parent_frame.on_sharpness_global,
            'sharpness_unwrapped': parent_frame.on_sharpness_global_unwrapped,
        }
        
        _process_selected_metrics(parent_frame, selected, handlers)


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
        'ssim_unwrapped': {'label': 'Structural Similarity (SSIM) - Unwrapped'},
        
        # MSE
        'mse': {'label': 'Mean Squared Error (MSE)'},
        'mse_unwrapped': {'label': 'Mean Squared Error (MSE) - Unwrapped'},
        
        # PSNR
        'psnr': {'label': 'Peak Signal-to-Noise Ratio (PSNR)'},
        'psnr_unwrapped': {'label': 'Peak Signal-to-Noise Ratio (PSNR) - Unwrapped'},
    }
    
    selected = show_metric_dialog(parent_frame, "Ground-Truth Comparisons - Select Metrics", config)
    
    if selected:
        handlers = {
            # SSIM
            'ssim': parent_frame.on_ssim_comparison,
            'ssim_unwrapped': parent_frame.on_ssim_comparison_unwrapped,
            
            # MSE
            'mse': parent_frame.on_mse_comparison,
            'mse_unwrapped': parent_frame.on_mse_comparison_unwrapped,
            
            # PSNR
            'psnr': parent_frame.on_psnr_comparison,
            'psnr_unwrapped': parent_frame.on_psnr_comparison_unwrapped,
        }
        
        _process_selected_metrics(parent_frame, selected, handlers)


# ==========================
# Helper function
# ==========================

def process_selected_metrics(parent_frame, module_metrics_config):
    """Process multiple selected metrics."""
    # Mostrar diálogo de selección de métricas
    dlg = MetricSelectionDialog(parent_frame, metrics_config=module_metrics_config)
    
    if dlg.ShowModal() == wx.ID_OK:
        selected_metrics = dlg.get_selected_metrics()
        
        if not selected_metrics:
            wx.MessageBox("No metrics selected", "Info", wx.ICON_INFORMATION)
            dlg.Destroy()
            return
        
        # Mostrar diálogo de selección de imágenes UNA SOLA VEZ
        selected_indices = show_image_selection_dialog(parent_frame, parent_frame.notebook)
        
        if selected_indices is None or len(selected_indices) == 0:
            dlg.Destroy()
            return
        
        # Calcular todas las métricas para todas las imágenes
        parent_frame._calculate_multiple_metrics(selected_metrics, selected_indices)
    
    dlg.Destroy()

# ==========================
# Helper function
# ==========================

def _process_selected_metrics(parent_frame, selected_metrics, handlers):
    """
    Process selected metrics using the new cached multi-metric system.
    
    Args:
        parent_frame: MainFrame instance
        selected_metrics: Dict of {metric_id: label_string}
        handlers: Dict of {metric_id: handler_function} (DEPRECATED for cached system)
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
