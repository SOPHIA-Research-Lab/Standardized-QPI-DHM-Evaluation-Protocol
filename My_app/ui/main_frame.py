import wx
import wx.aui
import wx.lib.dialogs
import numpy as np
from PIL import Image
import pandas as pd
from core import file_manager
from menus import file_menu, edit_menu, analysis_menu, help_menu, samples_menu
from detached_notebook import ImageNotebook
from analysis.module1 import utilitiesRBPV as utRBPV
from analysis.module1 import residual_background as rb
from analysis.module2 import global_phase as gp
from analysis.module3 import ground_truth_comparison as gtc
import matplotlib.pyplot as plt
import subprocess
import sys
import os
from ui.image_selection_dialog import show_image_selection_dialog
from ui.metric_selection_dialog import MetricSelectionDialog

class MainFrame(wx.Frame):
# Main application frame for image analysis.
    
    def __init__(self, parent, title, size):
        super().__init__(parent, title=title, size=size)
        self._initialize_components()
        self._setup_menus()
        self._setup_layout()
        self._bind_events()
        self.Show()

    def _initialize_components(self):
# Initialize main components and state variables.
        self.notebook = ImageNotebook(self)
        self.metric_table = None
        self.mask_table = None
        self.is_sample_image = False
        self.ground_truth_path = None
        self.ground_truth_data = None
        self._zones_cache = {} 
        self._shared_zones = None  
        self.CreateStatusBar()
        
        # Metric IDs to functions mapping
        self.metric_map = {
            # ===== MODULE 1: Residual Background Phase Variance =====
            # STD metrics
            'std_unwrapped': (rb.std_background, 'STD_Unwrapped_Background', True),
            'std_background': (rb.std_background, 'STD_Background', False),
            'std_zones': (rb.std_background, 'STD_Zones', False),
            
            # MAD metrics
            'mad_unwrapped': (rb.mean_absolute_deviation_background, 'MAD_Unwrapped_Background', True),
            'mad_background': (rb.mean_absolute_deviation_background, 'MAD_Background', False),
            'mad_zones': (rb.mean_absolute_deviation_background, 'MAD_Zones', False),
            
            # RMS metrics
            'rms_unwrapped': (rb.rms_background, 'RMS_Unwrapped_Background', True),
            'rms_background': (rb.rms_background, 'RMS_Background', False),
            'rms_zones': (rb.rms_background, 'RMS_Zones', False),
            
            # PV metrics
            'pv_unwrapped': (rb.pv_background, 'PV_Unwrapped_Background', True),
            'pv_background': (rb.pv_background, 'PV_Background', False),
            'pv_zones': (rb.pv_background, 'PV_Zones', False),
            
            # FWHM metrics
            'fwhm_unwrapped': (rb.fwhm_background, 'FWHM_Unwrapped_Background', True),
            'fwhm_background': (rb.fwhm_background, 'FWHM_Background', False),
            'fwhm_zones': (rb.fwhm_background, 'FWHM_Zones', False),
            
            # Entropy metrics
            'entropy_unwrapped': (rb.entropy_background, 'Entropy_Unwrapped_Background', True),
            'entropy_background': (rb.entropy_background, 'Entropy_Background', False),
            'entropy_zones': (rb.entropy_background, 'Entropy_Zones', False),
            
            # Legendre metrics 
            'legendre_background': (rb.legendre_background, 'Legendre-Background', False),
            'legendre_unwrapped': (rb.legendre_background, 'Legendre-UnwrappedBackground', True),
            'legendre_zones': (rb.legendre_background, 'Legendre_Zones', False),

            
            # ===== MODULE 2: Global Phase Distortion Metrics =====
            
            # Maximum-Minus-Minimum
            'mmm_global': (gp.maximum_minus_minimum, 'MMM_Global', False),
            'mmm_unwrapped': (gp.maximum_minus_minimum, 'MMM_Unwrapped', True),
            
            # Phase Gradient
            'gradient_global': (gp.global_phase_gradient, 'Gradient_Global', False),
            'gradient_unwrapped': (gp.global_phase_gradient, 'Gradient_Unwrapped', True),
            
            # TSM
            'tsm_global': (gp.tsm_global, 'TSM_Global', False),
            'tsm_unwrapped': (gp.tsm_global, 'TSM_Unwrapped', True),
            
            # Phase Curvature
            'curvature_global': (gp.reconstruction_background, 'Curvature_Global', False),
            'curvature_unwrapped': (gp.reconstruction_background, 'Curvature_Unwrapped', True),
            
            # Laplacian Energy
            'laplacian_global': (gp.laplacian_energy, 'Laplacian_Global', False),
            'laplacian_unwrapped': (gp.laplacian_energy, 'Laplacian_Unwrapped', True),
            
            # Spatial Frequency
            'spatial_freq_global': (gp.spatial_frequency_global, 'Spatial_Frequency_Global', False),
            'spatial_freq_unwrapped': (gp.spatial_frequency_global, 'Spatial_Frequency_Unwrapped', True),
            
            # Global Entropy
            'global_entropy': (gp.global_entropy_global, 'Global_Entropy', False),
            'global_entropy_unwrapped': (gp.global_entropy_global, 'Global_Entropy_Unwrapped', True),
            
            # GSM
            'sharpness_global': (gp.sharpness_global, 'Sharpness_Global', False),
            'sharpness_unwrapped': (gp.sharpness_global, 'Sharpness_Unwrapped', True),
            
            # ===== MODULE 3: Ground-Truth Comparisons =====
            
            # SSIM
            'ssim': (gtc.calculate_ssim, 'SSIM', False),
            'ssim_unwrapped': (gtc.calculate_ssim, 'SSIM_Unwrapped', True),
            
            # MSE
            'mse': (gtc.calculate_mse, 'MSE', False),
            'mse_unwrapped': (gtc.calculate_mse, 'MSE_Unwrapped', True),
            
            # PSNR
            'psnr': (gtc.calculate_psnr, 'PSNR', False),
            'psnr_unwrapped': (gtc.calculate_psnr, 'PSNR_Unwrapped', True),
        }

    def _setup_menus(self):
        # Set up the menu bar.
        menubar = wx.MenuBar()
        menubar.Append(file_menu.create(self, self.notebook), "&File")
        menubar.Append(edit_menu.create(self, self.notebook), "&Edit")
        menubar.Append(analysis_menu.create(self, self.notebook), "&Analysis")
        menubar.Append(help_menu.create(self), "&Help")
        self.SetMenuBar(menubar)

    def _setup_layout(self):
        # Set up the main layout.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.update_analysis_menu_state()

    def _bind_events(self):
        # Bind event handlers.
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self.on_tab_change)
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self.on_tab_close)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def update_analysis_menu_state(self):
        # Update Analysis menu state based on current image type.
        menubar = self.GetMenuBar()
        analysis_index = menubar.FindMenu("&Analysis")
        if analysis_index == wx.NOT_FOUND:
            return

        analysis_menu = menubar.GetMenu(analysis_index)
        has_images = self.notebook.GetPageCount() > 0
        
        # Get menu items by their IDs
        if hasattr(self, 'module1_metrics_id'):
            module1_item = analysis_menu.FindItemById(self.module1_metrics_id)
            module2_item = analysis_menu.FindItemById(self.module2_metrics_id)
            module3_item = analysis_menu.FindItemById(self.module3_metrics_id)
            complexity_item = analysis_menu.FindItemById(self.computational_complexity_id)
            
            if not has_images:
                # No images: disable module 1, 2, and 3
                if module1_item:
                    module1_item.Enable(False)
                if module2_item:
                    module2_item.Enable(False)
                if module3_item:
                    module3_item.Enable(False)
                if complexity_item:
                    complexity_item.Enable(True)  # Always enabled
            elif self.is_sample_image:
                # Sample image: only enable module 3
                if module1_item:
                    module1_item.Enable(False)
                if module2_item:
                    module2_item.Enable(False)
                if module3_item:
                    module3_item.Enable(True)
                if complexity_item:
                    complexity_item.Enable(True)
            else:
                # Regular images: enable all
                if module1_item:
                    module1_item.Enable(True)
                if module2_item:
                    module2_item.Enable(True)
                if module3_item:
                    module3_item.Enable(True)
                if complexity_item:
                    complexity_item.Enable(True)

    def mark_as_sample_image(self, is_sample=True):
        self.is_sample_image = is_sample
        self.update_analysis_menu_state()

    def ensure_metric_table(self):
        if self.metric_table is not None:
            return
        
        self.metric_table = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.metric_table.InsertColumn(0, "Image Name", width=250)
        
        
        if self.GetSizer():
            self.GetSizer().Add(self.metric_table, 0, wx.EXPAND | wx.ALL, 5)
            self.Layout()

    def _ensure_column(self, col_name: str, width: int = 120):
        if not self.metric_table:
            return
        
        for i in range(self.metric_table.GetColumnCount()):
            if self.metric_table.GetColumn(i).GetText() == col_name:
                return
        
        col_idx = self.metric_table.GetColumnCount()
        self.metric_table.InsertColumn(col_idx, col_name, width=width)

    def _get_column_index_by_name(self, col_name: str) -> int:
        if not self.metric_table:
            return -1
        
        for i in range(self.metric_table.GetColumnCount()):
            if self.metric_table.GetColumn(i).GetText() == col_name:
                return i
        return -1

    def _find_or_create_row(self, image_name):
        if not self.metric_table:
            return 0
        
        for row in range(self.metric_table.GetItemCount()):
            if self.metric_table.GetItem(row, 0).GetText() == image_name:
                return row
        
        new_row = self.metric_table.GetItemCount()
        self.metric_table.InsertItem(new_row, image_name)
        return new_row

    def update_table_with_zones(self, image_name, metric_column, value, zone_stats=None, num_zones=0):
        if self.metric_table is None:
            self.ensure_metric_table()

        row = self._find_or_create_row(image_name)
        
        # Ensure column exists
        self._ensure_column(metric_column)
        col_idx = self._get_column_index_by_name(metric_column)
        
        if col_idx < 0:
            return
        
        # Write metric value
        self.metric_table.SetItem(row, col_idx, f"{value:.4f}")
        
        if zone_stats:
            self._update_zone_columns_dynamic(row, zone_stats, metric_column, num_zones)

    def _update_zone_columns_dynamic(self, row, zone_stats, metric, num_zones=0):
        metric_label = metric.split("_")[0]
        
        for i in range(1, num_zones + 1):
            self._ensure_column(f"Zone {i} {metric_label}")
        
        self._ensure_column(f"{metric_label} Total Zones", width=120)
        self._ensure_column(f"{metric_label} Zone Details", width=200)
        
        col_indices = {
            self.metric_table.GetColumn(c).GetText(): c
            for c in range(self.metric_table.GetColumnCount())
        }
        
        self._populate_zone_values(row, zone_stats, metric_label, col_indices)
        
        self._populate_zone_summary(row, zone_stats, col_indices, num_zones, metric_label)

    def _populate_zone_values(self, row, zone_stats, metric_label, col_indices):
        # exctract the name of the base metric (e.g., 'STD' from 'STD_Zones')
        if isinstance(metric_label, str):
            base_metric = metric_label.split('_')[0]
        else:
            base_metric = str(metric_label)
        
        metric_name_lower = base_metric.lower()
        
        for stat in zone_stats:
            zone_num = stat.get('zone')
            
            if not isinstance(zone_num, int):
                continue
            
            col_name = f"Zone {zone_num} {base_metric}"
            
            if col_name not in col_indices:
                continue
            
            # Buscar el valor con diferentes variantes del nombre
            val = stat.get(metric_name_lower)  # 'std', 'mad', 'rms', etc.
            if val is None:
                val = stat.get(base_metric.lower())
            if val is None:
                val = stat.get(base_metric)
            
            if val is None:
                continue
    
            try:
                val = float(np.mean(val)) if not np.isscalar(val) else float(val)
                if np.isfinite(val):
                    self.metric_table.SetItem(row, col_indices[col_name], f"{val:.4f}")
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error setting value for {col_name}: {e}")
                continue

    def _populate_zone_summary(self, row, zone_stats, col_indices, num_zones=None, metric_label=None):
        if num_zones is None or num_zones == 0:
            num_zones = len([s for s in zone_stats if isinstance(s.get('zone'), int)])
        
        total_zones_col = f"{metric_label} Total Zones" if metric_label else "Total Zones"
        zone_details_col = f"{metric_label} Zone Details" if metric_label else "Zone Details"
        
        if total_zones_col in col_indices:
            self.metric_table.SetItem(row, col_indices[total_zones_col], str(num_zones))
        
        if zone_details_col in col_indices and zone_stats:
            coords_summary = self._create_coords_summary(zone_stats)
            self.metric_table.SetItem(row, col_indices[zone_details_col], coords_summary)

    def _create_coords_summary(self, zone_stats):
        # Create a summary string of zone coordinates.
        coords_parts = [
            f"Z{stat['zone']}:({stat['coords'][0]},{stat['coords'][2]})-({stat['coords'][1]},{stat['coords'][3]})"
            for stat in zone_stats[:3]
        ]
        
        coords_summary = "; ".join(coords_parts)
        if len(zone_stats) > 3:
            coords_summary += f"; +{len(zone_stats)-3} more"
        
        return coords_summary

    def ensure_mask_table(self):
        # Create mask table if it doesn't exist.
        if self.mask_table is not None:
            return
            
        self.mask_table = wx.ListCtrl(self, style=wx.LC_REPORT)
        
        columns = [
            ("Image Name", 200), ("Threshold", 100), ("Total Pixels", 100),
            ("Background Pixels", 120), ("Background %", 100),
            ("Sample Pixels", 120), ("Sample %", 100), 
        ]
        
        for i, (name, width) in enumerate(columns):
            self.mask_table.InsertColumn(i, name, width=width)

        self.GetSizer().Add(self.mask_table, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()
        self.mask_table.Bind(wx.EVT_CONTEXT_MENU, self.on_mask_table_context_menu)

    def update_mask_table(self, image_name, threshold, total_pixels, background_pixels, 
                         sample_pixels):
        # Update mask table with calculated data.
        self.ensure_mask_table()
        
        found_row = None
        for row in range(self.mask_table.GetItemCount()):
            if self.mask_table.GetItemText(row, 0) == image_name:
                found_row = row
                break

        bg_percent = 100 * background_pixels / total_pixels if total_pixels > 0 else 0
        sample_percent = 100 * sample_pixels / total_pixels if total_pixels > 0 else 0
        
        data = [
            f"{threshold:.4f}", f"{total_pixels}", f"{background_pixels}",
            f"{bg_percent:.1f}%", f"{sample_pixels}",
            f"{sample_percent:.1f}%",
        ]

        if found_row is not None:
            for col, value in enumerate(data, start=1):
                self.mask_table.SetItem(found_row, col, value)
        else:
            index = self.mask_table.InsertItem(self.mask_table.GetItemCount(), image_name)
            for col, value in enumerate(data, start=1):
                self.mask_table.SetItem(index, col, value)

    # Event handlers
    def on_open(self, event):
        initial_count = self.notebook.GetPageCount()
        file_manager.open_image(self, self.notebook)
        
        if self.notebook.GetPageCount() > initial_count:
            self.mark_as_sample_image(False)  

    def on_save(self, event):
        img = self.notebook.get_current_image()
        file_manager.save_image(self, img)

    def _get_current_image_data(self):
        pil_img = self.notebook.get_current_image()
        name = self.notebook.get_current_image_name()
        page = self.notebook.get_current_page()
        
        if pil_img is None or name is None:
            wx.MessageBox("No image selected", "Error", wx.ICON_ERROR)
            return None, None, None
            
        return pil_img, name, page

    def _convert_to_phase(self, pil_img):
        page = self.notebook.get_current_page()
        
        if page is not None and hasattr(page, 'is_mat_complex') and page.is_mat_complex:
            if hasattr(page, 'phase') and page.phase is not None:
                return page.phase
        
        grayscale = pil_img.convert("L")
        img_array = np.array(grayscale, dtype=float)
        return utRBPV.grayscaleToPhase(img_array)

    def _update_tables_with_mask_data(self, name, sample, background_mask, 
                                   threshold):
        if background_mask is None or threshold is None:
            return

        total_pixels = sample.size
        background_pixels = np.sum(background_mask)
        sample_pixels = np.sum(~background_mask)

        self.update_mask_table(
            name, threshold, total_pixels, background_pixels, 
            sample_pixels
        )

    def _get_zone_count_from_user(self, default=2):
        # Get number of zones from user input.
        with wx.NumberEntryDialog(
            self, 
            "How many zones do you want to select?", 
            "Number of zones (1-20):", 
            "Zone Selection", 
            default, 1, 20
        ) as dlg:
            return dlg.GetValue() if dlg.ShowModal() == wx.ID_OK else None
        
    def _get_limit_from_user(self, default=64):
        # Get limit parameter from user input for Legendre analysis.
        with wx.NumberEntryDialog(
            self, 
            "Enter the limit parameter for Fourier domain cropping:", 
            "Limit (64-1024):", 
            "Legendre Limit Parameter", 
            default, 64, 1024
        ) as dlg:
            return dlg.GetValue() if dlg.ShowModal() == wx.ID_OK else None    
        
    def _get_legendre_order_from_user(self, default=5):
        # Get number of Legendre coefficients from user input.
        with wx.NumberEntryDialog(
            self, 
            "How many Legendre coefficients do you want to calculate?", 
            "Number of coefficients (1-15):", 
            "Legendre Order", 
            default, 1, 15
        ) as dlg:
            return dlg.GetValue() if dlg.ShowModal() == wx.ID_OK else None

    def _handle_zone_analysis_result(self, name, metric, result, num_zones):
        # Handle the result from zone-based analysis.
        value, zone_stats = result if isinstance(result, tuple) else (result, [])

        if np.isnan(value):
            wx.MessageBox("No zones were selected.", "Info", wx.ICON_INFORMATION)
            return

        self.update_table_with_zones(name, metric, value, zone_stats, num_zones=num_zones)
        
        if zone_stats:
            self._show_zone_analysis_results(metric, num_zones, value, zone_stats)

    def _show_zone_analysis_results(self, metric, num_zones, avg_value, zone_stats):
        # Display zone analysis results to user.
        metric_name = metric.split('_')[0]
        zone_details = "\n".join([
        f"Zone {stat['zone']}: {metric_name} = "
        f"{stat.get(metric_name.lower(), 'N/A') if not isinstance(stat.get(metric_name.lower()), (int, float)) else f'{stat.get(metric_name.lower()):.6f}'}"
        for stat in zone_stats
        ])
        
        zone_info = (
            f"Analysis completed:\n\n"
            f"Requested zones: {num_zones}\n"
            f"Selected zones: {len(zone_stats)}\n"
            f"Average {metric_name}: {avg_value:.6f}\n\n"
            f"Detail per zone:\n{zone_details}"
        )

        wx.MessageBox(zone_info, "Analysis Results", wx.ICON_INFORMATION)

    def _process_metric_with_unwrap(self, event, metric_func, metric_name):
        # Generic handler for unwrapped metrics with multi-image selection.
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  
        
        progress = wx.ProgressDialog(
            "Processing Images",
            f"Calculating {metric_name}...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                progress.Update(idx, f"Processing {name}...")
                
                if hasattr(panel, 'is_mat_complex') and panel.is_mat_complex and hasattr(panel, 'phase'):
                    sample = panel.phase
                else:
                    grayscale = pil_img.convert("L")
                    img_array = np.array(grayscale, dtype=float)
                    sample = utRBPV.grayscaleToPhase(img_array)
                
                unwrapped = rb.unwrap_with_scikit(sample)
                background_mask,  threshold = utRBPV.create_background_mask(
                    unwrapped, method='otsu', parent=self
                )
                
                value = metric_func(sample, background_mask, manual=False, num_zones=2)
                
                # Update tables
                self._update_tables_with_mask_data(
                    name, sample, background_mask,  threshold
                )
                self.update_table_with_zones(name, f"{metric_name}_Unwrapped_Background", value)
                
                results[name] = value
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        if results:
            summary = f"Unwrapped background {metric_name} analysis completed:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.6f}\n"
            wx.MessageBox(summary, f"{metric_name} Background (Unwrapped) Results", wx.ICON_INFORMATION)
        
        plt.close('all')

    def _process_metric_without_unwrap(self, event, metric_func, metric_name):
        # Generic handler for non-unwrapped metrics with multi-image selection.
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  
        
        progress = wx.ProgressDialog(
            "Processing Images",
            f"Calculating {metric_name}...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                progress.Update(idx, f"Processing {name}...")
                
                if hasattr(panel, 'is_mat_complex') and panel.is_mat_complex and hasattr(panel, 'phase'):
                    sample = panel.phase
                else:
                    grayscale = pil_img.convert("L")
                    img_array = np.array(grayscale, dtype=float)
                    sample = utRBPV.grayscaleToPhase(img_array)
                
                background_mask,  threshold = utRBPV.create_background_mask(
                    sample, method='otsu', parent=self
                )
                
                value = metric_func(sample, background_mask, manual=False, num_zones=2)
                
                self._update_tables_with_mask_data(
                    name, sample, background_mask,  threshold
                )
                self.update_table_with_zones(name, f"{metric_name}_Background", value)
                
                results[name] = value
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        if results:
            summary = f"Background {metric_name} analysis completed:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.6f}\n"
            wx.MessageBox(summary, f"{metric_name} Background Results", wx.ICON_INFORMATION)
        
        plt.close('all')


    def on_clear(self, event):
        # Clear all images and tables.
        with wx.MessageDialog(
            self,
            message="Are you sure you want to close all images and remove all tables?",
            caption="Confirm Clear",
            style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING
        ) as dlg:
            if dlg.ShowModal() == wx.ID_YES:
                self._clear_all_data()

    def _clear_all_data(self):
        # Clear all application data and reset state.
        while self.notebook.GetPageCount() > 0:
            self.notebook.DeletePage(0)

        self._close_detached_windows()
        self._destroy_tables()
        
        self.is_sample_image = False
        self.Layout()
        self.update_analysis_menu_state()

    def _close_detached_windows(self):
        # Close all detached windows except main frame.
        for window in wx.GetTopLevelWindows():
            if isinstance(window, wx.Frame) and window != self:
                try:
                    window.Close()
                except:
                    pass

    def _destroy_tables(self):
        # Safely destroy both tables.
        if self.metric_table:
            self.metric_table.Destroy()
            self.metric_table = None

        if self.mask_table:
            self.mask_table.Destroy()
            self.mask_table = None

    # Export handlers
    def on_metric_table_context_menu(self, event):
        # Show context menu for metrics table.
        menu = wx.Menu()
        export_csv = menu.Append(wx.ID_ANY, "Export Metrics to CSV")
        export_xls = menu.Append(wx.ID_ANY, "Export Metrics to Excel")

        self.Bind(wx.EVT_MENU, self.on_export_metrics_csv, export_csv)
        self.Bind(wx.EVT_MENU, self.on_export_metrics_excel, export_xls)

        self.PopupMenu(menu)
        menu.Destroy()

    def on_mask_table_context_menu(self, event):
        # Show context menu for mask table.
        menu = wx.Menu()
        export_csv = menu.Append(wx.ID_ANY, "Export Mask Data to CSV")
        export_xls = menu.Append(wx.ID_ANY, "Export Mask Data to Excel")

        self.Bind(wx.EVT_MENU, self.on_export_mask_csv, export_csv)
        self.Bind(wx.EVT_MENU, self.on_export_mask_excel, export_xls)

        self.PopupMenu(menu)
        menu.Destroy()

    def _export_table_data(self, table_type, file_format):
        # Generic method to export table data.
        table_info = {
            "metrics": (self.metric_table, self._get_metric_table_data, "Metrics"),
            "mask": (self.mask_table, self._get_mask_table_data, "Mask Data")
        }
        
        table, data_method, file_desc = table_info[table_type]

        if table is None or table.GetItemCount() == 0:
            wx.MessageBox(f"No {file_desc.lower()} to export", "Error", wx.ICON_ERROR)
            return

        wildcard = "CSV files (*.csv)|*.csv" if file_format == "csv" else "Excel files (*.xlsx)|*.xlsx"

        with wx.FileDialog(
            self, 
            f"Save {file_desc} {file_format.upper()} file",
            wildcard=wildcard,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()

        try:
            data = data_method()
            df = pd.DataFrame(data)
            
            if file_format == "csv":
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)
                
            wx.MessageBox(f"{file_desc} exported to {path}", "Success", wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Export failed: {str(e)}", "Error", wx.ICON_ERROR)

    def on_export_metrics_csv(self, event):
        self._export_table_data("metrics", "csv")

    def on_export_metrics_excel(self, event):
        self._export_table_data("metrics", "excel")

    def on_export_mask_csv(self, event):
        self._export_table_data("mask", "csv")

    def on_export_mask_excel(self, event):
        self._export_table_data("mask", "excel")

    def _get_metric_table_data(self):
        # Extract metrics table data for export.
        if self.metric_table is None:
            return []
        
        return [
            {
                self.metric_table.GetColumn(col).GetText(): self.metric_table.GetItemText(row, col)
                for col in range(self.metric_table.GetColumnCount())
            }
            for row in range(self.metric_table.GetItemCount())
        ]

    def _get_mask_table_data(self):
        # Extract mask table data for export.
        if self.mask_table is None:
            return []
            
        column_names = [
            "Image Name", "Threshold", "Total Pixels", "Background Pixels",
            "Background %", "Sample Pixels", "Sample %"
        ]
        
        return [
            {col_name: self.mask_table.GetItemText(row, col) 
             for col, col_name in enumerate(column_names)}
            for row in range(self.mask_table.GetItemCount())
        ]

    # Tab and window management
    def on_tab_change(self, event):
        # Handle tab change event.
        # Check if the current page is a sample image
        page = self.notebook.get_current_page()
        if page and hasattr(page, 'is_sample'):
            self.mark_as_sample_image(page.is_sample)
        else:
            self.mark_as_sample_image(False)
        
        event.Skip()

    def on_tab_close(self, event):
        # Handle tab close event.
        wx.CallAfter(self._handle_tab_close_cleanup)
        event.Skip()

    def _handle_tab_close_cleanup(self):
        # Execute after closing a tab. Clean up if no images remain.
        page_count = self.notebook.GetPageCount()
        
        if page_count == 0:
            self._destroy_tables()
            self._close_detached_windows()
            self.is_sample_image = False
            self.update_analysis_menu_state()
            self.Layout()
        else:
            # Update menu state based on remaining images
            page = self.notebook.get_current_page()
            if page and hasattr(page, 'is_sample'):
                self.mark_as_sample_image(page.is_sample)
            else:
                self.mark_as_sample_image(False)

    def on_legendre_background_unwrapped(self, event):
        # Legendre background analysis with unwrapping and multi-image selection.
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Get limit parameter from user
        limit = self._get_limit_from_user(default=64)
        if limit is None:
            return
        
        # Get number of coefficients from user
        order_max = self._get_legendre_order_from_user(default=5)
        if order_max is None:
            return
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            "Calculating Legendre coefficients...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Convert to phase
                if hasattr(panel, 'is_mat_complex') and panel.is_mat_complex and hasattr(panel, 'phase'):
                    sample = panel.phase
                else:
                    grayscale = pil_img.convert("L")
                    img_array = np.array(grayscale, dtype=float)
                    sample = utRBPV.grayscaleToPhase(img_array)
                
                # Process image
                unwrapped = rb.unwrap_with_scikit(sample)
                background_mask,  threshold = utRBPV.create_background_mask(
                    unwrapped, method='otsu', parent=self
                )
                
                # Calculate Legendre coefficients
                coefficients = rb.legendre_background(sample, background_mask, manual=False, num_zones=2, limit=limit, order_max=order_max)
                
                # Update tables
                self._update_tables_with_mask_data(
                    name, sample, background_mask,  threshold
                )
                
                # Add each Legendre coefficient as a separate column
                self.ensure_metric_table()
                row = self._find_or_create_row(name)
                
                for i, coeff in enumerate(coefficients):
                    col_name = f"Legendre_C{i+1}_Unwrapped"
                    self._ensure_column(col_name, width=120)
                    col_idx = self._get_column_index_by_name(col_name)
                    if col_idx >= 0:
                        self.metric_table.SetItem(row, col_idx, f"{coeff:.6f}")
                
                results[name] = coefficients
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            summary = f"Unwrapped background Legendre analysis completed:\n\n"
            summary += f"Limit parameter: {limit}\n"
            summary += f"Order (coefficients): {order_max}\n\n"
            for img_name in results.keys():
                summary += f"{img_name}: {len(results[img_name])} coefficients calculated\n"
            wx.MessageBox(summary, "Legendre Background (Unwrapped) Results", wx.ICON_INFORMATION)
        
        plt.close('all')
    
    def on_legendre_background(self, event):
        # Legendre background analysis without unwrapping with multi-image selection.
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Get limit parameter from user
        limit = self._get_limit_from_user(default=64)
        if limit is None:
            return
        
        # Get number of coefficients from user
        order_max = self._get_legendre_order_from_user(default=5)
        if order_max is None:
            return
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            "Calculating Legendre coefficients...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Convert to phase
                if hasattr(panel, 'is_mat_complex') and panel.is_mat_complex and hasattr(panel, 'phase'):
                    sample = panel.phase
                else:
                    grayscale = pil_img.convert("L")
                    img_array = np.array(grayscale, dtype=float)
                    sample = utRBPV.grayscaleToPhase(img_array)
                
                # Process image
                background_mask,  threshold = utRBPV.create_background_mask(
                    sample, method='otsu', parent=self
                )
                
                # Calculate Legendre coefficients
                coefficients = rb.legendre_background(sample, background_mask, manual=False, num_zones=2, limit=limit, order_max=order_max, NoPistonCompensation=False, UsePCA=True)
                
                # Update tables
                self._update_tables_with_mask_data(
                    name, sample, background_mask,  threshold
                )
                
                # Add each Legendre coefficient as a separate column
                self.ensure_metric_table()
                row = self._find_or_create_row(name)
                
                for i, coeff in enumerate(coefficients):
                    col_name = f"Legendre_C{i+1}"
                    self._ensure_column(col_name, width=100)
                    col_idx = self._get_column_index_by_name(col_name)
                    if col_idx >= 0:
                        self.metric_table.SetItem(row, col_idx, f"{coeff:.6f}")
                
                results[name] = coefficients
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            summary = f"Background Legendre analysis completed:\n\n"
            summary += f"Limit parameter: {limit}\n"
            summary += f"Order (coefficients): {order_max}\n\n"
            for img_name in results.keys():
                summary += f"{img_name}: {len(results[img_name])} coefficients calculated\n"
            wx.MessageBox(summary, "Legendre Background Results", wx.ICON_INFORMATION)
        
        plt.close('all')
        
    def on_legendre_background_zones(self, event):
        # Legendre background analysis with zone selection.
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        sample = self._convert_to_phase(pil_img)
        
        # Get limit parameter from user
        limit = self._get_limit_from_user(default=64)
        if limit is None:
            return  # User cancelled
        
        # Get number of coefficients from user
        order_max = self._get_legendre_order_from_user(default=5)
        if order_max is None:
            return  # User cancelled
        
        num_zones = self._get_zone_count_from_user()
        if num_zones is None:
            return

        try:
            # Pass limit and order_max parameters to legendre_background
            result = rb.legendre_background(sample, mask=None, manual=True, num_zones=num_zones, limit=limit, order_max=order_max, NoPistonCompensation=False, UsePCA=False)
            
            if isinstance(result, tuple):
                mean_coeffs, zone_stats = result
            else:
                wx.MessageBox("No zones were selected.", "Info", wx.ICON_INFORMATION)
                return
            
            if len(zone_stats) == 0:
                wx.MessageBox("No zones were selected.", "Info", wx.ICON_INFORMATION)
                return
            
            # Add mean coefficients to table
            self.ensure_metric_table()
            row = self._find_or_create_row(name)
            
            for i, coeff in enumerate(mean_coeffs):
                col_name = f"Legendre_C{i+1}_Zones"
                self._ensure_column(col_name, width=120)
                col_idx = self._get_column_index_by_name(col_name)
                if col_idx >= 0:
                    self.metric_table.SetItem(row, col_idx, f"{coeff:.6f}")
            
            # Show results
            zone_details = "\n".join([
                f"Zone {stat['zone']}: {len(stat['legendre'])} coefficients computed"
                for stat in zone_stats
            ])
            
            zone_info = (
                f"Legendre analysis with zones completed:\n\n"
                f"Limit parameter: {limit}\n"
                f"Order (coefficients): {order_max}\n"
                f"Requested zones: {num_zones}\n"
                f"Selected zones: {len(zone_stats)}\n"
                f"Number of coefficients calculated: {len(mean_coeffs)}\n\n"
                f"Detail per zone:\n{zone_details}"
            )

            wx.MessageBox(zone_info, "Legendre Analysis Results", wx.ICON_INFORMATION)
            
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)

    def on_close(self, event):
        self.Destroy()

    def on_exit(self, event):
        self.Close()

    def on_computational_complexity(self, event):
        # Launch the computational complexity GUI.
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))  # ui/
            parent_dir = os.path.dirname(current_dir)  # My_app/
            complexity_repo_path = os.path.join(parent_dir, 'complexity_algorithm')
            gui_script = os.path.join(complexity_repo_path, 'main.py')

            # Verify if the file exists in Windows
            if not os.path.exists(gui_script):
                wx.MessageBox(
                    f"Complexity GUI not found at:\n{gui_script}\n\n"
                    "Please make sure the 'complexity_algorithm' repository "
                    "is located inside the My_app folder.",
                    "Error",
                    wx.ICON_ERROR
                )
                return

            # Convert the path from Windows format to WSL format (/mnt/c/Users/...)
            wsl_path = gui_script.replace("\\", "/")
            if wsl_path[1] == ":":
                drive = wsl_path[0].lower()
                wsl_path = f"/mnt/{drive}/{wsl_path[3:]}"  # /mnt/c/Users/...

            # Execute GUI within WSL (without CREATE_NEW_CONSOLE)
            #subprocess.Popen(["wsl", "python3", wsl_path])
            subprocess.Popen([sys.executable, gui_script])

            wx.MessageBox(
                "Computational Complexity GUI launched successfully!",
                "Success",
                wx.ICON_INFORMATION
            )

        except Exception as e:
            wx.MessageBox(
                f"Error launching the GUI:\n{str(e)}",
                "Error",
                wx.ICON_ERROR
            )
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)

    # ==========================================
    # Residual Background Metrics (Module 1)
    # ==========================================
   
    # STD handlers
    def on_std_background_unwrapped(self, event):
        self._process_metric_with_unwrap(event, rb.std_background, "STD")

    def on_std_background(self, event):
        self._process_metric_without_unwrap(event, rb.std_background, "STD")

    def on_std_background_zones(self, event):
        self._process_metric_zones(event, rb.std_background, "STD")

    # MAD handlers
    def on_mad_background_unwrapped(self, event):
        self._process_metric_with_unwrap(event, rb.mean_absolute_deviation_background, "MAD")

    def on_mad_background(self, event):
        self._process_metric_without_unwrap(event, rb.mean_absolute_deviation_background, "MAD")

    def on_mad_background_zones(self, event):
        self._process_metric_zones(event, rb.mean_absolute_deviation_background, "MAD")

    # RMS handlers
    def on_rms_background_unwrapped(self, event):
        self._process_metric_with_unwrap(event, rb.rms_background, "RMS")

    def on_rms_background(self, event):
        self._process_metric_without_unwrap(event, rb.rms_background, "RMS")

    def on_rms_background_zones(self, event):
        self._process_metric_zones(event, rb.rms_background, "RMS")

    # PV handlers
    def on_pv_background_unwrapped(self, event):
        self._process_metric_with_unwrap(event, rb.pv_background, "PV")

    def on_pv_background(self, event):
        self._process_metric_without_unwrap(event, rb.pv_background, "PV")

    def on_pv_background_zones(self, event):
        self._process_metric_zones(event, rb.pv_background, "PV")

    # FWHM handlers
    def on_fwhm_background_unwrapped(self, event):
        self._process_metric_with_unwrap(event, rb.fwhm_background, "FWHM")

    def on_fwhm_background(self, event):
        self._process_metric_without_unwrap(event, rb.fwhm_background, "FWHM")

    def on_fwhm_background_zones(self, event):
        self._process_metric_zones(event, rb.fwhm_background, "FWHM")

    # Entropy handlers
    def on_entropy_background_unwrapped(self, event):
        self._process_metric_with_unwrap(event, rb.entropy_background, "Entropy")

    def on_entropy_background(self, event):
        self._process_metric_without_unwrap(event, rb.entropy_background, "Entropy")

    def on_entropy_background_zones(self, event):
        self._process_metric_zones(event, rb.entropy_background, "Entropy")

    def on_all_metrics(self, event):
# Calculate all metrics for all opened images.
        count = self.notebook.GetPageCount()
        if count == 0:
            wx.MessageBox("No images opened", "Error", wx.ICON_ERROR)
            return

        for idx in range(count):
            pil_img = self.notebook.images[idx]
            name = self.notebook.GetPageText(idx)
            page = self.notebook.GetPage(idx)

            if pil_img is None or name is None:
                continue

            try:
                if hasattr(page, 'is_mat_complex') and page.is_mat_complex and hasattr(page, 'phase'):
                    phase_img = page.phase
                else:
                    grayscale = pil_img.convert("L")
                    img_array = np.array(grayscale, dtype=float)
                    phase_img = utRBPV.grayscaleToPhase(img_array)
                
                std_simple_val = rb.std_background(phase_img)
                mad_val = rb.mean_absolute_deviation_background(phase_img)

                self.update_table_with_zones(name, "STD_Simple", std_simple_val)
                self.update_table_with_zones(name, "MAD", mad_val)
                
            except Exception as e:
                continue

    def get_zones_mode_from_user(self, num_images, num_zone_metrics):
# Ask user which mode to use for zones based on context.
        # Case 1: Single image, single zone metric - no need to ask
        if num_images == 1 and num_zone_metrics == 1:
            return 'per_image'  # Default mode
        
        # Case 2: Multiple zone metrics, single image
        if num_images == 1 and num_zone_metrics > 1:
            dlg = wx.SingleChoiceDialog(
                self,
                f"You selected {num_zone_metrics} zone-based metrics for 1 image.\n\n"
                "How would you like to define zones?\n\n"
                "Shared Zones (Same for all metrics):\n"
                "  • Select zones ONCE\n"
                "  • Use the SAME zones for ALL zone metrics (STD, MAD, RMS, etc.)\n"
                "  • Faster\n\n"
                "Per-Metric Zones (Different for each metric):\n"
                "  • Select DIFFERENT zones for EACH metric\n"
                "  • More flexible\n"
                "  • Takes more time",
                "Zone Selection Mode - Multiple Metrics",
                ["Shared Zones (Same for all metrics)", "Per-Metric Zones (Different for each metric)"]
            )
            
            if dlg.ShowModal() == wx.ID_OK:
                selection = dlg.GetSelection()
                dlg.Destroy()
                return 'shared' if selection == 0 else 'per_metric'
            
            dlg.Destroy()
            return None
        
        # Case 3: Multiple images
        if num_images > 1:
            # Determine message based on whether there are also multiple metrics
            if num_zone_metrics > 1:
                message = (
                    f"You selected {num_images} images and {num_zone_metrics} zone-based metrics.\n\n"
                    "How would you like to define zones?\n\n"
                    "Shared Zones (Same for all):\n"
                    "  • Select zones ONCE on the first image\n"
                    "  • Use the SAME zones for ALL images AND ALL metrics\n"
                    "  • Fastest option\n\n"
                    "Per-Image Zones (Different for each image):\n"
                    "  • Select zones for EACH image\n"
                    "  • Same zones across metrics within each image\n"
                    "  • More flexible for different image structures"
                )
                choices = ["Shared Zones (Same for all)", "Per-Image Zones (Different for each image)"]
            else:
                message = (
                    f"You selected {num_images} images with zone-based metrics.\n\n"
                    "How would you like to define zones?\n\n"
                    "Shared Zones (Same for all images):\n"
                    "  • Select zones ONCE on the first image\n"
                    "  • Use the SAME zones for ALL images\n"
                    "  • Faster\n\n"
                    "Per-Image Zones (Different for each image):\n"
                    "  • Select DIFFERENT zones for EACH image\n"
                    "  • More flexible\n"
                    "  • Takes more time"
                )
                choices = ["Shared Zones (Same for all images)", "Per-Image Zones (Different for each image)"]
            
            dlg = wx.SingleChoiceDialog(
                self,
                message,
                "Zone Selection Mode",
                choices
            )
            
            if dlg.ShowModal() == wx.ID_OK:
                selection = dlg.GetSelection()
                dlg.Destroy()
                return 'shared' if selection == 0 else 'per_image'
            
            dlg.Destroy()
            return None
        
        # Default case
        return 'per_image'

    def _calculate_multiple_metrics(self, metrics_dict, image_indices):
# Calculate multiple metrics for multiple images with single threshold per image per state.
        # Verify ground-truth if needed
        if not self._ensure_ground_truth_loaded(metrics_dict):
            return
        
        # Detect if there are zone metrics and ask for mode
        has_zone_metrics = any('zones' in mid for mid in metrics_dict.keys())
        
        zones_mode = 'per_image'  # Default mode
        
        if has_zone_metrics:
            # Count number of zone metrics
            num_zone_metrics = sum(1 for mid in metrics_dict.keys() if 'zones' in mid)
            
            # Ask user which mode they prefer
            zones_mode = self.get_zones_mode_from_user(len(image_indices), num_zone_metrics)
            
            if zones_mode is None:
                wx.MessageBox(
                    "Zone mode selection cancelled.",
                    "Cancelled",
                    wx.ICON_INFORMATION
                )
                return
            
            # Initialize zones cache for this session
            if not hasattr(self, '_zones_cache'):
                self._zones_cache = {}
            
            # If shared mode, clear cache to start fresh and reset shared zones
            if zones_mode == 'shared':
                self._zones_cache.clear()
                self._shared_zones = None  # Will be set on first image
            # If per-image or per-metric mode, clear cache for fresh zones
            else:
                self._zones_cache.clear()
        
        total_operations = len(metrics_dict) * len(image_indices)
        
        # Progress bar
        progress = wx.ProgressDialog(
            "Calculating Metrics",
            "Processing...",
            maximum=total_operations,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        current_op = 0
        results_summary = []
        
        try:
            # Process each image
            for img_idx in image_indices:
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    current_op += len(metrics_dict)
                    continue
                
                results_summary.append(f"\n{'='*50}")
                results_summary.append(f"Image: {name}")
                results_summary.append(f"{'='*50}")
                
                # 1️⃣ Convert to phase ONCE per image
                if hasattr(panel, 'is_mat_complex') and panel.is_mat_complex and hasattr(panel, 'phase'):
                    sample = panel.phase
                else:
                    grayscale = pil_img.convert("L")
                    img_array = np.array(grayscale, dtype=float)
                    sample = utRBPV.grayscaleToPhase(img_array)
                
                # 2️⃣ Pre-calculate masks ONCE per image
                # Pass zone mode so cache handles it correctly
                masks_cache = self._precalculate_masks(
                    sample, name, metrics_dict, zones_mode=zones_mode
                )
                
                # 3️⃣ Calculate each metric using the cache
                for metric_id, metric_label in metrics_dict.items():
                    progress.Update(
                        current_op,
                        f"Calculating {metric_label} for {name}..."
                    )
                    
                    try:
                        value = self._calculate_single_metric_cached(
                            metric_id,
                            sample,
                            name,
                            masks_cache
                        )
                        
                        if value is not None:
                            results_summary.append(f" ✓ {metric_label}: {value:.6f}")
                        else:
                            results_summary.append(f" ⊘ {metric_label}: Skipped")
                            
                    except Exception as e:
                        results_summary.append(f" ✗ {metric_label}: ERROR - {str(e)}")
                    
                    current_op += 1
            
            progress.Update(total_operations)
        
        finally:
            progress.Destroy()
        
        # Show summary
        summary_text = "\n".join(results_summary)
        
        try:
            dlg = wx.lib.dialogs.ScrolledMessageDialog(
                self,
                summary_text,
                "Metrics Calculation Complete",
                size=(600, 400)
            )
            dlg.ShowModal()
            dlg.Destroy()
        except:
            wx.MessageBox(summary_text, "Metrics Calculation Complete", wx.ICON_INFORMATION)
        
        plt.close('all')

    def _precalculate_masks(self, sample, image_name, metrics_dict, zones_mode='per_image'):
        # Pre-calculate all needed masks for an image BEFORE processing metrics.
        masks_cache = {}
        
        # Detect which type of metrics are being calculated
        module1_metrics = []
        module2_metrics = []
        module3_metrics = []
        zone_metrics = []
        
        for metric_id in metrics_dict.keys():
            if 'zones' in metric_id:
                zone_metrics.append(metric_id)
            elif metric_id.startswith(('mmm_', 'gradient_', 'tsm_', 'curvature_',
                                    'laplacian_', 'spatial_freq_', 'global_entropy',
                                    'sharpness_')):
                module2_metrics.append(metric_id)
            elif metric_id.startswith(('ssim', 'mse', 'psnr')):
                module3_metrics.append(metric_id)
            else:
                module1_metrics.append(metric_id)
        

        apply_unwrap = False
        if zone_metrics: 
            unwrap_dlg = wx.MessageDialog(
                self,
                "Do you want to apply 'unwrap' (phase unwrapping) to images before zone analysis?\n\n"
                "Unwrapping helps remove 2π phase discontinuities.",
                "Apply Phase Unwrapping for Zones",
                style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION
            )
            unwrap_response = unwrap_dlg.ShowModal()
            unwrap_dlg.Destroy()
            apply_unwrap = unwrap_response == wx.ID_YES
        
        if apply_unwrap:
            try:
                sample = rb.unwrap_with_scikit(sample)
            except Exception as e:
                wx.MessageBox(f"Error applying unwrap to {image_name}: {e}", "Error", wx.ICON_ERROR)
                return masks_cache
        
        if module1_metrics:
            needs_unwrapped = any('unwrapped' in mid for mid in module1_metrics)
            needs_wrapped = any('unwrapped' not in mid and 'zones' not in mid for mid in module1_metrics)
            
            if needs_wrapped:
                background_mask, threshold = utRBPV.create_background_mask(
                    sample, method='otsu', parent=self
                )
                masks_cache[False] = (background_mask, threshold, sample)
                self._update_tables_with_mask_data(
                    image_name, sample, background_mask, threshold
                )
            
            if needs_unwrapped:
                try:
                    unwrapped_sample = rb.unwrap_with_scikit(sample)
                    background_mask, threshold = utRBPV.create_background_mask(
                        unwrapped_sample, method='otsu', parent=self
                    )
                    masks_cache[True] = (background_mask, threshold, unwrapped_sample)
                except Exception as e:
                    wx.MessageBox(f"Error applying unwrap: {e}", "Error", wx.ICON_ERROR)
        
        # Smart zone handling according to mode
        if zone_metrics:
            
            if zones_mode == 'shared':
                # SHARED MODE: Use same zones for all images and all metrics
                if self._shared_zones is None:
                    # First image: ask for zones
                    num_zones = self._get_zone_count_from_user()
                    if num_zones is None:
                        return masks_cache
                    
                    zones = rb.select_manual_zones(sample, num_zones)
                    if not zones or len(zones) == 0:
                        wx.MessageBox("No zones selected.", "Info", wx.ICON_INFORMATION)
                        return masks_cache
                    
                    # Save as shared zones
                    self._shared_zones = (zones, num_zones)
                else:
                    zones, num_zones = self._shared_zones
                
                masks_cache['zones'] = zones
                masks_cache['num_zones'] = num_zones
            
            elif zones_mode == 'per_metric':
                # PER-METRIC MODE: Each metric gets its own zones
                metric_zones = {}
                
                for metric_id in zone_metrics:
                    cache_key = f"{image_name}_{metric_id}"
                    
                    if cache_key in self._zones_cache:
                        zones, num_zones = self._zones_cache[cache_key]
                    else:
                        num_zones = self._get_zone_count_from_user()
                        if num_zones is None:
                            return masks_cache
                        
                        zones = rb.select_manual_zones(sample, num_zones)
                        if not zones or len(zones) == 0:
                            wx.MessageBox(f"No zones selected for {metric_id}.", "Info", wx.ICON_INFORMATION)
                            return masks_cache
                        
                        self._zones_cache[cache_key] = (zones, num_zones)
                    
                    metric_zones[metric_id] = (zones, num_zones)
                
                masks_cache['metric_zones'] = metric_zones
            
            else:  # PER-IMAGE MODE
                cache_key = f"{image_name}_zones"
                
                if cache_key in self._zones_cache:
                    zones, num_zones = self._zones_cache[cache_key]
                else:
                    num_zones = self._get_zone_count_from_user()
                    if num_zones is None:
                        return masks_cache
                    
                    zones = rb.select_manual_zones(sample, num_zones)
                    if not zones or len(zones) == 0:
                        wx.MessageBox(f"No zones selected for {image_name}.", "Info", wx.ICON_INFORMATION)
                        return masks_cache
                    
                    self._zones_cache[cache_key] = (zones, num_zones)
                
                masks_cache['zones'] = zones
                masks_cache['num_zones'] = num_zones
        
        return masks_cache


    def _calculate_single_metric_cached(self, metric_id, sample, image_name, masks_cache):
        """Calculate a single metric using pre-calculated cached masks."""
        
        # Search for metric in the map
        if metric_id not in self.metric_map:
            return None
        
        metric_func, table_column, needs_unwrap = self.metric_map[metric_id]
        
        # SPECIAL HANDLING FOR LEGENDRE METRICS
        if 'legendre' in metric_id.lower():
            # Get limit parameter
            limit = self._get_limit_from_user(default=64)
            if limit is None:
                return None
            
            # Get ordermax parameter
            ordermax = self._get_legendre_order_from_user(default=5)
            if ordermax is None:
                return None
            
            # For zone-based Legendre
            if 'zones' in metric_id:
                if 'zones' not in masks_cache or 'num_zones' not in masks_cache:
                    return None
                
                zones = masks_cache['zones']
                num_zones = masks_cache['num_zones']
                
                try:
                    result = metric_func(sample, mask=None, manual=True, num_zones=num_zones, 
                                    limit=limit, order_max=ordermax, zones=zones)
                    
                    if isinstance(result, tuple):
                        mean_coeffs, zone_stats = result
                    else:
                        return None
                    
                    # Add coefficients to table
                    self.ensure_metric_table()
                    row = self._find_or_create_row(image_name)
                    
                    for i, coeff in enumerate(mean_coeffs):
                        col_name = f"Legendre_C{i+1}_Zones"
                        self._ensure_column(col_name, width=120)
                        col_idx = self._get_column_index_by_name(col_name)
                        if col_idx >= 0:
                            self.metric_table.SetItem(row, col_idx, f"{coeff:.6f}")
                    
                    return float(np.mean(mean_coeffs))
                except Exception as e:
                    return None
            
            # For background Legendre (no zones) - USE CACHED MASK
            else:
                # Get the cached mask (already calculated in _precalculate_masks)
                if needs_unwrap not in masks_cache:
                    return None
                
                background_mask, threshold, processed_sample = masks_cache[needs_unwrap]
                
                try:
                    # Calculate Legendre coefficients using the CACHED mask
                    coefficients = metric_func(processed_sample, background_mask, manual=False,
                                            num_zones=2, limit=limit, order_max=ordermax)
                    
                    # Add coefficients to table
                    self.ensure_metric_table()
                    row = self._find_or_create_row(image_name)
                    
                    suffix = "_Unwrapped" if needs_unwrap else ""
                    for i, coeff in enumerate(coefficients):
                        col_name = f"Legendre_C{i+1}{suffix}"
                        self._ensure_column(col_name, width=120)
                        col_idx = self._get_column_index_by_name(col_name)
                        if col_idx >= 0:
                            self.metric_table.SetItem(row, col_idx, f"{coeff:.6f}")
                    
                    return float(np.mean(coefficients))
                except Exception as e:
                    return None
        
        # ZONE METRICS: Use cached zones (can be shared, per-image, or per-metric)
        if 'zones' in metric_id:
            # Check if this is per-metric mode
            if 'metric_zones' in masks_cache:
                # Per-metric mode: each metric has different zones
                if metric_id not in masks_cache['metric_zones']:
                    return None
                zones, num_zones = masks_cache['metric_zones'][metric_id]
            else:
                # Shared or per-image mode: all metrics use same zones
                if 'zones' not in masks_cache or 'num_zones' not in masks_cache:
                    return None
                zones = masks_cache['zones']
                num_zones = masks_cache['num_zones']
            
            # Extract metric name from metric_id (e.g., 'std_zones' -> 'std')
            metric_name = metric_id.split('_')[0].lower()
            
            # Calculate metric in each zone
            values = []
            zone_stats = []
            for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
                zone_img = sample[ymin:ymax, xmin:xmax]
                val = metric_func(zone_img)
                values.append(val)
                zone_stats.append({
                    'zone': i,
                    metric_name: val,
                    'coords': (xmin, xmax, ymin, ymax)
                })
            
            mean_value = float(np.mean(values))
            
            # Update table with mean value and individual zone values
            self.update_table_with_zones(
                image_name,
                table_column,
                mean_value,
                zone_stats,
                num_zones=num_zones
            )
            
            return mean_value
        
        # MODULE 3: Ground-truth metrics (need ground-truth loaded)
        if metric_id.startswith(('ssim', 'mse', 'psnr')):
            if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
                return None
            
            if needs_unwrap:
                try:
                    processed_sample = rb.unwrap_with_scikit(sample)
                except Exception as e:
                    return None
            else:
                processed_sample = sample
            
            try:
                value = metric_func(processed_sample, self.ground_truth_data, use_unwrap=needs_unwrap)
                if isinstance(value, tuple):
                    value = value
                self.update_table_with_zones(image_name, table_column, value)
                return value
            except Exception as e:
                return None
        
        # MODULE 2: Global metrics (NO need mask)
        elif metric_id.startswith(('mmm_', 'gradient_', 'tsm_', 'curvature_',
                                'laplacian_', 'spatial_freq_', 'global_entropy',
                                'sharpness_')):
            if needs_unwrap:
                try:
                    processed_sample = rb.unwrap_with_scikit(sample)
                except Exception as e:
                    return None
            else:
                processed_sample = sample
            
            import inspect
            params = inspect.signature(metric_func).parameters
            
            try:
                if 'use_unwrap' in params:
                    value = metric_func(processed_sample, use_unwrap=needs_unwrap)
                else:
                    value = metric_func(processed_sample)
                
                if isinstance(value, tuple):
                    value = value
                
                self.update_table_with_zones(image_name, table_column, value)
                return value
            except Exception as e:
                return None
        
        # MODULE 1: Background metrics (NEED mask)
        else:
            if needs_unwrap not in masks_cache:
                return None
            
            background_mask, threshold, processed_sample = masks_cache[needs_unwrap]
            
            try:
                value = metric_func(sample, background_mask, manual=False, num_zones=2)
                self.update_table_with_zones(image_name, table_column, value)
                return value
            except Exception as e:
                return None

    def _process_global_metric(self, metric_func, metric_name, use_unwrap):
        # Generic handler for global phase metrics with multi-image selection.
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Convert use_unwrap to boolean
        if isinstance(use_unwrap, str):
            use_unwrap = use_unwrap.strip().lower() in ("true", "1", "yes", "y", "t")
        elif use_unwrap is None:
            use_unwrap = False
        else:
            use_unwrap = bool(use_unwrap)
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            f"Calculating {metric_name}...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Convert to phase
                if hasattr(panel, 'is_mat_complex') and panel.is_mat_complex and hasattr(panel, 'phase'):
                    phase = panel.phase
                else:
                    grayscale = pil_img.convert("L")
                    img_array = np.array(grayscale, dtype=float)
                    phase = utRBPV.grayscaleToPhase(img_array)
                
                # Calculate metric
                result = metric_func(phase, use_unwrap=use_unwrap)
                
                # Extract main value
                if isinstance(result, tuple):
                    value = result[0]
                else:
                    value = result
                
                # Determine suffix dynamically
                suffix = "_Unwrapped" if use_unwrap else ""
                full_metric_name = f"{metric_name}{suffix}"
                
                # Update table
                self.update_table_with_zones(name, full_metric_name, value)
                
                results[name] = value
            
            progress.Update(len(selected_indices))
            
        except Exception as e:
            progress.Destroy()
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)
            return
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            unwrap_text = " (Unwrapped)" if use_unwrap else ""
            summary = f"{metric_name}{unwrap_text} calculated successfully:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.6f}\n"
            wx.MessageBox(summary, "Analysis Complete", wx.ICON_INFORMATION)

    # Finish of module 1

    
    # ==========================================
    # Global Phase Distortion Metrics (Module 2)
    # ==========================================
    def on_mmm_global(self, event):
        self._process_global_metric(gp.maximum_minus_minimum, "MMM_Global", False)

    def on_mmm_global_unwrapped(self, event):
        self._process_global_metric(gp.maximum_minus_minimum, "MMM_Global", True)

    def on_gradient_global(self, event):
        self._process_global_metric(gp.global_phase_gradient, "Gradient_Global", False)
  
    def on_gradient_global_unwrapped(self, event):
        self._process_global_metric(gp.global_phase_gradient, "Gradient_Global", True)
             
    def on_tsm_global(self, event):
        self._process_global_metric(gp.tsm_global, "TSM_Global", False)

    def on_tsm_global_unwrapped(self, event):
        self._process_global_metric(gp.tsm_global, "TSM_Global", True)

    def on_curvature_global(self, event):
        self._process_global_metric(gp.reconstruction_background, "Legendre coefficients", False)

    def on_curvature_global_unwrapped(self, event):
        self._process_global_metric(gp.reconstruction_background, "Legendre coefficients", True)

    def on_laplacian_global(self, event):
        self._process_global_metric(gp.laplacian_energy, "Laplacian_Global", False)

    def on_laplacian_global_unwrapped(self, event):
        self._process_global_metric(gp.laplacian_energy, "Laplacian_Global", True)

    def on_spatial_freq_global(self, event):
        self._process_global_metric(gp.spatial_frequency_global, "Spatial Frequency", False)

    def on_spatial_freq_global_unwrapped(self, event):
        self._process_global_metric(gp.spatial_frequency_global, "Spatial Frequency", True)

    def on_global_entropy_global(self, event):
        self._process_global_metric(gp.global_entropy_global, "Global Entropy", False)

    def on_global_entropy_global_unwrapped(self, event):
        self._process_global_metric(gp.global_entropy_global, "Global Entropy", True)

    def on_sharpness_global(self, event):
        self._process_global_metric(gp.sharpness_global, "Global Sharpness", False)
    
    def on_sharpness_global_unwrapped(self, event):
        self._process_global_metric(gp.sharpness_global, "Global Sharpness", True)

    def on_all_global_metrics_M2(self, event):
# Compute all global phase metrics Module 2 (wrapped and unwrapped) at once for the current image.
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        try:
            phase = self._convert_to_phase(pil_img)

            # List of metrics with display names
            metrics = [
                (gp.maximum_minus_minimum, "Maximum-Minus-Minimum"),
                (gp.global_phase_gradient, "Global phase Gradient"),
                (gp.tsm_global, "TSM"),
                (gp.reconstruction_background, "Phase Curvature"),
                (gp.laplacian_energy, "Laplacian Energy"),
                (gp.spatial_frequency_global, "Spatial Frequency"),
                (gp.global_entropy_global, "Global Entropy"),
                (gp.sharpness_global, "Sharpness"),
            ]

            import inspect
            results_summary = []

            # Calculate each metric with and without unwrap
            for func, display_name in metrics:
                for use_unwrap in [False, True]:
                    try:
                        # Suffix for display (space instead of underscore)
                        suffix = " Unwrapped" if use_unwrap else ""

                        # Check if the function accepts use_unwrap
                        params = inspect.signature(func).parameters
                        if 'use_unwrap' in params:
                            result = func(phase, use_unwrap=use_unwrap)
                        else:
                            result = func(phase)

                        value = result[0] if isinstance(result, tuple) else result

                        # Update table with full name
                        full_name = f"{display_name}{suffix}"
                        self.update_table_with_zones(name, full_name, value)

                        # Save to summary
                        results_summary.append(f"{full_name}: {value:.6f}")

                    except Exception as metric_err:
                        results_summary.append(f"{display_name}{suffix}: ERROR ({metric_err})")

            # Show summary box with all results
            wx.MessageBox(
                "✅ All global metrics calculated:\n\n" + "\n".join(results_summary),
                "Global Metrics Summary",
                wx.ICON_INFORMATION
            )

        except Exception as e:
            wx.MessageBox(f"Error calculating metrics:\n{str(e)}", "Error", wx.ICON_ERROR)
    # Finish of module 2

    # ==========================================
    # Ground-Truth Comparisons (Module 3)
    # ==========================================
    
    def on_load_ground_truth(self, event):
        # Load ground-truth image for comparison.
        with wx.FileDialog(
            self, 
            "Select Ground-Truth Image",
            wildcard="Image files (*.png;*.jpg;*.bmp;*.tif)|*.png;*.jpg;*.bmp;*.tif|MAT files (*.mat)|*.mat",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            
            path = dlg.GetPath()
            try:
                self.ground_truth_data = gtc.load_ground_truth(path)
                self.ground_truth_path = path
                wx.MessageBox(
                    f"Ground-truth loaded successfully:\n{path}\n\nShape: {self.ground_truth_data.shape}", 
                    "Success", 
                    wx.ICON_INFORMATION
                )
            except Exception as e:
                wx.MessageBox(f"Error loading ground-truth: {str(e)}", "Error", wx.ICON_ERROR)

    def _ensure_ground_truth_loaded(self, metrics_dict):
# Ensure ground-truth is loaded before calculating Module 3 metrics.
        # Verificar si hay metrics del Módulo 3
        has_module3 = any(mid.startswith(('ssim', 'mse', 'psnr')) for mid in metrics_dict.keys())
        
        if not has_module3:
            return True  # Ground-truth not needed
        
        # Check current ground-truth status
        gt_loaded = hasattr(self, 'ground_truth_data') and self.ground_truth_data is not None
        
        # 🔹 CASO 1: Ground-truth already loaded
        if gt_loaded:
            gt_filename = os.path.basename(self.ground_truth_path) if hasattr(self, 'ground_truth_path') else 'Unknown'
            gt_shape = self.ground_truth_data.shape
            
            message = (
                f"Ground-Truth Currently Loaded:\n\n"
                f"File: {gt_filename}\n"
                f"Shape: {gt_shape}\n\n"
                f"Do you want to update/change the ground-truth image?"
            )
            
            response = wx.MessageBox(
                message,
                "Ground-Truth Manager",
                wx.YES_NO | wx.ICON_QUESTION
            )
            
            if response == wx.YES:
                # User wants to change ground-truth
                self.on_load_ground_truth(None)
                
                # Verify if loaded correctly
                if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
                    wx.MessageBox(
                        "Ground-truth not loaded. Metrics calculation cancelled.",
                        "Error",
                        wx.ICON_ERROR
                    )
                    return False
            # If chooses NO, continue with current
            return True
        
        # 🔹 CASO 2: No ground-truth loaded
        else:
            message = (
                "⚠️ No Ground-Truth Loaded\n\n"
                "Module 3 metrics require a ground-truth image for comparison.\n\n"
                "Do you want to load a ground-truth image now?"
            )
            
            response = wx.MessageBox(
                message,
                "Ground-Truth Required",
                wx.YES_NO | wx.ICON_WARNING
            )
            
            if response == wx.YES:
                # User wants to load ground-truth
                self.on_load_ground_truth(None)
                
                # Verify if loaded correctly
                if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
                    wx.MessageBox(
                        "Ground-truth not loaded. Metrics calculation cancelled.",
                        "Error",
                        wx.ICON_ERROR
                    )
                    return False
                return True
            else:
                # User cancelled
                wx.MessageBox(
                    "Ground-truth not loaded. Metrics calculation cancelled.",
                    "Cancelled",
                    wx.ICON_INFORMATION
                )
                return False

    def on_ssim_comparison(self, event):
        # Calculate SSIM between selected images and ground-truth.
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            "Calculating SSIM...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Calculate SSIM
                ssim_value = gtc.calculate_ssim(pil_img, self.ground_truth_data, use_unwrap=False)
                self.update_table_with_zones(name, "SSIM", ssim_value)
                results[name] = ssim_value
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            summary = "SSIM calculation completed:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.6f}\n"
            summary += "\n(1.0 = perfect similarity)"
            wx.MessageBox(summary, "SSIM Result", wx.ICON_INFORMATION)

    def on_ssim_comparison_unwrapped(self, event):
        # Calculate SSIM with unwrapping for selected images.
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            "Calculating SSIM (Unwrapped)...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Calculate SSIM with unwrapping
                ssim_value = gtc.calculate_ssim(pil_img, self.ground_truth_data, use_unwrap=True)
                self.update_table_with_zones(name, "SSIM_Unwrapped", ssim_value)
                results[name] = ssim_value
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            summary = "SSIM (Unwrapped) calculation completed:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.6f}\n"
            wx.MessageBox(summary, "SSIM Unwrapped Result", wx.ICON_INFORMATION)

    def on_mse_comparison(self, event):
# Calculate MSE between selected images and ground-truth.
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            "Calculating MSE...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Calculate MSE
                mse_value = gtc.calculate_mse(pil_img, self.ground_truth_data, use_unwrap=False)
                self.update_table_with_zones(name, "MSE", mse_value)
                results[name] = mse_value
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            summary = "MSE calculation completed:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.6f}\n"
            summary += "\n(Lower is better)"
            wx.MessageBox(summary, "MSE Result", wx.ICON_INFORMATION)

    def on_mse_comparison_unwrapped(self, event):
        # Calculate MSE with unwrapping for selected images.
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            "Calculating MSE (Unwrapped)...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Calculate MSE with unwrapping
                mse_value = gtc.calculate_mse(pil_img, self.ground_truth_data, use_unwrap=True)
                self.update_table_with_zones(name, "MSE_Unwrapped", mse_value)
                results[name] = mse_value
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            summary = "MSE (Unwrapped) calculation completed:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.6f}\n"
            wx.MessageBox(summary, "MSE Unwrapped Result", wx.ICON_INFORMATION)

    def on_psnr_comparison(self, event):
        # Calculate PSNR between selected images and ground-truth.
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            "Calculating PSNR...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Calculate PSNR
                psnr_value = gtc.calculate_psnr(pil_img, self.ground_truth_data, use_unwrap=False)
                self.update_table_with_zones(name, "PSNR", psnr_value)
                results[name] = psnr_value
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            summary = "PSNR calculation completed:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.2f} dB\n"
            summary += "\n(Higher is better)"
            wx.MessageBox(summary, "PSNR Result", wx.ICON_INFORMATION)

    def on_psnr_comparison_unwrapped(self, event):
# Calculate PSNR with unwrapping for selected images.
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        # Show image selection dialog
        selected_indices = show_image_selection_dialog(self, self.notebook)
        
        if selected_indices is None:
            return  # User cancelled
        
        # Progress dialog
        progress = wx.ProgressDialog(
            "Processing Images",
            "Calculating PSNR (Unwrapped)...",
            maximum=len(selected_indices),
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
        )
        
        results = {}
        
        try:
            for idx, img_idx in enumerate(selected_indices):
                # Get image from tab
                panel = self.notebook.GetPage(img_idx)
                pil_img = getattr(panel, 'pil_img', None)
                name = self.notebook.GetPageText(img_idx)
                
                if pil_img is None:
                    continue
                
                # Update progress
                progress.Update(idx, f"Processing {name}...")
                
                # Calculate PSNR with unwrapping
                psnr_value = gtc.calculate_psnr(pil_img, self.ground_truth_data, use_unwrap=True)
                self.update_table_with_zones(name, "PSNR_Unwrapped", psnr_value)
                results[name] = psnr_value
            
            progress.Update(len(selected_indices))
            
        finally:
            progress.Destroy()
        
        # Show summary
        if results:
            summary = "PSNR (Unwrapped) calculation completed:\n\n"
            for img_name, val in results.items():
                summary += f"{img_name}: {val:.2f} dB\n"
            wx.MessageBox(summary, "PSNR Unwrapped Result", wx.ICON_INFORMATION)
    # Finish of module 3