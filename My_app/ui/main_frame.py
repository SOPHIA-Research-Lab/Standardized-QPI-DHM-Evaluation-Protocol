import wx
import wx.aui
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

class MainFrame(wx.Frame):
    """Main application frame for image analysis."""
    
    def __init__(self, parent, title, size):
        super().__init__(parent, title=title, size=size)
        self._initialize_components()
        self._setup_menus()
        self._setup_layout()
        self._bind_events()
        self.Show()

    def _initialize_components(self):
        """Initialize main components and state variables."""
        self.notebook = ImageNotebook(self)
        self.metric_table = None
        self.mask_table = None
        self.is_sample_image = False  
        self.ground_truth_path = None
        self.ground_truth_data = None
        self.CreateStatusBar()

    def _setup_menus(self):
        """Set up the menu bar."""
        menubar = wx.MenuBar()
        menubar.Append(file_menu.create(self, self.notebook), "&File")
        menubar.Append(edit_menu.create(self, self.notebook), "&Edit")
        menubar.Append(samples_menu.create(self, self.notebook), "&Samples")
        menubar.Append(analysis_menu.create(self, self.notebook), "&Analysis")
        menubar.Append(help_menu.create(self), "&Help")
        self.SetMenuBar(menubar)

    def _setup_layout(self):
        """Set up the main layout."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.update_analysis_menu_state()

    def _bind_events(self):
        """Bind event handlers."""
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self.on_tab_change)
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self.on_tab_close)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def update_analysis_menu_state(self):
        """Update Analysis menu state based on current image type."""
        menubar = self.GetMenuBar()
        analysis_index = menubar.FindMenu("&Analysis")
        if analysis_index == wx.NOT_FOUND:
            return

        analysis_menu = menubar.GetMenu(analysis_index)
        has_images = self.notebook.GetPageCount() > 0
        
        # Enable/disable menu items based on image type
        for i in range(analysis_menu.GetMenuItemCount()):
            item = analysis_menu.FindItemByPosition(i)
            if item:
                if not has_images:
                    # No images: disable all items
                    item.Enable(False)
                elif self.is_sample_image:
                    # Sample image: only enable the third item (index 2)
                    item.Enable(i == 2)
                else:
                    # Regular images: enable all items
                    item.Enable(True)

    def mark_as_sample_image(self, is_sample=True):
        """Mark current image as sample or regular image."""
        self.is_sample_image = is_sample
        self.update_analysis_menu_state()

    def ensure_metric_table(self):
        """Create basic table with only Image Name. Columns are added dynamically."""
        if self.metric_table is not None:
            return
            
        self.metric_table = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.metric_table.InsertColumn(0, "Image Name", width=250)
        self.GetSizer().Add(self.metric_table, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()
        self.metric_table.Bind(wx.EVT_CONTEXT_MENU, self.on_metric_table_context_menu)

    def _ensure_column(self, col_name: str, width: int = 120):
        """Ensure a column exists in the table."""
        if not self.metric_table or self._get_column_index_by_name(col_name) >= 0:
            return

        self.metric_table.InsertColumn(
            self.metric_table.GetColumnCount(), col_name, width=width
        )

    def _get_column_index_by_name(self, col_name: str) -> int:
        """Return column index by name, or -1 if not found."""
        if not self.metric_table:
            return -1

        for i in range(self.metric_table.GetColumnCount()):
            if self.metric_table.GetColumn(i).GetText() == col_name:
                return i
        return -1

    def _find_or_create_row(self, image_name):
        """Find existing row for image or create new one."""
        for row in range(self.metric_table.GetItemCount()):
            if self.metric_table.GetItemText(row, 0) == image_name:
                return row
        
        return self.metric_table.InsertItem(self.metric_table.GetItemCount(), image_name)

    def update_table_with_zones(self, image_name, metric, mean_value, zone_stats=None, num_zones=0):
        """Update table with metric data including zone statistics."""
        self.ensure_metric_table()
        row = self._find_or_create_row(image_name)
        
        self._ensure_column(metric)
        col_idx = self._get_column_index_by_name(metric)
        if col_idx >= 0:
            self.metric_table.SetItem(row, col_idx, f"{mean_value:.4f}")

        if zone_stats and "Zones" in metric:
            self._update_zone_columns_dynamic(row, zone_stats, metric, num_zones)

    def _update_zone_columns_dynamic(self, row, zone_stats, metric, num_zones=0):
        """Dynamically populate table columns for zone metrics."""
        metric_label = metric.split("_")[0]
        
        for i in range(1, num_zones + 1):
            self._ensure_column(f"Zone {i} {metric_label}")

        self._ensure_column("Total Zones")
        self._ensure_column("Zone Details")

        col_indices = {
            self.metric_table.GetColumn(c).GetText(): c
            for c in range(self.metric_table.GetColumnCount())
        }

        self._populate_zone_values(row, zone_stats, metric_label, col_indices)
        self._populate_zone_summary(row, zone_stats, col_indices)

    def _populate_zone_values(self, row, zone_stats, metric_label, col_indices):
        """Populate individual zone values in the table."""
        key = metric_label.lower()
        
        for stat in zone_stats:
            zone_num = stat['zone']
            col_name = f"Zone {zone_num} {metric_label}"
            
            if col_name not in col_indices:
                continue
                
            val = stat.get(key)
            if val is None:
                continue
            
            try:
                val = float(np.mean(val)) if not np.isscalar(val) else float(val)
                if np.isfinite(val):
                    self.metric_table.SetItem(row, col_indices[col_name], f"{val:.4f}")
            except (ValueError, TypeError):
                continue

    def _populate_zone_summary(self, row, zone_stats, col_indices):
        """Populate zone summary information."""
        if "Total Zones" in col_indices:
            self.metric_table.SetItem(row, col_indices["Total Zones"], str(len(zone_stats)))

        if "Zone Details" in col_indices and zone_stats:
            coords_summary = self._create_coords_summary(zone_stats)
            self.metric_table.SetItem(row, col_indices["Zone Details"], coords_summary)

    def _create_coords_summary(self, zone_stats):
        """Create a summary string of zone coordinates."""
        coords_parts = [
            f"Z{stat['zone']}:({stat['coords'][0]},{stat['coords'][2]})-({stat['coords'][1]},{stat['coords'][3]})"
            for stat in zone_stats[:3]
        ]
        
        coords_summary = "; ".join(coords_parts)
        if len(zone_stats) > 3:
            coords_summary += f"; +{len(zone_stats)-3} more"
        
        return coords_summary

    def ensure_mask_table(self):
        """Create mask table if it doesn't exist."""
        if self.mask_table is not None:
            return
            
        self.mask_table = wx.ListCtrl(self, style=wx.LC_REPORT)
        
        columns = [
            ("Image Name", 200), ("Threshold", 100), ("Total Pixels", 100),
            ("Background Pixels", 120), ("Background %", 100),
            ("Sample Pixels", 120), ("Sample %", 100), ("Background Values", 150),
        ]
        
        for i, (name, width) in enumerate(columns):
            self.mask_table.InsertColumn(i, name, width=width)

        self.GetSizer().Add(self.mask_table, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()
        self.mask_table.Bind(wx.EVT_CONTEXT_MENU, self.on_mask_table_context_menu)

    def update_mask_table(self, image_name, threshold, total_pixels, background_pixels, 
                         sample_pixels, background_values_count):
        """Update mask table with calculated data."""
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
            f"{sample_percent:.1f}%", f"{background_values_count}",
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
        """Handle file open event."""
        initial_count = self.notebook.GetPageCount()
        file_manager.open_image(self, self.notebook)
        
        # Only mark as regular image if a new image was actually added
        if self.notebook.GetPageCount() > initial_count:
            self.mark_as_sample_image(False)  # Regular image from file

    def on_save(self, event):
        """Handle file save event."""
        img = self.notebook.get_current_image()
        file_manager.save_image(self, img)

    def _get_current_image_data(self):
        """Get current image, name, and page with error checking."""
        pil_img = self.notebook.get_current_image()
        name = self.notebook.get_current_image_name()
        page = self.notebook.get_current_page()
        
        if pil_img is None or name is None:
            wx.MessageBox("No image selected", "Error", wx.ICON_ERROR)
            return None, None, None
            
        return pil_img, name, page

    def _convert_to_phase(self, pil_img):
        """Convert PIL image to phase array."""
        page = self.notebook.get_current_page()
        
        if page is not None and hasattr(page, 'is_mat_complex') and page.is_mat_complex:
            if hasattr(page, 'phase') and page.phase is not None:
                print("✓ Using phase data from complex .mat file")
                return page.phase
        
        print("Converting grayscale to phase using grayscaleToPhase()")
        grayscale = pil_img.convert("L")
        img_array = np.array(grayscale, dtype=float)
        return utRBPV.grayscaleToPhase(img_array)

    def _update_tables_with_mask_data(self, name, sample, background_mask, 
                                  background_values, threshold):
        """Update both metric and mask tables with calculated data."""
        if background_mask is None or background_values is None or threshold is None:
            print(f"⚠️ Skipping update for image '{name}': mask creation was cancelled or invalid data.")
            return

        total_pixels = sample.size
        background_pixels = np.sum(background_mask)
        sample_pixels = np.sum(~background_mask)

        self.update_mask_table(
            name, threshold, total_pixels, background_pixels, 
            sample_pixels, len(background_values)
        )

    def _get_zone_count_from_user(self, default=2):
        """Get number of zones from user input."""
        with wx.NumberEntryDialog(
            self, 
            "How many zones do you want to select?", 
            "Number of zones (1-20):", 
            "Zone Selection", 
            default, 1, 20
        ) as dlg:
            return dlg.GetValue() if dlg.ShowModal() == wx.ID_OK else None
        
    def _get_limit_from_user(self, default=64):
        """Get limit parameter from user input for Legendre analysis."""
        with wx.NumberEntryDialog(
            self, 
            "Enter the limit parameter for Fourier domain cropping:", 
            "Limit (64-1024):", 
            "Legendre Limit Parameter", 
            default, 64, 1024
        ) as dlg:
            return dlg.GetValue() if dlg.ShowModal() == wx.ID_OK else None    
        
    def _get_legendre_order_from_user(self, default=10):
        """Get number of Legendre coefficients from user input."""
        with wx.NumberEntryDialog(
            self, 
            "How many Legendre coefficients do you want to calculate?", 
            "Number of coefficients (1-15):", 
            "Legendre Order", 
            default, 1, 15
        ) as dlg:
            return dlg.GetValue() if dlg.ShowModal() == wx.ID_OK else None

    def _handle_zone_analysis_result(self, name, metric, result, num_zones):
        """Handle the result from zone-based analysis."""
        value, zone_stats = result if isinstance(result, tuple) else (result, [])

        if np.isnan(value):
            wx.MessageBox("No zones were selected.", "Info", wx.ICON_INFORMATION)
            return

        self.update_table_with_zones(name, metric, value, zone_stats, num_zones=num_zones)
        
        if zone_stats:
            self._show_zone_analysis_results(metric, num_zones, value, zone_stats)

    def _show_zone_analysis_results(self, metric, num_zones, avg_value, zone_stats):
        """Display zone analysis results to user."""
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
        """Generic handler for unwrapped metrics."""
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        sample = self._convert_to_phase(pil_img)
        unwrapped = rb.unwrap_with_scikit(sample)
        utRBPV.show_side_by_side(sample, unwrapped, 'Sample', 'Unwrapped', cmap='gray')
        
        background_mask, background_values, threshold = utRBPV.create_background_mask(
            unwrapped, method='otsu'
        )
        
        value = metric_func(sample, background_mask, manual=False, num_zones=2)
        
        self._update_tables_with_mask_data(
            name, sample, background_mask, background_values, threshold
        )
        self.update_table_with_zones(name, f"{metric_name}_Unwrapped_Background", value)

        info_text = (
            f"Unwrapped background {metric_name} analysis completed:\n\n"
            f"Method: Otsu (on unwrapped phase)\n"
            f"Applied threshold: {threshold:.6f}\n"
            f"{metric_name}: {value:.6f}\n"
            f"Number of background pixels analyzed: {len(background_values)}"
        )

        wx.MessageBox(info_text, f"{metric_name} Background (Unwrapped) Results", wx.ICON_INFORMATION)
        plt.close('all')

    def _process_metric_without_unwrap(self, event, metric_func, metric_name):
        """Generic handler for non-unwrapped metrics."""
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        sample = self._convert_to_phase(pil_img)
        
        background_mask, background_values, threshold = utRBPV.create_background_mask(
            sample, method='otsu'
        )
        
        value = metric_func(sample, background_mask, manual=False, num_zones=2)
        
        self._update_tables_with_mask_data(
            name, sample, background_mask, background_values, threshold
        )
        self.update_table_with_zones(name, f"{metric_name}_Background", value)
        
        info_text = (
            f"Background {metric_name} analysis completed:\n\n"
            f"Method: Otsu\n"
            f"Applied threshold: {threshold:.6f}\n"
            f"{metric_name}: {value:.6f}\n"
            f"Number of background pixels analyzed: {len(background_values)}"
        )

        wx.MessageBox(info_text, f"{metric_name} Background Results", wx.ICON_INFORMATION)
        plt.close('all')

    def _process_metric_zones(self, event, metric_func, metric_name):
        """Generic handler for zone-based metrics."""
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        sample = self._convert_to_phase(pil_img)
        num_zones = self._get_zone_count_from_user()
        if num_zones is None:
            return

        try:
            result = metric_func(sample, mask=None, manual=True, num_zones=num_zones)
            self._handle_zone_analysis_result(name, f"{metric_name}_Background_Zones", result, num_zones)
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)


    # Sample menu handlers  
    def on_usaf_sample(self, event):
        """Load USAF target sample (150 nm)"""
        # TODO: Implement sample loading logic
        # When implemented, call: self.mark_as_sample_image(True)
        wx.MessageBox("Loading USAF target (150 nm)", "Sample", wx.ICON_INFORMATION)
        self.mark_as_sample_image(True)
    
    def on_star_sample(self, event):
        """Load Star target sample (150 nm)"""
        # TODO: Implement sample loading logic
        # When implemented, call: self.mark_as_sample_image(True)
        wx.MessageBox("Loading Star target (150 nm)", "Sample", wx.ICON_INFORMATION)
        self.mark_as_sample_image(True)

    # Clear and cleanup
    def on_clear(self, event):
        """Clear all images and tables."""
        with wx.MessageDialog(
            self,
            message="Are you sure you want to close all images and remove all tables?",
            caption="Confirm Clear",
            style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING
        ) as dlg:
            if dlg.ShowModal() == wx.ID_YES:
                self._clear_all_data()

    def _clear_all_data(self):
        """Clear all application data and reset state."""
        while self.notebook.GetPageCount() > 0:
            self.notebook.DeletePage(0)

        self._close_detached_windows()
        self._destroy_tables()
        
        self.is_sample_image = False
        self.Layout()
        self.update_analysis_menu_state()

    def _close_detached_windows(self):
        """Close all detached windows except main frame."""
        for window in wx.GetTopLevelWindows():
            if isinstance(window, wx.Frame) and window != self:
                try:
                    window.Close()
                except:
                    pass

    def _destroy_tables(self):
        """Safely destroy both tables."""
        if self.metric_table:
            self.metric_table.Destroy()
            self.metric_table = None

        if self.mask_table:
            self.mask_table.Destroy()
            self.mask_table = None

    # Export handlers
    def on_metric_table_context_menu(self, event):
        """Show context menu for metrics table."""
        menu = wx.Menu()
        export_csv = menu.Append(wx.ID_ANY, "Export Metrics to CSV")
        export_xls = menu.Append(wx.ID_ANY, "Export Metrics to Excel")

        self.Bind(wx.EVT_MENU, self.on_export_metrics_csv, export_csv)
        self.Bind(wx.EVT_MENU, self.on_export_metrics_excel, export_xls)

        self.PopupMenu(menu)
        menu.Destroy()

    def on_mask_table_context_menu(self, event):
        """Show context menu for mask table."""
        menu = wx.Menu()
        export_csv = menu.Append(wx.ID_ANY, "Export Mask Data to CSV")
        export_xls = menu.Append(wx.ID_ANY, "Export Mask Data to Excel")

        self.Bind(wx.EVT_MENU, self.on_export_mask_csv, export_csv)
        self.Bind(wx.EVT_MENU, self.on_export_mask_excel, export_xls)

        self.PopupMenu(menu)
        menu.Destroy()

    def _export_table_data(self, table_type, file_format):
        """Generic method to export table data."""
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
        """Extract metrics table data for export."""
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
        """Extract mask table data for export."""
        if self.mask_table is None:
            return []
            
        column_names = [
            "Image Name", "Threshold", "Total Pixels", "Background Pixels",
            "Background %", "Sample Pixels", "Sample %", "Background Values"
        ]
        
        return [
            {col_name: self.mask_table.GetItemText(row, col) 
             for col, col_name in enumerate(column_names)}
            for row in range(self.mask_table.GetItemCount())
        ]

    # Tab and window management
    def on_tab_change(self, event):
        """Handle tab change event."""
        # Check if the current page is a sample image
        page = self.notebook.get_current_page()
        if page and hasattr(page, 'is_sample'):
            self.mark_as_sample_image(page.is_sample)
        else:
            self.mark_as_sample_image(False)
        
        event.Skip()

    def on_tab_close(self, event):
        """Handle tab close event."""
        wx.CallAfter(self._handle_tab_close_cleanup)
        event.Skip()

    def _handle_tab_close_cleanup(self):
        """Execute after closing a tab. Clean up if no images remain."""
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
        """Legendre background analysis with unwrapping."""
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        # Get limit parameter from user
        limit = self._get_limit_from_user(default=64)
        if limit is None:
            return  # User cancelled
        
        # Get number of coefficients from user
        order_max = self._get_legendre_order_from_user(default=5)
        if order_max is None:
            return  # User cancelled

        sample = self._convert_to_phase(pil_img)
        unwrapped = rb.unwrap_with_scikit(sample)
        utRBPV.show_side_by_side(sample, unwrapped, 'Sample', 'Unwrapped', cmap='gray')
        
        background_mask, background_values, threshold = utRBPV.create_background_mask(
            unwrapped, method='otsu'
        )
        
        # Pass limit and order_max parameters to legendre_background
        coefficients = rb.legendre_background(sample, background_mask, manual=False, num_zones=2, limit=limit, order_max=order_max)
        
        self._update_tables_with_mask_data(
            name, sample, background_mask, background_values, threshold
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

        info_text = (
            f"Unwrapped background Legendre analysis completed:\n\n"
            f"Method: Otsu (on unwrapped phase)\n"
            f"Applied threshold: {threshold:.6f}\n"
            f"Limit parameter: {limit}\n"
            f"Order (coefficients): {order_max}\n"
            f"Number of coefficients calculated: {len(coefficients)}\n"
            f"Number of background pixels analyzed: {len(background_values)}"
        )

        wx.MessageBox(info_text, "Legendre Background (Unwrapped) Results", wx.ICON_INFORMATION)
        plt.close('all')

    
    def on_legendre_background(self, event):
        """Legendre background analysis without unwrapping."""
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        # Get limit parameter from user
        limit = self._get_limit_from_user(default=64)
        if limit is None:
            return  # User cancelled
        
        # Get number of coefficients from user
        order_max = self._get_legendre_order_from_user(default=5)
        if order_max is None:
            return  # User cancelled

        sample = self._convert_to_phase(pil_img)
        
        background_mask, background_values, threshold = utRBPV.create_background_mask(
            sample, method='otsu'
        )
        
        # Pass limit and order_max parameters to legendre_background
        coefficients = rb.legendre_background(sample, background_mask, manual=False, num_zones=2, limit=limit, order_max=order_max)
        
        self._update_tables_with_mask_data(
            name, sample, background_mask, background_values, threshold
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
        
        info_text = (
            f"Background Legendre analysis completed:\n\n"
            f"Method: Otsu\n"
            f"Applied threshold: {threshold:.6f}\n"
            f"Limit parameter: {limit}\n"
            f"Order (coefficients): {order_max}\n"
            f"Number of coefficients calculated: {len(coefficients)}\n"
            f"Number of background pixels analyzed: {len(background_values)}"
        )

        wx.MessageBox(info_text, "Legendre Background Results", wx.ICON_INFORMATION)
        plt.close('all')
        
    def on_legendre_background_zones(self, event):
        """Legendre background analysis with zone selection."""
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
            result = rb.legendre_background(sample, mask=None, manual=True, num_zones=num_zones, limit=limit, order_max=order_max)
            
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
        """Handle main window close event."""
        self.Destroy()

    def on_exit(self, event):
        """Handle exit menu event."""
        self.Close()

    def on_computational_complexity(self, event):
        """Launch the computational complexity GUI."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))  # ui/
            parent_dir = os.path.dirname(current_dir)  # My_app/
            complexity_repo_path = os.path.join(parent_dir, 'complexity_algorithm')
            gui_script = os.path.join(complexity_repo_path, 'main.py')

            # Verificar si el archivo existe en Windows
            if not os.path.exists(gui_script):
                wx.MessageBox(
                    f"Complexity GUI not found at:\n{gui_script}\n\n"
                    "Please make sure the 'complexity_algorithm' repository "
                    "is located inside the My_app folder.",
                    "Error",
                    wx.ICON_ERROR
                )
                return

            # Convertir la ruta de Windows a formato WSL (/mnt/c/Users/...)
            wsl_path = gui_script.replace("\\", "/")
            if wsl_path[1] == ":":
                drive = wsl_path[0].lower()
                wsl_path = f"/mnt/{drive}/{wsl_path[3:]}"  # /mnt/c/Users/...

            # Ejecutar GUI dentro de WSL (sin CREATE_NEW_CONSOLE)
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
        """Calculate all metrics for all opened images."""
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
                    print(f"✓ Using phase data from complex .mat file for: {name}")
                else:
                    grayscale = pil_img.convert("L")
                    img_array = np.array(grayscale, dtype=float)
                    phase_img = utRBPV.grayscaleToPhase(img_array)
                    print(f"⚠ Converting grayscale to phase for: {name}")
                
                std_simple_val = rb.std_background(phase_img)
                mad_val = rb.mean_absolute_deviation_background(phase_img)

                self.update_table_with_zones(name, "STD_Simple", std_simple_val)
                self.update_table_with_zones(name, "MAD", mad_val)
                
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                continue

    def _process_global_metric(self, metric_func, metric_name, use_unwrap):
        """
        Generic handler for global phase metrics.
        Handles suffix '_Unwrapped' automatically based on use_unwrap.
        """
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        try:
            # --- Convertir use_unwrap en bool robustamente ---
            if isinstance(use_unwrap, str):
                use_unwrap = use_unwrap.strip().lower() in ("true", "1", "yes", "y", "t")
            elif use_unwrap is None:
                use_unwrap = False
            else:
                use_unwrap = bool(use_unwrap)

            # --- Calcular fase ---
            phase = self._convert_to_phase(pil_img)

            # --- Calcular métrica ---
            result = metric_func(phase, use_unwrap=use_unwrap)

            # --- Extraer valor principal ---
            if isinstance(result, tuple):
                value = result[0]
            else:
                value = result

            # --- Determinar sufijo dinámicamente ---
            suffix = "_Unwrapped" if use_unwrap else ""
            unwrap_text = " (Unwrapped)" if use_unwrap else ""

            # --- Actualizar tabla con nombre base + sufijo ---
            full_metric_name = f"{metric_name}{suffix}"
            self.update_table_with_zones(name, full_metric_name, value)

            # --- Mostrar mensaje ---
            wx.MessageBox(
                f"{metric_name}{unwrap_text} calculated successfully:\n\n"
                f"{full_metric_name}: {value:.6f}",
                "Analysis Complete",
                wx.ICON_INFORMATION
            )

        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)

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
        """Compute all global phase metrics Module 2 (wrapped and unwrapped) at once for the current image."""
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return

        try:
            phase = self._convert_to_phase(pil_img)

            # Lista de métricas con nombres finales
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

            # Calcular cada métrica con y sin unwrap
            for func, display_name in metrics:
                for use_unwrap in [False, True]:
                    try:
                        # Sufijo visual bonito (espacio, no guion bajo)
                        suffix = " Unwrapped" if use_unwrap else ""

                        # Detectar si la función acepta use_unwrap
                        params = inspect.signature(func).parameters
                        if 'use_unwrap' in params:
                            result = func(phase, use_unwrap=use_unwrap)
                        else:
                            result = func(phase)

                        value = result[0] if isinstance(result, tuple) else result

                        # Actualizar tabla con nombre completo
                        full_name = f"{display_name}{suffix}"
                        self.update_table_with_zones(name, full_name, value)

                        # Guardar en resumen
                        results_summary.append(f"{full_name}: {value:.6f}")

                    except Exception as metric_err:
                        results_summary.append(f"{display_name}{suffix}: ERROR ({metric_err})")

            # Mostrar cuadro resumen con todos los resultados
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
        """Load ground-truth image for comparison."""
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

    def on_ssim_comparison(self, event):
        """Calculate SSIM between current image and ground-truth."""
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return
        
        try:
            ssim_value = gtc.calculate_ssim(pil_img, self.ground_truth_data, use_unwrap=False)
            self.update_table_with_zones(name, "SSIM", ssim_value)
            wx.MessageBox(
                f"SSIM calculation completed:\n\nSSIM: {ssim_value:.6f}\n\n(1.0 = perfect similarity)", 
                "SSIM Result", 
                wx.ICON_INFORMATION
            )
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)

    def on_ssim_comparison_unwrapped(self, event):
        """Calculate SSIM with unwrapping."""
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return
        
        try:
            ssim_value = gtc.calculate_ssim(pil_img, self.ground_truth_data, use_unwrap=True)
            self.update_table_with_zones(name, "SSIM_Unwrapped", ssim_value)
            wx.MessageBox(
                f"SSIM (Unwrapped) calculation completed:\n\nSSIM: {ssim_value:.6f}", 
                "SSIM Unwrapped Result", 
                wx.ICON_INFORMATION
            )
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)

    def on_mse_comparison(self, event):
        """Calculate MSE between current image and ground-truth."""
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return
        
        try:
            mse_value = gtc.calculate_mse(pil_img, self.ground_truth_data, use_unwrap=False)
            self.update_table_with_zones(name, "MSE", mse_value)
            wx.MessageBox(
                f"MSE calculation completed:\n\nMSE: {mse_value:.6f}\n\n(Lower is better)", 
                "MSE Result", 
                wx.ICON_INFORMATION
            )
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)

    def on_mse_comparison_unwrapped(self, event):
        """Calculate MSE with unwrapping."""
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return
        
        try:
            mse_value = gtc.calculate_mse(pil_img, self.ground_truth_data, use_unwrap=True)
            self.update_table_with_zones(name, "MSE_Unwrapped", mse_value)
            wx.MessageBox(
                f"MSE (Unwrapped) calculation completed:\n\nMSE: {mse_value:.6f}", 
                "MSE Unwrapped Result", 
                wx.ICON_INFORMATION
            )
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)

    def on_psnr_comparison(self, event):
        """Calculate PSNR between current image and ground-truth."""
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return
        
        try:
            psnr_value = gtc.calculate_psnr(pil_img, self.ground_truth_data, use_unwrap=False)
            self.update_table_with_zones(name, "PSNR", psnr_value)
            wx.MessageBox(
                f"PSNR calculation completed:\n\nPSNR: {psnr_value:.2f} dB\n\n(Higher is better)", 
                "PSNR Result", 
                wx.ICON_INFORMATION
            )
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)

    def on_psnr_comparison_unwrapped(self, event):
        """Calculate PSNR with unwrapping."""
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            wx.MessageBox("Please load a ground-truth image first", "Error", wx.ICON_ERROR)
            return
        
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return
        
        try:
            psnr_value = gtc.calculate_psnr(pil_img, self.ground_truth_data, use_unwrap=True)
            self.update_table_with_zones(name, "PSNR_Unwrapped", psnr_value)
            wx.MessageBox(
                f"PSNR (Unwrapped) calculation completed:\n\nPSNR: {psnr_value:.2f} dB", 
                "PSNR Unwrapped Result", 
                wx.ICON_INFORMATION
            )
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.ICON_ERROR)
    # Finish of module 3