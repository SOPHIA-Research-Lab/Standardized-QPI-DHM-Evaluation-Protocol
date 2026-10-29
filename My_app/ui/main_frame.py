import traceback

import wx
import wx.aui
import wx.lib.dialogs
import numpy as np
from PIL import Image
import pandas as pd
from core import file_manager
from menus import file_menu,samples_menu, analysis_menu, help_menu
from detached_notebook import ImageNotebook
from analysis.module1 import utilitiesRBPV as utRBPV
from analysis.module1 import residual_background as rb
from analysis.module2 import global_phase as gp
from analysis.module3 import ground_truth_comparison as gtc
from analysis.catalog import METRIC_MAP
from analysis.cache import AnalysisCache
from ui.zone_selector import select_manual_zones
from ui.threshold_dialog import create_background_mask
from ui.legendre_viewer import show_legendre_reconstruction
import matplotlib.pyplot as plt
import subprocess
import sys
import os
import hashlib
import inspect

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
        # Masks, zone selections and unwrapped phase: one object, one
        # lifecycle. They used to be four attributes purged by hand.
        self._cache = AnalysisCache()
        # Raw, unrounded values behind each cell, keyed by image name and
        # column name. The tables show 4 decimals for readability; the export
        # must not inherit that truncation, and it used to because it read the
        # cell TEXT back out of the widget.
        self._metric_values = {}
        self._mask_values = {}
        self._skip_reason = None
        self._unwrap_warning_shown = False
        self._unwrap_choice = False
        self._legendre_params_shown = False
        self._legendre_limit = None
        self._legendre_order = None
        self.CreateStatusBar()
        
        # Metric IDs to functions mapping
        # The catalogue lives in analysis/catalog.py: it defines the
        # protocol, not this window. Read-only here.
        self.metric_map = METRIC_MAP

    def on_check_imagej_updates(self, event):
        """Periodically checks if the user edited any image inside ImageJ."""
        try:
            # Imported lazily and inside the try: pyimagej/scyjava may not be
            # installed at all, and this runs on a 1.5 s timer.
            from core.imagej_bridge import check_for_imagej_updates
            changes = check_for_imagej_updates()
        except Exception:
            return  # ImageJ not started or bridge unavailable; silently ignore
        
        for title, pil_img in changes:
            if title in self.imagej_tab_map:
                # A tab is already linked to this image: update it
                idx = self.imagej_tab_map[title]
                if idx < self.notebook.GetPageCount():
                    panel = self.notebook.GetPage(idx)
                    panel.pil_img = pil_img
                    self.notebook._update_bitmap(panel.bitmap_ctrl, pil_img, panel.scale_factor)
                    continue
            
            # Not linked yet: create a new tab and register it
            panel = self.notebook.add_image_tab(pil_img, f"{title}_ImageJ")
            self.imagej_tab_map[title] = self.notebook.GetPageCount() - 1

    def _setup_menus(self):
        # Set up the menu bar.
        menubar = wx.MenuBar()
        menubar.Append(file_menu.create(self, self.notebook), "&File")
        menubar.Append(samples_menu.create(self, self.notebook), "&Samples")
        menubar.Append(analysis_menu.create(self, self.notebook), "&Analysis")
        menubar.Append(self._create_imagej_menu(), "&ImageJ")
        menubar.Append(help_menu.create(self), "&Help")
        self.SetMenuBar(menubar)

        self.imagej_tab_map = {}  # Maps ImageJ image title -> notebook tab index

        # The timer is NOT started here: polling ImageJ from startup means a
        # failed import every 1.5 s for users without pyimagej installed, and
        # useless work for everyone who never opens ImageJ. It is started on
        # demand by _ensure_imagej_polling() the first time an image is sent.
        self.imagej_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_check_imagej_updates, self.imagej_timer)

    def _create_imagej_menu(self):
        """Create the ImageJ integration menu."""
        menu = wx.Menu()
        
        send_item = menu.Append(wx.ID_ANY, "Send current image to ImageJ")
        self.Bind(wx.EVT_MENU, self.on_send_to_imagej, send_item)
        
        return menu

    def on_send_to_imagej(self, event):
        """Send the currently selected image to ImageJ for editing."""
        from core.imagej_bridge import show_imagej_gui, pil_to_imageplus
        
        pil_img, name, page = self._get_current_image_data()
        if pil_img is None:
            return
        
        show_imagej_gui(parent_window=self)
        pil_to_imageplus(pil_img, title=name, parent_window=self)
        self._ensure_imagej_polling()
        
        # Only create a linked tab the FIRST time this image is sent
        if name not in self.imagej_tab_map:
            self.notebook.add_image_tab(pil_img, f"{name}_ImageJ")
            self.imagej_tab_map[name] = self.notebook.GetPageCount() - 1

    def _ensure_imagej_polling(self):
        """Start polling ImageJ for edits, once an image has actually been sent."""
        if not self.imagej_timer.IsRunning():
            self.imagej_timer.Start(1500)  # Check every 1.5 seconds

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
        self.Bind(wx.EVT_BUTTON, self.on_save)

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
                    module1_item.Enable(True)
                if module2_item:
                    module2_item.Enable(True)
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

        self.metric_table.Bind(wx.EVT_CONTEXT_MENU, self.on_metric_table_context_menu)
       
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

    @staticmethod
    def _record_value(store, image_name, col_name, value):
        """Keep the full-precision number behind a formatted cell."""
        try:
            store.setdefault(image_name, {})[col_name] = float(value)
        except (TypeError, ValueError):
            pass

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
        self._record_value(self._metric_values, image_name, metric_column, value)
        
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
           
            # Search for the value with different variants of the name
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
                    nombre_img = self.metric_table.GetItemText(row, 0)
                    self._record_value(self._metric_values, nombre_img, col_name, val)
            except (ValueError, TypeError) as e:
                print(f" Error setting value for {col_name}: {e}")
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
        for col_name, valor in (("Threshold", threshold),
                                ("Total Pixels", total_pixels),
                                ("Background Pixels", background_pixels),
                                ("Background %", bg_percent),
                                ("Sample Pixels", sample_pixels),
                                ("Sample %", sample_percent)):
            self._record_value(self._mask_values, image_name, col_name, valor)

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

    def _get_current_image_data(self):
        pil_img = self.notebook.get_current_image()
        name = self.notebook.get_current_image_name()
        page = self.notebook.get_current_page()
        
        if pil_img is None or name is None:
            wx.MessageBox("No image selected", "Error", wx.ICON_ERROR)
            return None, None, None
            
        return pil_img, name, page


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

        self._cache.clear()
        self._metric_values.clear()
        self._mask_values.clear()

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
        solo_excel = menu.Append(wx.ID_ANY, "Export Metrics table to Excel")
        solo_csv = menu.Append(wx.ID_ANY, "Export Metrics table to CSV")
        menu.AppendSeparator()
        export_excel = menu.Append(wx.ID_ANY, "Export All Tables to Excel")
        export_csv = menu.Append(wx.ID_ANY, "Export All Tables to CSV")

        self.Bind(wx.EVT_MENU, self.on_export_metrics_excel, solo_excel)
        self.Bind(wx.EVT_MENU, self.on_export_metrics_csv, solo_csv)
        self.Bind(wx.EVT_MENU, self.on_export_all_excel, export_excel)
        self.Bind(wx.EVT_MENU, self.on_export_all_csv, export_csv)
        self.PopupMenu(menu)
        menu.Destroy()

    def on_mask_table_context_menu(self, event):
        # Show context menu for mask table.
        menu = wx.Menu()
        solo_excel = menu.Append(wx.ID_ANY, "Export Mask table to Excel")
        solo_csv = menu.Append(wx.ID_ANY, "Export Mask table to CSV")
        menu.AppendSeparator()
        export_excel = menu.Append(wx.ID_ANY, "Export All Tables to Excel")
        export_csv = menu.Append(wx.ID_ANY, "Export All Tables to CSV")

        self.Bind(wx.EVT_MENU, self.on_export_mask_excel, solo_excel)
        self.Bind(wx.EVT_MENU, self.on_export_mask_csv, solo_csv)
        self.Bind(wx.EVT_MENU, self.on_export_all_excel, export_excel)
        self.Bind(wx.EVT_MENU, self.on_export_all_csv, export_csv)

        self.PopupMenu(menu)
        menu.Destroy()

    def export_all_tables(self, file_format, event=None, only=None):
        """
        Export the result tables.

        Parameters
        ----------
        file_format : {'excel', 'csv'}
        only : tuple of str, optional
            Restrict the export to these tables ("Metrics", "Mask Data").
            Omitted, every table holding data is exported.
        """
        table_info = {
            "Metrics": (self.metric_table, self._get_metric_table_data),
            "Mask Data": (self.mask_table, self._get_mask_table_data),
        }
        if only:
            table_info = {k: v for k, v in table_info.items() if k in only}

        # Check if there is at least one table with data
        has_data = any(
            table is not None and table.GetItemCount() > 0
            for table, _ in table_info.values()
        )

        if not has_data:
            wx.MessageBox(
                "No tables with data to export.",
                "Error",
                wx.OK | wx.ICON_ERROR
            )
            return

        try:

            # Export to Excel
            if file_format.lower() == "excel":

                with wx.FileDialog(
                    self,
                    "Save Excel file",
                    wildcard="Excel files (*.xlsx)|*.xlsx",
                    style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                ) as dlg:

                    if dlg.ShowModal() == wx.ID_CANCEL:
                        return

                    path = dlg.GetPath()

                exported = 0

                with pd.ExcelWriter(path, engine="openpyxl") as writer:

                    for sheet_name, (table, data_method) in table_info.items():

                        if table is None or table.GetItemCount() == 0:
                            continue

                        data = data_method()

                        if not data:
                            continue

                        df = pd.DataFrame(data)

                        df.to_excel(
                            writer,
                            sheet_name=sheet_name[:31],   # Excel sheet name limit
                            index=False
                        )

                        exported += 1

                wx.MessageBox(
                    f"{exported} table(s) exported successfully.",
                    "Success",
                    wx.OK | wx.ICON_INFORMATION
                )
            # Export to CSV
            elif file_format.lower() == "csv":

                with wx.DirDialog(
                    self,
                    "Select folder to save CSV files",
                    style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
                ) as dlg:

                    if dlg.ShowModal() == wx.ID_CANCEL:
                        return

                    folder = dlg.GetPath()

                exported = 0

                for file_name, (table, data_method) in table_info.items():

                    if table is None or table.GetItemCount() == 0:
                        continue

                    data = data_method()

                    if not data:
                        continue

                    df = pd.DataFrame(data)

                    csv_path = os.path.join(
                        folder,
                        f"{file_name}.csv"
                    )

                    df.to_csv(csv_path, index=False)

                    exported += 1

                wx.MessageBox(
                    f"{exported} CSV file(s) exported successfully.",
                    "Success",
                    wx.OK | wx.ICON_INFORMATION
                )

        except Exception as e:

            wx.MessageBox(
                f"Export failed:\n{str(e)}",
                "Error",
                wx.OK | wx.ICON_ERROR
            )

    def on_export_all_excel(self, event):
        self.export_all_tables("excel")

    def on_export_all_csv(self, event):
        self.export_all_tables("csv")

    def on_export_metrics_excel(self, event):
        self.export_all_tables("excel", only=("Metrics",))

    def on_export_metrics_csv(self, event):
        self.export_all_tables("csv", only=("Metrics",))

    def on_export_mask_excel(self, event):
        self.export_all_tables("excel", only=("Mask Data",))

    def on_export_mask_csv(self, event):
        self.export_all_tables("csv", only=("Mask Data",))


    def _get_metric_table_data(self):
        # Extract metrics table data for export.
        if self.metric_table is None:
            return []
        
        return self._table_rows(self.metric_table, self._metric_values)

    def _table_rows(self, table, raw_store):
        """
        Build the export rows, preferring the stored full-precision number over
        the text shown in the cell. Columns with no raw value (image name, zone
        coordinates) fall back to the displayed text.
        """
        filas = []
        n_cols = table.GetColumnCount()
        nombres = [table.GetColumn(c).GetText() for c in range(n_cols)]

        for row in range(table.GetItemCount()):
            nombre_img = table.GetItemText(row, 0)
            crudos = raw_store.get(nombre_img, {})
            fila = {}
            for col, col_name in enumerate(nombres):
                texto = table.GetItemText(row, col)
                if col_name in crudos and texto != "":
                    fila[col_name] = crudos[col_name]
                else:
                    fila[col_name] = texto
            filas.append(fila)
        return filas

    def _get_mask_table_data(self):
        # Extract mask table data for export.
        if self.mask_table is None:
            return []

        # Column names now come from the widget itself. They match the list this
        # used to hardcode, and can no longer drift apart from it.
        return self._table_rows(self.mask_table, self._mask_values)

    # Tab and window management
    def on_tab_change(self, event):
        page = self.notebook.get_current_page()
        if page and hasattr(page, 'is_sample'):
            self.mark_as_sample_image(page.is_sample)
        else:
            self.mark_as_sample_image(False)
        
        event.Skip()

    def on_tab_close(self, event):
        # Handle tab close event.
        idx = event.GetSelection()
        closed_image_name = None
        if idx != wx.NOT_FOUND and idx < self.notebook.GetPageCount():
            closed_image_name = self.notebook.GetPageText(idx)
        
        wx.CallAfter(self._handle_tab_close_cleanup, closed_image_name)
        event.Skip()

    def _handle_tab_close_cleanup(self, closed_image_name=None):

        if closed_image_name:
            self._cache.forget_image(closed_image_name)
            
            # Also clear any per-image zone selections tied to this image
            keys_to_remove = [
                k for k in self._cache.zones
                if k == f"{closed_image_name}_zones" or k.startswith(f"{closed_image_name}_")
            ]
            for k in keys_to_remove:
                del self._cache.zones[k]
        
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

    
    def on_save(self, event):
    
                table_info = {
                    "Metrics": self.metric_table,
                    "Mask Data": self.mask_table,
                }
                has_data = any(
                    table is not None and table.GetItemCount() > 0
                    for table in table_info.values()
                )
    
                if not has_data:
                    wx.MessageBox(
                        "There are no results to save.",
                        "No Data",
                        wx.OK | wx.ICON_INFORMATION
                    )
                    return
                self.export_all_tables("excel")    

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
            # 'Main.py' with a capital M: that is the real file name. Spelling it
            # 'main.py' only worked because Windows paths are case
            # insensitive; on Linux/macOS it failed with a message blaming
            # a missing repository. Module IV needs Linux 'perf', so those
            # are exactly the platforms where this has to work.
            gui_script = os.path.join(complexity_repo_path, 'Main.py')

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

            subprocess.Popen([sys.executable, gui_script])

            # No confirmation dialog here on purpose: it used to pop up the
            # instant the process was launched, which pulled the focus back to
            # this window and left the complexity window opening behind it.
            # The status bar says the same thing without taking the foreground.
            self.SetStatusText(
                "Computational Complexity launched. Profiling needs Linux "
                "or WSL2; the window itself opens anywhere."
            )

        except Exception as e:
            wx.MessageBox(
                f"Error launching the GUI:\n{str(e)}",
                "Error",
                wx.ICON_ERROR
            )


    # Residual Background Metrics (Module 1)
  

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

    # Each RMS variant drags its matching mean along (see _RMS_MEAN_PAIRS below)
    _RMS_MEAN_PAIRS = (
        ('rms_background', 'mean_background', 'Mean - Background'),
        ('rms_unwrapped', 'mean_unwrapped', 'Mean - Unwrapped Background'),
        ('rms_zones', 'mean_zones', 'Mean - Background Zones'),
    )

    def _ask_unwrap_once(self):
        """
        Ask ONCE per batch whether phase unwrapping may be applied.

        The three paths used to behave differently: zone metrics got a Yes/No
        question, Module 1 unwrapped metrics got an OK-only notice they could
        not refuse, and Module 2 unwrapped metrics got nothing at all and were
        unwrapped silently. Every path now shares this single answer.

        Answering No is binding: metrics that REQUIRE unwrapped phase are
        skipped instead of being silently computed on wrapped data. Zone
        metrics, where unwrapping is optional, run on the wrapped phase.
        """
        if self._unwrap_warning_shown:
            return self._unwrap_choice

        dlg = wx.MessageDialog(
            self,
            "Some of the selected metrics use phase unwrapping.\n\n"
            "Unwrapping removes 2π discontinuities, but it can jeopardize "
            "phase compensation and introduce errors of its own.\n\n"
            "Yes - apply unwrapping where the metric calls for it.\n"
            "No  - do not unwrap. Metrics that require unwrapped phase will be "
            "skipped; zone metrics will use the wrapped phase.",
            "Apply Phase Unwrapping?",
            style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION
        )
        respuesta = dlg.ShowModal()
        dlg.Destroy()

        self._unwrap_choice = (respuesta == wx.ID_YES)
        self._unwrap_warning_shown = True
        return self._unwrap_choice

    def _calculate_multiple_metrics(self, metrics_dict, image_indices):
        # Calculate multiple metrics for multiple images with single threshold per image per state.

        # The mean is exactly what RMS adds over STD, because
        # RMS^2 = mean^2 + STD^2. On its own it is not a quality figure, so it
        # is not offered as a separate checkbox: it is computed whenever RMS
        # is, and the three columns can then be read (and checked) together.
        metrics_dict = dict(metrics_dict)
        for rms_id, mean_id, mean_label in self._RMS_MEAN_PAIRS:
            if rms_id in metrics_dict and mean_id not in metrics_dict:
                metrics_dict[mean_id] = mean_label

        # Verify ground-truth if needed
        if not self._ensure_ground_truth_loaded(metrics_dict):
            return
        self._unwrap_warning_shown = False

        # One question for the whole batch, asked up front and only when
        # something in the selection can actually use unwrapping.
        necesita_unwrap = any(
            'zones' in mid or self.metric_map.get(mid, (None, None, False))[2]
            for mid in metrics_dict
        )
        if necesita_unwrap:
            self._ask_unwrap_once()
        self._legendre_params_shown = False
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
            
            # If shared mode, clear cache to start fresh and reset shared zones
            if zones_mode == 'shared':
                self._cache.reset_zones(shared=True)
            # If per-image or per-metric mode, clear cache for fresh zones
            else:
                self._cache.reset_zones()
        
        total_operations = len(metrics_dict) * len(image_indices)
        
        # Progress bar
        progress = wx.ProgressDialog(
            "Calculating Metrics",
            "Processing...",
            maximum=total_operations,
            parent=self,
            # PD_CAN_ABORT gives the dialog a Cancel button, and Update() then
            # reports whether it was pressed. Without it, a long batch could
            # only be stopped by killing the process.
            # No PD_REMAINING_TIME: wx estimates it from the progress rate and
            # shows "Unknown" until it has enough samples, which for a short
            # batch is the whole run.
            style=(wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
                   | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME)
        )
        
        current_op = 0
        results_summary = []
        cancelado = False
        
        try:
            # Process each image
            for img_idx in image_indices:
                if cancelado:
                    break
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
                    # Update() refreshes the bar (which pumps the event loop, so
                    # the window keeps repainting) and reports Cancel presses.
                    seguir, _ = progress.Update(
                        current_op,
                        f"Calculating {metric_label} for {name}..."
                    )
                    if not seguir:
                        cancelado = True
                        results_summary.append(
                            "\n*** Cancelled by the user. Everything above was "
                            "already written to the table. ***"
                        )
                        break
                    
                    try:
                        value = self._calculate_single_metric_cached(
                            metric_id,
                            sample,
                            name,
                            masks_cache
                        )
                        
                        if value is None:
                            motivo = self._skip_reason or "skipped"
                            results_summary.append(f" ⊘ {metric_label}: {motivo}")
                        elif not np.isfinite(value):
                            results_summary.append(
                                f" ⊘ {metric_label}: undefined (NaN) — the estimator "
                                f"does not apply to this data"
                            )
                        else:
                            results_summary.append(f" ✓ {metric_label}: {value:.6f}")
                            
                    except Exception as e:
                        results_summary.append(f" ✗ {metric_label}: ERROR - {str(e)}")
                    
                    current_op += 1
            
            if not cancelado:
                progress.Update(total_operations)
        
        finally:
            progress.Destroy()
        
        # Show summary
        summary_text = "\n".join(results_summary)
        
        try:
            dlg = wx.lib.dialogs.ScrolledMessageDialog(
                self,
                summary_text,
                "Metrics Calculation Cancelled" if cancelado else "Metrics Calculation Complete",
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
            elif metric_id.startswith(('mmm_', 'gradient_', 'tsm_',
                                    'laplacian_', 'spatial_freq_', 'global_entropy',
                                    'sharpness_', 'fwhm_global')):
                module2_metrics.append(metric_id)
            elif metric_id.startswith(('ssim', 'mse', 'psnr')):
                module3_metrics.append(metric_id)
            else:
                module1_metrics.append(metric_id)
        

        apply_unwrap = self._ask_unwrap_once() if zone_metrics else False

        zone_sample = sample  # copia separada para las métricas de zonas
        if apply_unwrap:
            try:
                zone_sample = self._get_unwrapped_sample(image_name, sample)  # NO pisa 'sample'
            except Exception as e:
                wx.MessageBox(f"Error applying unwrap to {image_name}: {e}", "Error", wx.ICON_ERROR)
                return masks_cache

        if zone_metrics:
            # Las metricas de zonas leen de aqui, no de 'sample'
            masks_cache['zone_sample'] = zone_sample
            masks_cache['zones_unwrapped'] = apply_unwrap
        
        if module1_metrics:
            needs_unwrapped = any('unwrapped' in mid for mid in module1_metrics)
            needs_wrapped = any('unwrapped' not in mid and 'zones' not in mid for mid in module1_metrics)

            if needs_wrapped:
                background_mask, threshold, processed_sample = self._get_or_create_background_mask(
                    image_name, sample, needs_unwrap=False
                )
                if background_mask is not None:
                    masks_cache[False] = (background_mask, threshold, processed_sample)
                    self._update_tables_with_mask_data(
                        image_name, sample, background_mask, threshold
                    )

            if needs_unwrapped and self._ask_unwrap_once():
                try:
                    background_mask, threshold, unwrapped_sample = self._get_or_create_background_mask(
                        image_name, sample, needs_unwrap=True
                    )
                    if background_mask is not None:
                        masks_cache[True] = (background_mask, threshold, unwrapped_sample)
                except Exception as e:
                    import traceback
                    print(f"[DEBUG] Excepción calculando unwrap para {image_name}:")
                    traceback.print_exc()
                    masks_cache['unwrap_error'] = str(e)
                    wx.MessageBox(f"Error applying unwrap for {image_name}: {e}", "Error", wx.ICON_ERROR)
            
        # Smart zone handling according to mode
        if zone_metrics:
            
            if zones_mode == 'shared':
                # SHARED MODE: Use same zones for all images and all metrics
                if self._cache.shared_zones is None:
                    # First image: ask for zones
                    num_zones = self._get_zone_count_from_user()
                    if num_zones is None:
                        return masks_cache
                    
                    zones = select_manual_zones(zone_sample, num_zones)
                    if not zones or len(zones) == 0:
                        wx.MessageBox("No zones selected.", "Info", wx.ICON_INFORMATION)
                        return masks_cache
                    
                    # Save as shared zones
                    self._cache.shared_zones = (zones, num_zones)
                else:
                    zones, num_zones = self._cache.shared_zones
                
                masks_cache['zones'] = zones
                masks_cache['num_zones'] = num_zones
            
            elif zones_mode == 'per_metric':
                # PER-METRIC MODE: Each metric gets its own zones
                metric_zones = {}
                
                for metric_id in zone_metrics:
                    cache_key = f"{image_name}_{metric_id}"
                    
                    if cache_key in self._cache.zones:
                        zones, num_zones = self._cache.zones[cache_key]
                    else:
                        num_zones = self._get_zone_count_from_user()
                        if num_zones is None:
                            return masks_cache
                        
                        zones = select_manual_zones(zone_sample, num_zones)
                        if not zones or len(zones) == 0:
                            wx.MessageBox(f"No zones selected for {metric_id}.", "Info", wx.ICON_INFORMATION)
                            return masks_cache
                        
                        self._cache.zones[cache_key] = (zones, num_zones)
                    
                    metric_zones[metric_id] = (zones, num_zones)
                
                masks_cache['metric_zones'] = metric_zones
            
            else:  # PER-IMAGE MODE
                cache_key = f"{image_name}_zones"
                
                if cache_key in self._cache.zones:
                    zones, num_zones = self._cache.zones[cache_key]
                else:
                    num_zones = self._get_zone_count_from_user()
                    if num_zones is None:
                        return masks_cache
                    
                    zones = select_manual_zones(zone_sample, num_zones)
                    if not zones or len(zones) == 0:
                        wx.MessageBox(f"No zones selected for {image_name}.", "Info", wx.ICON_INFORMATION)
                        return masks_cache
                    
                    self._cache.zones[cache_key] = (zones, num_zones)
                
                masks_cache['zones'] = zones
                masks_cache['num_zones'] = num_zones
        
        return masks_cache


    def _skip(self, reason):
        """
        Record WHY a metric could not be computed, and return None.

        Every failure path used to be 'except Exception as e: return None',
        discarding the exception, so the summary only ever said "Skipped" and
        a cancelled dialog looked exactly like a real bug. The reason is read
        back by _calculate_multiple_metrics when it builds the report.
        """
        self._skip_reason = reason
        return None

    def _calculate_single_metric_cached(self, metric_id, sample, image_name, masks_cache):
        """Calculate a single metric using pre-calculated cached masks."""
        
        self._skip_reason = None

        # Search for metric in the map
        if metric_id not in self.metric_map:
            return self._skip("unknown metric id (not in metric_map)")
        
        metric_func, table_column, needs_unwrap = self.metric_map[metric_id]

        if needs_unwrap and not self._ask_unwrap_once():
            return self._skip("phase unwrapping was declined for this batch")
        
        # SPECIAL HANDLING FOR LEGENDRE METRICS
        if 'legendre' in metric_id.lower():
            if not self._legendre_params_shown:
                limit = self._get_limit_from_user(default=64)
                if limit is None:
                    return self._skip("Legendre limit dialog was cancelled")
                
                ordermax = self._get_legendre_order_from_user(default=5)
                if ordermax is None:
                    return self._skip("Legendre limit dialog was cancelled")
                
                self._legendre_limit = limit
                self._legendre_order = ordermax
                self._legendre_params_shown = True
            else:
                limit = self._legendre_limit
                ordermax = self._legendre_order
            
            # # Get ordermax parameter
            # ordermax = self._get_legendre_order_from_user(default=5)
            # if ordermax is None:
            #     return None
            
            # For zone-based Legendre
            if 'zones' in metric_id:
                if 'zones' not in masks_cache or 'num_zones' not in masks_cache:
                    return self._skip("Legendre order dialog was cancelled")
                
                zones = masks_cache['zones']
                num_zones = masks_cache['num_zones']
                
                try:
                    result = metric_func(sample, mask=None, manual=True, num_zones=num_zones, 
                                    limit=limit, order_max=ordermax, zones=zones)
                    
                    if isinstance(result, tuple):
                        mean_coeffs, zone_stats = result
                    else:
                        return self._skip("Legendre returned no per-zone coefficients")
                    
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
                    return self._skip(f"{type(e).__name__}: {e}")
            
            # For background Legendre (no zones) - USE CACHED MASK
            else:
                # Get the cached mask (already calculated in _precalculate_masks)
                if needs_unwrap not in masks_cache:
                    return self._skip("no background mask: the threshold dialog was cancelled, or unwrapping failed")
                
                background_mask, threshold, processed_sample = masks_cache[needs_unwrap]
                
                try:
                    # Calculate Legendre coefficients using the CACHED mask
                    coefficients = metric_func(
                        processed_sample, background_mask, manual=False,
                        num_zones=2, limit=limit, order_max=ordermax,
                        # The analysis layer no longer knows about viewers:
                        # it hands the surface back here to be displayed.
                        show_reconstruction=show_legendre_reconstruction,
                    )
                    
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
                    return self._skip(f"{type(e).__name__}: {e}")
        
        # ZONE METRICS: Use cached zones (can be shared, per-image, or per-metric)
        if 'zones' in metric_id:
            # Check if this is per-metric mode
            if 'metric_zones' in masks_cache:
                # Per-metric mode: each metric has different zones
                if metric_id not in masks_cache['metric_zones']:
                    return self._skip("no zones were selected for this metric")
                zones, num_zones = masks_cache['metric_zones'][metric_id]
            else:
                # Shared or per-image mode: all metrics use same zones
                if 'zones' not in masks_cache or 'num_zones' not in masks_cache:
                    return self._skip("no zones were selected for this metric")
                zones = masks_cache['zones']
                num_zones = masks_cache['num_zones']
            
            # Extract metric name from metric_id (e.g., 'std_zones' -> 'std')
            metric_name = metric_id.split('_')[0].lower()
            
            # Unwrapped copy if the user asked for it in _precalculate_masks
            zone_sample = masks_cache.get('zone_sample', sample)

            # Calculate metric in each zone
            values = []
            zone_stats = []
            for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
                zone_img = zone_sample[ymin:ymax, xmin:xmax]
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
                return self._skip("no ground-truth image loaded")
            
            try:
                # Module 3 never unwraps: see _prepare_sample_and_gt.
                value = metric_func(sample, self.ground_truth_data)
                if isinstance(value, tuple):
                    value = value
                self.update_table_with_zones(image_name, table_column, value)
                return value
            except Exception as e:
                return self._skip(f"{type(e).__name__}: {e}")
        
        # MODULE 2: Global metrics (NO need mask)
        elif metric_id.startswith(('mmm_', 'gradient_', 'tsm_', 
                                'laplacian_', 'spatial_freq_', 'global_entropy',
                                'sharpness_', 'fwhm_global')):
            if needs_unwrap:
                try:
                    processed_sample = self._get_unwrapped_sample(image_name, sample)
                except Exception as e:
                    return self._skip(f"{type(e).__name__}: {e}")
            else:
                processed_sample = sample
            
            params = inspect.signature(metric_func).parameters
            
            try:
                if 'use_unwrap' in params:
                    # Already unwrapped above when needed: asking again here
                    # would unwrap twice and change the result.
                    value = metric_func(processed_sample, use_unwrap=False)
                else:
                    value = metric_func(processed_sample)
                
                if isinstance(value, tuple):
                    value = value
                
                self.update_table_with_zones(image_name, table_column, value)
                return value
            except Exception as e:
                return self._skip(f"{type(e).__name__}: {e}")
        
        # MODULE 1: Background metrics (NEED mask)
        else:
            if needs_unwrap not in masks_cache:
                return self._skip("no background mask: the threshold dialog was cancelled, or unwrapping failed")
            
            background_mask, threshold, processed_sample = masks_cache[needs_unwrap]
            
            try:
                # Module 1 metrics take (image, mask). The manual/num_zones
                # parameters they used to carry were for a zone branch that
                # this frame has handled itself for a long time.
                value = metric_func(processed_sample, background_mask)
                self.update_table_with_zones(image_name, table_column, value)
                return value
            except Exception as e:
                return self._skip(f"{type(e).__name__}: {e}")


    def _sample_fingerprint(self, sample):
        """
        Cheap content fingerprint used to detect whether the phase data for
        an image actually changed (edited via ImageJ, reloaded, or a
        different image reusing the same tab name), so the cached background
        mask can be safely reused when nothing changed.
        """
        arr = np.ascontiguousarray(sample)
        h = hashlib.md5()
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        if arr.size > 4_000_000:
            # Hashing every byte of a 4096x4096 float array costs ~0.3 s and
            # this runs on every cache lookup. A strided sample plus shape and
            # dtype is enough to notice that an image was edited or replaced.
            flat = arr.reshape(-1)
            h.update(flat[::max(1, flat.size // 1_000_000)].tobytes())
        else:
            h.update(arr.tobytes())
        return h.hexdigest()
    
    
    def _get_unwrapped_sample(self, image_name, sample):
        """
        Unwrap a phase image ONCE per image and reuse the result.

        Unwrapping is by far the most expensive step of the pipeline
        (~0.85 s for 1920x1200, i.e. 4-40x any single metric), so running
        it per metric made a batch of 8 unwrapped metrics 16x slower than
        needed.

        It is also NOT idempotent: skimage's unwrap_phase expects input in
        [-pi, pi], so feeding it already-unwrapped data changes the values
        (differences of tens of radians). Everything downstream must
        therefore receive the cached array with use_unwrap=False, instead
        of unwrapping here and asking the metric to unwrap again.
        """
        fingerprint = self._sample_fingerprint(sample)
        cached = self._cache.get_unwrapped(image_name, fingerprint)
        if cached is not None:
            return cached

        # Eviction is the cache's business, not this method's.
        return self._cache.store_unwrapped(
            image_name, fingerprint, rb.unwrap_with_scikit(sample)
        )


    def _get_or_create_background_mask(self, image_name, sample, needs_unwrap):
        """
        Return (background_mask, threshold, processed_sample).

        The Otsu background mask is computed ONLY ONCE per image, regardless
        of whether wrapped or unwrapped metrics are requested. Unwrapping
        only changes phase VALUES (removes 2*pi discontinuities); it does
        NOT change which pixels belong to the background vs the sample, so
        the same spatial mask is valid for both wrapped and unwrapped
        metrics. Recomputing Otsu separately for each state was causing the
        binarization dialog to appear twice when calculating "all Module 1
        metrics" (which includes both wrapped and unwrapped variants).

        The unwrapped version of the sample is also cached (once computed),
        so repeated unwrapped metrics for the same image don't re-run
        phase unwrapping unnecessarily either.
        """
        fingerprint = self._sample_fingerprint(sample)
        cached = self._cache.masks.get(image_name)

        if cached is None or cached['fingerprint'] != fingerprint:
            # New image, or this image's data changed: (re)compute Otsu
            # exactly once, always on the original (wrapped) sample.
            background_mask, threshold = create_background_mask(
                sample, method='otsu', parent=self
            )
            if background_mask is None:
                # User cancelled the threshold dialog: nothing to cache and
                # no valid mask, so the caller must skip this metric.
                return None, None, None
            cached = {
                'fingerprint': fingerprint,
                'mask': background_mask,
                'threshold': threshold,
            }
            self._cache.masks[image_name] = cached

        if needs_unwrap:
            processed_sample = self._get_unwrapped_sample(image_name, sample)
        else:
            processed_sample = sample

        return cached['mask'], cached['threshold'], processed_sample

    # Finish of module 1

    
    # Global Phase Distortion Metrics (Module 2)


    # def on_curvature_global(self, event):
    #     self._process_global_metric(gp.reconstruction_background, "Legendre coefficients", False)

    # def on_curvature_global_unwrapped(self, event):
    #     self._process_global_metric(gp.reconstruction_background, "Legendre coefficients", True)


    # Finish of module 2


    # Ground-Truth Comparisons (Module 3)
    
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
        has_module3 = any(mid.startswith(('ssim', 'mse', 'psnr')) for mid in metrics_dict.keys())
        
        if not has_module3:
            return True  # Ground-truth not needed
        
        # Check current ground-truth status
        gt_loaded = hasattr(self, 'ground_truth_data') and self.ground_truth_data is not None
        
        # 🔹 CASE 1: Ground-truth already loaded
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
        
        # 🔹 CASE 2: No ground-truth loaded
        else:
            message = (
                "No Ground-Truth Loaded\n\n"
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


    # Finish of module 3