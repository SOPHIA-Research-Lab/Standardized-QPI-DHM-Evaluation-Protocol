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
from analysis.module3 import ground_truth_comparison as gtc
from analysis.catalog import METRIC_MAP
from analysis.cache import AnalysisCache
from ui.zone_selector import select_manual_zones
from ui.threshold_dialog import create_background_mask
from ui.legendre_viewer import show_legendre_reconstruction
from ui.tsm_dialog import ask_tsm_threshold
from ui import icons
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
        # One root panel: a control on a bare wx.Frame paints a different grey.
        self.root_panel = wx.Panel(self)

        # Image on top, result tables below, with a sash the user can drag.
        self.splitter = wx.SplitterWindow(
            self.root_panel, style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)
        self.splitter.SetMinimumPaneSize(80)

        self.tables_panel = wx.Panel(self.splitter)

        # A second sash, so the one-row mask table does not take half the pane.
        self.tables_splitter = wx.SplitterWindow(
            self.tables_panel, style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)
        self.tables_splitter.SetMinimumPaneSize(60)

        tablesSizer = wx.BoxSizer(wx.VERTICAL)
        tablesSizer.Add(self.tables_splitter, 1, wx.EXPAND)
        self.tables_panel.SetSizer(tablesSizer)

        # An empty AuiNotebook paints its own grey, so it is hidden until use.
        self.workspace = wx.Panel(self.splitter)

        self.welcome_panel = wx.Panel(self.workspace)
        welcomeSizer = wx.BoxSizer(wx.VERTICAL)
        message = wx.StaticText(
            self.welcome_panel,
            label="No image open\n\n"
                  "File > Open (Ctrl+O) to load a compensated phase map,\n"
                  "or pick one from the Samples menu.",
            style=wx.ALIGN_CENTRE_HORIZONTAL)
        source = message.GetFont()
        source.SetPointSize(source.GetPointSize() + 1)
        message.SetFont(source)
        message.SetForegroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT))
        welcomeSizer.AddStretchSpacer()
        welcomeSizer.Add(message, 0, wx.ALIGN_CENTER)
        welcomeSizer.AddStretchSpacer()
        self.welcome_panel.SetSizer(welcomeSizer)

        self.notebook = ImageNotebook(self.workspace)
        self.notebook.Hide()

        workspaceSizer = wx.BoxSizer(wx.VERTICAL)
        workspaceSizer.Add(self.welcome_panel, 1, wx.EXPAND)
        workspaceSizer.Add(self.notebook, 1, wx.EXPAND)
        self.workspace.SetSizer(workspaceSizer)

        # Unsplit until there is something to show below.
        self.splitter.Initialize(self.workspace)
        self.metric_table = None
        self.mask_table = None
        self.is_sample_image = False
        self.ground_truth_path = None
        self.ground_truth_data = None
        # Masks, zone selections and unwrapped phase: one object, one lifecycle.
        self._cache = AnalysisCache()
        # Full precision behind each cell: the tables show only 4 decimals.
        self._metric_values = {}
        self._mask_values = {}
        self._skip_reason = None
        self._unwrap_warning_shown = False
        self._unwrap_choice = False
        self._legendre_params_shown = False
        self._legendre_limit = None
        self._legendre_order = None
        # Three fields: last action, current image, and what has been measured.
        self.CreateStatusBar(3)
        self.SetStatusWidths([-1, 230, 170])
        self.SetStatusText("Ready")
        self._sort_state = {}
        
        # The catalogue is in analysis/catalog.py: it defines the protocol.
        self.metric_map = METRIC_MAP

    def on_check_imagej_updates(self, event):
        """Periodically checks if the user edited any image inside ImageJ."""
        try:
            # Imported lazily: pyimagej may not be installed at all.
            from core.imagej_bridge import check_for_imagej_updates
            changes = check_for_imagej_updates()
        except Exception:
            return  # ImageJ not started or bridge unavailable; silently ignore
        
        for imageTitle, pilImage in changes:
            if imageTitle in self.imagej_tab_map:
                # A tab is already linked to this image: update it
                pageIndex = self.imagej_tab_map[imageTitle]
                if pageIndex < self.notebook.GetPageCount():
                    panel = self.notebook.GetPage(pageIndex)
                    panel.pil_img = pilImage
                    self.notebook._update_bitmap(panel.bitmap_ctrl, pilImage, panel.scale_factor)
                    continue
            
            # Not linked yet: create a new tab and register it
            panel = self.notebook.add_image_tab(pilImage, f"{imageTitle}_ImageJ")
            self.imagej_tab_map[imageTitle] = self.notebook.GetPageCount() - 1

    def _setup_menus(self):
        # Set up the menu bar.
        menuBar = wx.MenuBar()
        menuBar.Append(file_menu.create(self), "&File")
        menuBar.Append(samples_menu.create(self, self.notebook), "Hologram &Samples")
        menuBar.Append(analysis_menu.create(self), "&Analysis")
        menuBar.Append(self._create_imagej_menu(), "&ImageJ")
        menuBar.Append(self._create_view_menu(), "&View")
        menuBar.Append(help_menu.create(self), "&Help")
        self.SetMenuBar(menuBar)
        self._setup_toolbar()

        self.imagej_tab_map = {}  # Maps ImageJ image title -> notebook tab index

        # NOT started here: _ensure_imagej_polling() starts it on demand.
        self.imagej_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_check_imagej_updates, self.imagej_timer)

    def _setup_toolbar(self):
        """The actions that make up most of the use, one click away."""
        # An AuiToolBar for its chevron: tools that do not fit fold away.
        from menus import analysis_menu, help_menu

        # The fold button lives OUTSIDE the bar, or it would hide with it.
        self.toolbar_row = wx.Panel(self.root_panel)

        self.toolbar = wx.aui.AuiToolBar(
            self.toolbar_row,
            style=wx.aui.AUI_TB_DEFAULT_STYLE | wx.aui.AUI_TB_HORZ_TEXT)
        # No SetToolBitmapSize: in wxAUI it is a no-op, the bitmaps decide.

        def addTool(etiqueta, nombre_icono, manejador, ayuda):
            toolItem = self.toolbar.AddTool(wx.ID_ANY, etiqueta,
                                        icons.icon(nombre_icono, 24, self),
                                        short_help_string=ayuda)
            self.Bind(wx.EVT_MENU, manejador, toolItem)
            return toolItem

        def addSeparator():
            self.toolbar.AddSeparator()

        addTool("Open", 'open', self.on_open,
                    "Open a phase image or .mat file (Ctrl+O)")
        addTool("Save", 'save', self.on_save,
                    "Export the result tables to Excel (Ctrl+S)")
        addTool("ImageJ", 'imagej', self.on_send_to_imagej,
                    "Send the current image to ImageJ and watch it for edits")
        addSeparator()
        # Kept so the toolbar can be greyed out with the Analysis menu; a
        # disabled menu entry beside a live button would be no restriction.
        self.module_tool_ids = [
            addTool("Background", 'background',
                    lambda evt: analysis_menu._show_module1_dialog(self),
                    "Module I - residual background phase variance").GetId(),
            addTool("Global", 'global',
                    lambda evt: analysis_menu._show_module2_dialog(self),
                    "Module II - global phase distortion").GetId(),
            addTool("Ground truth", 'ground_truth',
                    lambda evt: analysis_menu._show_module3_dialog(self),
                    "Module III - comparison against a reference image").GetId(),
        ]
        addTool("Complexity", 'complexity', self.on_computational_complexity,
                    "Module IV - computational complexity (needs Linux or WSL2)")
        addSeparator()
        addTool("Clear", 'clear', self.on_clear,
                    "Close every image and drop the tables (Ctrl+L)")
        addSeparator()
        addTool("Help", 'help', lambda evt: help_menu.show_help(self),
                    "What every metric measures and how to read it (F1)")

        self.toolbar.Realize()

        self.toolbar_toggle = wx.BitmapButton(
            self.toolbar_row, bitmap=icons.icon('chevron_up', 16, self),
            style=wx.BORDER_NONE)
        self.toolbar_toggle.SetToolTip("Hide the toolbar (Ctrl+T)")
        self.toolbar_toggle.Bind(wx.EVT_BUTTON,
                                 lambda evt: self.on_toggle_toolbar(None))

        toolbarRow = wx.BoxSizer(wx.HORIZONTAL)
        toolbarRow.Add(self.toolbar, 1, wx.EXPAND)
        toolbarRow.Add(self.toolbar_toggle, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.toolbar_row.SetSizer(toolbarRow)

    def _create_view_menu(self):
        """A place to fold the toolbar away."""
        menu = wx.Menu()
        self.toolbar_item = menu.AppendCheckItem(
            wx.ID_ANY, "&Toolbar\tCtrl+T", "Show or hide the toolbar")
        self.toolbar_item.Check(True)
        self.Bind(wx.EVT_MENU, self.on_toggle_toolbar, self.toolbar_item)
        return menu

    def on_toggle_toolbar(self, event):
        """Fold the bar away, or bring it back."""
        # Shared by the button, View > Toolbar and Ctrl+T, so they stay in step.
        visible = not self.toolbar.IsShown()
        self.toolbar.Show(visible)
        self.toolbar_item.Check(visible)
        self.toolbar_toggle.SetBitmap(
            icons.icon('chevron_up' if visible else 'chevron_down', 16, self))
        self.toolbar_toggle.SetToolTip(
            "Hide the toolbar (Ctrl+T)" if visible
            else "Show the toolbar (Ctrl+T)")
        self.toolbar_row.Layout()
        self.root_panel.Layout()

    def _create_imagej_menu(self):
        """Create the ImageJ integration menu."""
        menu = wx.Menu()
        
        sendItem = menu.Append(wx.ID_ANY, "Send current image to ImageJ")
        self.Bind(wx.EVT_MENU, self.on_send_to_imagej, sendItem)
        
        return menu

    def on_send_to_imagej(self, event):
        """Send the currently selected image to ImageJ for editing."""
        # The bridge needs pyimagej, JPype and a JVM. Without them the import
        # raised into the event loop and the button simply did nothing, which
        # is the worst way to fail: silently.
        try:
            from core.imagej_bridge import show_imagej_gui, pil_to_imageplus
        except ImportError as error:
            wx.MessageBox(
                "ImageJ support is not available in this Python environment.\n\n"
                f"{error}\n\n"
                "It needs pyimagej and JPype1 (see requirements.txt), plus a "
                "Java runtime. Install them, or start the application from an "
                "environment that already has them.",
                "ImageJ unavailable", wx.OK | wx.ICON_WARNING)
            return

        pilImage, imageName, page = self._get_current_image_data()
        if pilImage is None:
            return

        try:
            show_imagej_gui(parent_window=self)
            pil_to_imageplus(pilImage, title=imageName, parent_window=self)
        except Exception as error:
            wx.MessageBox(
                f"ImageJ could not be started.\n\n{type(error).__name__}: {error}",
                "ImageJ unavailable", wx.OK | wx.ICON_ERROR)
            return

        self._ensure_imagej_polling()
        
        # Only create a linked tab the FIRST time this image is sent
        if imageName not in self.imagej_tab_map:
            self.notebook.add_image_tab(pilImage, f"{imageName}_ImageJ")
            self.imagej_tab_map[imageName] = self.notebook.GetPageCount() - 1

    def _ensure_imagej_polling(self):
        """Start polling ImageJ for edits, once an image has actually been sent."""
        if not self.imagej_timer.IsRunning():
            self.imagej_timer.Start(1500)  # Check every 1.5 seconds

    def _setup_layout(self):
        # Set up the main layout.
        panelSizer = wx.BoxSizer(wx.VERTICAL)
        panelSizer.Add(self.toolbar_row, 0, wx.EXPAND)
        panelSizer.Add(self.splitter, 1, wx.EXPAND)
        self.root_panel.SetSizer(panelSizer)

        frameSizer = wx.BoxSizer(wx.VERTICAL)
        frameSizer.Add(self.root_panel, 1, wx.EXPAND)
        self.SetSizer(frameSizer)
        self.update_analysis_menu_state()

    def _arrange_tables(self):
        """Put whichever tables exist into the lower pane."""
        # The mask table takes only the height it needs; a sash splits the rest.
        maskTable, metricTable = self.mask_table, self.metric_table

        if maskTable and metricTable:
            if not self.tables_splitter.IsSplit():
                self.tables_splitter.SplitHorizontally(maskTable, metricTable,
                                                       self.FromDIP(130))
        elif maskTable or metricTable:
            singleZoneMetric = maskTable or metricTable
            if self.tables_splitter.IsSplit():
                self.tables_splitter.Unsplit()
            self.tables_splitter.Initialize(singleZoneMetric)
            singleZoneMetric.Show()

        self.tables_panel.Layout()

    def _show_tables_pane(self):
        """Reveal the lower pane the first time a table is created."""
        if not self.splitter.IsSplit():
            paneHeight = max(self.FromDIP(220), self.GetClientSize().height // 3)
            self.splitter.SplitHorizontally(self.workspace, self.tables_panel,
                                            -paneHeight)
        self.tables_panel.Layout()
        self.root_panel.Layout()

    def _hide_tables_pane(self):
        """Give the whole window back to the image when no table is left."""
        if self.splitter.IsSplit():
            self.splitter.Unsplit(self.tables_panel)
        self.root_panel.Layout()

    def _bind_events(self):
        # Bind event handlers.
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self.on_tab_change)
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self.on_tab_close)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_BUTTON, self.on_save)

    #: Shade for the odd rows, painted by hand in _stripe().
    ALTERNATE_ROW = wx.Colour(0xF5, 0xF7, 0xFA)

    @staticmethod
    def _stripe(table):
        """Re-apply the alternating row shade, after inserting or sorting."""
        sampleIsWhite = wx.Colour(0xFF, 0xFF, 0xFF)
        for rowIndex in range(table.GetItemCount()):
            table.SetItemBackgroundColour(
                rowIndex, MainFrame.ALTERNATE_ROW if rowIndex % 2 else sampleIsWhite)

    @staticmethod
    def _style_table(table):
        """A bold header: twenty columns are hard to track otherwise."""
        header = wx.ItemAttr()
        source = table.GetFont()
        source = source.Bold()
        header.SetFont(source)
        table.SetHeaderAttr(header)

    def _on_column_click(self, event, table, raw_store):
        """Sort by the clicked column; click again to reverse."""
        columnIndex = event.GetColumn()
        previous = self._sort_state.get(id(table))
        ascending = not (previous is not None and previous[0] == columnIndex and previous[1])
        self._sort_state[id(table)] = (columnIndex, ascending)
        self._sort_table(table, raw_store, columnIndex, ascending)

    @staticmethod
    def _sort_table(table, raw_store, col, ascending):
        """Reorder the rows by one column."""
        # Sorted on the stored number, so "10" comes after "9", not before.
        columnCount = table.GetColumnCount()
        columnName = table.GetColumn(col).GetText()

        rows = []
        for rowIndex in range(table.GetItemCount()):
            cellTexts = [table.GetItemText(rowIndex, columnIndex)
                         for columnIndex in range(columnCount)]
            rawValue = raw_store.get(cellTexts[0], {}).get(columnName)
            rows.append((rawValue, cellTexts))

        def sortKey(pair):
            cellValue, cellTexts = pair
            if isinstance(cellValue, float):
                # NaN last, whichever way round the sort runs
                if cellValue != cellValue:
                    return (2, 0.0, "")
                return (0, cellValue, "")
            return (1, 0.0, cellTexts[col].lower())

        rows.sort(key=sortKey, reverse=not ascending)

        table.DeleteAllItems()
        for index, (_, cellTexts) in enumerate(rows):
            table.InsertItem(index, cellTexts[0])
            for columnIndex in range(1, columnCount):
                table.SetItem(index, columnIndex, cellTexts[columnIndex])
        MainFrame._stripe(table)

    def _update_status_image(self):
        """Second field: the image in front and how it was loaded."""
        page = self.notebook.get_current_page()
        imageName = self.notebook.get_current_image_name()
        if page is None or imageName is None:
            self.SetStatusText("", 1)
            return

        phase = getattr(page, 'phase', None)
        if phase is not None:
            paneHeight, columnWidth = phase.shape[:2]
            source = "complex field" if getattr(page, 'is_mat_complex', False) else "image"
            self.SetStatusText("%s  -  %d x %d  -  %s" % (imageName, columnWidth, paneHeight, source), 1)
        else:
            self.SetStatusText(imageName, 1)

    def _update_status_counts(self):
        """Third field: how much of the table is filled."""
        if self.metric_table is None or self.metric_table.GetItemCount() == 0:
            self.SetStatusText("", 2)
            return
        rows = self.metric_table.GetItemCount()
        columnNames = max(0, self.metric_table.GetColumnCount() - 1)
        self.SetStatusText("%d image(s), %d metric column(s)" % (rows, columnNames), 2)

    def _update_workspace(self):
        """Show the notebook only while it actually holds an image."""
        hasImages = self.notebook.GetPageCount() > 0
        if self.notebook.IsShown() != hasImages:
            self.notebook.Show(hasImages)
            self.welcome_panel.Show(not hasImages)
            self.workspace.Layout()

    def update_analysis_menu_state(self):
        self._update_workspace()
        self._update_status_image()
        self._update_status_counts()

        # Update Analysis menu state based on current image type.
        menuBar = self.GetMenuBar()
        analysisIndex = menuBar.FindMenu("&Analysis")
        if analysisIndex == wx.NOT_FOUND:
            return
        analysis_menu = menuBar.GetMenu(analysisIndex)
        hasImages = self.notebook.GetPageCount() > 0

        # Modules I-III read the phase. A bundled sample is a hologram, so
        # they would be measuring interference fringes, not a compensated phase
        # map: they stay off until the user opens one. Module IV never looks at
        # the workspace, so it is always available.
        allowMetrics = hasImages and not self.is_sample_image

        # Get menu items by their IDs
        if hasattr(self, 'module1_metrics_id'):
            module1Item = analysis_menu.FindItemById(self.module1_metrics_id)
            module2Item = analysis_menu.FindItemById(self.module2_metrics_id)
            module3Item = analysis_menu.FindItemById(self.module3_metrics_id)
            complexityItem = analysis_menu.FindItemById(self.computational_complexity_id)

            for metricItem in (module1Item, module2Item, module3Item):
                if metricItem:
                    metricItem.Enable(allowMetrics)
            if complexityItem:
                complexityItem.Enable(True)

        self._enable_module_tools(allowMetrics)

    def _enable_module_tools(self, enabled):
        """Keep the three module buttons in step with the Analysis menu."""
        toolbar = getattr(self, 'toolbar', None)
        if toolbar is None:
            return
        for toolId in getattr(self, 'module_tool_ids', []):
            toolbar.EnableTool(toolId, enabled)
        toolbar.Refresh()

    def mark_as_sample_image(self, is_sample=True):
        self.is_sample_image = is_sample
        self.update_analysis_menu_state()

    def ensure_metric_table(self):
        if self.metric_table is not None:
            return
        
        self.metric_table = wx.ListCtrl(self.tables_splitter, style=wx.LC_REPORT)
        self._style_table(self.metric_table)
        self.metric_table.InsertColumn(0, "Image Name", width=self.FromDIP(250))
        self.metric_table.Bind(
            wx.EVT_LIST_COL_CLICK,
            lambda evt: self._on_column_click(evt, self.metric_table,
                                              self._metric_values))

        self.metric_table.Bind(wx.EVT_CONTEXT_MENU, self.on_metric_table_context_menu)
       
        self._arrange_tables()
        self._show_tables_pane()

    def _ensure_column(self, col_name: str, width: int = 120):
        width = self.FromDIP(width)
        if not self.metric_table:
            return
        
        for index in range(self.metric_table.GetColumnCount()):
            if self.metric_table.GetColumn(index).GetText() == col_name:
                return
        
        columnIndex = self.metric_table.GetColumnCount()
        self.metric_table.InsertColumn(columnIndex, col_name, width=width)

    def _get_column_index_by_name(self, col_name: str) -> int:
        if not self.metric_table:
            return -1
        
        for index in range(self.metric_table.GetColumnCount()):
            if self.metric_table.GetColumn(index).GetText() == col_name:
                return index
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
        
        for rowIndex in range(self.metric_table.GetItemCount()):
            if self.metric_table.GetItem(rowIndex, 0).GetText() == image_name:
                return rowIndex
        
        newRowIndex = self.metric_table.GetItemCount()
        self.metric_table.InsertItem(newRowIndex, image_name)
        self._stripe(self.metric_table)
        return newRowIndex

    def update_table_with_zones(self, image_name, metric_column, value, zone_stats=None, num_zones=0):
        if self.metric_table is None:
            self.ensure_metric_table()

        rowIndex = self._find_or_create_row(image_name)
        
        # Ensure column exists
        self._ensure_column(metric_column)
        columnIndex = self._get_column_index_by_name(metric_column)
        
        if columnIndex < 0:
            return
        
        # Write metric value
        self.metric_table.SetItem(rowIndex, columnIndex, f"{value:.4f}")
        self._record_value(self._metric_values, image_name, metric_column, value)
        
        if zone_stats:
            self._update_zone_columns_dynamic(rowIndex, zone_stats, metric_column, num_zones)

    def _update_zone_columns_dynamic(self, row, zone_stats, metric, num_zones=0):
        metricLabel = metric.split("_")[0]
        
        for index in range(1, num_zones + 1):
            self._ensure_column(f"Zone {index} {metricLabel}")
        
        self._ensure_column(f"{metricLabel} Total Zones", width=120)
        self._ensure_column(f"{metricLabel} Zone Details", width=200)
        
        columnIndices = {
            self.metric_table.GetColumn(columnIndex).GetText(): columnIndex
            for columnIndex in range(self.metric_table.GetColumnCount())
        }
        
        self._populate_zone_values(row, zone_stats, metricLabel, columnIndices)
    
        self._populate_zone_summary(row, zone_stats, columnIndices, num_zones, metricLabel)

    def _populate_zone_values(self, row, zone_stats, metric_label, col_indices):
        # exctract the name of the base metric (e.g., 'STD' from 'STD_Zones')
        if isinstance(metric_label, str):
            baseMetric = metric_label.split('_')[0]
        else:
            baseMetric = str(metric_label)
        
        metricNameLower = baseMetric.lower()
        
        for zoneStat in zone_stats:
            zoneNumber = zoneStat.get('zone')
            
            if not isinstance(zoneNumber, int):
                continue
            
            columnName = f"Zone {zoneNumber} {baseMetric}"
            
            if columnName not in col_indices:
                continue
           
            # Search for the value with different variants of the name
            zoneValue = zoneStat.get(metricNameLower)  # 'std', 'mad', 'rms', etc.
            if zoneValue is None:
                zoneValue = zoneStat.get(baseMetric.lower())
            if zoneValue is None:
                zoneValue = zoneStat.get(baseMetric)
            
            if zoneValue is None:
                continue
    
            try:
                zoneValue = float(np.mean(zoneValue)) if not np.isscalar(zoneValue) else float(zoneValue)
                if np.isfinite(zoneValue):
                    self.metric_table.SetItem(row, col_indices[columnName], f"{zoneValue:.4f}")
                    imageName = self.metric_table.GetItemText(row, 0)
                    self._record_value(self._metric_values, imageName, columnName, zoneValue)
            except (ValueError, TypeError) as error:
                print(f" Error setting value for {columnName}: {error}")
                continue

    def _populate_zone_summary(self, row, zone_stats, col_indices, num_zones=None, metric_label=None):
        if num_zones is None or num_zones == 0:
            num_zones = len([strideStep for strideStep in zone_stats if isinstance(strideStep.get('zone'), int)])
        
        totalZonesColumn = f"{metric_label} Total Zones" if metric_label else "Total Zones"
        zoneDetailsColumn = f"{metric_label} Zone Details" if metric_label else "Zone Details"
        
        if totalZonesColumn in col_indices:
            self.metric_table.SetItem(row, col_indices[totalZonesColumn], str(num_zones))
        
        if zoneDetailsColumn in col_indices and zone_stats:
            coordinateSummary = self._create_coords_summary(zone_stats)
            self.metric_table.SetItem(row, col_indices[zoneDetailsColumn], coordinateSummary)

    def _create_coords_summary(self, zone_stats):
        # Create a summary string of zone coordinates.
        coordinateParts = [
            f"Z{zoneStat['zone']}:({zoneStat['coords'][0]},{zoneStat['coords'][2]})-({zoneStat['coords'][1]},{zoneStat['coords'][3]})"
            for zoneStat in zone_stats[:3]
        ]
        
        coordinateSummary = "; ".join(coordinateParts)
        if len(zone_stats) > 3:
            coordinateSummary += f"; +{len(zone_stats)-3} more"
        
        return coordinateSummary

    def ensure_mask_table(self):
        # Create mask table if it doesn't exist.
        if self.mask_table is not None:
            return
            
        self.mask_table = wx.ListCtrl(self.tables_splitter, style=wx.LC_REPORT)
        self._style_table(self.mask_table)
        self.mask_table.Bind(
            wx.EVT_LIST_COL_CLICK,
            lambda evt: self._on_column_click(evt, self.mask_table,
                                              self._mask_values))
        
        # Segmentation records how the mask was obtained. Without it the
        # numbers beside it cannot be reproduced by anyone else.
        columnNames = [
            ("Image Name", 200), ("Threshold", 100), ("Total Pixels", 100),
            ("Background Pixels", 120), ("Background %", 100),
            ("Sample Pixels", 120), ("Sample %", 100),
            # Narrow on purpose: wider than this and the row outgrows the
            # window, which puts a scrollbar inside a 130 px pane. The full
            # recipe is in the tooltip and in the export.
            ("Segmentation", 200),
        ]
        
        for index, (imageName, columnWidth) in enumerate(columnNames):
            self.mask_table.InsertColumn(index, imageName, width=columnWidth)

        self._arrange_tables()
        self._show_tables_pane()
        self.mask_table.Bind(wx.EVT_CONTEXT_MENU, self.on_mask_table_context_menu)

    def update_mask_table(self, image_name, threshold, total_pixels, background_pixels,
                         sample_pixels, segmentation=""):
        # Update mask table with calculated data.
        self.ensure_mask_table()
        
        foundRowIndex = None
        for rowIndex in range(self.mask_table.GetItemCount()):
            if self.mask_table.GetItemText(rowIndex, 0) == image_name:
                foundRowIndex = rowIndex
                break

        backgroundPercent = 100 * background_pixels / total_pixels if total_pixels > 0 else 0
        samplePercent = 100 * sample_pixels / total_pixels if total_pixels > 0 else 0
        
        pageData = [
            f"{threshold:.4f}", f"{total_pixels}", f"{background_pixels}",
            f"{backgroundPercent:.1f}%", f"{sample_pixels}",
            f"{samplePercent:.1f}%", segmentation,
        ]
        for columnName, cellValue in (("Threshold", threshold),
                                ("Total Pixels", total_pixels),
                                ("Background Pixels", background_pixels),
                                ("Background %", backgroundPercent),
                                ("Sample Pixels", sample_pixels),
                                ("Sample %", samplePercent),
                                ("Segmentation", segmentation)):
            self._record_value(self._mask_values, image_name, columnName, cellValue)

        if foundRowIndex is not None:
            for columnIndex, value in enumerate(pageData, start=1):
                self.mask_table.SetItem(foundRowIndex, columnIndex, value)
        else:
            index = self.mask_table.InsertItem(self.mask_table.GetItemCount(), image_name)
            self._stripe(self.mask_table)
            for columnIndex, value in enumerate(pageData, start=1):
                self.mask_table.SetItem(index, columnIndex, value)

    # Event handlers
    def on_open(self, event):
        initialCount = self.notebook.GetPageCount()
        file_manager.open_image(self, self.notebook)
        
        if self.notebook.GetPageCount() > initialCount:
            self.mark_as_sample_image(False)  

    def _get_current_image_data(self):
        pilImage = self.notebook.get_current_image()
        imageName = self.notebook.get_current_image_name()
        page = self.notebook.get_current_page()
        
        if pilImage is None or imageName is None:
            wx.MessageBox("No image selected", "Error", wx.ICON_ERROR)
            return None, None, None
            
        return pilImage, imageName, page


    def _update_tables_with_mask_data(self, name, sample, background_mask,
                                   threshold, recipe=None):
        if background_mask is None or threshold is None:
            return

        totalPixels = sample.size
        backgroundPixels = np.sum(background_mask)
        samplePixels = np.sum(~background_mask)

        # A row segmented differently from the rest of the batch is marked, so
        # that it is not read as directly comparable with the others.
        # With the level in it, the column alone reproduces the mask.
        segmentation = "" if recipe is None else recipe.describe(threshold)
        batchRecipe = self._cache.segmentation_recipe
        if recipe is not None and batchRecipe is not None \
                and not recipe.same_as(batchRecipe):
            segmentation = "* " + segmentation

        self.update_mask_table(
            name, threshold, totalPixels, backgroundPixels,
            samplePixels, segmentation
        )

    def _get_zone_count_from_user(self, default=2):
        # Get number of zones from user input.
        with wx.NumberEntryDialog(
            self, 
            "How many zones do you want to select?", 
            "Number of zones (1-20):", 
            "Zone Selection", 
            default, 1, 20
        ) as dialog:
            return dialog.GetValue() if dialog.ShowModal() == wx.ID_OK else None
        
    def _get_limit_from_user(self, default=64):
        # Get limit parameter from user input for Legendre analysis.
        with wx.NumberEntryDialog(
            self, 
            "Enter the limit parameter for Fourier domain cropping:", 
            "Limit (64-1024):", 
            "Legendre Limit Parameter", 
            default, 64, 1024
        ) as dialog:
            return dialog.GetValue() if dialog.ShowModal() == wx.ID_OK else None    
        
    def _get_legendre_order_from_user(self, default=5):
        # Get number of Legendre coefficients from user input.
        with wx.NumberEntryDialog(
            self, 
            "How many Legendre coefficients do you want to calculate?", 
            "Number of coefficients (1-15):", 
            "Legendre Order", 
            default, 1, 15
        ) as dialog:
            return dialog.GetValue() if dialog.ShowModal() == wx.ID_OK else None


    def on_clear(self, event):
        # Clear all images and tables.
        with wx.MessageDialog(
            self,
            message="Are you sure you want to close all images and remove all tables?",
            caption="Confirm Clear",
            style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING
        ) as dialog:
            if dialog.ShowModal() == wx.ID_YES:
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
        # Detach from the sash BEFORE destroying, or the process segfaults.
        if self.tables_splitter.IsSplit():
            self.tables_splitter.Unsplit()

        if self.metric_table:
            self.metric_table.Destroy()
            self.metric_table = None

        if self.mask_table:
            self.mask_table.Destroy()
            self.mask_table = None

        self._hide_tables_pane()

    # Export handlers
    def on_metric_table_context_menu(self, event):
        # Show context menu for metrics table.
        menu = wx.Menu()
        exportThisExcelItem = menu.Append(wx.ID_ANY, "Export Metrics table to Excel")
        exportThisCsvItem = menu.Append(wx.ID_ANY, "Export Metrics table to CSV")
        menu.AppendSeparator()
        exportAllExcelItem = menu.Append(wx.ID_ANY, "Export All Tables to Excel")
        exportAllCsvItem = menu.Append(wx.ID_ANY, "Export All Tables to CSV")

        self.Bind(wx.EVT_MENU, self.on_export_metrics_excel, exportThisExcelItem)
        self.Bind(wx.EVT_MENU, self.on_export_metrics_csv, exportThisCsvItem)
        self.Bind(wx.EVT_MENU, self.on_export_all_excel, exportAllExcelItem)
        self.Bind(wx.EVT_MENU, self.on_export_all_csv, exportAllCsvItem)
        self.PopupMenu(menu)
        menu.Destroy()

    def on_mask_table_context_menu(self, event):
        # Show context menu for mask table.
        menu = wx.Menu()
        exportThisExcelItem = menu.Append(wx.ID_ANY, "Export Mask table to Excel")
        exportThisCsvItem = menu.Append(wx.ID_ANY, "Export Mask table to CSV")
        menu.AppendSeparator()
        exportAllExcelItem = menu.Append(wx.ID_ANY, "Export All Tables to Excel")
        exportAllCsvItem = menu.Append(wx.ID_ANY, "Export All Tables to CSV")

        self.Bind(wx.EVT_MENU, self.on_export_mask_excel, exportThisExcelItem)
        self.Bind(wx.EVT_MENU, self.on_export_mask_csv, exportThisCsvItem)
        self.Bind(wx.EVT_MENU, self.on_export_all_excel, exportAllExcelItem)
        self.Bind(wx.EVT_MENU, self.on_export_all_csv, exportAllCsvItem)

        self.PopupMenu(menu)
        menu.Destroy()

    def export_all_tables(self, file_format, event=None, only=None):
        """Export the result tables, or just the ones named in `only`."""
        tableInfo = {
            "Metrics": (self.metric_table, self._get_metric_table_data),
            "Mask Data": (self.mask_table, self._get_mask_table_data),
        }
        if only:
            tableInfo = {tableName: value for tableName, value in tableInfo.items()
                         if tableName in only}

        # Check if there is at least one table with data
        hasData = any(
            table is not None and table.GetItemCount() > 0
            for table, _ in tableInfo.values()
        )

        if not hasData:
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
                ) as dialog:

                    if dialog.ShowModal() == wx.ID_CANCEL:
                        return

                    exportPath = dialog.GetPath()

                exportedSheets = 0

                with pd.ExcelWriter(exportPath, engine="openpyxl") as excelWriter:

                    for sheetName, (table, pageDataMethod) in tableInfo.items():

                        if table is None or table.GetItemCount() == 0:
                            continue

                        pageData = pageDataMethod()

                        if not pageData:
                            continue

                        dataFrame = pd.DataFrame(pageData)

                        dataFrame.to_excel(
                            excelWriter,
                            sheet_name=sheetName[:31],   # Excel sheet name limit
                            index=False
                        )

                        exportedSheets += 1

                wx.MessageBox(
                    f"{exportedSheets} table(s) exported successfully.",
                    "Success",
                    wx.OK | wx.ICON_INFORMATION
                )
            # Export to CSV
            elif file_format.lower() == "csv":

                with wx.DirDialog(
                    self,
                    "Select folder to save CSV files",
                    style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
                ) as dialog:

                    if dialog.ShowModal() == wx.ID_CANCEL:
                        return

                    folder = dialog.GetPath()

                exportedSheets = 0

                for fileName, (table, pageDataMethod) in tableInfo.items():

                    if table is None or table.GetItemCount() == 0:
                        continue

                    pageData = pageDataMethod()

                    if not pageData:
                        continue

                    dataFrame = pd.DataFrame(pageData)

                    csvPath = os.path.join(
                        folder,
                        f"{fileName}.csv"
                    )

                    dataFrame.to_csv(csvPath, index=False)

                    exportedSheets += 1

                wx.MessageBox(
                    f"{exportedSheets} CSV file(s) exported successfully.",
                    "Success",
                    wx.OK | wx.ICON_INFORMATION
                )

        except Exception as error:

            wx.MessageBox(
                f"Export failed:\n{str(error)}",
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
        """Export rows, using the stored full-precision number where there is one."""
        rows = []
        columnCount = table.GetColumnCount()
        imageNames = [table.GetColumn(columnIndex).GetText()
                      for columnIndex in range(columnCount)]

        for rowIndex in range(table.GetItemCount()):
            imageName = table.GetItemText(rowIndex, 0)
            rawValues = raw_store.get(imageName, {})
            rowValues = {}
            for columnIndex, columnName in enumerate(imageNames):
                cellText = table.GetItemText(rowIndex, columnIndex)
                if columnName in rawValues and cellText != "":
                    rowValues[columnName] = rawValues[columnName]
                else:
                    rowValues[columnName] = cellText
            rows.append(rowValues)
        return rows

    def _get_mask_table_data(self):
        # Extract mask table data for export.
        if self.mask_table is None:
            return []

        # Column names come from the widget, so they cannot drift apart from it.
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
        pageIndex = event.GetSelection()
        closedImageName = None
        if pageIndex != wx.NOT_FOUND and pageIndex < self.notebook.GetPageCount():
            closedImageName = self.notebook.GetPageText(pageIndex)
        
        wx.CallAfter(self._handle_tab_close_cleanup, closedImageName)
        event.Skip()

    def _handle_tab_close_cleanup(self, closed_image_name=None):

        if closed_image_name:
            self._cache.forget_image(closed_image_name)
            
            # Also clear any per-image zone selections tied to this image
            keysToRemove = [
                cacheKey for cacheKey in self._cache.zones
                if cacheKey == f"{closed_image_name}_zones" or cacheKey.startswith(f"{closed_image_name}_")
            ]
            for cacheKey in keysToRemove:
                del self._cache.zones[cacheKey]
        
        # Execute after closing a tab. Clean up if no images remain.
        pageCount = self.notebook.GetPageCount()
        
        if pageCount == 0:
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
    
                tableInfo = {
                    "Metrics": self.metric_table,
                    "Mask Data": self.mask_table,
                }
                hasData = any(
                    table is not None and table.GetItemCount() > 0
                    for table in tableInfo.values()
                )
    
                if not hasData:
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
            currentDir = os.path.dirname(os.path.abspath(__file__))  # ui/
            parentDir = os.path.dirname(currentDir)  # My_app/
            complexityRepoPath = os.path.join(parentDir, 'complexity_algorithm')
            # 'Main.py' with a capital M: lowercase only works on Windows.
            guiScript = os.path.join(complexityRepoPath, 'Main.py')

            # Verify if the file exists in Windows
            if not os.path.exists(guiScript):
                wx.MessageBox(
                    f"Complexity GUI not found at:\n{guiScript}\n\n"
                    "Please make sure the 'complexity_algorithm' repository "
                    "is located inside the My_app folder.",
                    "Error",
                    wx.ICON_ERROR
                )
                return

            subprocess.Popen([sys.executable, guiScript])

            # No confirmation dialog: it left the new window opening behind.
            self.SetStatusText(
                "Computational Complexity launched. Profiling needs Linux "
                "or WSL2; the window itself opens anywhere."
            )

        except Exception as error:
            wx.MessageBox(
                f"Error launching the GUI:\n{str(error)}",
                "Error",
                wx.ICON_ERROR
            )


    # Residual Background Metrics (Module 1)
  

    def get_zones_mode_from_user(self, num_images, num_zone_metrics):
        # One image and one zone metric: nothing to ask.
        if num_images == 1 and num_zone_metrics == 1:
            return 'per_image'  # Default mode
        
        # Case 2: Multiple zone metrics, single image
        if num_images == 1 and num_zone_metrics > 1:
            # The two choices say what they do; the bullet lists that used to
            # sit above them only pushed the buttons off a short screen.
            dialog = wx.SingleChoiceDialog(
                self,
                "",
                "Zone Selection Mode - Multiple Metrics",
                ["Shared Zones (Same for all metrics)", "Per-Metric Zones (Different for each metric)"]
            )
            
            if dialog.ShowModal() == wx.ID_OK:
                selection = dialog.GetSelection()
                dialog.Destroy()
                return 'shared' if selection == 0 else 'per_metric'
            
            dialog.Destroy()
            return None
        
        # Case 3: Multiple images
        if num_images > 1:
            # Determine message based on whether there are also multiple metrics
            if num_zone_metrics > 1:
                choices = ["Shared Zones (Same for all)", "Per-Image Zones (Different for each image)"]
            else:
                choices = ["Shared Zones (Same for all images)", "Per-Image Zones (Different for each image)"]

            dialog = wx.SingleChoiceDialog(
                self,
                "",
                "Zone Selection Mode",
                choices
            )
            
            if dialog.ShowModal() == wx.ID_OK:
                selection = dialog.GetSelection()
                dialog.Destroy()
                return 'shared' if selection == 0 else 'per_image'
            
            dialog.Destroy()
            return None
        
        # Default case
        return 'per_image'

    # Each RMS variant drags its matching mean along (see _RMS_MEAN_PAIRS below)
    _RMS_MEAN_PAIRS = (
        ('rms_background', 'mean_background', 'Mean - Background'),
        ('rms_unwrapped', 'mean_unwrapped', 'Mean - Unwrapped Background'),
        ('rms_zones', 'mean_zones', 'Mean - Background Zones'),
    )

    def _ask_unwrap_once(self, metrics_dict=None):
        """Ask ONCE per batch whether phase unwrapping may be applied.

        Declining means two different things, so the button has to say which
        one applies: an Unwrapped variant is skipped, while a Zones metric is
        still measured, on the wrapped phase, inside the same zones."""
        if self._unwrap_warning_shown:
            return self._unwrap_choice

        selected = list(metrics_dict or ())
        hasZones = any('zones' in candidateId for candidateId in selected)
        hasUnwrapped = any('unwrapped' in candidateId and 'zones' not in candidateId
                           for candidateId in selected)

        message = ("Some of the selected metrics use phase unwrapping.\n\n"
                   "Unwrapping removes 2π discontinuities, but it can "
                   "jeopardize phase compensation and introduce errors of "
                   "its own.")
        if hasZones and hasUnwrapped:
            # The only case one label cannot carry, so it is said in a line.
            message += ("\n\nDeclining leaves the zone metrics on the wrapped "
                        "phase and skips the Unwrapped variants.")
            noLabel = "Do not unwrap"
        elif hasZones:
            noLabel = "Do not unwrap"
        else:
            noLabel = "Skip"

        dialog = wx.MessageDialog(
            self, message, "Apply Phase Unwrapping?",
            style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION
        )
        # Named after what they do, not Yes/No. As a side effect the two
        # labels stay in English on a Windows whose own Yes/No is translated.
        dialog.SetYesNoLabels("Unwrap", noLabel)
        answer = dialog.ShowModal()
        dialog.Destroy()

        self._unwrap_choice = (answer == wx.ID_YES)
        self._unwrap_warning_shown = True
        return self._unwrap_choice

    def _calculate_multiple_metrics(self, metrics_dict, image_indices):
        # Every selected metric over every selected image, one threshold each.

        # RMS^2 = mean^2 + STD^2, so the mean rides along with RMS.
        metrics_dict = dict(metrics_dict)
        for rmsMetricId, meanMetricId, meanLabel in self._RMS_MEAN_PAIRS:
            if rmsMetricId in metrics_dict and meanMetricId not in metrics_dict:
                metrics_dict[meanMetricId] = meanLabel

        # Verify ground-truth if needed
        if not self._ensure_ground_truth_loaded(metrics_dict):
            return
        self._unwrap_warning_shown = False

        # One question for the whole batch, asked up front.
        needsUnwrap = any(
            'zones' in candidateId or self.metric_map.get(candidateId, (None, None, False))[2]
            for candidateId in metrics_dict
        )
        if needsUnwrap:
            # Asked here, where the whole selection is known: the wording of
            # the decline button depends on what was selected.
            self._ask_unwrap_once(metrics_dict)
        self._legendre_params_shown = False
        # Detect if there are zone metrics and ask for mode
        hasZoneMetrics = any('zones' in candidateId for candidateId in metrics_dict.keys())
        
        zonesMode = 'per_image'  # Default mode
        
        if hasZoneMetrics:
            # Count number of zone metrics
            zoneMetricCount = sum(1 for candidateId in metrics_dict.keys() if 'zones' in candidateId)
            
            # Ask user which mode they prefer
            zonesMode = self.get_zones_mode_from_user(len(image_indices), zoneMetricCount)
            
            if zonesMode is None:
                wx.MessageBox(
                    "Zone mode selection cancelled.",
                    "Cancelled",
                    wx.ICON_INFORMATION
                )
                return
            
            # Initialize zones cache for this session
            
            # If shared mode, clear cache to start fresh and reset shared zones
            if zonesMode == 'shared':
                self._cache.reset_zones(shared=True)
            # If per-image or per-metric mode, clear cache for fresh zones
            else:
                self._cache.reset_zones()
        
        totalOperations = len(metrics_dict) * len(image_indices)
        
        # Progress bar
        progressDialog = wx.ProgressDialog(
            "Calculating Metrics",
            "Processing...",
            maximum=totalOperations,
            parent=self,
            # PD_CAN_ABORT so a long batch can be stopped part way.
            style=(wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH
                   | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME)
        )
        
        currentOperation = 0
        resultLines = []
        cancelled = False
        
        try:
            # Process each image
            for imageIndex in image_indices:
                if cancelled:
                    break
                panel = self.notebook.GetPage(imageIndex)
                pilImage = getattr(panel, 'pil_img', None)
                imageName = self.notebook.GetPageText(imageIndex)
                
                if pilImage is None:
                    currentOperation += len(metrics_dict)
                    continue
                
                resultLines.append(f"\n{'='*50}")
                resultLines.append(f"Image: {imageName}")
                resultLines.append(f"{'='*50}")
                
                # 1️⃣ The phase, computed once at load time. Never re-derived
                # from the picture on screen: that picture is rescaled for
                # display, so reading it back would feed the metrics something
                # that is not the phase.
                sample = getattr(panel, 'phase', None)
                if sample is None:
                    grayscale = pilImage.convert("L")
                    grayLevels = np.array(grayscale, dtype=float)
                    sample = utRBPV.gray_to_phase(grayLevels)
                
                # Masks computed ONCE per image, zone mode included in the key.
                maskCache = self._precalculate_masks(
                    sample, imageName, metrics_dict, zones_mode=zonesMode
                )
                
                # 3️⃣ Calculate each metric using the cache
                for metricId, metricLabel in metrics_dict.items():
                    # Update() pumps the event loop and reports Cancel presses.
                    keepGoing, _ = progressDialog.Update(
                        currentOperation,
                        f"Calculating {metricLabel} for {imageName}..."
                    )
                    if not keepGoing:
                        cancelled = True
                        resultLines.append(
                            "\n*** Cancelled by the user. Everything above was "
                            "already written to the table. ***"
                        )
                        break
                    
                    try:
                        value = self._calculate_single_metric_cached(
                            metricId,
                            sample,
                            imageName,
                            maskCache
                        )
                        
                        if value is None:
                            reason = self._skip_reason or "skipped"
                            resultLines.append(f" ⊘ {metricLabel}: {reason}")
                        elif not np.isfinite(value):
                            resultLines.append(
                                f" ⊘ {metricLabel}: undefined (NaN) — the estimator "
                                f"does not apply to this data"
                            )
                        else:
                            resultLines.append(f" ✓ {metricLabel}: {value:.6f}")
                            
                    except Exception as error:
                        resultLines.append(f" ✗ {metricLabel}: ERROR - {str(error)}")
                    
                    currentOperation += 1
            
            if not cancelled:
                progressDialog.Update(totalOperations)
        
        finally:
            progressDialog.Destroy()
        
        # Show summary
        summaryText = "\n".join(resultLines)
        
        try:
            dialog = wx.lib.dialogs.ScrolledMessageDialog(
                self,
                summaryText,
                "Metrics Calculation Cancelled" if cancelled else "Metrics Calculation Complete",
                size=self.FromDIP(wx.Size(600, 400))
            )
            dialog.ShowModal()
            dialog.Destroy()
        except:
            wx.MessageBox(summaryText, "Metrics Calculation Complete", wx.ICON_INFORMATION)
        
        self._update_status_counts()
        self.SetStatusText(
            "Cancelled" if cancelled else "Metrics calculated")

    def _precalculate_masks(self, sample, image_name, metrics_dict, zones_mode='per_image'):
        # Pre-calculate all needed masks for an image BEFORE processing metrics.
        maskCache = {}
        
        # Detect which type of metrics are being calculated
        module1Metrics = []
        module2Metrics = []
        module3Metrics = []
        zoneMetrics = []
        
        for metricId in metrics_dict.keys():
            if 'zones' in metricId:
                zoneMetrics.append(metricId)
            elif metricId.startswith(('mmm_', 'gradient_', 'tsm_',
                                    'laplacian_', 'spatial_freq_', 'global_entropy',
                                    'sharpness_', 'fwhm_global')):
                module2Metrics.append(metricId)
            elif metricId.startswith(('ssim', 'mse', 'psnr')):
                module3Metrics.append(metricId)
            else:
                module1Metrics.append(metricId)
        

        applyUnwrap = self._ask_unwrap_once() if zoneMetrics else False

        zoneSample = sample  # copia separada para las métricas de zonas
        if applyUnwrap:
            try:
                zoneSample = self._get_unwrapped_sample(image_name, sample)  # NO pisa 'sample'
            except Exception as error:
                wx.MessageBox(f"Error applying unwrap to {image_name}: {error}", "Error", wx.ICON_ERROR)
                return maskCache

        if zoneMetrics:
            # Zone metrics read from here, not from 'sample'.
            maskCache['zone_sample'] = zoneSample
            maskCache['zones_unwrapped'] = applyUnwrap
        
        if module1Metrics:
            needsUnwrapped = any('unwrapped' in candidateId for candidateId in module1Metrics)
            needsWrapped = any('unwrapped' not in candidateId and 'zones' not in candidateId for candidateId in module1Metrics)

            if needsWrapped:
                backgroundMask, threshold, processedSample = self._get_or_create_background_mask(
                    image_name, sample, needs_unwrap=False
                )
                if backgroundMask is not None:
                    maskCache[False] = (backgroundMask, threshold, processedSample)

            if needsUnwrapped and self._ask_unwrap_once():
                try:
                    backgroundMask, threshold, unwrappedSample = self._get_or_create_background_mask(
                        image_name, sample, needs_unwrap=True
                    )
                    if backgroundMask is not None:
                        maskCache[True] = (backgroundMask, threshold, unwrappedSample)
                except Exception as error:
                    import traceback
                    print(f"[DEBUG] Excepción calculando unwrap para {image_name}:")
                    traceback.print_exc()
                    maskCache['unwrap_error'] = str(error)
                    wx.MessageBox(f"Error applying unwrap for {image_name}: {error}", "Error", wx.ICON_ERROR)

            # One row per image, whichever lane produced the mask. Unwrapping
            # changes the values, not which pixels are background, so both
            # lanes share the one mask -- but only the wrapped branch used to
            # record it, and an Unwrapped-only run left no trace of how the
            # image had been segmented.
            segmented = self._cache.masks.get(image_name)
            if segmented is not None:
                self._update_tables_with_mask_data(
                    image_name, sample, segmented['mask'], segmented['threshold'],
                    segmented.get('recipe')
                )

        # Smart zone handling according to mode
        if zoneMetrics:
            
            if zones_mode == 'shared':
                # SHARED MODE: Use same zones for all images and all metrics
                if self._cache.shared_zones is None:
                    # First image: ask for zones
                    zoneCount = self._get_zone_count_from_user()
                    if zoneCount is None:
                        return maskCache
                    
                    zones = select_manual_zones(zoneSample, zoneCount)
                    if not zones or len(zones) == 0:
                        wx.MessageBox("No zones selected.", "Info", wx.ICON_INFORMATION)
                        return maskCache
                    
                    # Save as shared zones
                    self._cache.shared_zones = (zones, zoneCount)
                else:
                    zones, zoneCount = self._cache.shared_zones
                
                maskCache['zones'] = zones
                maskCache['num_zones'] = zoneCount
            
            elif zones_mode == 'per_metric':
                # PER-METRIC MODE: Each metric gets its own zones
                zonesByMetric = {}
                
                for metricId in zoneMetrics:
                    cacheKey = f"{image_name}_{metricId}"
                    
                    if cacheKey in self._cache.zones:
                        zones, zoneCount = self._cache.zones[cacheKey]
                    else:
                        zoneCount = self._get_zone_count_from_user()
                        if zoneCount is None:
                            return maskCache
                        
                        zones = select_manual_zones(zoneSample, zoneCount)
                        if not zones or len(zones) == 0:
                            wx.MessageBox(f"No zones selected for {metricId}.", "Info", wx.ICON_INFORMATION)
                            return maskCache
                        
                        self._cache.zones[cacheKey] = (zones, zoneCount)
                    
                    zonesByMetric[metricId] = (zones, zoneCount)
                
                maskCache['metric_zones'] = zonesByMetric
            
            else:  # PER-IMAGE MODE
                cacheKey = f"{image_name}_zones"
                
                if cacheKey in self._cache.zones:
                    zones, zoneCount = self._cache.zones[cacheKey]
                else:
                    zoneCount = self._get_zone_count_from_user()
                    if zoneCount is None:
                        return maskCache
                    
                    zones = select_manual_zones(zoneSample, zoneCount)
                    if not zones or len(zones) == 0:
                        wx.MessageBox(f"No zones selected for {image_name}.", "Info", wx.ICON_INFORMATION)
                        return maskCache
                    
                    self._cache.zones[cacheKey] = (zones, zoneCount)
                
                maskCache['zones'] = zones
                maskCache['num_zones'] = zoneCount
        
        return maskCache


    def _skip(self, reason):
        """Record WHY a metric could not be computed, and return None."""
        # Discarding the exception made a cancelled dialog look like a bug.
        self._skip_reason = reason
        return None

    def _calculate_single_metric_cached(self, metric_id, sample, image_name, masks_cache):
        """Calculate a single metric using pre-calculated cached masks."""
        
        self._skip_reason = None

        # Search for metric in the map
        if metric_id not in self.metric_map:
            return self._skip("unknown metric id (not in metric_map)")
        
        metricFunction, tableColumn, needsUnwrap = self.metric_map[metric_id]

        if needsUnwrap and not self._ask_unwrap_once():
            return self._skip("phase unwrapping was declined for this batch")
        
        # SPECIAL HANDLING FOR LEGENDRE METRICS
        if 'legendre' in metric_id.lower():
            if not self._legendre_params_shown:
                limit = self._get_limit_from_user(default=64)
                if limit is None:
                    return self._skip("Legendre limit dialog was cancelled")
                
                orderMax = self._get_legendre_order_from_user(default=5)
                if orderMax is None:
                    return self._skip("Legendre limit dialog was cancelled")
                
                self._legendre_limit = limit
                self._legendre_order = orderMax
                self._legendre_params_shown = True
            else:
                limit = self._legendre_limit
                orderMax = self._legendre_order
            
            
            # For zone-based Legendre
            if 'zones' in metric_id:
                if 'zones' not in masks_cache or 'num_zones' not in masks_cache:
                    return self._skip("Legendre order dialog was cancelled")
                
                zones = masks_cache['zones']
                zoneCount = masks_cache['num_zones']
                
                # The unwrapped copy when the user allowed it, like every other
                # zone metric; see _precalculate_masks.
                zoneSample = masks_cache.get('zone_sample', sample)

                try:
                    result = metricFunction(zoneSample, mask=None, manual=True, num_zones=zoneCount,
                                    limit=limit, order_max=orderMax, zones=zones)
                    
                    if isinstance(result, tuple):
                        meanCoefficients, zoneStats = result
                    else:
                        return self._skip("Legendre returned no per-zone coefficients")
                    
                    # Add coefficients to table
                    self.ensure_metric_table()
                    rowIndex = self._find_or_create_row(image_name)
                    
                    for index, coefficient in enumerate(meanCoefficients):
                        columnName = f"Legendre_C{index+1}_Zones"
                        self._ensure_column(columnName, width=120)
                        columnIndex = self._get_column_index_by_name(columnName)
                        if columnIndex >= 0:
                            self.metric_table.SetItem(rowIndex, columnIndex, f"{coefficient:.6f}")
                    
                    return float(np.mean(meanCoefficients))
                except Exception as error:
                    return self._skip(f"{type(error).__name__}: {error}")
            
            # For background Legendre (no zones) - USE CACHED MASK
            else:
                # Get the cached mask (already calculated in _precalculate_masks)
                if needsUnwrap not in masks_cache:
                    return self._skip("no background mask: the threshold dialog was cancelled, or unwrapping failed")
                
                backgroundMask, threshold, processedSample = masks_cache[needsUnwrap]
                
                try:
                    # Calculate Legendre coefficients using the CACHED mask
                    coefficients = metricFunction(
                        processedSample, backgroundMask, manual=False,
                        num_zones=2, limit=limit, order_max=orderMax,
                        # The analysis layer hands the surface back to be shown here.
                        show_reconstruction=show_legendre_reconstruction,
                    )
                    
                    # Add coefficients to table
                    self.ensure_metric_table()
                    rowIndex = self._find_or_create_row(image_name)
                    
                    suffix = "_Unwrapped" if needsUnwrap else ""
                    for index, coefficient in enumerate(coefficients):
                        columnName = f"Legendre_C{index+1}{suffix}"
                        self._ensure_column(columnName, width=120)
                        columnIndex = self._get_column_index_by_name(columnName)
                        if columnIndex >= 0:
                            self.metric_table.SetItem(rowIndex, columnIndex, f"{coefficient:.6f}")
                    
                    return float(np.mean(coefficients))
                except Exception as error:
                    return self._skip(f"{type(error).__name__}: {error}")
        
        # ZONE METRICS: Use cached zones (can be shared, per-image, or per-metric)
        if 'zones' in metric_id:
            # Check if this is per-metric mode
            if 'metric_zones' in masks_cache:
                # Per-metric mode: each metric has different zones
                if metric_id not in masks_cache['metric_zones']:
                    return self._skip("no zones were selected for this metric")
                zones, zoneCount = masks_cache['metric_zones'][metric_id]
            else:
                # Shared or per-image mode: all metrics use same zones
                if 'zones' not in masks_cache or 'num_zones' not in masks_cache:
                    return self._skip("no zones were selected for this metric")
                zones = masks_cache['zones']
                zoneCount = masks_cache['num_zones']
            
            # Extract metric name from metric_id (e.g., 'std_zones' -> 'std')
            metricName = metric_id.split('_')[0].lower()
            
            # Unwrapped copy if the user asked for it in _precalculate_masks
            zoneSample = masks_cache.get('zone_sample', sample)

            # Calculate metric in each zone
            values = []
            zoneStats = []
            for index, (xMin, xMax, yMin, yMax) in enumerate(zones, start=1):
                zoneImage = zoneSample[yMin:yMax, xMin:xMax]
                zoneValue = metricFunction(zoneImage)
                values.append(zoneValue)
                zoneStats.append({
                    'zone': index,
                    metricName: zoneValue,
                    'coords': (xMin, xMax, yMin, yMax)
                })
            
            meanValue = float(np.mean(values))
            
            # Update table with mean value and individual zone values
            self.update_table_with_zones(
                image_name,
                tableColumn,
                meanValue,
                zoneStats,
                num_zones=zoneCount
            )
            
            return meanValue
        
        # MODULE 3: Ground-truth metrics (need ground-truth loaded)
        if metric_id.startswith(('ssim', 'mse', 'psnr')):
            if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
                return self._skip("no ground-truth image loaded")
            
            try:
                # Module 3 never unwraps: see _prepare_sample_and_gt.
                value = metricFunction(sample, self.ground_truth_data)
                if isinstance(value, tuple):
                    value = value
                self.update_table_with_zones(image_name, tableColumn, value)
                return value
            except Exception as error:
                return self._skip(f"{type(error).__name__}: {error}")
        
        # MODULE 2: Global metrics (NO need mask)
        elif metric_id.startswith(('mmm_', 'gradient_', 'tsm_', 
                                'laplacian_', 'spatial_freq_', 'global_entropy',
                                'sharpness_', 'fwhm_global')):
            if needsUnwrap:
                try:
                    processedSample = self._get_unwrapped_sample(image_name, sample)
                except Exception as error:
                    return self._skip(f"{type(error).__name__}: {error}")
            else:
                processedSample = sample
            
            parameters = inspect.signature(metricFunction).parameters
            
            try:
                if metric_id.startswith('tsm_'):
                    tsmThreshold = self._get_tsm_threshold(
                        image_name, processedSample, needsUnwrap)
                    if tsmThreshold is None:
                        return self._skip(
                            "the TSM threshold dialog was cancelled")
                    value = metricFunction(processedSample, use_unwrap=False,
                                        threshold=tsmThreshold)
                elif 'use_unwrap' in parameters:
                    # Already unwrapped above: asking again would unwrap twice.
                    value = metricFunction(processedSample, use_unwrap=False)
                else:
                    value = metricFunction(processedSample)
                
                if isinstance(value, tuple):
                    value = value
                
                self.update_table_with_zones(image_name, tableColumn, value)
                return value
            except Exception as error:
                return self._skip(f"{type(error).__name__}: {error}")
        
        # MODULE 1: Background metrics (NEED mask)
        else:
            if needsUnwrap not in masks_cache:
                return self._skip("no background mask: the threshold dialog was cancelled, or unwrapping failed")
            
            backgroundMask, threshold, processedSample = masks_cache[needsUnwrap]
            
            try:
                # Module 1 metrics take (image, mask); this frame handles zones.
                value = metricFunction(processedSample, backgroundMask)
                self.update_table_with_zones(image_name, tableColumn, value)
                return value
            except Exception as error:
                return self._skip(f"{type(error).__name__}: {error}")


    def _sample_fingerprint(self, sample):
        """Cheap fingerprint telling whether an image's phase data changed."""
        pixels = np.ascontiguousarray(sample)
        digest = hashlib.md5()
        digest.update(str(pixels.shape).encode())
        digest.update(str(pixels.dtype).encode())
        if pixels.size > 4_000_000:
            # Hashing every byte costs ~0.3 s here, so a strided sample will do.
            flatPixels = pixels.reshape(-1)
            digest.update(flatPixels[::max(1, flatPixels.size // 1_000_000)].tobytes())
        else:
            digest.update(pixels.tobytes())
        return digest.hexdigest()
    
    
    def _get_tsm_threshold(self, image_name, sample, unwrapped):
        """Where to binarise for TSM, asked once per image and reused."""
        # The formula fixes everything but the threshold; a mean read ~50 %.
        cacheKey = (image_name, bool(unwrapped))
        fingerprint = self._sample_fingerprint(sample)

        cached = self._cache.tsm_thresholds.get(cacheKey)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

        tsmThreshold = ask_tsm_threshold(self, sample, image_name)
        if tsmThreshold is None:
            return None

        self._cache.tsm_thresholds[cacheKey] = (fingerprint, tsmThreshold)
        return tsmThreshold

    def _get_unwrapped_sample(self, image_name, sample):
        """Unwrap a phase image ONCE per image and reuse the result."""
        # NOT idempotent: unwrapping twice shifts values by tens of radians.
        fingerprint = self._sample_fingerprint(sample)
        cached = self._cache.get_unwrapped(image_name, fingerprint)
        if cached is not None:
            return cached

        # Eviction is the cache's business, not this method's.
        return self._cache.store_unwrapped(
            image_name, fingerprint, rb.unwrap_with_scikit(sample)
        )


    def _get_or_create_background_mask(self, image_name, sample, needs_unwrap):
        """Return (background_mask, threshold, processed_sample) for one image."""
        # ONCE per image: unwrapping changes values, not the background pixels.
        fingerprint = self._sample_fingerprint(sample)
        cached = self._cache.masks.get(image_name)

        if cached is None or cached['fingerprint'] != fingerprint:
            # New image or changed data: segment once, on the wrapped sample.
            # The recipe comes from the batch when one has been settled; only
            # the threshold is tuned per image, so the rows stay comparable.
            batchRecipe = self._cache.segmentation_recipe
            backgroundMask, threshold, recipe = create_background_mask(
                sample, recipe=batchRecipe, recipe_locked=batchRecipe is not None,
                image_name=image_name, parent=self)
            if backgroundMask is None:
                # Segmentation dialog cancelled: nothing to cache, skip it.
                return None, None, None
            if batchRecipe is None:
                self._cache.segmentation_recipe = recipe
            cached = {
                'fingerprint': fingerprint,
                'mask': backgroundMask,
                'threshold': threshold,
                'recipe': recipe,
            }
            self._cache.masks[image_name] = cached

        if needs_unwrap:
            processedSample = self._get_unwrapped_sample(image_name, sample)
        else:
            processedSample = sample

        return cached['mask'], cached['threshold'], processedSample

    # Finish of module 1

    
    # Global Phase Distortion Metrics (Module 2)




    # Finish of module 2


    # Ground-Truth Comparisons (Module 3)
    
    def on_load_ground_truth(self, event):
        # Load ground-truth image for comparison.
        with wx.FileDialog(
            self, 
            "Select Ground-Truth Image",
            wildcard="Image files (*.png;*.jpg;*.bmp;*.tif)|*.png;*.jpg;*.bmp;*.tif|MAT files (*.mat)|*.mat",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            
            exportPath = dialog.GetPath()
            try:
                self.ground_truth_data = gtc.load_ground_truth(exportPath)
                self.ground_truth_path = exportPath
                wx.MessageBox(
                    f"Ground-truth loaded successfully:\n{exportPath}\n\nShape: {self.ground_truth_data.shape}", 
                    "Success", 
                    wx.ICON_INFORMATION
                )
            except Exception as error:
                wx.MessageBox(f"Error loading ground-truth: {str(error)}", "Error", wx.ICON_ERROR)

    def _ensure_ground_truth_loaded(self, metrics_dict):
        # Ensure ground-truth is loaded before calculating Module 3 metrics.
        hasModule3 = any(candidateId.startswith(('ssim', 'mse', 'psnr')) for candidateId in metrics_dict.keys())
        
        if not hasModule3:
            return True  # Ground-truth not needed
        
        # Check current ground-truth status
        groundTruth = hasattr(self, 'ground_truth_data') and self.ground_truth_data is not None
        
        # 🔹 CASE 1: Ground-truth already loaded
        if groundTruth:
            groundTruthName = os.path.basename(self.ground_truth_path) if hasattr(self, 'ground_truth_path') else 'Unknown'
            groundTruthShape = self.ground_truth_data.shape
            
            message = (
                f"Ground-Truth Currently Loaded:\n\n"
                f"File: {groundTruthName}\n"
                f"Shape: {groundTruthShape}\n\n"
                f"Do you want to update/change the ground-truth image?"
            )
            
            answer = wx.MessageBox(
                message,
                "Ground-Truth Manager",
                wx.YES_NO | wx.ICON_QUESTION
            )
            
            if answer == wx.YES:
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
        
        # Case 2: no ground truth loaded.
        else:
            message = (
                "No Ground-Truth Loaded\n\n"
                "Module 3 metrics require a ground-truth image for comparison.\n\n"
                "Do you want to load a ground-truth image now?"
            )
            
            answer = wx.MessageBox(
                message,
                "Ground-Truth Required",
                wx.YES_NO | wx.ICON_WARNING
            )
            
            if answer == wx.YES:
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