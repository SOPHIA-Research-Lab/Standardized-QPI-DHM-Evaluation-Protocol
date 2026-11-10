"""The run summary: one row per image, one column per metric."""
# A list down the page gave each image its own block, so comparing the same
# metric across two images meant scrolling between them. The grid puts those
# numbers side by side, which is the only reason to run a batch at all.
import wx
import wx.grid

#: Shade for the odd rows, as in the tables of the main window.
ALTERNATE_ROW = wx.Colour(0xF5, 0xF7, 0xFA)
WHITE = wx.Colour(0xFF, 0xFF, 0xFF)
#: A value that could not be computed is greyed; an error is marked.
MUTED_INK = wx.Colour(0x77, 0x77, 0x77)
ERROR_INK = wx.Colour(0xC0, 0x39, 0x2B)

_INK_BY_KIND = {'skipped': MUTED_INK, 'nan': MUTED_INK, 'error': ERROR_INK}


class ResultsGridDialog(wx.Dialog):
    """Images down the side, metrics across the top, values in the cells."""

    def __init__(self, parent, image_names, metric_labels, cells, cancelled=False):
        title = "Metrics Calculation Cancelled" if cancelled \
            else "Metrics Calculation Complete"
        super().__init__(parent, title=title,
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.image_names = list(image_names)
        self.metric_labels = list(metric_labels)
        self.cells = cells

        mainSizer = wx.BoxSizer(wx.VERTICAL)

        if cancelled:
            notice = wx.StaticText(
                self, label="Cancelled by the user. Everything below was "
                            "already written to the tables.")
            notice.SetForegroundColour(ERROR_INK)
            mainSizer.Add(notice, 0, wx.LEFT | wx.RIGHT | wx.TOP, 12)

        self.grid = self._build_grid()
        mainSizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 12)

        details = self._detail_lines()
        if details:
            label = wx.StaticText(self, label="Cells without a number:")
            label.SetFont(label.GetFont().Bold())
            mainSizer.Add(label, 0, wx.LEFT | wx.RIGHT, 12)
            detailBox = wx.TextCtrl(
                self, value="\n".join(details),
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
            detailBox.SetMinSize(self.FromDIP(wx.Size(-1, 90)))
            mainSizer.Add(detailBox, 0, wx.EXPAND | wx.ALL, 12)

        buttonRow = wx.BoxSizer(wx.HORIZONTAL)
        buttonRow.AddStretchSpacer()
        closeButton = wx.Button(self, wx.ID_OK, "Close")
        buttonRow.Add(closeButton, 0)
        mainSizer.Add(buttonRow, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        self.SetSizer(mainSizer)
        closeButton.SetDefault()
        self._fit_to_contents(mainSizer)
        self.CentreOnParent()

    # ----------------------------------------------------------------- grid
    def _build_grid(self):
        """The matrix itself: one row per metric, one column per image."""
        # This way round because a metric is read ACROSS the images: the numbers
        # being compared end up side by side on one line, and a long metric name
        # is written once down the side instead of widening a column.
        grid = wx.grid.Grid(self)
        grid.CreateGrid(max(1, len(self.metric_labels)),
                        max(1, len(self.image_names)))
        grid.EnableEditing(False)
        grid.DisableDragRowSize()
        grid.SetCellHighlightPenWidth(0)
        grid.SetSelectionMode(wx.grid.Grid.GridSelectRows)

        headerFont = grid.GetLabelFont().Bold()
        grid.SetLabelFont(headerFont)
        grid.SetLabelBackgroundColour(ALTERNATE_ROW)
        # The metric name reads as a label, not as a centred heading.
        grid.SetRowLabelAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTRE)

        for column, imageName in enumerate(self.image_names):
            grid.SetColLabelValue(column, imageName)

        for row, label in enumerate(self.metric_labels):
            grid.SetRowLabelValue(row, label)
            shade = ALTERNATE_ROW if row % 2 else WHITE
            for column, imageName in enumerate(self.image_names):
                # A cell with no entry at all is one the run never reached,
                # which is not the same as one that was skipped: leave it blank.
                text, kind, _ = self.cells.get((imageName, label),
                                               ("", 'none', ''))
                grid.SetCellValue(row, column, text)
                grid.SetCellAlignment(row, column, wx.ALIGN_RIGHT, wx.ALIGN_CENTRE)
                grid.SetCellBackgroundColour(row, column, shade)
                ink = _INK_BY_KIND.get(kind)
                if ink is not None:
                    grid.SetCellTextColour(row, column, ink)

        self._size_columns(grid)
        return grid

    def _size_columns(self, grid):
        """Wide enough for the numbers AND for the image name above them."""
        padding = self.FromDIP(24)
        for column, imageName in enumerate(self.image_names):
            grid.AutoSizeColumn(column, setAsMin=False)
            byLabel = grid.GetTextExtent(imageName).width + padding
            grid.SetColSize(column, max(grid.GetColSize(column), byLabel))

        longest = max((grid.GetTextExtent(label).width
                       for label in self.metric_labels), default=0)
        grid.SetRowLabelSize(max(self.FromDIP(90), longest + padding))

    def _fit_to_contents(self, sizer):
        """Ask for the size the grid wants, within what the screen can show."""
        gridWidth = self.grid.GetRowLabelSize() + sum(
            self.grid.GetColSize(column)
            for column in range(self.grid.GetNumberCols()))
        gridHeight = self.grid.GetColLabelSize() + sum(
            self.grid.GetRowSize(row)
            for row in range(self.grid.GetNumberRows()))
        # Exactly the content: a margin here would read as an empty extra row.
        # When the cap below bites the grid scrolls, and the dialog resizes.
        self.grid.SetMinSize(wx.Size(gridWidth + 2, gridHeight + 2))

        sizer.Fit(self)
        screen = wx.Display(
            max(0, wx.Display.GetFromWindow(self))).GetClientArea()
        size = self.GetSize()
        self.SetSize(wx.Size(min(size.width, int(screen.width * 0.9)),
                             min(size.height, int(screen.height * 0.9))))

    # -------------------------------------------------------------- details
    def _detail_lines(self):
        """Why each non-numeric cell has no number: the grid has no room for it."""
        lines = []
        # In reading order: down the metrics, across the images.
        for label in self.metric_labels:
            for imageName in self.image_names:
                _, kind, detail = self.cells.get((imageName, label),
                                                 ("", 'none', ''))
                if kind != 'value' and detail:
                    lines.append(f"{label}  -  {imageName}: {detail}")
        return lines


def show_results_grid(parent, image_names, metric_labels, cells, cancelled=False):
    """Show the run summary as a grid. `cells` maps (image, label) -> (text, kind, detail)."""
    dialog = ResultsGridDialog(parent, image_names, metric_labels, cells, cancelled)
    try:
        dialog.ShowModal()
    finally:
        dialog.Destroy()
