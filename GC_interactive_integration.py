"""
Created on Fri Jul 24 16:21:54 2026

@author: Garrett

from pathlib import Path


GC-FID CSV Peak Explorer (Qt)

Features
- Choose a CSV/TXT file with two numeric columns: RT and signal
- Plot chromatogram in a Qt window
- Adjust start time, prominence, width, and distance
- Detect and integrate peaks
- Show detected peaks with apex RT, height, area, and area %
- Export peak table to CSV

Install:
    pip install numpy scipy matplotlib PySide6

Run:
    python gc_fid_qt_peak_gui.py
"""

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import percentile_filter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFontComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QCheckBox,
    QWidget,
)

from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


_SPLIT_RE = re.compile(r"[\\t,; ]+")


@dataclass
class PeakResult:
    peak_index: int
    rt_min: float
    height: float
    left_min: float
    right_min: float
    area: float
    area_percent: float


def _to_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def parse_two_column_chromatogram(text: str) -> Tuple[np.ndarray, np.ndarray]:
    x_vals: List[float] = []
    y_vals: List[float] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p for p in _SPLIT_RE.split(line) if p]
        if len(parts) < 2:
            continue

        x = _to_float(parts[0])
        y = _to_float(parts[1])
        if x is None or y is None:
            continue

        x_vals.append(x)
        y_vals.append(y)

    if len(x_vals) < 3:
        raise ValueError("Could not parse at least 3 numeric rows from the file.")

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    order = np.argsort(x)
    return x[order], y[order]


def apply_start_cutoff(x: np.ndarray, y: np.ndarray, start_min: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = x >= start_min
    return x[mask], y[mask]

def _make_odd_window(window: int, n: int, minimum: int = 3) -> int:
    """Return a valid odd window length smaller than the data length."""
    window = max(minimum, int(window))

    if window % 2 == 0:
        window += 1

    max_window = n if n % 2 == 1 else n - 1

    if max_window < minimum:
        return 0

    return min(window, max_window)


def estimate_baseline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    smooth_window: int = 11,
    baseline_window: int = 101,
    baseline_percentile: float = 10.0,
    baseline_smooth_window: int = 101,
) -> np.ndarray:
    """
    Estimate a slowly varying chromatogram baseline.

    1. Smooth short-scale detector noise.
    2. Use a low rolling percentile to follow the lower envelope.
    3. Smooth that envelope to remove small wiggles.
    """
    if len(y) < 5:
        return np.zeros_like(y, dtype=float)

    # ------------------------------------------------------------
    # Step 1: lightly smooth the raw signal.
    # This is ONLY for baseline estimation.
    # ------------------------------------------------------------
    sw = _make_odd_window(smooth_window, len(y))

    if sw >= 5:
        y_smooth = savgol_filter(
            y,
            window_length=sw,
            polyorder=2,
        )
    else:
        y_smooth = y.copy()

    # ------------------------------------------------------------
    # Step 2: estimate the lower envelope using a rolling
    # percentile. This is much less sensitive to peaks than
    # using the local mean.
    # ------------------------------------------------------------
    bw = _make_odd_window(baseline_window, len(y))

    if bw == 0:
        return np.zeros_like(y, dtype=float)

    baseline_raw = percentile_filter(
        y_smooth,
        percentile=float(baseline_percentile),
        size=bw,
        mode="nearest",
    )

    # ------------------------------------------------------------
    # Step 3: smooth the estimated baseline so detector noise
    # does not become part of the baseline itself.
    # ------------------------------------------------------------
    bs = _make_odd_window(baseline_smooth_window, len(y))

    if bs >= 5:
        baseline = savgol_filter(
            baseline_raw,
            window_length=bs,
            polyorder=2,
        )
    else:
        baseline = baseline_raw

    return baseline

def find_peak_boundaries(
    signal: np.ndarray,
    peaks: np.ndarray,
    *,
    search_window: int = 200,
    smooth_window: int = 11,
    valley_prominence_fraction: float = 0.02,
) -> List[Tuple[int, int]]:
    """
    Find start/end boundaries independently of the main peak
    prominence setting.

    For each detected peak, scan left and right for a meaningful
    local valley in the baseline-corrected signal.

    valley_prominence_fraction is the minimum valley prominence
    expressed as a fraction of the peak height.
    """

    if len(peaks) == 0:
        return []

    # Smooth only for boundary/valley detection.
    sw = _make_odd_window(smooth_window, len(signal))

    if sw >= 5:
        search_signal = savgol_filter(
            signal,
            window_length=sw,
            polyorder=2,
        )
    else:
        search_signal = signal.copy()

    boundaries: List[Tuple[int, int]] = []

    for peak in peaks:
        peak = int(peak)

        peak_height = max(
            float(search_signal[peak]),
            0.0,
        )

        # Valley prominence is relative to this peak.
        valley_prominence = max(
            peak_height * float(valley_prominence_fraction),
            1e-12,
        )

        # --------------------------------------------------------
        # LEFT SIDE
        # --------------------------------------------------------
        left_start = max(
            0,
            peak - int(search_window),
        )

        left_segment = search_signal[left_start:peak + 1]

        if len(left_segment) >= 3:
            valleys_left, valley_props_left = find_peaks(
                -left_segment,
                prominence=valley_prominence,
                distance=3,
            )

            if len(valleys_left) > 0:
                # Choose the valley closest to the apex.
                left = left_start + int(
                    valleys_left[-1]
                )
            else:
                # No significant valley found.
                # Fall back to the lowest point in the search range.
                left = left_start + int(
                    np.argmin(left_segment)
                )
        else:
            left = left_start

        # --------------------------------------------------------
        # RIGHT SIDE
        # --------------------------------------------------------
        right_end = min(
            len(signal) - 1,
            peak + int(search_window),
        )

        right_segment = search_signal[peak:right_end + 1]

        if len(right_segment) >= 3:
            valleys_right, valley_props_right = find_peaks(
                -right_segment,
                prominence=valley_prominence,
                distance=3,
            )

            if len(valleys_right) > 0:
                # Choose the valley closest to the apex.
                right = peak + int(
                    valleys_right[0]
                )
            else:
                # No significant valley found.
                # Fall back to the lowest point in the search range.
                right = peak + int(
                    np.argmin(right_segment)
                )
        else:
            right = right_end

        # --------------------------------------------------------
        # Safety checks
        # --------------------------------------------------------
        left = max(
            0,
            min(int(left), peak),
        )

        right = min(
            len(signal) - 1,
            max(int(right), peak),
        )

        boundaries.append((left, right))

    return boundaries

def detect_and_integrate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    start_min: float,
    prominence: float,
    distance: int,
    width: float,
    baseline_smooth_window: int,
    baseline_window: int,
    baseline_percentile: float,
    baseline_smooth2_window: int,
    boundary_window: int,
    boundary_prominence_fraction: float,
) -> List[PeakResult]:
    x2, y2 = apply_start_cutoff(x, y, start_min)
    if len(x2) < 5:
        return []
    
    baseline = estimate_baseline(
        x2,
        y2,
        smooth_window=baseline_smooth_window,
        baseline_window=baseline_window,
        baseline_percentile=baseline_percentile,
        baseline_smooth_window=baseline_smooth2_window,
    )

    corrected = y2 - baseline
    corrected = np.maximum(corrected, 0.0)
    
    kwargs = {
        "prominence": float(prominence),
        "distance": max(1, int(distance)),
    }

    if width and width > 0:
        kwargs["width"] = float(width)

    # Detect peak apexes.
    peaks, props = find_peaks(corrected, **kwargs)
    
    boundaries = find_peak_boundaries(
        corrected,
        peaks,
        search_window=int(boundary_window),
        smooth_window=11,
        valley_prominence_fraction=(
            float(boundary_prominence_fraction)
        ),
    )

    if len(peaks) == 0:
        return []

    temp_rows = []
    raw_areas = []

    for i, p in enumerate(peaks):
        p = int(p)
    
        left, right = boundaries[i]
    
        if right <= left + 1:
            continue
    
        xseg = x2[left:right + 1]
        yseg = y2[left:right + 1]
    
        if len(xseg) < 3:
            continue
    
        # Use the whole-chromatogram baseline calculated above.
        baseline_seg = baseline[left:right + 1]

        corrected_peak = yseg - baseline_seg
        corrected_peak = np.maximum(
            corrected_peak,
            0.0,
        )

        area = float(np.trapz(corrected_peak, xseg))    
        raw_areas.append(area)
    
        temp_rows.append(
            {
                "peak_index": p,
                "rt_min": float(x2[p]),
                "height": float(y2[p]),
                "left_min": float(x2[left]),
                "right_min": float(x2[right]),
                "area": area,
            }
        )

    total_area = float(np.sum(raw_areas)) if raw_areas else 0.0

    results: List[PeakResult] = []

    for row in temp_rows:
        area_percent = (
            0.0
            if total_area == 0
            else 100.0 * row["area"] / total_area
        )

        results.append(
            PeakResult(
                peak_index=row["peak_index"],
                rt_min=row["rt_min"],
                height=row["height"],
                left_min=row["left_min"],
                right_min=row["right_min"],
                area=row["area"],
                area_percent=area_percent,
            )
        )

    results.sort(key=lambda r: r.rt_min)
    return results


class PeakExplorer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GC-FID CSV Peak Explorer")
        self.resize(1500, 950)

        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.current_peaks: List[PeakResult] = []
        self.current_path: Optional[Path] = None

        self._build_ui()
        
    def save_clean_png(self) -> None:
        if self.x is None or self.y is None:
            QMessageBox.information(self, "No data", "Load a file first.")
            return
    
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save chromatogram PNG",
            "",
            "PNG files (*.png);;All files (*.*)",
        )
        if not path:
            return
    
        if not path.lower().endswith(".png"):
            path += ".png"
    
        start_min = float(self.start_spin.value())
        x2, y2 = apply_start_cutoff(self.x, self.y, start_min)
    
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(x2, y2, linewidth=1.0)
        ax.set_xlabel(self.xlabel_edit.text())
        ax.set_ylabel(self.ylabel_edit.text())
        ax.set_title(self.title_edit.text())
        ax.grid(True, alpha=0.25)
        
        figs = self._figure_settings()
        
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(x2, y2, linewidth=figs["line_width"])
        ax.set_xlabel(figs["xlabel"], fontsize=figs["axis_size"])
        ax.set_ylabel(figs["ylabel"], fontsize=figs["axis_size"])
        ax.set_title(figs["title"], fontsize=figs["title_size"])
        ax.tick_params(axis="both", labelsize=figs["tick_size"])
        ax.grid(True, alpha=0.25)

        fig.savefig(path, dpi=200, bbox_inches="tight")
        self.status.setText(f"Saved PNG: {path}")

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        main.addWidget(splitter)

        left = QWidget()
        right = QWidget()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([470, 1030])

        left_layout = QVBoxLayout(left)
        right_layout = QVBoxLayout(right)

        file_box = QGroupBox("Input file")
        file_layout = QVBoxLayout(file_box)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        file_btn_row = QHBoxLayout()
        self.btn_choose = QPushButton("Choose CSV/TXT...")
        self.btn_export = QPushButton("Export peak table...")
        self.btn_save_png = QPushButton("Save PNG...")
        
        self.btn_save_png.clicked.connect(self.save_clean_png)
        file_layout.addLayout(file_btn_row)

        left_layout.addWidget(file_box)

        param_box = QGroupBox("Peak settings")
        param_layout = QFormLayout(param_box)

        self.title_edit = QLineEdit("GC-FID Chromatogram")
        self.xlabel_edit = QLineEdit("Retention time (min)")
        self.ylabel_edit = QLineEdit("Signal")
        
        figure_box = QGroupBox("Figure settings")
        figure_layout = QFormLayout(figure_box)
        
        left_layout.addWidget(figure_box)
        
        self.shade_toggle = QCheckBox("Shade integrated areas")
        self.shade_toggle.setChecked(False)

        self.boundary_toggle = QCheckBox("Show peak start/end markers")
        self.boundary_toggle.setChecked(False)

        figure_layout.addRow("Area shading", self.shade_toggle)
        figure_layout.addRow("Peak boundaries", self.boundary_toggle)

        left_layout.addWidget(figure_box)
        
        baseline_box = QGroupBox("Baseline correction")
        baseline_layout = QFormLayout(baseline_box)
        
        self.baseline_toggle = QCheckBox("Show baseline")
        self.baseline_toggle.setChecked(True)
        
        self.corrected_toggle = QCheckBox("Show corrected signal")
        self.corrected_toggle.setChecked(False)
        
        self.corrected_toggle.stateChanged.connect(
            self._visual_options_changed
        )
        
        baseline_layout.addRow(
            "Corrected signal",
            self.corrected_toggle,
        )
        
        self.baseline_smooth_spin = QSpinBox()
        self.baseline_smooth_spin.setRange(3, 1001)
        self.baseline_smooth_spin.setSingleStep(2)
        self.baseline_smooth_spin.setValue(11)
        
        self.baseline_window_spin = QSpinBox()
        self.baseline_window_spin.setRange(5, 5001)
        self.baseline_window_spin.setSingleStep(10)
        self.baseline_window_spin.setValue(101)
        
        self.baseline_percentile_spin = QDoubleSpinBox()
        self.baseline_percentile_spin.setRange(1.0, 50.0)
        self.baseline_percentile_spin.setSingleStep(1.0)
        self.baseline_percentile_spin.setValue(10.0)
        
        self.baseline_smooth2_spin = QSpinBox()
        self.baseline_smooth2_spin.setRange(3, 5001)
        self.baseline_smooth2_spin.setSingleStep(10)
        self.baseline_smooth2_spin.setValue(101)
        
        self.baseline_toggle.stateChanged.connect(
            self._visual_options_changed
        )
        
        self.baseline_smooth_spin.valueChanged.connect(
            self._visual_options_changed
        )
        
        self.baseline_window_spin.valueChanged.connect(
            self._visual_options_changed
        )
        
        self.baseline_percentile_spin.valueChanged.connect(
            self._visual_options_changed
        )
        
        self.baseline_smooth2_spin.valueChanged.connect(
            self._visual_options_changed
        )
        
        baseline_layout.addRow(
            "Show baseline",
            self.baseline_toggle,
        )
        baseline_layout.addRow(
            "Noise smoothing (points)",
            self.baseline_smooth_spin,
        )
        baseline_layout.addRow(
            "Baseline window (points)",
            self.baseline_window_spin,
        )
        baseline_layout.addRow(
            "Baseline percentile (%)",
            self.baseline_percentile_spin,
        )
        baseline_layout.addRow(
            "Baseline smoothing (points)",
            self.baseline_smooth2_spin,
        )
        
        left_layout.addWidget(baseline_box)
        
        self.boundary_window_spin = QSpinBox()
        self.boundary_window_spin.setRange(5, 10000)
        self.boundary_window_spin.setSingleStep(10)
        self.boundary_window_spin.setValue(200)
        
        baseline_layout.addRow(
            "Boundary search window (points)",
            self.boundary_window_spin,
        )
        
        self.boundary_prominence_spin = QDoubleSpinBox()
        self.boundary_prominence_spin.setRange(0.1, 25.0)
        self.boundary_prominence_spin.setSingleStep(0.5)
        self.boundary_prominence_spin.setValue(2.0)

        baseline_layout.addRow(
            "Boundary valley sensitivity (%)",
            self.boundary_prominence_spin,
        )        
        
        file_btn_row.addWidget(self.btn_choose)
        file_btn_row.addWidget(self.btn_export)
        file_btn_row.addWidget(self.btn_save_png)
        
        self.title_edit = QLineEdit("GC-FID Chromatogram")
        self.xlabel_edit = QLineEdit("Retention time (min)")
        self.ylabel_edit = QLineEdit("Signal")
        
        self.font_box = QFontComboBox()
        self.font_box.setCurrentFont(QFont("Arial"))

        figure_layout.addRow("Font", self.font_box)
        
        self.title_size_spin = QSpinBox()
        self.title_size_spin.setRange(6, 48)
        self.title_size_spin.setValue(16)
        
        self.axis_size_spin = QSpinBox()
        self.axis_size_spin.setRange(6, 36)
        self.axis_size_spin.setValue(14)
        
        self.tick_size_spin = QSpinBox()
        self.tick_size_spin.setRange(6, 24)
        self.tick_size_spin.setValue(12)
        
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.1, 10.0)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setValue(1.0)
        
        figure_layout.addRow("Title", self.title_edit)
        figure_layout.addRow("X-axis label", self.xlabel_edit)
        figure_layout.addRow("Y-axis label", self.ylabel_edit)
        figure_layout.addRow("Title size", self.title_size_spin)
        figure_layout.addRow("Axis size", self.axis_size_spin)
        figure_layout.addRow("Tick size", self.tick_size_spin)
        figure_layout.addRow("Line width", self.line_width_spin)
        
        left_layout.addWidget(figure_box)
        
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(6)
        self.start_spin.setRange(0.0, 1000.0)
        self.start_spin.setSingleStep(0.01)
        self.start_spin.setValue(0.0)

        self.prom_spin = QDoubleSpinBox()
        self.prom_spin.setDecimals(6)
        self.prom_spin.setRange(0.0, 1e9)
        self.prom_spin.setSingleStep(1.0)
        self.prom_spin.setValue(20.0)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setDecimals(3)
        self.width_spin.setRange(0.0, 1e6)
        self.width_spin.setSingleStep(1.0)
        self.width_spin.setValue(0.0)

        self.dist_spin = QSpinBox()
        self.dist_spin.setRange(1, 1_000_000)
        self.dist_spin.setValue(5)

        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(1, 500)
        self.topn_spin.setValue(20)

        param_layout.addRow("Start time (min)", self.start_spin)
        param_layout.addRow("Prominence", self.prom_spin)
        param_layout.addRow("Width (points, 0=off)", self.width_spin)
        param_layout.addRow("Distance (points)", self.dist_spin)
        param_layout.addRow("Show top N peaks", self.topn_spin)

        param_layout.addRow("Plot title", self.title_edit)
        param_layout.addRow("X-axis label", self.xlabel_edit)
        param_layout.addRow("Y-axis label", self.ylabel_edit)
        
        left_layout.addWidget(param_box)

        action_box = QGroupBox("Actions")
        action_layout = QHBoxLayout(action_box)
        self.btn_update = QPushButton("Parse / Update")
        self.btn_clear = QPushButton("Clear")
        action_layout.addWidget(self.btn_update)
        action_layout.addWidget(self.btn_clear)
        left_layout.addWidget(action_box)

        self.status = QLabel("Choose a CSV/TXT file to begin.")
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(300)
        left_layout.addWidget(self.log, stretch=1)

        self.figure = Figure(figsize=(10, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas, right)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        table_box = QGroupBox("Detected peaks (sample)")
        table_layout = QVBoxLayout(table_box)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Apex RT (min)", "Height", "Area", "Area %"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        table_layout.addWidget(self.table)

        right_layout.addWidget(table_box)

        self.btn_choose.clicked.connect(self.choose_file)
        self.btn_export.clicked.connect(self.export_peak_table)
        self.btn_update.clicked.connect(self.update_analysis)
        self.btn_clear.clicked.connect(self.clear_all)
        self.shade_toggle.stateChanged.connect(
            self._visual_options_changed
        )
        self.boundary_toggle.stateChanged.connect(
            self._visual_options_changed
            )

    def log_message(self, msg: str) -> None:
        self.log.append(msg)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose chromatogram CSV/TXT",
            "",
            "CSV/TXT files (*.csv *.txt *.tsv);;All files (*.*)",
        )
        if not path:
            return
        self.current_path = Path(path)
        self.file_label.setText(str(self.current_path))
        self.load_and_parse()

    def load_and_parse(self) -> None:
        if self.current_path is None:
            QMessageBox.warning(self, "Missing file", "Choose a CSV/TXT file first.")
            return

        path = self.current_path
        if not path.exists():
            QMessageBox.critical(self, "File not found", f"File does not exist:\n{path}")
            return

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            x, y = parse_two_column_chromatogram(content)
        except Exception as e:
            QMessageBox.critical(self, "Parse error", str(e))
            return

        self.x = x
        self.y = y
        self.log.clear()
        self.log_message(f"Loaded {len(x)} points from {path.name}")
        self.update_analysis()

    def clear_all(self) -> None:
        self.x = None
        self.y = None
        self.current_peaks = []
        self.current_path = None
        self.file_label.setText("No file selected")
        self.table.setRowCount(0)
        self.log.clear()
        self.ax.clear()
        self.ax.set_title("")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.canvas.draw_idle()
        self.status.setText("Cleared.")

    def update_analysis(self) -> None:
        if self.x is None or self.y is None:
            return

        try:
            start_min = float(self.start_spin.value())
            prominence = float(self.prom_spin.value())
            width = float(self.width_spin.value())
            distance = int(self.dist_spin.value())
            top_n = int(self.topn_spin.value())
            baseline_smooth_window = int(
                self.baseline_smooth_spin.value()
            )
            
            baseline_window = int(
                self.baseline_window_spin.value()
            )
            
            baseline_percentile = float(
                self.baseline_percentile_spin.value()
            )
            
            baseline_smooth2_window = int(
                self.baseline_smooth2_spin.value()
            )
            boundary_window = int(
                self.boundary_window_spin.value()
            )
            boundary_prominence_fraction = (
                float(
                    self.boundary_prominence_spin.value()
                ) / 100.0
            )
        except Exception as e:
            self.status.setText(f"Invalid settings: {e}")
            return

        try:
            peaks = detect_and_integrate(
                self.x,
                self.y,
                start_min=start_min,
                prominence=prominence,
                distance=distance,
                width=width,
                baseline_smooth_window=baseline_smooth_window,
                baseline_window=baseline_window,
                baseline_percentile=baseline_percentile,
                baseline_smooth2_window=baseline_smooth2_window,
                boundary_window=boundary_window,
                boundary_prominence_fraction=boundary_prominence_fraction,
            )
        except Exception as e:
            self.status.setText(f"Peak detection failed: {e}")
            return

        self.current_peaks = peaks
        self._update_plot(start_min=start_min, peaks=peaks)
        self._update_table(peaks, top_n=top_n)

        total_area = sum(p.area for p in peaks)
        self.status.setText(
            f"Points: {len(self.x)} | Peaks: {len(peaks)} | Total area: {total_area:.6g}"
        )
        self.log_message(
            f"Updated: start={start_min:.4g} min, prominence={prominence:.4g}, width={width:.4g}, distance={distance}, peaks={len(peaks)}"
        )
    
    def _figure_settings(self):
        return {
            "title": self.title_edit.text(),
            "xlabel": self.xlabel_edit.text(),
            "ylabel": self.ylabel_edit.text(),
            "font": self.font_box.currentFont().family(),
            "title_size": self.title_size_spin.value(),
            "axis_size": self.axis_size_spin.value(),
            "tick_size": self.tick_size_spin.value(),
            "line_width": self.line_width_spin.value(),
        }
        
    def _update_plot(
        self,
        *,
        start_min: float,
        peaks: List[PeakResult],
    ) -> None:
        x2, y2 = apply_start_cutoff(self.x, self.y, start_min)

        self.ax.clear()
        
        baseline = estimate_baseline(
            x2,
            y2,
            smooth_window=int(self.baseline_smooth_spin.value()),
            baseline_window=int(self.baseline_window_spin.value()),
            baseline_percentile=float(
                self.baseline_percentile_spin.value()
            ),
            baseline_smooth_window=int(
                self.baseline_smooth2_spin.value()
            ),
        )
        
        corrected = y2 - baseline
        corrected = np.maximum(corrected, 0.0)
        
        if self.corrected_toggle.isChecked():
            self.ax.plot(
                x2,
                corrected,
                linewidth=1.0,
                linestyle="-",
                label="Baseline-corrected signal",
            )

        # Main chromatogram
        self.ax.plot(
            x2,
            y2,
            linewidth=self.line_width_spin.value(),
            label="Signal",
        )
        if self.baseline_toggle.isChecked():
            self.ax.plot(
                x2,
                baseline,
                linewidth=1.5,
                linestyle="--",
                label="Estimated baseline",
            )

        # --------------------------------------------------------
        # Visualize each integrated peak
        # --------------------------------------------------------
        shaded_label_used = False
        boundary_label_used = False

        for peak in peaks:
            left_min = peak.left_min
            right_min = peak.right_min

            mask = (x2 >= left_min) & (x2 <= right_min)

            if not np.any(mask):
                continue

            xseg = x2[mask]
            yseg = y2[mask]

            if len(xseg) < 2:
                continue

            # Use the same whole-chromatogram baseline used
            # for the actual area calculation.
            baseline_seg = baseline[mask]

            # ----------------------------------------------------
            # Shade integrated area
            # ----------------------------------------------------
            if self.shade_toggle.isChecked():
                y_fill = np.maximum(yseg, baseline_seg)

                self.ax.fill_between(
                    xseg,
                    baseline_seg,
                    y_fill,
                    alpha=0.30,
                    label=(
                        "Integrated area"
                        if not shaded_label_used
                        else None
                    ),
                )
                
                shaded_label_used = True

            # ----------------------------------------------------
            # Start/end markers
            # ----------------------------------------------------
            if self.boundary_toggle.isChecked():
                left_y = float(np.interp(left_min, x2, y2))
                right_y = float(np.interp(right_min, x2, y2))

                self.ax.plot(
                    left_min,
                    left_y,
                    marker="o",
                    markersize=5,
                    linestyle="None",
                    label=(
                        "Peak start/end"
                        if not boundary_label_used
                        else None
                    ),
                )

                self.ax.plot(
                    right_min,
                    right_y,
                    marker="s",
                    markersize=5,
                    linestyle="None",
                )

                boundary_label_used = True

        # Apex markers
        if peaks:
            peak_rts = np.array(
                [p.rt_min for p in peaks],
                dtype=float,
            )
            peak_heights = np.array(
                [p.height for p in peaks],
                dtype=float,
            )

            self.ax.plot(
                peak_rts,
                peak_heights,
                "ro",
                markersize=3,
                label="Detected peaks",
            )

        # --------------------------------------------------------
        # Figure formatting
        # --------------------------------------------------------
        fig = self._figure_settings()

        self.ax.set_title(
            fig["title"],
            fontsize=fig["title_size"],
            fontname=fig["font"],
        )

        self.ax.set_xlabel(
            fig["xlabel"],
            fontsize=fig["axis_size"],
            fontname=fig["font"],
        )

        self.ax.set_ylabel(
            fig["ylabel"],
            fontsize=fig["axis_size"],
            fontname=fig["font"],
        )

        self.ax.tick_params(
            axis="both",
            labelsize=fig["tick_size"],
        )

        for tick in self.ax.get_xticklabels():
            tick.set_fontname(fig["font"])

        for tick in self.ax.get_yticklabels():
            tick.set_fontname(fig["font"])

        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="best")

        self.canvas.draw_idle()
        
    def _visual_options_changed(self, _state=None) -> None:
        if self.x is None or self.y is None:
            return

        self._update_plot(
            start_min=float(self.start_spin.value()),
            peaks=self.current_peaks,
        )

    def _update_table(self, peaks: List[PeakResult], top_n: int) -> None:
        self.table.setRowCount(0)
    
        # Always order by apex time, then show the first N peaks by time.
        display = sorted(peaks, key=lambda p: p.rt_min)[: max(1, top_n)]
    
        self.table.setRowCount(len(display))
        for row, p in enumerate(display):
            values = [
                f"{p.rt_min:.6f}",
                f"{p.height:.6g}",
                f"{p.area:.6g}",
                f"{p.area_percent:.2f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, item)
    
        self.table.resizeColumnsToContents()

    def export_peak_table(self) -> None:
        if not self.current_peaks:
            QMessageBox.information(self, "No peaks", "No peaks have been detected yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save peak table as CSV",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["peak_index", "rt_min", "height", "left_min", "right_min", "area", "area_percent"])
                for p in sorted(self.current_peaks, key=lambda x: x.rt_min):
                    writer.writerow([
                        p.peak_index,
                        p.rt_min,
                        p.height,
                        p.left_min,
                        p.right_min,
                        p.area,
                        p.area_percent,
                    ])
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return

        self.status.setText(f"Exported peak table: {path}")
        self.log_message(f"Exported peak table to {path}")


def main() -> int:
    app = QApplication(sys.argv)
    window = PeakExplorer()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
