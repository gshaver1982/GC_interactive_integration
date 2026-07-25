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
from scipy.signal import find_peaks

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


def detect_and_integrate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    start_min: float,
    prominence: float,
    distance: int,
    width: float,
) -> List[PeakResult]:
    x2, y2 = apply_start_cutoff(x, y, start_min)
    if len(x2) < 5:
        return []

    kwargs = {
        "prominence": float(prominence),
        "distance": max(1, int(distance)),
    }
    if width and width > 0:
        kwargs["width"] = float(width)

    peaks, props = find_peaks(y2, **kwargs)
    if len(peaks) == 0:
        return []

    temp_rows = []
    raw_areas = []

    for i, p in enumerate(peaks):
        left = int(props["left_bases"][i])
        right = int(props["right_bases"][i])

        if right <= left + 1:
            continue

        xseg = x2[left : right + 1]
        yseg = y2[left : right + 1]
        if len(xseg) < 3:
            continue

        baseline = np.interp(xseg, [xseg[0], xseg[-1]], [yseg[0], yseg[-1]])
        corrected = yseg - baseline
        area = float(np.trapz(corrected, xseg))

        raw_areas.append(area)
        temp_rows.append(
            {
                "peak_index": int(p),
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
        area_percent = 0.0 if total_area == 0 else 100.0 * row["area"] / total_area
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
        
        file_btn_row.addWidget(self.btn_choose)
        file_btn_row.addWidget(self.btn_export)
        file_btn_row.addWidget(self.btn_save_png)
        
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
        
    def _update_plot(self, *, start_min: float, peaks: List[PeakResult]) -> None:
        x2, y2 = apply_start_cutoff(self.x, self.y, start_min)

        self.ax.clear()
        self.ax.plot(x2, y2, linewidth=1.0, label="Signal")

        if peaks:
            peak_rts = np.array([p.rt_min for p in peaks], dtype=float)
            peak_heights = np.array([p.height for p in peaks], dtype=float)
            self.ax.plot(peak_rts, peak_heights, "ro", markersize=2, label="Detected peaks")

        self.ax.set_xlabel(self.xlabel_edit.text())
        self.ax.set_ylabel(self.ylabel_edit.text())
        self.ax.set_title(self.title_edit.text())
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="best")
        self.canvas.draw_idle()
        fig = self._figure_settings()
        
        self.ax.clear()
        self.ax.plot(x2, y2, linewidth=fig["line_width"], label="Signal")
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
        
        self.ax.tick_params(labelsize=fig["tick_size"])
        
        for tick in self.ax.get_xticklabels():
            tick.set_fontname(fig["font"])
        
        for tick in self.ax.get_yticklabels():
            tick.set_fontname(fig["font"])

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
