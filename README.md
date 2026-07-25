# GC-FID CSV Peak Explorer

A Qt-based desktop application for exploring chromatogram CSV/TXT exports.

## Features

- Open a chromatogram file with a file picker
- Parse two numeric columns: retention time and signal
- Plot the chromatogram in a Qt window
- Adjust peak detection settings:
  - start time cutoff
  - prominence
  - width
  - distance
- Detect and integrate peaks
- Show a sample peak table with:
  - apex retention time
  - height
  - area
  - area percent
- Export the full peak table to CSV
- Save a clean PNG of the chromatogram

## Requirements

- Python 3.10 or newer
- PyQt5
- NumPy
- SciPy
- Matplotlib

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application from a terminal:

```bash
python gc_fid_qt_peak_gui.py
```

Then:

1. Click **Choose CSV/TXT...**
2. Select a chromatogram CSV, TXT, or TSV file
3. Adjust the peak settings as needed
4. Review the detected peaks in the table
5. Export the peak table or save a PNG if needed

## Expected input format

The file should contain two numeric columns:

- retention time in minutes
- detector signal

A header row is acceptable. The parser ignores non-numeric rows and accepts comma-, tab-, semicolon-, or whitespace-delimited input.

Example:

```text
RT (min),Signal
0.001659383,-0.000623486
0.00332605,-0.000936729
0.004992717,-0.000117258
```

## Peak calculation

Peaks are detected with `scipy.signal.find_peaks()` and integrated with a straight-line baseline between the left and right peak bases.

The peak table includes:

- `peak_index`
- `rt_min`
- `height`
- `left_min`
- `right_min`
- `area`
- `area_percent`

## Notes

- The app is intended for quick chromatogram review and routine peak exploration.
- Peak detection settings may need to be adjusted from file to file.
- If the chromatogram includes a strong solvent peak, use the start time cutoff to exclude the early region.

## License

Add your preferred license before publishing the repository.
