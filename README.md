# Nanoindentation analysis

This repository contains NI_Analysis_NIvMPDpaper.py, a Python workflow used to process Optics11 Chiaro nanoindentation measurements for the NI–MPD cardiac mechanics study. The workflow processes exported force–indentation curves, performs Hertz model fitting and fit-quality filtering, extracts Young’s modulus (E), and generates diagnostic plots and summary outputs.

## Code availability

The analysis code is available in this repository. Download the repository
archive or clone it from its hosting page to access the script. When citing
the code in a paper or data-availability statement, provide the repository's
public URL and, where available, its archived DOI or release.

## Requirements

- Python 3.9.7 or newer is recommended. The script does not specify a tested
  Python version or dependency versions.
- Packages: `numpy`, `pandas`, `scipy`, `matplotlib` and 'openpyxl'
- `openpyxl` is needed by pandas to write the Excel map workbook.

Create an environment and install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy matplotlib openpyxl
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Input data

The script expects raw Optics11 Chiaro point exports as tab-delimited `.txt`
files. Each file should contain a metadata section followed by tabular
measurement data, including time, load/force, and indentation columns. The
metadata must provide scan coordinates and XYZ positions; the `P [max]` field
is used to identify points without an indentation.

Place the script and one directory per slice in the analysis root:

```text
analysis-root/
├── NI_Analysis_NIvMPDpaper.py
├── slice-name/
│   ├── point-001.txt
│   ├── point-002.txt
│   └── ... (subdirectories are supported)
└── another-slice/
    └── ...
```

The script treats every non-hidden directory directly under its working
directory as a slice, regardless of its name, and searches recursively within
each slice for `.txt` files. Keep unrelated directories outside the analysis
root.

## Reproducing the analysis

1. Download or clone this repository and arrange the raw data as shown above.
2. Install the Python dependencies.
3. From the analysis root, run:

   ```bash
   python NI_Analysis_NIvMPDpaper.py
   ```

The script processes all discovered slices and their `.txt` point files in
parallel. For each slice, it creates a date-stamped `output_YYYYMMDD`
directory containing per-point `result.csv` files and diagnostic plots. It
also combines point results into `all samples.csv` and produces
`correlations.csv`, `maps.xlsx`, and spatial maps in TIFF, PNG, PDF, and JPEG
formats. Exact results depend on the input exports and analysis parameters
defined in the script.
