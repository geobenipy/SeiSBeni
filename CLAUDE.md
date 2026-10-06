# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-file PyQt5 desktop app (`SeiSBeni.py`, ~560 lines) for viewing SEGY/SGY seismic files: a trace header table, a grayscale seismic section with gain control, and a Folium map of survey coordinates. `README.md` is the user-facing doc. `requirements.txt` lists the dependencies.

## Commands

- Install dependencies: `pip install numpy PyQt5 PyQtWebEngine matplotlib folium pyproj`
- Run: `python SeiSBeni.py`
- `SeiSBeni_alt.py` is the original version, kept unchanged so it still runs: `python SeiSBeni_alt.py`. Don't edit it.
- There is no test suite, linter, formatter, or build step configured, so there is no single-test command.

## Architecture

Loading a file runs through four pieces:

1. `SEGYViewerApp` (main window) collects the settings: UTM zone and hemisphere, trace skip, downsample factor, and the chosen file.
2. `SEGYLoaderThread` (a `QThread`) reads the 3200-byte text header and the 400-byte binary header (big-endian). Sample count is at bytes 21–22, sample interval at 17–18, and format code at 25–26. It derives the trace count from file size and fixed-length trace records, then runs `load_trace_chunk` over chunks in a `multiprocessing.Pool` created inside `run()` with `cpu_count() - 1` workers.
3. The loader emits a single `dict` through its `finished` signal. Keys: `filepath`, `text_header`, `n_samples`, `dt_us`, `n_traces`, `trace_headers` (list of raw 240-byte `bytes`), `data` (float32, traces × samples), `trace_skip`, `downsample_factor`, `coord_scalar` (header bytes 71–72 of the first trace).
4. `on_load_finished` creates a new `SEGYViewerWindow` for each load. The window must be kept in `self.viewer_windows`, or Qt/Python will garbage-collect it.

`SEGYViewerWindow` draws three things from that dict. The layout is nested splitters: header table | (seismic plot / map), so every panel can be resized by dragging:

- **Header table:** all 60 four-byte fields of the 240-byte trace header, with min and max across traces. `TRACE_HEADERS` keys are 1-based byte positions, while slicing is 0-based (`header[byte-1:byte+3]`).
- **Seismic section:** `imshow` in gray with 2–98 percentile clipping. Gain is changed in ×1.5 / ÷1.5 steps, and each change re-applies gain and redraws the whole figure from `segy_data['data']`.
- **Map:** source (bytes 73–80) or group coordinates are multiplied by the coordinate factor (`coord_factor_from_scalar` on the header scalar if "Header-Skalar nutzen" is checked, otherwise the GUI factor, default 0.1), converted from UTM to WGS84 with pyproj (`EPSG:326xx` for north, `327xx` for south), written to a temp HTML file with Folium, and shown in a `QWebEngineView`. Only every 50th trace is used. If the headers carry no coordinates, `load_sidecar_navigation` reads `<basename>_nav.csv` next to the SEG-Y file (columns `ffid,lat,lon`, interpolated per trace FFID); if that is missing too, the map shows "Keine Koordinaten gefunden".

## Data formats

- Format 1 is IBM float (`ibm2ieee`), format 5 is IEEE float32, format 3 is int16, format 2 is int32.
- Defaults target the example survey `86161` (outside the repo, in `C:\Users\haime\OneDrive\Desktop\86161`): UTM zone 34N, trace skip 1, downsample 1, coordinate factor 0.1.
- `D:\91698.sgy` (marine, 48-channel first record then single-trace records, IBM float, 500 samples at 4 ms) has header coordinates in WGS 84 / UTM 34N, scalar -100. Its positions match `D:\Jaunie_Kdzp_profili\91on.shp` (`X_WGS84`/`Y_WGS84`, profile 91698). It needs no sidecar.
- Its trace headers carry no coordinates. Sidecar files `<basename>_nav.csv` (`ffid,lon,lat`, WGS84) were generated from each line's observer log (FFID→shot point) and `Nav_Bath_86161.xls` (UTM 34N = EPSG:32634): `SEG-Y_DVD_59\p86161_nav.csv` for `p86161.sgy`, `SEG-Y_DVD_04\L86161_nav.csv` for `L86161`, and `86161_nav.csv` in `SEG-Y_DVD_16` for `86161.rec`. The two logs have different FFID numbering, so the sidecars are not interchangeable.

## Known issues

- **Coordinate factor.** Header scalar -100 means ×0.01 (e.g. `91698.sgy`); scalar -10 means ×0.1 (the old hardcoded value, still the GUI default). The UTM zone must be set per survey.
- **Unknown format codes** are read as 4-byte IEEE float, which is only correct for format 5.

## Conventions

- Comments, UI labels, and status messages are in German. Match that for new user-facing text; the README is in English.
- The whole trace set is held in RAM. Trace skip and downsampling are the only ways to reduce memory use.
