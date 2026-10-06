# SEGY/SGY Viewer (Python + PyQt5)

A fast and lightweight GUI application for viewing SEGY/SGY seismic data.
The program supports **multi-core loading**, **trace skipping**, **sample downsampling**, **header inspection**, **amplitude gain control**, and an integrated **map view** based on UTM coordinates.

## Features

* **High-performance SEGY/SGY loading** using multiprocessing
* **Trace Skip** (load every Nth trace)
* **Downsampling** to reduce memory usage
* **Interactive PyQt5 interface**
* **Seismic section viewer** with real-time amplitude gain control
* **Full 240-byte trace header table** (min/max for each field)
* **Map view** (Folium) with automatic UTM → WGS84 conversion
* Supports multiple viewer windows simultaneously

## Installation

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

## Running the Viewer

```bash
python SeiSBeni.py
```

## Requirements

Python ≥ 3.8
The main dependencies are:

```
numpy
PyQt5
PyQtWebEngine
matplotlib
folium
pyproj
```

(Other imports are standard-library modules.)

## Notes

* For large datasets, using a **trace skip > 1** can significantly speed up loading.
* Coordinate scaling: with **Header-Skalar nutzen** (default on) the factor comes from the SEG-Y scalar in the trace headers (bytes 71–72). Otherwise the **Koordinaten-Faktor** is used (default 0.1). Header coordinates × factor = metres.
* The viewer panels are resizable: drag the dividers between the header table, the seismic section, and the map.
* `SeiSBeni_alt.py` is the original version, kept so it still runs.
* Map view uses Source/Group coordinates if available in the headers. If the headers have none, it uses a sidecar file `<name>_nav.csv` next to the SEG-Y file, with columns `ffid,lon,lat` (WGS84).
* IBM float (SEG-Y format 1), IEEE float (5), int16 (3), int32 (2) and int8 (8) are supported.
* Default UTM zone is 34 N.
