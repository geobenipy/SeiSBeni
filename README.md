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
python SeisBeni.py
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
* Coordinate scaling may need adjustment depending on the survey (`scale = 0.1` in the code).
* Map view uses Source/Group coordinates if available in the headers.
