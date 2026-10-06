import sys
import os
import struct
import numpy as np
from multiprocessing import Pool, cpu_count
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QProgressBar, QTableWidget, QTableWidgetItem, 
                             QSplitter, QScrollArea, QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import folium
import tempfile
import warnings
warnings.filterwarnings('ignore') 

# -------------------------------------------------------
# SEGY Header Definitionen
# -------------------------------------------------------
TRACE_HEADERS = {
    1: "Trace sequence number (line)",
    5: "Trace sequence number (file)",
    9: "Original field record number (FFID)",
    13: "Trace number (field record)",
    17: "Source point number",
    21: "CDP ensemble number",
    25: "Trace number (CDP)",
    29: "Trace ID code",
    71: "Scalar for coordinates",
    73: "Source X",
    77: "Source Y",
    81: "Group X (CDP X)",
    85: "Group Y (CDP Y)",
    89: "Coordinate units",
    115: "Number of samples",
    117: "Sample interval (µs)",
    181: "X coordinate",
    185: "Y coordinate",
    189: "Inline number",
    193: "Crossline number",
    113: "MuteTimeEND"
}

# -------------------------------------------------------
# Helper function für Multiprocessing
# -------------------------------------------------------
def load_trace_chunk(args):
    """Lädt einen Chunk von Traces (für Multiprocessing)."""
    filepath, start_idx, end_idx, trace_size, n_samples, bytes_per_sample, format_code, downsample_factor = args
    
    try:
        with open(filepath, 'rb') as f:
            n_samples_ds = n_samples // downsample_factor
            chunk_data = []
            chunk_headers = []
            
            for i in range(start_idx, end_idx):
                f.seek(3600 + i * trace_size)
                header = f.read(240)
                if len(header) < 240:
                    break
                chunk_headers.append(header)
                
                # Trace Daten lesen
                if format_code == 1 or format_code == 5:
                    trace_data = np.frombuffer(f.read(n_samples * 4), dtype='>f4')
                elif format_code == 3:
                    trace_data = np.frombuffer(f.read(n_samples * 2), dtype='>i2').astype(np.float32)
                else:
                    trace_data = np.frombuffer(f.read(n_samples * 4), dtype='>f4')
                
                # Downsampling
                if downsample_factor > 1:
                    trace_data = trace_data[::downsample_factor]
                
                chunk_data.append(trace_data[:n_samples_ds])
            
            return chunk_headers, np.array(chunk_data, dtype=np.float32)
    except:
        return [], np.array([])

# -------------------------------------------------------
# Worker Thread für SEGY Laden
# -------------------------------------------------------
class SEGYLoaderThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, filepath, trace_skip=1, downsample_factor=1):
        super().__init__()
        self.filepath = filepath
        self.trace_skip = trace_skip
        self.downsample_factor = downsample_factor
        
    def run(self):
        try:
            self.progress.emit(10, "Öffne Datei...")
            
            with open(self.filepath, 'rb') as f:
                # Text Header
                self.progress.emit(20, "Lese Text Header...")
                text_header = f.read(3200)
                
                # Binary Header
                self.progress.emit(30, "Lese Binary Header...")
                binary_header = f.read(400)
                n_samples = struct.unpack('>H', binary_header[20:22])[0]
                dt_us = struct.unpack('>H', binary_header[16:18])[0]
                format_code = struct.unpack('>H', binary_header[24:26])[0]
                
                bytes_per_sample = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(format_code, 4)
                
                self.progress.emit(40, "Zähle Traces...")
                
                # Traces zählen
                trace_size = 240 + (n_samples * bytes_per_sample)
                file_size = os.path.getsize(self.filepath)
                data_size = file_size - 3600
                n_traces = data_size // trace_size
                
                self.progress.emit(50, f"Lade {n_traces} Traces (Skip={self.trace_skip}, Multicore)...")
                
                # Trace-Indizes mit Skip
                trace_indices = list(range(0, n_traces, self.trace_skip))
                n_traces_to_load = len(trace_indices)
                
                # MULTIPROCESSING: Traces in Chunks aufteilen
                n_cores = max(1, cpu_count() - 1)  # Ein Core frei lassen
                chunk_size = max(1, n_traces_to_load // n_cores)
                
                chunks = []
                for i in range(0, n_traces_to_load, chunk_size):
                    chunk_indices = trace_indices[i:i+chunk_size]
                    if chunk_indices:
                        chunks.append((
                            self.filepath,
                            chunk_indices[0],
                            chunk_indices[-1] + 1,
                            trace_size,
                            n_samples,
                            bytes_per_sample,
                            format_code,
                            self.downsample_factor
                        ))
                
                self.progress.emit(60, f"Lade parallel auf {n_cores} Cores...")
                
                # Parallel laden
                with Pool(processes=n_cores) as pool:
                    results = pool.map(load_trace_chunk, chunks)
                
                self.progress.emit(80, "Kombiniere Daten...")
                
                # Ergebnisse kombinieren
                trace_headers = []
                data_list = []
                for headers, data_chunk in results:
                    if len(headers) > 0:
                        trace_headers.extend(headers)
                        data_list.append(data_chunk)
                
                if not data_list:
                    raise ValueError("Keine Daten geladen")
                
                data = np.vstack(data_list)
                n_samples_ds = data.shape[1]
                
                self.progress.emit(100, "Fertig!")
                
                result = {
                    'filepath': self.filepath,
                    'text_header': text_header,
                    'n_samples': n_samples_ds,
                    'dt_us': dt_us * self.downsample_factor,
                    'n_traces': len(trace_headers),
                    'trace_headers': trace_headers,
                    'data': data,
                    'trace_skip': self.trace_skip,
                    'downsample_factor': self.downsample_factor
                }
                
                self.finished.emit(result)
                
        except Exception as e:
            self.error.emit(str(e))

# -------------------------------------------------------
# Viewer Fenster
# -------------------------------------------------------
class SEGYViewerWindow(QMainWindow):
    def __init__(self, segy_data, utm_zone=33, utm_hemisphere='N'):
        super().__init__()
        self.segy_data = segy_data
        self.utm_zone = utm_zone
        self.utm_hemisphere = utm_hemisphere
        self.amp_gain = 1.0  # Amplituden-Verstärkung
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(f"SEGY Viewer - {os.path.basename(self.segy_data['filepath'])}")
        self.setGeometry(100, 100, 1600, 900)
        
        # Main Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Splitter für Header und Seismic
        splitter = QSplitter(Qt.Horizontal)
        
        # === LINKE SEITE: Header Tabelle ===
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        
        header_label = QLabel("<b>Trace Headers</b>")
        header_layout.addWidget(header_label)
        
        self.header_table = QTableWidget()
        self.populate_header_table()
        header_layout.addWidget(self.header_table)
        
        splitter.addWidget(header_widget)
        
        # === RECHTE SEITE: Seismic Display ===
        seismic_widget = QWidget()
        seismic_layout = QVBoxLayout(seismic_widget)
        
        # Amplitude Controls
        amp_control_layout = QHBoxLayout()
        amp_label = QLabel('<b>Amplitude:</b>')
        amp_control_layout.addWidget(amp_label)
        
        minus_btn = QPushButton('-')
        minus_btn.setMaximumWidth(40)
        minus_btn.clicked.connect(self.decrease_amplitude)
        amp_control_layout.addWidget(minus_btn)
        
        self.amp_label = QLabel(f'Gain: {self.amp_gain:.1f}x')
        amp_control_layout.addWidget(self.amp_label)
        
        plus_btn = QPushButton('+')
        plus_btn.setMaximumWidth(40)
        plus_btn.clicked.connect(self.increase_amplitude)
        amp_control_layout.addWidget(plus_btn)
        
        reset_btn = QPushButton('Reset')
        reset_btn.setMaximumWidth(60)
        reset_btn.clicked.connect(self.reset_amplitude)
        amp_control_layout.addWidget(reset_btn)
        
        # Info Label
        skip_info = self.segy_data.get('trace_skip', 1)
        ds_info = self.segy_data.get('downsample_factor', 1)
        info_text = f"Trace Skip: {skip_info} | Downsample: {ds_info}x"
        info_label = QLabel(info_text)
        amp_control_layout.addWidget(info_label)
        
        amp_control_layout.addStretch()
        seismic_layout.addLayout(amp_control_layout)
        
        # Matplotlib Canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        seismic_layout.addWidget(self.toolbar)
        seismic_layout.addWidget(self.canvas)
        
        # Karte (optional)
        self.map_view = QWebEngineView()
        self.create_map()
        seismic_layout.addWidget(self.map_view)
        
        splitter.addWidget(seismic_widget)
        splitter.setSizes([400, 1200])
        
        main_layout.addWidget(splitter)
        
        # Plot Seismic
        self.plot_seismic()
        
    def increase_amplitude(self):
        """Erhöht die Amplitudenverstärkung."""
        self.amp_gain *= 1.5
        self.amp_label.setText(f'Gain: {self.amp_gain:.1f}x')
        self.plot_seismic()
    
    def decrease_amplitude(self):
        """Verringert die Amplitudenverstärkung."""
        self.amp_gain /= 1.5
        self.amp_label.setText(f'Gain: {self.amp_gain:.1f}x')
        self.plot_seismic()
    
    def reset_amplitude(self):
        """Setzt Amplitudenverstärkung zurück."""
        self.amp_gain = 1.0
        self.amp_label.setText(f'Gain: {self.amp_gain:.1f}x')
        self.plot_seismic()
    
    def populate_header_table(self):
        """Füllt die Header-Tabelle mit ALLEN Werten (alle 240 Bytes)."""
        headers = self.segy_data['trace_headers']
        
        self.header_table.setColumnCount(4)
        self.header_table.setHorizontalHeaderLabels(['Byte', 'Header', 'Min', 'Max'])
        
        # Alle 4-Byte Integer im 240-Byte Header (60 Einträge)
        n_entries = 60
        self.header_table.setRowCount(n_entries)
        
        for idx in range(n_entries):
            byte_pos = idx * 4 + 1  # 1-basiert: 1, 5, 9, 13, ...
            
            # Header-Name wenn bekannt, sonst "Unknown"
            header_name = TRACE_HEADERS.get(byte_pos, f"Byte {byte_pos}-{byte_pos+3}")
            
            # Werte aus allen Traces sammeln
            values = []
            for header in headers:
                if byte_pos + 3 <= len(header):
                    val = struct.unpack('>i', header[byte_pos-1:byte_pos+3])[0]
                    values.append(val)
            
            if values:
                self.header_table.setItem(idx, 0, QTableWidgetItem(f"{byte_pos}-{byte_pos+3}"))
                self.header_table.setItem(idx, 1, QTableWidgetItem(header_name))
                self.header_table.setItem(idx, 2, QTableWidgetItem(str(min(values))))
                self.header_table.setItem(idx, 3, QTableWidgetItem(str(max(values))))
        
        self.header_table.resizeColumnsToContents()
    
    def plot_seismic(self):
        """Plottet die seismischen Daten mit Amplitudenverstärkung."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Daten mit Gain multiplizieren
        data = self.segy_data['data'].T * self.amp_gain
        n_traces = self.segy_data['n_traces']
        n_samples = self.segy_data['n_samples']
        dt = self.segy_data['dt_us'] / 1000  # ms
        
        # Amplitude clipping für bessere Darstellung
        vmin, vmax = np.percentile(data[np.isfinite(data)], [2, 98])
        
        # Fix für matplotlib cursor bug
        data = np.nan_to_num(data, nan=0.0, posinf=vmax, neginf=vmin)
        
        im = ax.imshow(data, aspect='auto', cmap='gray', 
                      extent=[0, n_traces, n_samples * dt, 0],
                      vmin=vmin, vmax=vmax, interpolation='bilinear')
        
        # Cursor deaktivieren um Fehler zu vermeiden
        ax.format_coord = lambda x, y: f'Trace: {int(x)}, Time: {y:.1f} ms'
        
        ax.set_xlabel('Trace Number')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Seismic Section - {os.path.basename(self.segy_data["filepath"])}')
        
        self.figure.colorbar(im, ax=ax, label='Amplitude')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def create_map(self):
        """Erstellt eine Karte mit den Koordinaten."""
        try:
            from pyproj import Transformer
            
            # Koordinaten extrahieren
            coords = []
            headers = self.segy_data['trace_headers']
            
            for i, header in enumerate(headers[::50]):  # Jeden 50. Trace
                scale = 0.1   # change for every survey!
                
                x = struct.unpack('>i', header[72:76])[0] * scale
                y = struct.unpack('>i', header[76:80])[0] * scale
                
                if x != 0 and y != 0:
                    # UTM zu Lat/Lon
                    utm_crs = f"EPSG:326{self.utm_zone}" if self.utm_hemisphere == 'N' else f"EPSG:327{self.utm_zone}"
                    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(x, y)
                    coords.append((lat, lon))
            
            if coords:
                center_lat = np.mean([c[0] for c in coords])
                center_lon = np.mean([c[1] for c in coords])
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                
                # Linie zeichnen
                folium.PolyLine(coords, color='red', weight=3, opacity=0.8).add_to(m)
                
                # Temporäre HTML speichern
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
                    m.save(f.name)
                    self.map_view.setUrl(QUrl.fromLocalFile(f.name))
            else:
                self.map_view.setHtml("<h3>Keine Koordinaten gefunden</h3>")
                
        except Exception as e:
            self.map_view.setHtml(f"<h3>Karte nicht verfügbar: {e}</h3>")

# -------------------------------------------------------
# Main GUI
# -------------------------------------------------------
class SEGYViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer_windows = []
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('SEGY Viewer - Main')
        self.setGeometry(300, 300, 600, 300)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel('<h1>SEGY/SGY Viewer</h1>')
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # UTM Settings
        utm_layout = QHBoxLayout()
        utm_layout.addWidget(QLabel('UTM Zone:'))
        self.utm_zone_spin = QSpinBox()
        self.utm_zone_spin.setRange(1, 60)
        self.utm_zone_spin.setValue(33)
        utm_layout.addWidget(self.utm_zone_spin)
        
        self.utm_north_check = QCheckBox('Northern Hemisphere')
        self.utm_north_check.setChecked(True)
        utm_layout.addWidget(self.utm_north_check)
        utm_layout.addStretch()
        layout.addLayout(utm_layout)
        
        # Trace Skip Settings
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel('Trace Skip (jeder N-te):'))
        self.trace_skip_spin = QSpinBox()
        self.trace_skip_spin.setRange(1, 100)
        self.trace_skip_spin.setValue(1)
        self.trace_skip_spin.setToolTip('Nur jeden N-ten Trace laden (schneller)')
        skip_layout.addWidget(self.trace_skip_spin)
        skip_layout.addStretch()
        layout.addLayout(skip_layout)
        
        # Downsample Settings
        ds_layout = QHBoxLayout()
        ds_layout.addWidget(QLabel('Downsample Factor:'))
        self.downsample_combo = QSpinBox()
        self.downsample_combo.setRange(1, 8)
        self.downsample_combo.setValue(1)
        self.downsample_combo.setToolTip('1=Original, 2=Halbe Samples, 4=Viertel Samples')
        ds_layout.addWidget(self.downsample_combo)
        ds_layout.addStretch()
        layout.addLayout(ds_layout)
        
        # File Selection
        self.file_label = QLabel('Keine Datei ausgewählt')
        layout.addWidget(self.file_label)
        
        select_btn = QPushButton('SEGY/SGY Datei auswählen')
        select_btn.clicked.connect(self.select_file)
        layout.addWidget(select_btn)
        
        # Load Button
        self.load_btn = QPushButton('Datei laden und anzeigen')
        self.load_btn.clicked.connect(self.load_file)
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel('')
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        self.current_file = None
        
    def select_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'SEGY/SGY Datei öffnen', '', 
            'SEGY Files (*.sgy *.segy);;All Files (*)'
        )
        
        if filepath:
            self.current_file = filepath
            self.file_label.setText(f'Ausgewählt: {os.path.basename(filepath)}')
            self.load_btn.setEnabled(True)
    
    def load_file(self):
        if not self.current_file:
            return
        
        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Settings holen
        trace_skip = self.trace_skip_spin.value()
        downsample = self.downsample_combo.value()
        
        # Start Worker Thread
        self.loader_thread = SEGYLoaderThread(self.current_file, trace_skip, downsample)
        self.loader_thread.progress.connect(self.update_progress)
        self.loader_thread.finished.connect(self.on_load_finished)
        self.loader_thread.error.connect(self.on_load_error)
        self.loader_thread.start()
    
    def update_progress(self, value, status):
        self.progress_bar.setValue(value)
        self.status_label.setText(status)
    
    def on_load_finished(self, segy_data):
        self.progress_bar.setVisible(False)
        self.status_label.setText('Erstelle Viewer...')
        self.load_btn.setEnabled(True)
        
        # Neues Viewer-Fenster öffnen
        utm_zone = self.utm_zone_spin.value()
        utm_hemisphere = 'N' if self.utm_north_check.isChecked() else 'S'
        
        viewer = SEGYViewerWindow(segy_data, utm_zone, utm_hemisphere)
        viewer.show()
        self.viewer_windows.append(viewer)
        
        self.status_label.setText(f'Viewer geöffnet für {os.path.basename(segy_data["filepath"])}')
    
    def on_load_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f'Fehler: {error_msg}')
        self.load_btn.setEnabled(True)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    window = SEGYViewerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()