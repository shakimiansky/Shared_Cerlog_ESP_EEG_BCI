# FINAL PRODUCTION PYTHON SCRIPT - High-Performance 4-Thread Architecture for 250Hz
# Corrected data ingestion logic and added robust error reporting.

import asyncio
import struct
from collections import deque
import numpy as np
from bleak import BleakClient, BleakScanner
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State
import threading
import time
import queue

# --- BLE Configuration (Unchanged) ---
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
HANDSHAKE_CHARACTERISTIC_UUID = "c8a91599-7144-4903-b3c9-f1e1e47e38ab"
DEVICE_NAME = "Cerelog-X8-BLE"

# --- Performance & Data Configuration ---
SAMPLING_RATE_HZ = 250.0
FFT_BUFFER_SIZE = 15000  # 60 seconds of data at 250Hz
TIME_SERIES_DEFAULT_SECONDS = 2
TIME_SERIES_POINTS = int(TIME_SERIES_DEFAULT_SECONDS * SAMPLING_RATE_HZ)

# --- UI Refresh Rate Configuration (OPTIMIZED) ---
TIME_SERIES_UPDATE_INTERVAL_MS = 500
FFT_UPDATE_INTERVAL_MS = 2000

# --- Device-Specific Configuration (Unchanged) ---
FFT_MAX_HZ = 100
ADS1299_NUM_CHANNELS = 8
ADS1299_NUM_STATUS_BYTES = 3
ADS1299_BYTES_PER_CHANNEL = 3
PACKET_TOTAL_SIZE = 37

# --- Data Buffers and State ---
data_queue = queue.Queue()
fft_buffers = [deque(maxlen=FFT_BUFFER_SIZE) for _ in range(ADS1299_NUM_CHANNELS)]
time_series_buffers = [deque(maxlen=TIME_SERIES_POINTS) for _ in range(ADS1299_NUM_CHANNELS)]
buffer_lock = threading.Lock()

fft_results = {ch: {} for ch in range(ADS1299_NUM_CHANNELS)}
fft_results_lock = threading.Lock()

last_packet_count = -1
packets_dropped = 0
total_packets_received = 0

# --- Brainwave Frequency Bands ---
BRAINWAVE_BANDS = {
    "Delta": ([0.5, 4], "hsl(225, 39%, 30%)", "(Deep Sleep)"),
    "Theta": ([4, 8], "hsl(225, 39%, 50%)", "(Drowsy)"),
    "Alpha": ([8, 13], "hsl(205, 56%, 70%)", "(Relaxed)"),
    "Beta": ([13, 30], "hsl(190, 56%, 60%)", "(Active)"),
    "Gamma": ([30, FFT_MAX_HZ], "hsl(175, 56%, 50%)", "(Insight)"),
}

# --- Helper Functions (Unchanged) ---
VREF = 4.5; GAIN = 24.0; ADC_RESOLUTION = 2**23 - 1
MICROVOLTS_SCALE_FACTOR = (VREF / (GAIN * ADC_RESOLUTION)) * 1_000_000
def convert_to_microvolts(raw_val): return raw_val * MICROVOLTS_SCALE_FACTOR

# --- Thread 1: BLE Data Reception (Main Thread with Asyncio) ---
async def ble_data_stream_loop():
    """Main BLE connection and data streaming loop."""
    print(f"Scanning for '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)
    if not device:
        print(f"Device '{DEVICE_NAME}' not found. Please check if it's on and in range.")
        return

    print(f"Connecting to {device.name} at {device.address}...")
    async with BleakClient(device) as client:
        print("Connected. Sending handshake...")
        await asyncio.sleep(1.0)
        current_unix_time = int(time.time())
        payload = struct.pack('>BI', 0x02, current_unix_time) + bytes([0x00, 0x00])
        checksum = sum(payload) & 0xFF
        handshake_packet = struct.pack('>BB', 0xAA, 0xBB) + payload + struct.pack('>B', checksum) + struct.pack('>BB', 0xCC, 0xDD)
        await client.write_gatt_char(HANDSHAKE_CHARACTERISTIC_UUID, handshake_packet, response=False)
        print("Handshake sent.")

        def notification_handler(sender, data: bytearray):
            data_queue.put(bytes(data))

        await client.start_notify(DATA_CHARACTERISTIC_UUID, notification_handler)
        print("Streaming data (250Hz)... Dash UI is running.")
        
        while client.is_connected:
            # NEW: Add a heartbeat to confirm the loop is running and connected
            if total_packets_received > 0 and total_packets_received % (SAMPLING_RATE_HZ * 10) == 0:
                 print(f"Heartbeat: Still connected. {total_packets_received} packets received.")
                 # This sleep prevents the message from spamming the console
                 await asyncio.sleep(1.1) 
            else:
                 await asyncio.sleep(1.0)

    print("Disconnected.")


# --- Thread 2: Data Ingestion Worker (CORRECTED) ---
def data_ingestion_worker():
    """Pulls raw data from the queue, parses it, and populates shared buffers."""
    print("Data ingestion worker started.")
    global last_packet_count, packets_dropped, total_packets_received

    while True:
        try:
            packet = data_queue.get(timeout=1)
            try:
                # *** FIX: Corrected b'\xdc\xba' and added explicit logging for discarded packets ***
                if len(packet) != PACKET_TOTAL_SIZE or packet[:2] != b'\xab\xcd' or packet[35:] != b'\xdc\xba':
                    print(f"Discarding invalid packet: len={len(packet)}, start={packet[:2].hex()}, end={packet[35:].hex()}")
                    continue

                _, _, msg_len, packet_count = struct.unpack('>BBBI', packet[:7])

                if last_packet_count != -1 and packet_count > last_packet_count + 1:
                    dropped_now = packet_count - (last_packet_count + 1)
                    packets_dropped += dropped_now
                    print(f"Warning: Dropped {dropped_now} packets. Total dropped: {packets_dropped}")
                last_packet_count = packet_count
                total_packets_received += 1

                ads_data = packet[7:34]
                current_time_s = packet_count / SAMPLING_RATE_HZ
                with buffer_lock:
                    for ch in range(ADS1299_NUM_CHANNELS):
                        idx = ADS1299_NUM_STATUS_BYTES + ch * ADS1299_BYTES_PER_CHANNEL
                        raw_val = int.from_bytes(ads_data[idx:idx + 3], byteorder='big', signed=True)
                        microvolts = convert_to_microvolts(raw_val)
                        fft_buffers[ch].append(microvolts)
                        time_series_buffers[ch].append((current_time_s, microvolts))
            except Exception as e:
                print(f"Error parsing packet: {e}")
            finally:
                # *** FIX: Ensure task_done is always called for every item from the queue ***
                data_queue.task_done()
        except queue.Empty:
            continue

# --- Thread 3: FFT Calculator Worker ---
def fft_calculator_worker():
    """Performs heavy FFT calculations in the background at a slow, regular pace."""
    print("FFT calculator worker started.")
    band_power_history = {f"ch{ch}_{band}": deque(maxlen=100) for ch in range(ADS1299_NUM_CHANNELS) for band in BRAINWAVE_BANDS}

    while True:
        time.sleep(FFT_UPDATE_INTERVAL_MS / 1000.0)

        for ch_idx in range(ADS1299_NUM_CHANNELS):
            with buffer_lock:
                data_snapshot = list(fft_buffers[ch_idx])

            N = len(data_snapshot)
            if N < SAMPLING_RATE_HZ:
                continue

            y_detrended = np.array(data_snapshot) - np.mean(data_snapshot)
            yf = np.fft.fft(y_detrended)
            amplitude = 2.0 / N * np.abs(yf[0:N//2])
            xf = np.fft.fftfreq(N, 1.0 / SAMPLING_RATE_HZ)[:N//2]

            channel_results = {}
            for band, (freq_range, color, desc) in BRAINWAVE_BANDS.items():
                band_mask = (xf >= freq_range[0]) & (xf < freq_range[1])
                power = np.mean(amplitude[band_mask]) if np.any(band_mask) else 0

                history_key = f"ch{ch_idx}_{band}"
                band_power_history[history_key].append(power)
                max_power = max(band_power_history[history_key])
                if max_power < 0.1: max_power = 1.0

                channel_results[band] = {'power': power, 'max_power': max_power}
            
            with fft_results_lock:
                fft_results[ch_idx] = channel_results

# --- Thread 4: Dash App and UI Callbacks ---
app = Dash(__name__)
app.title = "Cerelog EEG Data Logger (BLE 250Hz - Optimized)"

def create_initial_figure(title="Waiting for data...", y_axis_title="Amplitude (μV)"):
    fig = go.Figure()
    fig.update_layout(title=title, yaxis_title=y_axis_title, margin=dict(l=60, r=20, t=50, b=50))
    return fig

app.layout = html.Div(style={'backgroundColor': '#f0f2f5', 'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("Cerelog: 8-Channel EEG Data Log (250Hz BLE)", style={'textAlign': 'center', 'color': '#333333', 'marginBottom': '30px'}),
    html.Div([
        html.Div(style={'width': '100%', 'marginBottom': '20px'}, children=[
            html.H2(f"Channel {i+1}", style={'fontSize': '1.5em', 'color': '#333333', 'textAlign': 'center'}),
            html.Div(style={'display': 'flex', 'flexDirection': 'row', 'width': '100%', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.1)', 'borderRadius': '8px', 'overflow': 'hidden', 'backgroundColor': '#ffffff'}, children=[
                html.Div(style={'width': '50%', 'padding': '10px'}, children=[dcc.Graph(id=f'channel-graph-{i+1}', figure=create_initial_figure())]),
                html.Div(id=f'fft-div-{i+1}', style={'width': '50%', 'padding': '10px'})])])
        for i in range(ADS1299_NUM_CHANNELS)], style={'maxWidth': '1600px', 'margin': '0 auto'}),
    dcc.Interval(id='fast-interval-time-series', interval=TIME_SERIES_UPDATE_INTERVAL_MS, n_intervals=0),
    dcc.Interval(id='slow-interval-fft', interval=FFT_UPDATE_INTERVAL_MS, n_intervals=0),
])

def generate_time_series_callback(ch_idx):
    @app.callback(Output(f'channel-graph-{ch_idx+1}', 'figure'), Input('fast-interval-time-series', 'n_intervals'), State(f'channel-graph-{ch_idx+1}', 'relayoutData'))
    def update_time_series(n, relayout_data):
        with buffer_lock:
            channel_data = list(time_series_buffers[ch_idx])
        if len(channel_data) < 2:
            raise PreventUpdate

        x_data, y_data_raw = zip(*channel_data)
        y_data_centered = np.array(y_data_raw) - np.mean(y_data_raw)

        fig = go.Figure(go.Scatter(x=x_data, y=y_data_centered, mode='lines', line=dict(shape='linear')))
        fig.update_layout(title='Time Domain', xaxis_title='Time (s)', yaxis_title='Amplitude (μV)', margin=dict(l=60, r=20, t=50, b=50), uirevision=f'time-series-{ch_idx}')
        if not (relayout_data and 'xaxis.autorange' in relayout_data and not relayout_data['xaxis.autorange']):
            fig.update_layout(xaxis_range=[min(x_data), max(x_data)])
        return fig
    return update_time_series

def generate_fft_callback(ch_idx):
    @app.callback(Output(f'fft-div-{ch_idx+1}', 'children'), Input('slow-interval-fft', 'n_intervals'))
    def update_fft_display(n):
        with fft_results_lock:
            channel_results = fft_results[ch_idx].copy()

        if not channel_results:
            return html.Div("Calculating FFT (gathering data)...", style={'textAlign': 'center', 'paddingTop': '50px'})

        gauges = [html.H4("Brainwave Band Power", style={'width': '100%', 'textAlign': 'center', 'marginBottom': 0})]
        gauge_row = []
        for band, (freq_range, color, desc) in BRAINWAVE_BANDS.items():
            result = channel_results.get(band)
            if not result: continue

            power = result['power']; max_power = result['max_power']

            gauge_fig = go.Figure(go.Indicator(mode="gauge+number", value=power, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text':f"<b>{band}</b><br>{desc}",'font':{'size':16}}, number={'suffix':" μV",'font':{'size':28},'valueformat':'.2f'},
                gauge={'axis':{'range':[0, max_power]}, 'bar':{'color':color,'thickness':0.75}}))
            gauge_fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20), uirevision=f'fft-gauge-{ch_idx}-{band}')
            gauge_row.append(html.Div(dcc.Graph(figure=gauge_fig), style={'width': '33%', 'minWidth': '150px'}))

        gauges.append(html.Div(gauge_row, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}))
        return gauges
    return update_fft_display

for i in range(ADS1299_NUM_CHANNELS):
    generate_time_series_callback(i)
    generate_fft_callback(i)

def run_dash_app():
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    ingestion_thread = threading.Thread(target=data_ingestion_worker, daemon=True)
    fft_thread = threading.Thread(target=fft_calculator_worker, daemon=True)
    dash_thread = threading.Thread(target=run_dash_app, daemon=True)

    print("Starting worker threads...")
    ingestion_thread.start()
    fft_thread.start()
    dash_thread.start()

    try:
        asyncio.run(ble_data_stream_loop())
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except Exception as e:
        print(f"An error occurred in the main BLE loop: {e}")