import serial
import struct
import threading
import time
from collections import deque
import serial.tools.list_ports
import numpy as np
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State

# --- DIAGNOSTIC CONFIGURATION ---
ENABLE_TERMINAL_DEBUG = True
DEBUG_PRINT_INTERVAL_PACKETS = 250  # Print one full packet breakdown every 250 packets (approx. 1 per second)
debug_packet_counter = 0

# --- General Configuration ---
INITIAL_BAUD_RATE = 9600
FINAL_BAUD_RATE = 115200
FIRMWARE_BAUD_RATE_INDEX = 0x04
SAMPLING_RATE_HZ = 250.0

# --- Plotting and FFT Constants ---
FFT_BUFFER_SIZE = 15000
TIME_SERIES_DEFAULT_SECONDS = 2
TIME_SERIES_POINTS = int(TIME_SERIES_DEFAULT_SECONDS * SAMPLING_RATE_HZ)
FFT_MAX_HZ = 100

# --- Packet Structures ---
DATA_PACKET_START_MARKER = 0xABCD
DATA_PACKET_END_MARKER = 0xDCBA
DATA_PACKET_TOTAL_SIZE = 37
HANDSHAKE_START_MARKER_1 = 0xAA
HANDSHAKE_END_MARKER_1 = 0xCC

# Packet Indices
PACKET_IDX_LENGTH = 2
PACKET_IDX_CHECKSUM = 34

# --- ADS1299 Config ---
ADS1299_NUM_CHANNELS = 8
ADS1299_NUM_STATUS_BYTES = 3
ADS1299_BYTES_PER_CHANNEL = 3

# --- Heuristics for Port Detection ---
BOARD_USB_IDS = [{'vid': 0x1A86, 'pid': 0x7523}]
BOARD_DESCRIPTIONS = ["USB-SERIAL CH340", "CH340"]

# --- Data Buffers ---
fft_buffers = [deque(maxlen=FFT_BUFFER_SIZE) for _ in range(ADS1299_NUM_CHANNELS)]
time_series_buffers = [deque(maxlen=TIME_SERIES_POINTS) for _ in range(ADS1299_NUM_CHANNELS)]
band_power_history = {f"ch{ch}_{band}": deque(maxlen=100) for ch in range(ADS1299_NUM_CHANNELS) for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]}
buffer_lock = threading.Lock()
total_sample_count = 0

# --- Brainwave Frequency Bands ---
BRAINWAVE_BANDS = {
    "Delta": ([0.5, 4], "hsl(225, 39%, 30%)", "(Deep Sleep)"),
    "Theta": ([4, 8], "hsl(225, 39%, 50%)", "(Drowsy)"),
    "Alpha": ([8, 13], "hsl(205, 56%, 70%)", "(Relaxed)"),
    "Beta": ([13, 30], "hsl(190, 56%, 60%)", "(Active)"),
    "Gamma": ([30, FFT_MAX_HZ], "hsl(175, 56%, 50%)", "(Insight)"),
}

# --- Helper Functions ---
def convert_to_microvolts(raw_val, vref=4.5, gain=24):
    """
    Converts the raw 24-bit integer from the ADS1299 to microvolts.
    This formula matches the known-working C++ driver for consistency.
    """
    scale_factor = (2 * vref / gain) / (2**24)
    return raw_val * scale_factor * 1_000_000

def parse_data_packet(packet):
    """Parses a validated 37-byte data packet and updates the data buffers."""
    global total_sample_count
    try:
        ads_data = packet[7:34]
        with buffer_lock:
            total_sample_count += 1
            current_time_s = total_sample_count / SAMPLING_RATE_HZ
            for ch in range(ADS1299_NUM_CHANNELS):
                idx = ADS1299_NUM_STATUS_BYTES + ch * ADS1299_BYTES_PER_CHANNEL
                raw_bytes = ads_data[idx:idx + ADS1299_BYTES_PER_CHANNEL]
                value = int.from_bytes(raw_bytes, byteorder='big', signed=True)
                microvolts = convert_to_microvolts(value)
                fft_buffers[ch].append(microvolts)
                time_series_buffers[ch].append((current_time_s, microvolts))
    except Exception as e:
        print(f"Error parsing packet: {e}")

def find_and_open_board():
    """Scans serial ports, connects, and negotiates the final baud rate."""
    print("Searching for the ADS1299 board...")
    ports = serial.tools.list_ports.comports()
    candidate_ports = [p.device for p in ports if (p.vid and p.pid and {'vid': p.vid, 'pid': p.pid} in BOARD_USB_IDS) or \
                       (p.description and any(desc.lower() in p.description.lower() for desc in BOARD_DESCRIPTIONS))]
    if not candidate_ports:
        print("No specific candidate ports found. Testing all available serial ports...")
        candidate_ports = [p.device for p in ports]

    for port_name in candidate_ports:
        print(f"--- Testing port: {port_name} ---")
        ser = None
        try:
            ser = serial.Serial(port_name, INITIAL_BAUD_RATE, timeout=2)
            print("Port opened. Waiting 5 seconds for board to initialize...")
            time.sleep(5)
            if ser.in_waiting > 0: ser.read(ser.in_waiting)

            print(f"Sending handshake to negotiate baud rate: {FINAL_BAUD_RATE} bps...")
            current_unix_time = int(time.time())
            checksum_payload = struct.pack('>BI', 0x02, current_unix_time) + bytes([0x01, FIRMWARE_BAUD_RATE_INDEX])
            checksum = sum(checksum_payload) & 0xFF
            handshake_packet = struct.pack('>BB', HANDSHAKE_START_MARKER_1, 0xBB) + checksum_payload + struct.pack('>B', checksum) + struct.pack('>BB', HANDSHAKE_END_MARKER_1, 0xDD)
            ser.write(handshake_packet)
            time.sleep(0.1)
            
            ser.baudrate = FINAL_BAUD_RATE
            print(f"Switched to {ser.baudrate} baud. Waiting for stream to stabilize...")
            time.sleep(0.5)
            ser.reset_input_buffer()
            
            print("Verifying data stream...")
            bytes_received = ser.read(DATA_PACKET_TOTAL_SIZE * 5)
            if bytes_received and DATA_PACKET_START_MARKER.to_bytes(2, 'big') in bytes_received:
                print(f"Success! Board found and streaming on port: {port_name}")
                return ser
            else:
                print("Handshake sent, but no valid data stream detected.")
                ser.close()
        except serial.SerialException as e:
            print(f"Could not test port {port_name}: {e}")
            if ser and ser.is_open: ser.close()
    return None

def serial_read_loop(ser):
    """Reads data from the serial port, validates, and prints debug info."""
    global debug_packet_counter
    if not ser: return
    
    buffer = bytearray()
    start_marker = DATA_PACKET_START_MARKER.to_bytes(2, 'big')
    end_marker = DATA_PACKET_END_MARKER.to_bytes(2, 'big')

    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                time.sleep(0.001)
                continue
            buffer.extend(data)

            while True:
                start_idx = buffer.find(start_marker)
                if start_idx == -1: break
                
                if len(buffer) < start_idx + DATA_PACKET_TOTAL_SIZE:
                    if start_idx > 0: buffer = buffer[start_idx:]
                    break

                potential_packet = buffer[start_idx : start_idx + DATA_PACKET_TOTAL_SIZE]
                
                if potential_packet.endswith(end_marker):
                    payload_for_checksum = potential_packet[PACKET_IDX_LENGTH:PACKET_IDX_CHECKSUM]
                    calculated_checksum = sum(payload_for_checksum) & 0xFF
                    received_checksum = potential_packet[PACKET_IDX_CHECKSUM]

                    if calculated_checksum == received_checksum:
                        debug_packet_counter += 1
                        
                        # --- START OF DEBUG BLOCK ---
                        if ENABLE_TERMINAL_DEBUG and (debug_packet_counter % DEBUG_PRINT_INTERVAL_PACKETS == 0):
                            print(f"\n--- [DEBUG] Valid Packet #{debug_packet_counter} ---")
                            hex_string = ' '.join(f'{b:02x}' for b in potential_packet)
                            print(f"Raw Bytes ({len(potential_packet)}): {hex_string}")
                            print(f"Checksum OK: 0x{received_checksum:02x}")
                            
                            ads_data = potential_packet[7:34]
                            print("--- Channel Data Breakdown ---")
                            for ch in range(ADS1299_NUM_CHANNELS):
                                idx = ADS1299_NUM_STATUS_BYTES + ch * ADS1299_BYTES_PER_CHANNEL
                                raw_bytes = ads_data[idx:idx + ADS1299_BYTES_PER_CHANNEL]
                                value = int.from_bytes(raw_bytes, byteorder='big', signed=True)
                                microvolts = convert_to_microvolts(value)
                                
                                ch_hex_bytes = ' '.join(f'{b:02x}' for b in raw_bytes)
                                print(f"  Ch {ch}: Bytes: [{ch_hex_bytes}] -> Raw Int: {value:<10} -> uV: {microvolts:.2f}")
                            print("--------------------------------")
                        # --- END OF DEBUG BLOCK ---

                        # Still parse the packet to update the Dash GUI
                        parse_data_packet(potential_packet)
                    else:
                        if ENABLE_TERMINAL_DEBUG:
                            print(f"Checksum mismatch! Expected: 0x{calculated_checksum:02x}, Got: 0x{received_checksum:02x}. Discarding.")
                    
                    buffer = buffer[start_idx + DATA_PACKET_TOTAL_SIZE:]
                else:
                    buffer = buffer[start_idx + 1:]

    except serial.SerialException as e:
        print(f"Serial Error during read loop: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("Serial port closed.")

# --- Dash App Layout and Callbacks (Largely Unchanged) ---
app = Dash(__name__)
app.title = "Cerelog 8-Channel EEG Data Logger"

def create_initial_figure(title="Waiting for data...", y_axis_title="Amplitude (μV)"):
    fig = go.Figure(); fig.update_layout(title=title, yaxis_title=y_axis_title, margin=dict(l=60, r=20, t=50, b=50)); return fig

app.layout = html.Div(style={'backgroundColor': '#f0f2f5', 'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("Cerelog: 8-Channel Brain-Computer Interface EEG Data Log", style={'textAlign': 'center', 'color': '#333333', 'marginBottom': '30px', 'fontSize': '2.5em', 'fontWeight': 'bold'}),
    html.Div([
        html.Div(style={'width': '100%', 'marginBottom': '20px'},
            children=[
                html.H2(f"Channel {i+1}", style={'fontSize': '1.5em', 'fontWeight': 'bold', 'color': '#333333', 'width': '100%', 'textAlign': 'center'}),
                html.Div(style={'display': 'flex', 'flexDirection': 'row', 'width': '100%', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.1)', 'borderRadius': '8px', 'overflow': 'hidden', 'backgroundColor': '#ffffff'}, children=[
                    html.Div(style={'width': '50%', 'padding': '10px'}, children=[dcc.Graph(id=f'channel-graph-{i+1}', figure=create_initial_figure())]),
                    html.Div(id=f'fft-div-{i+1}', style={'width': '50%', 'padding': '10px'})])])
        for i in range(ADS1299_NUM_CHANNELS)], style={'maxWidth': '1600px', 'margin': '0 auto'}),
    dcc.Interval(id='fast-interval-time-series', interval=500, n_intervals=0),
    dcc.Interval(id='slow-interval-fft', interval=2000, n_intervals=0),
])

def generate_time_series_callback(ch_idx):
    @app.callback(Output(f'channel-graph-{ch_idx+1}', 'figure'), Input('fast-interval-time-series', 'n_intervals'), State(f'channel-graph-{ch_idx+1}', 'relayoutData'))
    def update_time_series(n, relayout_data):
        with buffer_lock: channel_data = list(time_series_buffers[ch_idx])
        if len(channel_data) < 2: raise PreventUpdate
        x_data, y_data_raw = zip(*channel_data); y_data_centered = np.array(y_data_raw) - np.mean(y_data_raw)
        fig = go.Figure(go.Scatter(x=x_data, y=y_data_centered, mode='lines', line=dict(shape='linear')))
        fig.update_layout(title='Time Domain', xaxis_title='Time Elapsed (seconds)', yaxis_title='Amplitude (μV, Centered)',
                          margin=dict(l=60, r=20, t=50, b=50), uirevision=f'time-series-{ch_idx}')
        is_zoomed = relayout_data and 'xaxis.range[0]' in relayout_data and not relayout_data.get('xaxis.autorange', False)
        if is_zoomed:
            zoom_width = relayout_data['xaxis.range[1]'] - relayout_data['xaxis.range[0]']
            new_xaxis_end = max(x_data); new_xaxis_start = new_xaxis_end - zoom_width
            fig.update_layout(xaxis_range=[new_xaxis_start, new_xaxis_end], yaxis_range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']])
        else: fig.update_layout(xaxis_range=[min(x_data), max(x_data)])
        return fig
    return update_time_series

def generate_fft_callback(ch_idx):
    @app.callback(Output(f'fft-div-{ch_idx+1}', 'children'), Input('slow-interval-fft', 'n_intervals'))
    def update_fft(n):
        with buffer_lock: full_buffer = list(fft_buffers[ch_idx])
        N = len(full_buffer)
        if N < SAMPLING_RATE_HZ: return html.Div("Calculating FFT...")
        y_detrended = np.array(full_buffer) - np.mean(full_buffer)
        yf = np.fft.fft(y_detrended); amplitude = 2.0/N * np.abs(yf[0:N//2])
        xf = np.fft.fftfreq(N, 1.0 / SAMPLING_RATE_HZ)[:N//2]
        gauges = [html.H4("Brainwave Band Power Levels", style={'width': '100%', 'textAlign': 'center', 'marginBottom': 0})]
        gauge_row = []
        for band, (freq_range, color, description) in BRAINWAVE_BANDS.items():
            band_mask = (xf >= freq_range[0]) & (xf < freq_range[1])
            power = np.mean(amplitude[band_mask]) if np.any(band_mask) else 0
            history_key = f"ch{ch_idx}_{band}"; band_power_history[history_key].append(power)
            min_power = min(band_power_history[history_key]); max_power = max(band_power_history[history_key]) if max(band_power_history[history_key]) > 0.1 else 1
            gauge_fig = go.Figure(go.Indicator(mode="gauge+number", value=power, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': f"<b>{band}</b><br>{description}", 'font': {'size': 16}}, number={'suffix': " μV", 'font': {'size': 28}, 'valueformat': '.2f'}, gauge={'axis': {'range': [min_power, max_power]}, 'bar': {'color': color, 'thickness': 0.75}}))
            gauge_fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20), uirevision=f'fft-gauge-{ch_idx}-{band}')
            gauge_row.append(html.Div(dcc.Graph(figure=gauge_fig), style={'width': '33%', 'minWidth': '150px'}))
        gauges.append(html.Div(gauge_row, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}))
        return gauges
    return update_fft

for i in range(ADS1299_NUM_CHANNELS):
    generate_time_series_callback(i)
    generate_fft_callback(i)

def main():
    serial_port_object = find_and_open_board()
    if serial_port_object:
        threading.Thread(target=serial_read_loop, args=(serial_port_object,), daemon=True).start()
        app.run(debug=True, use_reloader=False)
    else:
        print("Could not start application: No board was found or data stream failed verification.")

if __name__ == "__main__":
    main()