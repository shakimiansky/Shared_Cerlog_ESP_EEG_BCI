import serial
import struct
import threading
import time
from collections import deque
import serial.tools.list_ports
import numpy as np
from dash.exceptions import PreventUpdate
import logging

import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State

# --- NEW: Import ML and Signal Processing libraries ---
from scipy.signal import butter, lfilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- Serial/Data Acquisition Configuration (Unchanged) ---
INITIAL_BAUD_RATE = 9600
FINAL_BAUD_RATE = 115200
FIRMWARE_BAUD_RATE_INDEX = 0x04
SAMPLING_RATE_HZ = 250.0
STREAM_TIMEOUT_SECONDS = 5.0
DATA_PACKET_START_MARKER = 0xABCD
DATA_PACKET_END_MARKER = 0xDCBA
DATA_PACKET_TOTAL_SIZE = 37
PACKET_IDX_LENGTH = 2
PACKET_IDX_CHECKSUM = 34
ADS1299_NUM_CHANNELS = 8
ADS1299_NUM_STATUS_BYTES = 3
ADS1299_BYTES_PER_CHANNEL = 3
BOARD_USB_IDS = [{'vid': 0x1A86, 'pid': 0x7523}]
BOARD_DESCRIPTIONS = ["USB-SERIAL CH340", "CH340"]

# --- MODIFIED: BCI Configuration with ML and Filtering ---
CHANNELS_TO_USE = [1, 2, 3] # Using channels 1, 2, 3
SSVEP_FREQ_LEFT = 12.0
SSVEP_FREQ_RIGHT = 17.0
FFT_WINDOW_SECONDS = 1.5 # How many seconds of data to analyze at a time
FFT_MAXLEN = int(SAMPLING_RATE_HZ * FFT_WINDOW_SECONDS)
# Filter design for cleaning EEG signal
FILTER_LOW_CUT_HZ = 5.0
FILTER_HIGH_CUT_HZ = 50.0
FILTER_ORDER = 5

# --- Visual Gradient Stimulus Configuration (Unchanged) ---
GRADIENT_X_RESOLUTION = 100
GRADIENT_Y_RESOLUTION = 50
FLICKER_INTENSITY = 0.7
GRADIENT_COLOR_MIN = 'hsl(220, 25%, 15%)'
GRADIENT_COLOR_MAX = 'hsl(190, 100%, 95%)'

# --- Game Configuration (Unchanged) ---
GAME_INTERVAL_MS = 50
PADDLE_SPEED = 25 # Paddle speed is now more direct
AI_PADDLE_SPEED = 8
INITIAL_BALL_SPEED_Y = -4
BALL_SPIN_FACTOR = 0.05
GAME_WIDTH = 800
GAME_HEIGHT = 600
PADDLE_WIDTH = 150
PADDLE_HEIGHT = 20
BALL_RADIUS = 10

# --- Data Buffers ---
# Use 0-based indexing for the channels to use in Python lists
CHANNELS_IDX = [c - 1 for c in CHANNELS_TO_USE]
fft_buffers = [deque(maxlen=FFT_MAXLEN) for _ in range(ADS1299_NUM_CHANNELS)]
buffer_lock = threading.Lock()

# --- NEW: ML and Calibration Globals ---
classifier = LinearDiscriminantAnalysis()
calibration_data = {'features': [], 'labels': []}
calibration_state = 'idle' # Can be 'idle', 'collecting_left', 'collecting_right', 'collecting_neutral', 'ready'
calibration_start_time = 0
CALIBRATION_DURATION_S = 10 # Seconds to collect data for each class

# --- NEW: Signal Processing Functions ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(FILTER_LOW_CUT_HZ, FILTER_HIGH_CUT_HZ, SAMPLING_RATE_HZ, order=FILTER_ORDER)

def bandpass_filter(data):
    return lfilter(b, a, data)

def extract_features(eeg_data_window, freqs_of_interest):
    """Extracts PSD features for a single channel's data window."""
    if len(eeg_data_window) < FFT_MAXLEN: return []
    
    # 1. Filter and apply a windowing function to the data
    y_data = np.array(eeg_data_window)
    y_filtered = bandpass_filter(y_data)
    y_win = y_filtered * np.hanning(len(y_filtered))

    # 2. Compute Power Spectral Density (PSD)
    N = len(y_win)
    yf = np.fft.fft(y_win)
    xf = np.fft.fftfreq(N, 1.0 / SAMPLING_RATE_HZ)
    psd = (2 / (SAMPLING_RATE_HZ * np.sum(np.hanning(N)**2))) * np.abs(yf[:N//2])**2

    # 3. Extract power at specific frequencies
    features = []
    for freq in freqs_of_interest:
        idx = np.argmin(np.abs(xf[:N//2] - freq))
        # Average power in a small bin around the target frequency for stability
        power = np.mean(psd[max(0, idx-1) : idx+2])
        features.append(power)
    return features

# --- Data Acquisition Backend (Unchanged) ---
def convert_to_microvolts(raw_val, vref=4.5, gain=24):
    scale_factor = (vref * 2) / (gain * (2**24)); return raw_val * scale_factor * 1_000_000

def parse_data_packet(packet):
    try:
        ads_data = packet[7:34]
        with buffer_lock:
            for ch in range(ADS1299_NUM_CHANNELS):
                idx = ADS1299_NUM_STATUS_BYTES + ch * ADS1299_BYTES_PER_CHANNEL
                raw_bytes = ads_data[idx:idx + ADS1299_BYTES_PER_CHANNEL]
                value = int.from_bytes(raw_bytes, byteorder='big', signed=True)
                microvolts = convert_to_microvolts(value)
                fft_buffers[ch].append(microvolts)
    except Exception as e: print(f"Error parsing packet: {e}")

def find_and_open_board():
    print("Searching for the ADS1299 board..."); ports = serial.tools.list_ports.comports()
    candidate_ports = [p.device for p in ports if (p.vid and p.pid and {'vid': p.vid, 'pid': p.pid} in BOARD_USB_IDS) or \
                       (p.description and any(desc.lower() in p.description.lower() for desc in BOARD_DESCRIPTIONS))]
    if not candidate_ports: print("No specific candidates found. Testing all available serial ports..."); candidate_ports = [p.device for p in ports]
    for port_name in candidate_ports:
        print(f"--- Testing port: {port_name} ---"); ser = None
        try:
            ser = serial.Serial(port_name, INITIAL_BAUD_RATE, timeout=2)
            print("Port opened. Waiting 5 seconds..."); time.sleep(5)
            if ser.in_waiting > 0: ser.read(ser.in_waiting)
            print(f"Sending handshake to negotiate baud rate: {FINAL_BAUD_RATE} bps...")
            current_unix_time = int(time.time())
            checksum_payload = struct.pack('>BI', 0x02, current_unix_time) + bytes([0x01, FIRMWARE_BAUD_RATE_INDEX])
            checksum = sum(checksum_payload) & 0xFF
            handshake_packet = struct.pack('>BB', 0xAA, 0xBB) + checksum_payload + struct.pack('>B', checksum) + struct.pack('>BB', 0xCC, 0xDD)
            ser.write(handshake_packet); time.sleep(0.1)
            ser.baudrate = FINAL_BAUD_RATE
            print(f"Switched to {ser.baudrate} baud. Verifying stream..."); time.sleep(0.5); ser.reset_input_buffer()
            bytes_received = ser.read(DATA_PACKET_TOTAL_SIZE * 5)
            if bytes_received and DATA_PACKET_START_MARKER.to_bytes(2, 'big') in bytes_received:
                print(f"Success! Board found on port: {port_name}"); return ser
            else: print("No valid data stream detected."); ser.close()
        except serial.SerialException as e: print(f"Could not test port {port_name}: {e}"); ser.close()
    return None

def serial_read_loop(ser):
    if not ser: return
    buffer = bytearray(); start_marker = DATA_PACKET_START_MARKER.to_bytes(2, 'big'); end_marker = DATA_PACKET_END_MARKER.to_bytes(2, 'big'); last_data_time = time.time()
    try:
        while True:
            if time.time() - last_data_time > STREAM_TIMEOUT_SECONDS: print(f"\nStream timed out."); break
            if ser.in_waiting > 0: buffer.extend(ser.read(ser.in_waiting))
            else: time.sleep(0.005); continue
            while True:
                start_idx = buffer.find(start_marker)
                if start_idx == -1: break
                if len(buffer) < start_idx + DATA_PACKET_TOTAL_SIZE: break
                potential_packet = buffer[start_idx : start_idx + DATA_PACKET_TOTAL_SIZE]
                if potential_packet.endswith(end_marker):
                    payload = potential_packet[PACKET_IDX_LENGTH:PACKET_IDX_CHECKSUM]
                    if (sum(payload) & 0xFF) == potential_packet[PACKET_IDX_CHECKSUM]: parse_data_packet(potential_packet); last_data_time = time.time(); buffer = buffer[start_idx + DATA_PACKET_TOTAL_SIZE:]; continue
                buffer = buffer[start_idx + 1:]
    except Exception as e: print(f"Error in serial_read_loop: {e}")
    finally:
        if ser and ser.is_open: ser.close(); print("Serial port closed.")

# --- Pong Game Setup ---
app = Dash(__name__)
app.title = "SSVEP BCI Pong (ML Enhanced)"

def get_initial_game_state():
    return {'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2,
            'ball_y': GAME_HEIGHT / 2, 'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y,
            'player_score': 0, 'ai_score': 0, 'prediction': 'N/A'}

# --- MODIFIED: New Layout with Calibration Controls ---
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#DDDDDD', 'textAlign': 'center', 'fontFamily': 'monospace'}, children=[
    html.H1("SSVEP BCI Pong (ML Enhanced)"),
    html.Div(id='status-bar'),
    dcc.Graph(id='pong-game-graph', config={'staticPlot': True}),
    html.Button('Start Calibration', id='calibration-button', n_clicks=0, style={'margin': '10px', 'fontSize': '16px'}),
    dcc.Interval(id='game-interval', interval=GAME_INTERVAL_MS, n_intervals=0),
    dcc.Interval(id='calibration-interval', interval=200, n_intervals=0), # Interval for collecting calibration data
    dcc.Store(id='game-state-store', data=get_initial_game_state()),
])

# --- NEW: Calibration Callback ---
@app.callback(
    Output('status-bar', 'children'),
    Input('calibration-button', 'n_clicks'),
    Input('calibration-interval', 'n_intervals'),
    prevent_initial_call=True
)
def handle_calibration(n_clicks, n_intervals):
    global calibration_state, calibration_start_time, classifier, calibration_data

    # This logic triggers only when the calibration button is clicked
    if n_clicks > 0 and calibration_state == 'idle':
        print("Starting calibration...")
        calibration_state = 'collecting_left'
        calibration_start_time = time.time()
        calibration_data = {'features': [], 'labels': []} # Reset data
        return html.H2(f"CALIBRATION: Focus on the LEFT for {CALIBRATION_DURATION_S} seconds...", style={'color': 'cyan'})

    # This logic runs on the calibration interval timer
    if calibration_state not in ['idle', 'ready']:
        elapsed_time = time.time() - calibration_start_time
        time_left = CALIBRATION_DURATION_S - elapsed_time

        if time_left <= 0:
            if calibration_state == 'collecting_left':
                calibration_state = 'collecting_right'
                calibration_start_time = time.time()
            elif calibration_state == 'collecting_right':
                calibration_state = 'collecting_neutral'
                calibration_start_time = time.time()
            elif calibration_state == 'collecting_neutral':
                print("Training model...")
                X = np.array(calibration_data['features'])
                y = np.array(calibration_data['labels'])
                if len(np.unique(y)) < 3:
                    print("Error: Not enough data for all 3 classes. Please restart calibration.")
                    calibration_state = 'idle'
                    return html.H2("Calibration Failed. Please try again.", style={'color': 'red'})
                
                classifier.fit(X, y)
                print("Model trained successfully!")
                calibration_state = 'ready'
                return html.H2("Calibration Complete! Ready to play.", style={'color': 'lime'})

        # Data collection part
        label_map = {'collecting_left': 0, 'collecting_right': 1, 'collecting_neutral': 2}
        current_label = label_map.get(calibration_state)
        
        # We only collect data after the first couple of seconds to let the user settle
        if elapsed_time > 2.0:
            with buffer_lock:
                if all(len(fft_buffers[ch]) == FFT_MAXLEN for ch in CHANNELS_IDX):
                    channel_features = []
                    # Use fundamental frequencies AND their first harmonic as features
                    freqs = [SSVEP_FREQ_LEFT, SSVEP_FREQ_RIGHT, SSVEP_FREQ_LEFT*2, SSVEP_FREQ_RIGHT*2]
                    for ch_idx in CHANNELS_IDX:
                        features = extract_features(list(fft_buffers[ch_idx]), freqs)
                        if features:
                            channel_features.extend(features)
                    
                    if channel_features:
                        calibration_data['features'].append(channel_features)
                        calibration_data['labels'].append(current_label)

        # Update status message
        if calibration_state == 'collecting_left':
            return html.H2(f"CALIBRATION: Focus on the LEFT... {time_left:.1f}s", style={'color': 'cyan'})
        elif calibration_state == 'collecting_right':
            return html.H2(f"CALIBRATION: Focus on the RIGHT... {time_left:.1f}s", style={'color': 'magenta'})
        elif calibration_state == 'collecting_neutral':
            return html.H2(f"CALIBRATION: Look at the center (ball)... {time_left:.1f}s", style={'color': 'yellow'})

    if calibration_state == 'idle':
        return html.H2("Click 'Start Calibration' before playing.")
    elif calibration_state == 'ready':
        return html.H2("Game On! Control the paddle with your mind.", style={'color': 'lime'})
    
    raise PreventUpdate


# --- MODIFIED & CORRECTED: Main Game Loop using ML Predictions ---
@app.callback(
    Output('pong-game-graph', 'figure'),
    Output('game-state-store', 'data'),
    Input('game-interval', 'n_intervals'),
    State('game-state-store', 'data')
)
def update_game(n, state):
    global calibration_state
    if n is None: raise PreventUpdate

    # Check for sufficient data before doing anything
    with buffer_lock:
        if any(len(fft_buffers[ch]) < FFT_MAXLEN for ch in CHANNELS_IDX):
            fig = go.Figure()
            fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', annotations=[dict(text="Waiting for EEG Data...", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=24, color='white'))])
            return fig, state

    control_signal = 0
    if calibration_state == 'ready':
        with buffer_lock:
            channel_features = []
            freqs = [SSVEP_FREQ_LEFT, SSVEP_FREQ_RIGHT, SSVEP_FREQ_LEFT*2, SSVEP_FREQ_RIGHT*2]
            for ch_idx in CHANNELS_IDX:
                features = extract_features(list(fft_buffers[ch_idx]), freqs)
                if features:
                    channel_features.extend(features)
        
        if channel_features:
            prediction = classifier.predict([channel_features])[0]
            if prediction == 0: # Left
                control_signal = -1
                state['prediction'] = 'Left'
            elif prediction == 1: # Right
                control_signal = 1
                state['prediction'] = 'Right'
            else: # Neutral
                control_signal = 0
                state['prediction'] = 'Neutral'
    
    # --- Game Logic ---
    if calibration_state == 'ready': # Only move paddle if calibrated
        state['player_x'] += PADDLE_SPEED * control_signal
        state['player_x'] = max(PADDLE_WIDTH / 2, min(GAME_WIDTH - PADDLE_WIDTH / 2, state['player_x']))

    if state['ai_x'] < state['ball_x']: state['ai_x'] += AI_PADDLE_SPEED
    if state['ai_x'] > state['ball_x']: state['ai_x'] -= AI_PADDLE_SPEED
    state['ball_x'] += state['ball_vx']; state['ball_y'] += state['ball_vy']
    if state['ball_x'] <= BALL_RADIUS or state['ball_x'] >= GAME_WIDTH - BALL_RADIUS: state['ball_vx'] *= -1
    if state['ball_vy'] < 0 and state['ball_y'] - BALL_RADIUS < PADDLE_HEIGHT and abs(state['player_x'] - state['ball_x']) < PADDLE_WIDTH / 2:
        state['ball_vy'] *= -1; state['ball_vx'] += (state['ball_x'] - state['player_x']) * BALL_SPIN_FACTOR
    if state['ball_vy'] > 0 and state['ball_y'] + BALL_RADIUS > GAME_HEIGHT - PADDLE_HEIGHT and abs(state['ai_x'] - state['ball_x']) < PADDLE_WIDTH / 2:
        state['ball_vy'] *= -1; state['ball_vx'] += (state['ball_x'] - state['ai_x']) * BALL_SPIN_FACTOR
    if state['ball_y'] < -BALL_RADIUS or state['ball_y'] > GAME_HEIGHT + BALL_RADIUS:
        if state['ball_y'] < 0: state['ai_score'] += 1
        else: state['player_score'] += 1
        p_score, a_score = state['player_score'], state['ai_score']
        state = get_initial_game_state(); state.update({'player_score': p_score, 'ai_score': a_score})

    # --- Drawing Logic ---
    fig = go.Figure()
    current_time_s = n * GAME_INTERVAL_MS / 1000.0
    flicker_val_left = 0.5 + (np.sin(2 * np.pi * SSVEP_FREQ_LEFT * current_time_s) * (FLICKER_INTENSITY / 2))
    flicker_val_right = 0.5 + (np.sin(2 * np.pi * SSVEP_FREQ_RIGHT * current_time_s) * (FLICKER_INTENSITY / 2))
    left_weights = np.linspace(1, 0, GRADIENT_X_RESOLUTION)
    gradient_row = (flicker_val_left * left_weights) + (flicker_val_right * (1 - left_weights))
    gradient_z_values = np.tile(gradient_row, (GRADIENT_Y_RESOLUTION, 1))
    
    fig.add_trace(go.Heatmap(
        z=gradient_z_values, x=np.linspace(0, GAME_WIDTH, GRADIENT_X_RESOLUTION),
        y=np.linspace(0, GAME_HEIGHT, GRADIENT_Y_RESOLUTION), colorscale=[[0, GRADIENT_COLOR_MIN], [1, GRADIENT_COLOR_MAX]],
        showscale=False, zmin=0, zmax=1
    ))
    
    fig.add_shape(type="rect", x0=state['player_x']-PADDLE_WIDTH/2, y0=0, x1=state['player_x']+PADDLE_WIDTH/2, y1=PADDLE_HEIGHT, fillcolor="cyan", line=dict(width=0))
    fig.add_shape(type="rect", x0=state['ai_x']-PADDLE_WIDTH/2, y0=GAME_HEIGHT-PADDLE_HEIGHT, x1=state['ai_x']+PADDLE_WIDTH/2, y1=GAME_HEIGHT, fillcolor="magenta", line=dict(width=0))
    fig.add_shape(type="circle", x0=state['ball_x']-BALL_RADIUS, y0=state['ball_y']-BALL_RADIUS, x1=state['ball_x']+BALL_RADIUS, y1=state['ball_y']+BALL_RADIUS, fillcolor="white", line=dict(width=0))
    
    fig.add_annotation(text=f"Prediction: {state.get('prediction', 'N/A')}",
                       xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False,
                       font=dict(size=16, color="white"))

    fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', margin=dict(l=10, r=10, t=40, b=10),
                      annotations=[dict(text=str(state['player_score']), x=30, y=GAME_HEIGHT/2 - 30, showarrow=False, font=dict(size=40, color='cyan')),
                                   dict(text=str(state['ai_score']), x=30, y=GAME_HEIGHT/2 + 30, showarrow=False, font=dict(size=40, color='magenta'))])
    
    # This function now correctly returns two values, matching the two Outputs
    return fig, state

def main():
    serial_port_object = find_and_open_board()
    if serial_port_object:
        threading.Thread(target=serial_read_loop, args=(serial_port_object,), daemon=True).start()
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        print("\nDash server is running. Open http://127.0.0.1:8050/ in your browser.")
        app.run(debug=False, use_reloader=False)
    else:
        print("Could not start application: No board was found or data stream failed verification.")

if __name__ == "__main__":
    main()