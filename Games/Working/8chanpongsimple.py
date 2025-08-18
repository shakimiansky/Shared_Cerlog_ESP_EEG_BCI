import serial
import struct
import threading
import time
from collections import deque
import serial.tools.list_ports
import numpy as np
from dash.exceptions import PreventUpdate

# --- NEW: Import scikit-learn for AI calibration ---
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State

# --- Configuration to match your new plotter script ---
INITIAL_BAUD_RATE = 9600
FINAL_BAUD_RATE = 115200
FIRMWARE_BAUD_RATE_INDEX = 0x04
SAMPLING_RATE_HZ = 250.0
STREAM_TIMEOUT_SECONDS = 5.0

# --- Packet Structures ---
DATA_PACKET_START_MARKER = 0xABCD
DATA_PACKET_END_MARKER = 0xDCBA
DATA_PACKET_TOTAL_SIZE = 37
HANDSHAKE_START_MARKER_1 = 0xAA
HANDSHAKE_END_MARKER_1 = 0xCC
PACKET_IDX_LENGTH = 2
PACKET_IDX_CHECKSUM = 34

# --- ADS1299 Config ---
ADS1299_NUM_CHANNELS = 8
ADS1299_NUM_STATUS_BYTES = 3
ADS1299_BYTES_PER_CHANNEL = 3

# --- Heuristics for Port Detection ---
BOARD_USB_IDS = [{'vid': 0x1A86, 'pid': 0x7523}]
BOARD_DESCRIPTIONS = ["USB-SERIAL CH340", "CH340"]

# --- Game and BCI Configuration ---
GAME_INTERVAL_MS = 50
FOCUS_SMOOTHING_WINDOW = 15 # How many readings to average for smooth control

# --- Game Feel Tuning ---
PADDLE_SPEED = 25
AI_PADDLE_SPEED = 8
INITIAL_BALL_SPEED_Y = -3
BALL_SPIN_FACTOR = 0.05
CALIBRATION_SECONDS_PER_PHASE = 10

# Game dimensions
GAME_WIDTH = 800
GAME_HEIGHT = 600
PADDLE_WIDTH = 150
PADDLE_HEIGHT = 20
BALL_RADIUS = 10

# --- Data Buffers & Global AI Models ---
# The buffer now holds 1 second of data at the new 250Hz sample rate
fft_buffers = [deque(maxlen=int(SAMPLING_RATE_HZ)) for _ in range(ADS1299_NUM_CHANNELS)]
feature_vector_history = deque(maxlen=FOCUS_SMOOTHING_WINDOW)
buffer_lock = threading.Lock()
ml_model = None
ml_scaler = None

# --- Brainwave Frequency Bands (using the plotter's 8-12Hz Alpha range) ---
BRAINWAVE_BANDS = { "Alpha": [8, 12], "Beta": [13, 30] }

# --- Data Acquisition Backend ---
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
    except Exception as e:
        print(f"Error parsing packet: {e}")

def find_and_open_board():
    print("Searching for the ADS1299 board...")
    ports = serial.tools.list_ports.comports()
    candidate_ports = [p.device for p in ports if (p.vid and p.pid and {'vid': p.vid, 'pid': p.pid} in BOARD_USB_IDS) or \
                       (p.description and any(desc.lower() in p.description.lower() for desc in BOARD_DESCRIPTIONS))]
    if not candidate_ports:
        print("No specific candidates found. Testing all available serial ports..."); candidate_ports = [p.device for p in ports]
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
            print(f"Switched to {ser.baudrate} baud. Verifying stream...")
            time.sleep(0.5); ser.reset_input_buffer()
            bytes_received = ser.read(DATA_PACKET_TOTAL_SIZE * 5)
            if bytes_received and DATA_PACKET_START_MARKER.to_bytes(2, 'big') in bytes_received:
                print(f"Success! Board found on port: {port_name}"); return ser
            else: print("No valid data stream detected."); ser.close()
        except serial.SerialException as e:
            print(f"Could not test port {port_name}: {e}")
            if ser and ser.is_open: ser.close()
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
                    if (sum(payload) & 0xFF) == potential_packet[PACKET_IDX_CHECKSUM]:
                        parse_data_packet(potential_packet); last_data_time = time.time()
                        buffer = buffer[start_idx + DATA_PACKET_TOTAL_SIZE:]; continue
                buffer = buffer[start_idx + 1:]
    except Exception as e: print(f"Error in serial_read_loop: {e}")
    finally:
        if ser and ser.is_open: ser.close(); print("Serial port closed.")

# --- Pong Game Setup ---
app = Dash(__name__)
app.title = "BCI Pong"

def get_initial_game_state():
    return {
        'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2, 'ball_y': GAME_HEIGHT / 2,
        'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y, 'player_score': 0, 'ai_score': 0,
        'game_mode': 'CALIBRATE_RELAX', 'calibration_start_time': None,
        'relax_readings': [], 'focus_readings': [],
    }

app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#DDDDDD', 'textAlign': 'center', 'fontFamily': 'monospace'}, children=[
    html.H1("BCI Pong (8-Channel Control)"),
    html.P(id='instruction-text', children="Starting Calibration..."),
    dcc.Graph(id='pong-game-graph', config={'staticPlot': True}),
    dcc.Interval(id='game-interval', interval=GAME_INTERVAL_MS, n_intervals=0),
    dcc.Store(id='game-state-store', data=get_initial_game_state()),
    html.Div(id='focus-metric-display')
])

@app.callback(
    Output('pong-game-graph', 'figure'), Output('game-state-store', 'data'),
    Output('focus-metric-display', 'children'), Output('instruction-text', 'children'),
    Input('game-interval', 'n_intervals'), State('game-state-store', 'data')
)
def update_game(n, state):
    global ml_model, ml_scaler
    if n is None: raise PreventUpdate

    with buffer_lock:
        if any(len(buf) < SAMPLING_RATE_HZ for buf in fft_buffers):
            fig = go.Figure(); fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', annotations=[dict(text="Waiting for EEG Data...", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=24, color='white'))])
            return fig, state, "Focus Metric: Waiting...", "Connecting to your brain..."

    feature_vector = []
    with buffer_lock:
        for ch in range(ADS1299_NUM_CHANNELS):
            eeg_data = list(fft_buffers[ch])
            y_data = np.array(eeg_data) - np.mean(eeg_data); N = len(y_data)
            
            win = np.hanning(N)
            y_win = y_data * win
            yf = np.fft.fft(y_win)
            xf = np.fft.fftfreq(N, 1.0 / SAMPLING_RATE_HZ)[:N//2]
            psd = (2 / (SAMPLING_RATE_HZ * np.sum(win**2))) * np.abs(yf[0:N//2])**2
            df = SAMPLING_RATE_HZ / N

            alpha_mask = (xf >= BRAINWAVE_BANDS['Alpha'][0]) & (xf < BRAINWAVE_BANDS['Alpha'][1])
            beta_mask = (xf >= BRAINWAVE_BANDS['Beta'][0]) & (xf < BRAINWAVE_BANDS['Beta'][1])
            
            alpha_power = np.sum(psd[alpha_mask]) * df if np.any(alpha_mask) else 0.0
            beta_power = np.sum(psd[beta_mask]) * df if np.any(beta_mask) else 0.0
            
            focus_metric = beta_power / alpha_power if alpha_power > 1e-9 else 0
            feature_vector.append(focus_metric)

    current_time = n * GAME_INTERVAL_MS / 1000.0
    if state['calibration_start_time'] is None: state['calibration_start_time'] = current_time

    # --- STATE MACHINE ---
    if state['game_mode'] == 'CALIBRATE_RELAX' or state['game_mode'] == 'CALIBRATE_FOCUS':
        time_in_phase = current_time - state['calibration_start_time']
        seconds_left = max(0, CALIBRATION_SECONDS_PER_PHASE - time_in_phase)
        instruction = "RELAX. Clear your mind." if state['game_mode'] == 'CALIBRATE_RELAX' else "FOCUS. Do mental math."
        
        if any(feature_vector):
            readings_key = 'relax_readings' if state['game_mode'] == 'CALIBRATE_RELAX' else 'focus_readings'
            state[readings_key].append(feature_vector)

        fig = go.Figure(); fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', annotations=[dict(text=f"{instruction}\n{int(seconds_left)}s left", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=30, color='white'))])
        focus_text = f"Avg Raw Focus Power Ratio: {np.mean(feature_vector):.2f}"

        if seconds_left == 0:
            if state['game_mode'] == 'CALIBRATE_RELAX':
                state['game_mode'] = 'CALIBRATE_FOCUS'; state['calibration_start_time'] = current_time
            else:
                state['game_mode'] = 'TRAINING_MODEL'
        return fig, state, focus_text, instruction

    elif state['game_mode'] == 'TRAINING_MODEL':
        print("Training AI model on 8-channel data...")
        X_relax = np.array(state['relax_readings'])
        X_focus = np.array(state['focus_readings'])
        
        if len(X_relax) < 20 or len(X_focus) < 20:
            print("Calibration failed: not enough data. Restarting...")
            return go.Figure(), get_initial_game_state(), "Calibration failed", "Not enough data captured. Restarting."

        X = np.vstack((X_relax, X_focus)); y = np.array([-1] * len(X_relax) + [1] * len(X_focus))
        print(f"Training with data of shape: {X.shape}")

        ml_scaler = StandardScaler().fit(X)
        X_scaled = ml_scaler.transform(X)
        ml_model = SVC(kernel='rbf', C=1.0, probability=True).fit(X_scaled, y)
       
        print("Model trained successfully.")
        state['game_mode'] = 'PLAYING'
   
    if state['game_mode'] == 'PLAYING':
        # <<< FIX: Check if the model exists in memory before trying to use it. >>>
        # This handles cases where the Python script restarts but the browser state is still 'PLAYING'.
        if ml_model is None or ml_scaler is None:
            print("Model not found in memory. Forcing recalibration...")
            fig = go.Figure()
            fig.update_layout(
                xaxis=dict(range=[0, GAME_WIDTH], visible=False),
                yaxis=dict(range=[0, GAME_HEIGHT], visible=False),
                plot_bgcolor='#000', paper_bgcolor='#111',
                annotations=[dict(
                    text="Server restarted. Recalibrating...",
                    x=GAME_WIDTH/2, y=GAME_HEIGHT/2,
                    showarrow=False, font=dict(size=24, color='white')
                )]
            )
            # Reset the entire game state and return to the beginning
            return fig, get_initial_game_state(), "Recalibrating...", "Server restarted. Please recalibrate."

        feature_vector_history.append(feature_vector)
        smoothed_feature_vector = np.mean(feature_vector_history, axis=0)

        features_scaled = ml_scaler.transform([smoothed_feature_vector])
        control_signal = ml_model.decision_function(features_scaled)[0]
        control_signal = np.clip(control_signal, -1.5, 1.5)

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
            state.update(get_initial_game_state())
            state.update({'game_mode': 'PLAYING', 'player_score': p_score, 'ai_score': a_score})

        fig = go.Figure()
        fig.add_shape(type="rect", x0=state['player_x']-PADDLE_WIDTH/2, y0=0, x1=state['player_x']+PADDLE_WIDTH/2, y1=PADDLE_HEIGHT, fillcolor="cyan", line=dict(width=0))
        fig.add_shape(type="rect", x0=state['ai_x']-PADDLE_WIDTH/2, y0=GAME_HEIGHT-PADDLE_HEIGHT, x1=state['ai_x']+PADDLE_WIDTH/2, y1=GAME_HEIGHT, fillcolor="magenta", line=dict(width=0))
        fig.add_shape(type="circle", x0=state['ball_x']-BALL_RADIUS, y0=state['ball_y']-BALL_RADIUS, x1=state['ball_x']+BALL_RADIUS, y1=state['ball_y']+BALL_RADIUS, fillcolor="white", line=dict(width=0))
        fig.add_shape(type="line", x0=0, y0=GAME_HEIGHT/2, x1=GAME_WIDTH, y1=GAME_HEIGHT/2, line=dict(color="grey", width=2, dash="dot"))
        fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', margin=dict(l=10, r=10, t=10, b=10),
                          annotations=[dict(text=str(state['player_score']), x=30, y=GAME_HEIGHT/2 - 30, showarrow=False, font=dict(size=40, color='cyan')),
                                       dict(text=str(state['ai_score']), x=30, y=GAME_HEIGHT/2 + 30, showarrow=False, font=dict(size=40, color='magenta'))])
       
        focus_text = f"Control Signal (8-Ch): {control_signal:.2f}"
        instruction_text = "Relax to move left. Focus to move right."
        return fig, state, focus_text, instruction_text

    return PreventUpdate

def main():
    serial_port_object = find_and_open_board()
    if serial_port_object:
        threading.Thread(target=serial_read_loop, args=(serial_port_object,), daemon=True).start()
        print("\nDash server is running. Open http://127.0.0.1:8050/ in your browser.")
        app.run(debug=False, use_reloader=False)
    else:
        print("Could not start application: No board was found or data stream failed verification.")

if __name__ == "__main__":
    main()