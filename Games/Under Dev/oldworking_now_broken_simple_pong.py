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


# --- Serial/Data Acquisition Configuration (Unchanged) ---
INITIAL_BAUD_RATE = 9600
FINAL_BAUD_RATE = 230400
FIRMWARE_BAUD_RATE_INDEX = 0x05
SAMPLING_RATE_HZ = 500.0
ADS1299_NUM_CHANNELS = 8
ADS1299_NUM_STATUS_BYTES = 3
ADS1299_BYTES_PER_CHANNEL = 3
DATA_PACKET_START_MARKER = 0xABCD
DATA_PACKET_END_MARKER = 0xDCBA
DATA_PACKET_TOTAL_SIZE = 37
HANDSHAKE_START_MARKER_1 = 0xAA
HANDSHAKE_END_MARKER_1 = 0xCC
BOARD_USB_IDS = [{'vid': 0x1A86, 'pid': 0x7523}]
BOARD_DESCRIPTIONS = ["USB-SERIAL CH340", "CH340"]


# --- Game and BCI Configuration ---
GAME_INTERVAL_MS = 50
FOCUS_CHANNEL = 0
FOCUS_SMOOTHING_WINDOW = 15


# <<< CHANGE #1: Tuned for better game feel >>>
PADDLE_SPEED = 25                      # Increased paddle responsiveness
AI_PADDLE_SPEED = 8                    # Slower AI paddle
INITIAL_BALL_SPEED_Y = -3              # Very slow initial ball speed
BALL_SPIN_FACTOR = 0.05                # Minimal "spin" for predictable bounces
CALIBRATION_SECONDS_PER_PHASE = 10     # How long to calibrate each state (relax/focus)


# Game dimensions
GAME_WIDTH = 800
GAME_HEIGHT = 600
PADDLE_WIDTH = 150
PADDLE_HEIGHT = 20
BALL_RADIUS = 10


# --- Data Buffers & Global AI Models ---
fft_buffers = [deque(maxlen=int(SAMPLING_RATE_HZ)) for _ in range(ADS1299_NUM_CHANNELS)]
focus_metric_history = deque(maxlen=FOCUS_SMOOTHING_WINDOW)
buffer_lock = threading.Lock()
# We will store the trained AI model and scaler globally for the app to use
ml_model = None
ml_scaler = None


# --- Brainwave Frequency Bands ---
BRAINWAVE_BANDS = { "Alpha": [8, 13], "Beta": [13, 30] }


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
    except Exception: pass
def find_and_open_board():
    print("Searching for board..."); ports = serial.tools.list_ports.comports()
    candidate_ports = [p.device for p in ports if (p.vid and p.pid and {'vid': p.vid, 'pid': p.pid} in BOARD_USB_IDS) or \
                       (p.description and any(desc.lower() in p.description.lower() for desc in BOARD_DESCRIPTIONS))]
    if not candidate_ports: candidate_ports = [p.device for p in ports]
    for port_name in candidate_ports:
        print(f"--- Testing port: {port_name} ---"); ser = None
        try:
            ser = serial.Serial(port_name, INITIAL_BAUD_RATE, timeout=2)
            time.sleep(4); ser.read(ser.in_waiting or 1)
            print(f"Sending handshake for {FINAL_BAUD_RATE} bps...")
            current_unix_time = int(time.time())
            checksum_payload = struct.pack('>BI', 0x02, current_unix_time) + bytes([0x01, FIRMWARE_BAUD_RATE_INDEX])
            checksum = sum(checksum_payload) & 0xFF
            handshake_packet = struct.pack('>BB', HANDSHAKE_START_MARKER_1, 0xBB) + checksum_payload + struct.pack('>B', checksum) + struct.pack('>BB', HANDSHAKE_END_MARKER_1, 0xDD)
            ser.write(handshake_packet); time.sleep(0.1)
            ser.baudrate = FINAL_BAUD_RATE; time.sleep(0.1); ser.reset_input_buffer()
            print(f"Verifying data stream..."); bytes_received = ser.read(DATA_PACKET_TOTAL_SIZE * 5)
            if bytes_received and DATA_PACKET_START_MARKER.to_bytes(2, 'big') in bytes_received:
                print(f"Success! Board found on port: {port_name}"); return ser
            ser.close()
        except Exception:
            if ser and ser.is_open: ser.close()
    return None
def serial_read_loop(ser):
    if not ser: return
    buffer = bytearray(); start_marker = DATA_PACKET_START_MARKER.to_bytes(2, 'big'); end_marker = DATA_PACKET_END_MARKER.to_bytes(2, 'big')
    try:
        while True:
            data = ser.read(ser.in_waiting or 1);
            if not data: time.sleep(0.001); continue
            buffer.extend(data)
            while True:
                start_idx = buffer.find(start_marker)
                if start_idx == -1: break
                if len(buffer) < start_idx + DATA_PACKET_TOTAL_SIZE:
                    if start_idx > 0: buffer = buffer[start_idx:]
                    break
                potential_packet = buffer[start_idx : start_idx + DATA_PACKET_TOTAL_SIZE]
                if potential_packet.endswith(end_marker):
                    parse_data_packet(potential_packet)
                    buffer = buffer[start_idx + DATA_PACKET_TOTAL_SIZE:]
                else: buffer = buffer[start_idx + 1:]
    finally: ser.close()


# --- Pong Game Setup ---
app = Dash(__name__)
app.title = "BCI Pong"


def get_initial_game_state():
    return {
        'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2, 'ball_y': GAME_HEIGHT / 2,
        'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y, 'player_score': 0, 'ai_score': 0,
        # <<< CHANGE #2: New state machine for AI calibration >>>
        'game_mode': 'CALIBRATE_RELAX', 'calibration_start_time': None,
        'relax_readings': [], 'focus_readings': [],
    }


app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#DDDDDD', 'textAlign': 'center', 'fontFamily': 'monospace'}, children=[
    html.H1("BCI Pong"),
    html.P(id='instruction-text', children="Starting Calibration..."),
    dcc.Graph(id='pong-game-graph', config={'staticPlot': True}), # staticPlot is faster
    dcc.Interval(id='game-interval', interval=GAME_INTERVAL_MS, n_intervals=0),
    dcc.Store(id='game-state-store', data=get_initial_game_state()),
    html.Div(id='focus-metric-display')
])


# <<< CHANGE #3: The main game loop now includes an AI training phase >>>
@app.callback(
    Output('pong-game-graph', 'figure'), Output('game-state-store', 'data'),
    Output('focus-metric-display', 'children'), Output('instruction-text', 'children'),
    Input('game-interval', 'n_intervals'), State('game-state-store', 'data')
)
def update_game(n, state):
    global ml_model, ml_scaler # Use global model variables
    if n is None: raise PreventUpdate
    with buffer_lock: eeg_data = list(fft_buffers[FOCUS_CHANNEL])
    if len(eeg_data) < SAMPLING_RATE_HZ:
        fig = go.Figure(); fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', annotations=[dict(text="Waiting for EEG Data...", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=24, color='white'))])
        return fig, state, "Focus Metric: Waiting...", "Connecting to your brain..."


    y_data = np.array(eeg_data) - np.mean(eeg_data); N = len(y_data)
    yf = np.fft.fft(y_data); positive_freqs = np.fft.fftfreq(N, 1.0 / SAMPLING_RATE_HZ)[:N//2]
    amplitude = 2.0/N * np.abs(yf[:N//2])
    alpha_mask = (positive_freqs >= BRAINWAVE_BANDS['Alpha'][0]) & (positive_freqs < BRAINWAVE_BANDS['Alpha'][1])
    beta_mask = (positive_freqs >= BRAINWAVE_BANDS['Beta'][0]) & (positive_freqs < BRAINWAVE_BANDS['Beta'][1])
    alpha_power = np.mean(amplitude[alpha_mask]) if np.any(alpha_mask) else 0.0
    beta_power = np.mean(amplitude[beta_mask]) if np.any(beta_mask) else 0.0
    focus_metric = beta_power / alpha_power if alpha_power > 0.01 else 0


    current_time = n * GAME_INTERVAL_MS / 1000.0
    if state['calibration_start_time'] is None: state['calibration_start_time'] = current_time


    # --- STATE MACHINE ---
    if state['game_mode'] == 'CALIBRATE_RELAX' or state['game_mode'] == 'CALIBRATE_FOCUS':
        time_in_phase = current_time - state['calibration_start_time']
        seconds_left = max(0, CALIBRATION_SECONDS_PER_PHASE - time_in_phase)
        instruction = "RELAX. Clear your mind." if state['game_mode'] == 'CALIBRATE_RELAX' else "FOCUS. Do mental math."
        if focus_metric > 0:
            readings_key = 'relax_readings' if state['game_mode'] == 'CALIBRATE_RELAX' else 'focus_readings'
            state[readings_key].append(focus_metric)


        fig = go.Figure(); fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', annotations=[dict(text=f"{instruction}\n{int(seconds_left)}s left", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=30, color='white'))])
        focus_text = f"Current Raw Focus: {focus_metric:.2f}"


        if seconds_left == 0:
            if state['game_mode'] == 'CALIBRATE_RELAX':
                state['game_mode'] = 'CALIBRATE_FOCUS'; state['calibration_start_time'] = current_time
            else:
                state['game_mode'] = 'TRAINING_MODEL'
        return fig, state, focus_text, instruction


    elif state['game_mode'] == 'TRAINING_MODEL':
        print("Training AI model...")
        X_relax = np.array(state['relax_readings']).reshape(-1, 1)
        X_focus = np.array(state['focus_readings']).reshape(-1, 1)
        if len(X_relax) < 10 or len(X_focus) < 10:
            print("Calibration failed: not enough data. Restarting...")
            return go.Figure(), get_initial_game_state(), "Calibration failed", "Not enough data captured. Restarting calibration."


        X = np.vstack((X_relax, X_focus))
        y = np.array([-1] * len(X_relax) + [1] * len(X_focus))


        ml_scaler = StandardScaler().fit(X)
        X_scaled = ml_scaler.transform(X)
        ml_model = SVC(kernel='linear', C=1.0).fit(X_scaled, y)
       
        print("Model trained successfully.")
        state['game_mode'] = 'PLAYING'
        # Fall through to the 'PLAYING' state immediately
   
    if state['game_mode'] == 'PLAYING':
        focus_metric_history.append(focus_metric); smoothed_focus = np.mean(focus_metric_history)


        # Use the trained AI model for proportional control
        focus_scaled = ml_scaler.transform([[smoothed_focus]])
        control_signal = ml_model.decision_function(focus_scaled)[0]
        control_signal = np.clip(control_signal, -1.5, 1.5) # Clip to prevent extreme speeds


        state['player_x'] += PADDLE_SPEED * control_signal


        state['player_x'] = max(PADDLE_WIDTH / 2, min(GAME_WIDTH - PADDLE_WIDTH / 2, state['player_x']))


        # AI and Ball logic
        if state['ai_x'] < state['ball_x']: state['ai_x'] += AI_PADDLE_SPEED
        if state['ai_x'] > state['ball_x']: state['ai_x'] -= AI_PADDLE_SPEED
        state['ball_x'] += state['ball_vx']; state['ball_y'] += state['ball_vy']


        # Collision detection & Scoring
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


        # Draw the game
        fig = go.Figure()
        fig.add_shape(type="rect", x0=state['player_x']-PADDLE_WIDTH/2, y0=0, x1=state['player_x']+PADDLE_WIDTH/2, y1=PADDLE_HEIGHT, fillcolor="cyan", line=dict(width=0))
        fig.add_shape(type="rect", x0=state['ai_x']-PADDLE_WIDTH/2, y0=GAME_HEIGHT-PADDLE_HEIGHT, x1=state['ai_x']+PADDLE_WIDTH/2, y1=GAME_HEIGHT, fillcolor="magenta", line=dict(width=0))
        fig.add_shape(type="circle", x0=state['ball_x']-BALL_RADIUS, y0=state['ball_y']-BALL_RADIUS, x1=state['ball_x']+BALL_RADIUS, y1=state['ball_y']+BALL_RADIUS, fillcolor="white", line=dict(width=0))
        fig.add_shape(type="line", x0=0, y0=GAME_HEIGHT/2, x1=GAME_WIDTH, y1=GAME_HEIGHT/2, line=dict(color="grey", width=2, dash="dot"))
        fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False),
                          plot_bgcolor='#000', paper_bgcolor='#111', margin=dict(l=10, r=10, t=10, b=10),
                          annotations=[dict(text=str(state['player_score']), x=30, y=GAME_HEIGHT/2 - 30, showarrow=False, font=dict(size=40, color='cyan')),
                                       dict(text=str(state['ai_score']), x=30, y=GAME_HEIGHT/2 + 30, showarrow=False, font=dict(size=40, color='magenta'))])
       
        focus_text = f"Control Signal: {control_signal:.2f}"
        instruction_text = "Relax to move left. Focus to move right."
        return fig, state, focus_text, instruction_text


    return PreventUpdate


def main():
    serial_port_object = find_and_open_board()
    if serial_port_object:
        threading.Thread(target=serial_read_loop, args=(serial_port_object,), daemon=True).start()
        app.run(debug=True, use_reloader=False)
    else:
        print("Could not start application: No board was found.")


if __name__ == "__main__":
    main()

