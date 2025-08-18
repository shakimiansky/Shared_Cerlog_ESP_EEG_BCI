import serial
import struct
import threading
import time
from collections import deque
import serial.tools.list_ports
import numpy as np
from dash.exceptions import PreventUpdate
import dash
import logging

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State

# --- Serial/Data Acquisition Configuration ---
INITIAL_BAUD_RATE = 9600; FINAL_BAUD_RATE = 115200; FIRMWARE_BAUD_RATE_INDEX = 0x04
SAMPLING_RATE_HZ = 250.0; STREAM_TIMEOUT_SECONDS = 5.0
DATA_PACKET_START_MARKER = 0xABCD; DATA_PACKET_END_MARKER = 0xDCBA; DATA_PACKET_TOTAL_SIZE = 37
PACKET_IDX_LENGTH = 2; PACKET_IDX_CHECKSUM = 34
ADS1299_NUM_CHANNELS = 8; ADS1299_NUM_STATUS_BYTES = 3; ADS1299_BYTES_PER_CHANNEL = 3
BOARD_USB_IDS = [{'vid': 0x1A86, 'pid': 0x7523}]; BOARD_DESCRIPTIONS = ["USB-SERIAL CH340", "CH340"]

# --- BCI Configuration ---
SSVEP_FREQ_LEFT = 12.0; SSVEP_FREQ_RIGHT = 17.0
CALIBRATION_SECONDS_PER_PHASE = 8; FEATURE_SLICE_HZ = 1.0
CONTROL_SMOOTHING_WINDOW = 8; PADDLE_SPEED = 40

# --- Visual Stimulus Configuration ---
GRADIENT_X_RESOLUTION = 100; GRADIENT_Y_RESOLUTION = 50
FLICKER_INTENSITY = 0.8
GRADIENT_COLOR_MIN = 'hsl(220, 25%, 15%)'; GRADIENT_COLOR_MAX = 'hsl(190, 100%, 95%)'

# --- Game Configuration ---
GAME_INTERVAL_MS = 50; AI_PADDLE_SPEED = 9; INITIAL_BALL_SPEED_Y = -4.5
BALL_SPIN_FACTOR = 0.05; GAME_WIDTH = 800; GAME_HEIGHT = 600
PADDLE_WIDTH = 150; PADDLE_HEIGHT = 20; BALL_RADIUS = 10

# --- Data Buffers & Global ML Models ---
fft_buffers = [deque(maxlen=int(SAMPLING_RATE_HZ * 2)) for _ in range(ADS1299_NUM_CHANNELS)]
control_signal_history = deque(maxlen=CONTROL_SMOOTHING_WINDOW)
buffer_lock = threading.Lock(); ml_model = None; ml_scaler = None

# --- Data Acquisition Backend ---
def convert_to_microvolts(raw_val, vref=4.5, gain=24):
    scale_factor = (vref * 2) / (gain * (2**24)); return raw_val * scale_factor * 1_000_000
def parse_data_packet(packet):
    try:
        ads_data = packet[7:34]
        with buffer_lock:
            for ch in range(ADS1299_NUM_CHANNELS):
                idx = ADS1299_NUM_STATUS_BYTES + ch * ADS1299_BYTES_PER_CHANNEL
                raw_bytes = ads_data[idx:idx + 3]
                value = int.from_bytes(raw_bytes, byteorder='big', signed=True)
                fft_buffers[ch].append(convert_to_microvolts(value))
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
            time.sleep(5); ser.read(ser.in_waiting or 1)
            print(f"Sending handshake for {FINAL_BAUD_RATE} bps...")
            current_unix_time = int(time.time())
            checksum_payload = struct.pack('>BI', 0x02, current_unix_time) + bytes([0x01, FIRMWARE_BAUD_RATE_INDEX])
            checksum = sum(checksum_payload) & 0xFF
            handshake_packet = struct.pack('>BB', 0xAA, 0xBB) + checksum_payload + struct.pack('>B', checksum) + struct.pack('>BB', 0xCC, 0xDD)
            ser.write(handshake_packet); time.sleep(0.1)
            ser.baudrate = FINAL_BAUD_RATE; time.sleep(0.5); ser.reset_input_buffer()
            bytes_received = ser.read(DATA_PACKET_TOTAL_SIZE * 5)
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
            buffer.extend(ser.read(ser.in_waiting or 1));
            while True:
                start_idx = buffer.find(start_marker)
                if start_idx == -1: break
                if len(buffer) < start_idx + DATA_PACKET_TOTAL_SIZE: break
                packet = buffer[start_idx : start_idx + DATA_PACKET_TOTAL_SIZE]
                if packet.endswith(end_marker):
                    payload = packet[PACKET_IDX_LENGTH:PACKET_IDX_CHECKSUM]
                    if (sum(payload) & 0xFF) == packet[PACKET_IDX_CHECKSUM]: parse_data_packet(packet)
                    buffer = buffer[start_idx + DATA_PACKET_TOTAL_SIZE:]
                else: buffer = buffer[start_idx + 1:]
    finally: ser.close()

# --- Pong Game Setup ---
app = Dash(__name__); app.title = "ML-Enhanced SSVEP BCI Pong"
def get_initial_game_state():
    return { 'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2, 'ball_y': GAME_HEIGHT / 2,
             'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y, 'player_score': 0, 'ai_score': 0,
             'game_mode': 'CALIBRATE_LEFT', 'calibration_start_time': None,
             'left_readings': [], 'right_readings': [], 'neutral_readings': [] }
app.layout = html.Div(style={'backgroundColor': '#111', 'color': '#DDD', 'textAlign': 'center', 'fontFamily': 'monospace'}, children=[
    html.H1("ML-Enhanced SSVEP BCI Pong"), html.P(id='instruction-text'),
    dcc.Graph(id='pong-game-graph', config={'staticPlot': True}),
    dcc.Interval(id='game-interval', interval=GAME_INTERVAL_MS, n_intervals=0, disabled=False),
    dcc.Store(id='game-state-store', data=get_initial_game_state()), html.Div(id='focus-metric-display')])

# --- Helper functions for drawing ---
def generate_flicker_stimulus(flicker_mode, current_time):
    if flicker_mode == 'none': return None
    flicker_val_left = 0.5 + (np.sin(2 * np.pi * SSVEP_FREQ_LEFT * current_time) * (FLICKER_INTENSITY / 2))
    flicker_val_right = 0.5 + (np.sin(2 * np.pi * SSVEP_FREQ_RIGHT * current_time) * (FLICKER_INTENSITY / 2))
    if flicker_mode == 'left': left_weights = np.ones(GRADIENT_X_RESOLUTION)
    elif flicker_mode == 'right': left_weights = np.zeros(GRADIENT_X_RESOLUTION)
    else: left_weights = np.linspace(1, 0, GRADIENT_X_RESOLUTION)
    gradient_row = (flicker_val_left * left_weights) + (flicker_val_right * (1 - left_weights))
    gradient_z = np.tile(gradient_row, (GRADIENT_Y_RESOLUTION, 1))
    return go.Heatmap(z=gradient_z, x=np.linspace(0, GAME_WIDTH, GRADIENT_X_RESOLUTION), y=np.linspace(0, GAME_HEIGHT, GRADIENT_Y_RESOLUTION),
                      colorscale=[[0, GRADIENT_COLOR_MIN], [1, GRADIENT_COLOR_MAX]], showscale=False, zmin=0, zmax=1)
def draw_game_elements(fig, state):
    fig.add_shape(type="rect", x0=state['player_x']-PADDLE_WIDTH/2, y0=0, x1=state['player_x']+PADDLE_WIDTH/2, y1=PADDLE_HEIGHT, fillcolor="cyan", line=dict(width=0))
    fig.add_shape(type="rect", x0=state['ai_x']-PADDLE_WIDTH/2, y0=GAME_HEIGHT-PADDLE_HEIGHT, x1=state['ai_x']+PADDLE_WIDTH/2, y1=GAME_HEIGHT, fillcolor="magenta", line=dict(width=0))
    fig.add_shape(type="circle", x0=state['ball_x']-BALL_RADIUS, y0=state['ball_y']-BALL_RADIUS, x1=state['ball_x']+BALL_RADIUS, y1=state['ball_y']+BALL_RADIUS, fillcolor="white", line=dict(width=0))
    fig.update_layout(annotations=[dict(text=str(state['player_score']), x=30, y=GAME_HEIGHT/2 - 30, showarrow=False, font=dict(size=40, color='cyan')),
                                   dict(text=str(state['ai_score']), x=30, y=GAME_HEIGHT/2 + 30, showarrow=False, font=dict(size=40, color='magenta'))])

@app.callback(
    Output('pong-game-graph', 'figure'), Output('game-state-store', 'data'),
    Output('focus-metric-display', 'children'), Output('instruction-text', 'children'),
    Output('game-interval', 'disabled'),
    Input('game-interval', 'n_intervals'), State('game-state-store', 'data')
)
def update_game(n, state):
    global ml_model, ml_scaler
    if n is None: raise PreventUpdate
    if 'game_mode' not in state:
        print("Invalid state from browser. Forcing reset."); state = get_initial_game_state()

    with buffer_lock:
        if any(len(buf) < SAMPLING_RATE_HZ * 2 for buf in fft_buffers):
            fig = go.Figure(); fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', annotations=[dict(text="Waiting for EEG Data...", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=24, color='white'))])
            return fig, state, "Status: Waiting for data...", "Connecting to your brain...", False

    feature_vector = []
    with buffer_lock:
        for ch in range(ADS1299_NUM_CHANNELS):
            y_data = np.array(list(fft_buffers[ch])) - np.mean(list(fft_buffers[ch])); N = len(y_data)
            win = np.hanning(N); y_win = y_data * win
            yf = np.fft.fft(y_win); xf = np.fft.fftfreq(N, 1.0 / SAMPLING_RATE_HZ)[:N//2]
            psd = (2 / (SAMPLING_RATE_HZ * np.sum(win**2))) * np.abs(yf[0:N//2])**2
            left_mask = (xf >= SSVEP_FREQ_LEFT - FEATURE_SLICE_HZ/2) & (xf < SSVEP_FREQ_LEFT + FEATURE_SLICE_HZ/2)
            right_mask = (xf >= SSVEP_FREQ_RIGHT - FEATURE_SLICE_HZ/2) & (xf < SSVEP_FREQ_RIGHT + FEATURE_SLICE_HZ/2)
            feature_vector.extend(psd[left_mask]); feature_vector.extend(psd[right_mask])
    if len(feature_vector) == 0: raise PreventUpdate

    current_time = n * GAME_INTERVAL_MS / 1000.0
    fig = go.Figure(); fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', margin=dict(l=10, r=10, t=10, b=10))

    if state['game_mode'].startswith('CALIBRATE'):
        if state['calibration_start_time'] is None: state['calibration_start_time'] = current_time
        time_in_phase = current_time - state['calibration_start_time']
        seconds_left = max(0, CALIBRATION_SECONDS_PER_PHASE - time_in_phase)
        flicker_mode, instruction = None, ""
        if state['game_mode'] == 'CALIBRATE_LEFT':
            flicker_mode, instruction = 'left', "Stare at the flickering screen."; state['left_readings'].append(feature_vector)
        elif state['game_mode'] == 'CALIBRATE_RIGHT':
            flicker_mode, instruction = 'right', "Stare at the flickering screen."; state['right_readings'].append(feature_vector)
        elif state['game_mode'] == 'CALIBRATE_NEUTRAL':
            flicker_mode, instruction = 'none', "Look at the ball and relax."; state['neutral_readings'].append(feature_vector)
            draw_game_elements(fig, state)
        stimulus = generate_flicker_stimulus(flicker_mode, current_time)
        if stimulus: fig.add_trace(stimulus)
        fig.add_annotation(text=f"CALIBRATION: {instruction}\n{int(seconds_left)}s", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=24, color='white'))
        if seconds_left == 0:
            state['calibration_start_time'] = current_time
            if state['game_mode'] == 'CALIBRATE_LEFT': state['game_mode'] = 'CALIBRATE_RIGHT'
            elif state['game_mode'] == 'CALIBRATE_RIGHT': state['game_mode'] = 'CALIBRATE_NEUTRAL'
            elif state['game_mode'] == 'CALIBRATE_NEUTRAL': state['game_mode'] = 'TRAINING_MODEL'
        return fig, state, "Status: Calibrating...", instruction, False

    elif state['game_mode'] == 'TRAINING_MODEL':
        X_left, X_right, X_neutral = (np.array(state[r]) for r in ['left_readings', 'right_readings', 'neutral_readings'])
        if len(X_left)<10 or len(X_right)<10 or len(X_neutral)<10: return fig, get_initial_game_state(), "Calibration Failed", "Restarting.", False
        y_left, y_right, y_neutral = (np.full(len(X), lab) for X, lab in [(X_left, -1), (X_right, 1), (X_neutral, 0)])
        X = np.vstack((X_left, X_right, X_neutral)); y = np.concatenate((y_left, y_right, y_neutral))
        print(f"Training model with feature matrix of shape: {X.shape}")
        ml_scaler = StandardScaler().fit(X); X_scaled = ml_scaler.transform(X)
        ml_model = SVC(kernel='rbf', C=1.0, probability=True, decision_function_shape='ovr').fit(X_scaled, y)
        state['game_mode'] = 'GET_READY'; state['calibration_start_time'] = current_time
        print("Training complete.")
        state['left_readings']=[]; state['right_readings']=[]; state['neutral_readings']=[]
        fig.add_annotation(text="Training Complete!", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=24, color='white'))
        return fig, state, "Status: Ready", "Get ready to play!", True

    elif state['game_mode'] == 'GET_READY':
        time_in_phase = current_time - state['calibration_start_time']
        interval_disabled = True
        if time_in_phase > 2:
            state['game_mode'] = 'PLAYING'; interval_disabled = False
        draw_game_elements(fig, state)
        fig.add_annotation(text=f"Get Ready... {max(0, 2-int(time_in_phase))}", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=30, color='white'))
        return fig, state, "Status: Ready", "Get ready to play!", interval_disabled

    if state['game_mode'] == 'PLAYING':
        if ml_model is None: return fig, get_initial_game_state(), "Error", "Server restarted. Recalibrating.", False
        feature_vector_scaled = ml_scaler.transform([feature_vector])
        scores = ml_model.decision_function(feature_vector_scaled)[0]
        control_signal = scores[1] - scores[0] 
        control_signal_history.append(control_signal); smoothed_signal = np.mean(control_signal_history)
        state['player_x'] += PADDLE_SPEED * smoothed_signal
        state['player_x'] = max(PADDLE_WIDTH / 2, min(GAME_WIDTH - PADDLE_WIDTH / 2, state['player_x']))
        state['ai_x'] += AI_PADDLE_SPEED if state['ai_x'] < state['ball_x'] else -AI_PADDLE_SPEED
        state['ball_x'] += state['ball_vx']; state['ball_y'] += state['ball_vy']
        if state['ball_x'] <= BALL_RADIUS or state['ball_x'] >= GAME_WIDTH-BALL_RADIUS: state['ball_vx'] *= -1
        
        # <<< THIS IS THE CORRECTED LINE WITH THE PARENTHESIS CLOSED >>>
        if state['ball_vy'] < 0 and abs(state['player_x']-state['ball_x']) < PADDLE_WIDTH/2 and state['ball_y']-BALL_RADIUS < PADDLE_HEIGHT:
            state['ball_vy'] *= -1; state['ball_vx'] += (state['ball_x'] - state['player_x']) * BALL_SPIN_FACTOR
        if state['ball_vy'] > 0 and abs(state['ai_x']-state['ball_x']) < PADDLE_WIDTH/2 and state['ball_y']+BALL_RADIUS > GAME_HEIGHT-PADDLE_HEIGHT:
            state['ball_vy'] *= -1; state['ball_vx'] += (state['ball_x'] - state['ai_x']) * BALL_SPIN_FACTOR
            
        if state['ball_y'] < -BALL_RADIUS or state['ball_y'] > GAME_HEIGHT+BALL_RADIUS:
            if state['ball_y'] < 0: state['ai_score'] += 1
            else: state['player_score'] += 1
            p_score, a_score = state['player_score'], state['ai_score']
            state = get_initial_game_state()
            state['player_score'] = p_score; state['ai_score'] = a_score
            state['game_mode'] = 'PLAYING'
        stimulus = generate_flicker_stimulus('both', current_time)
        if stimulus: fig.add_trace(stimulus)
        draw_game_elements(fig, state)
        return fig, state, f"Control Signal: {smoothed_signal:.2f}", "Look Left/Right to move.", False

    raise PreventUpdate

def main():
    serial_port_object = find_and_open_board()
    if serial_port_object:
        threading.Thread(target=serial_read_loop, args=(serial_port_object,), daemon=True).start()
        log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
        print("\nDash server is running. Open http://127.0.0.1:8050/ in your browser.")
        app.run(debug=False, use_reloader=False)
    else:
        print("Could not start application: No board was found.")

if __name__ == "__main__":
    main()