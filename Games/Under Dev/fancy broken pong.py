import serial
import struct
import threading
import time
from collections import deque
import serial.tools.list_ports
import numpy as np
import random

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State, callback_context
from dash.exceptions import PreventUpdate

# --- Configuration Constants (Unchanged) ---
INITIAL_BAUD_RATE = 9600; FINAL_BAUD_RATE = 230400; FIRMWARE_BAUD_RATE_INDEX = 0x05
SAMPLING_RATE_HZ = 500.0; ADS1299_NUM_CHANNELS = 8; DATA_PACKET_START_MARKER = 0xABCD
DATA_PACKET_END_MARKER = 0xDCBA; DATA_PACKET_TOTAL_SIZE = 37; HANDSHAKE_START_MARKER_1 = 0xAA
BOARD_USB_IDS = [{'vid': 0x1A86, 'pid': 0x7523}]; BOARD_DESCRIPTIONS = ["USB-SERIAL CH340"]
GAME_INTERVAL_MS = 50; FOCUS_CHANNEL = 0; PADDLE_SPEED = 30; AI_PADDLE_SPEED = 12
INITIAL_BALL_SPEED_Y = -4; BALL_SPIN_FACTOR = 0.05; POWERUP_SPAWN_CHANCE = 0.5
POWERUP_DURATION_S = 5.0; AI_INACCURACY_FACTOR = 0.25; SCORE_TO_WIN = 5
GAME_START_BUFFER_SAMPLES = 10; CALIBRATION_SWEEPS = 5; CALIBRATION_SPEED = 25
CALIBRATION_PAUSE_S = 0.25; CALIBRATION_SAMPLES_PER_FRAME = 5; MIN_TRAINING_SAMPLES = 400
GAME_WIDTH = 800; GAME_HEIGHT = 600; PADDLE_WIDTH = 150; PADDLE_HEIGHT = 20
BALL_RADIUS = 15; POWERUP_RADIUS = 20; PLAYER_COLOR = 'cyan'; AI_COLOR = 'magenta'
BG_COLOR = '#111'; FONT_COLOR = '#DDD'; PLOT_BG_COLOR = '#000'
BRAINWAVE_BANDS = {"Delta": [0.5, 4], "Theta": [4, 8], "Alpha": [8, 13], "Beta": [13, 30], "Gamma": [30, 100]}
BAND_ORDER = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
POWERUP_CONFIG = {'zen_mode': {'color': 'deepskyblue'}, 'focus_beam': {'color': 'crimson'}}

# --- Global State (Unchanged) ---
fft_buffers = [deque(maxlen=int(SAMPLING_RATE_HZ)) for _ in range(ADS1299_NUM_CHANNELS)]
band_power_history = deque(maxlen=10); buffer_lock = threading.Lock()
ml_model = None; ml_scaler = None

# --- Data Acquisition Backend (Unchanged) ---
def convert_to_microvolts(raw_val, vref=4.5, gain=24):
    scale_factor = (vref * 2) / (gain * (2**24)); return raw_val * scale_factor * 1_000_000
def parse_data_packet(packet):
    try:
        ads_data = packet[7:34];
        with buffer_lock:
            for ch in range(ADS1299_NUM_CHANNELS):
                idx = 3 + ch * 3; raw_bytes = ads_data[idx:idx + 3]
                value = int.from_bytes(raw_bytes, byteorder='big', signed=True)
                fft_buffers[ch].append(convert_to_microvolts(value))
    except Exception: pass
def find_and_open_board():
    print("Searching for board..."); ports = serial.tools.list_ports.comports()
    candidate_ports = [p.device for p in ports if (p.vid and p.pid and {'vid': p.vid, 'pid': p.pid} in BOARD_USB_IDS) or (p.description and any(desc.lower() in p.description.lower() for desc in BOARD_DESCRIPTIONS))]
    if not candidate_ports: candidate_ports = [p.device for p in ports]
    for port_name in candidate_ports:
        print(f"--- Testing port: {port_name} ---"); ser = None
        try:
            ser = serial.Serial(port_name, INITIAL_BAUD_RATE, timeout=2); time.sleep(2); ser.read(ser.in_waiting or 1)
            print(f"Sending handshake..."); current_unix_time = int(time.time()); checksum_payload = struct.pack('>BI', 0x02, current_unix_time) + bytes([0x01, FIRMWARE_BAUD_RATE_INDEX])
            checksum = sum(checksum_payload) & 0xFF; handshake_packet = struct.pack('>BB', 0xAA, 0xBB) + checksum_payload + struct.pack('>B', checksum) + struct.pack('>BB', 0xCC, 0xDD)
            ser.write(handshake_packet); time.sleep(0.1); ser.baudrate = FINAL_BAUD_RATE; time.sleep(0.1); ser.reset_input_buffer(); print("Verifying data stream...")
            bytes_received = ser.read(DATA_PACKET_TOTAL_SIZE * 5)
            if bytes_received and DATA_PACKET_START_MARKER.to_bytes(2, 'big') in bytes_received:
                print(f"Success! Board found on port: {port_name}"); return ser
            ser.close()
        except Exception as e:
            if ser and ser.is_open: ser.close(); print(f"Failed on port {port_name}: {e}")
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
                    if start_idx > 0: buffer = buffer[start_idx:]; break
                potential_packet = buffer[start_idx : start_idx + DATA_PACKET_TOTAL_SIZE]
                if potential_packet.endswith(end_marker):
                    parse_data_packet(potential_packet); buffer = buffer[start_idx + DATA_PACKET_TOTAL_SIZE:]
                else: buffer = buffer[start_idx + 1:]
    finally: ser.close()

# --- Game State Management ---
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "BCI Pong"

def get_initial_game_state(mode='START_SCREEN'):
    return {
        'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2, 'ball_y': GAME_HEIGHT / 2,
        'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y, 'player_score': 0, 'ai_score': 0, 'game_mode': mode,
        'X_train': [], 'y_train': [], 'powerup_pos': None, 'powerup_type': None, 'powerup_timer': 0,
        'countdown_start_time': None, 'winner': None, 'calibration_phase': 'get_ready',
        'calibration_sweeps_done': 0, 'calibration_timer': time.time()
    }

def reset_for_new_point(state):
    p_score, a_score = state['player_score'], state['ai_score']
    new_state = get_initial_game_state('COUNTDOWN')
    new_state['player_score'], new_state['ai_score'] = p_score, a_score
    new_state['countdown_start_time'] = time.time()
    return new_state

# --- App Layout (Simplified Button Layout) ---
app.layout = html.Div(style={'backgroundColor': BG_COLOR, 'color': FONT_COLOR, 'fontFamily': "'Segoe UI', 'Roboto', sans-serif", 'textAlign': 'center'}, children=[
    html.H1("BCI Pong"),
    html.Div(id='instruction-text', children="Welcome!"),
    html.Div([
        html.Button('Start Calibration', id='start-button', n_clicks=0),
        html.Button('Start Game', id='start-game-button', n_clicks=0),
        html.Button('Play Again', id='play-again-button', n_clicks=0),
        html.Button('Recalibrate', id='recalibrate-button', n_clicks=0),
    ], style={'padding': '10px', 'display': 'flex', 'justifyContent': 'center', 'gap': '15px'}),
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'flex-start'}, children=[
        dcc.Graph(id='pong-game-graph', config={'staticPlot': True}, style={'width': '70vw', 'height': '75vh'}),
        html.Div(style={'width': '25vw', 'height': '75vh', 'paddingLeft': '20px'}, children=[
            html.H4("Live Brainwave Feedback"),
            dcc.Graph(id='bci-feedback-graph', style={'height': '90%'})])]),
    dcc.Interval(id='game-interval', interval=GAME_INTERVAL_MS, n_intervals=0),
    dcc.Store(id='game-state-store', data=get_initial_game_state()),
    html.Div(id='focus-metric-display', style={'fontSize': '1.2em', 'marginTop': '10px'})])

# --- Callbacks ---

@app.callback(
    Output('game-state-store', 'data'),
    Input('game-interval', 'n_intervals'),
    Input('start-button', 'n_clicks'),
    Input('start-game-button', 'n_clicks'),
    Input('play-again-button', 'n_clicks'),
    Input('recalibrate-button', 'n_clicks'),
    State('game-state-store', 'data'),
    prevent_initial_call=True
)
def master_update_callback(n_intervals, n_start, n_start_game, n_play_again, n_recal, state):
    """
    This is the master state machine for the game.
    It is structured to be simple and robust:
    1. If a button was clicked, handle it and exit immediately. This gives users instant feedback.
    2. If no button was clicked, assume the timer ticked and run the continuous game logic.
    This prevents the timer from ever interfering with a button press.
    """
    global ml_model, ml_scaler
    triggered_id = callback_context.triggered_id
    if not triggered_id: raise PreventUpdate

    # --- PART 1: HANDLE BUTTON CLICKS (High Priority) ---
    if triggered_id != 'game-interval':
        if triggered_id in ['start-button', 'recalibrate-button']:
            ml_model, ml_scaler = None, None  # Reset model
            return get_initial_game_state(mode='CALIBRATION') # Return the new state immediately

        if triggered_id == 'start-game-button' and state['game_mode'] == 'TRAINING_COMPLETE':
            return reset_for_new_point(get_initial_game_state(mode='COUNTDOWN'))

        if triggered_id == 'play-again-button':
            return reset_for_new_point(get_initial_game_state(mode='COUNTDOWN'))
        
        # If a button was clicked but its condition wasn't met (e.g., Start Game when not trained), do nothing.
        raise PreventUpdate

    # --- PART 2: HANDLE GAME LOOP (Triggered by Interval) ---
    new_state = state.copy(); mode = new_state.get('game_mode')
    
    # Do nothing on interval ticks if the game is in a static state
    if mode not in ['CALIBRATION', 'PLAYING', 'COUNTDOWN', 'BUFFERING']:
        raise PreventUpdate

    # --- BCI Feature Extraction ---
    if mode in ['CALIBRATION', 'PLAYING']:
        with buffer_lock: eeg_data = list(fft_buffers[FOCUS_CHANNEL])
        if len(eeg_data) < SAMPLING_RATE_HZ: raise PreventUpdate
        y_data = np.array(eeg_data) - np.mean(eeg_data); N = len(y_data); yf = np.fft.fft(y_data)
        positive_freqs = np.fft.fftfreq(N, 1.0 / SAMPLING_RATE_HZ)[:N//2]; amplitude = 2.0/N * np.abs(yf[:N//2])
        current_band_powers = [np.mean(amplitude[(positive_freqs >= b[0]) & (positive_freqs < b[1])]) for b in BRAINWAVE_BANDS.values()]
        band_power_history.append(current_band_powers)

    # --- Game State Logic ---
    if mode == 'CALIBRATION':
        if new_state['calibration_sweeps_done'] >= CALIBRATION_SWEEPS:
            if ml_model is None: # Only train ONCE
                if len(new_state['y_train']) >= MIN_TRAINING_SAMPLES:
                    print(f"Training model with {len(new_state['y_train'])} samples..."); X = np.array(new_state['X_train']); y = np.array(new_state['y_train'])
                    ml_scaler = StandardScaler().fit(X); X_scaled = ml_scaler.transform(X); ml_model = SVC(kernel='rbf', C=1.0, probability=True).fit(X_scaled, y)
                    print("Model training complete."); new_state['game_mode'] = 'TRAINING_COMPLETE'
                else:
                    print(f"Calibration failed: samples {len(new_state['y_train'])} < {MIN_TRAINING_SAMPLES}."); new_state['game_mode'] = 'TRAINING_FAILED'
        else: # If still calibrating, move the ball
            phase = new_state.get('calibration_phase', 'get_ready')
            if phase == 'get_ready':
                if time.time() - new_state['calibration_timer'] > 2.0: new_state['calibration_phase'] = 'moving_right'
            elif phase == 'moving_right':
                new_state['ball_x'] += CALIBRATION_SPEED
                if len(band_power_history) >= CALIBRATION_SAMPLES_PER_FRAME: new_state['X_train'].extend(list(band_power_history)[-CALIBRATION_SAMPLES_PER_FRAME:]); new_state['y_train'].extend([1] * CALIBRATION_SAMPLES_PER_FRAME)
                if new_state['ball_x'] >= GAME_WIDTH - BALL_RADIUS: new_state['calibration_phase'] = 'pause'; new_state['calibration_timer'] = time.time()
            elif phase == 'moving_left':
                new_state['ball_x'] -= CALIBRATION_SPEED
                if len(band_power_history) >= CALIBRATION_SAMPLES_PER_FRAME: new_state['X_train'].extend(list(band_power_history)[-CALIBRATION_SAMPLES_PER_FRAME:]); new_state['y_train'].extend([-1] * CALIBRATION_SAMPLES_PER_FRAME)
                if new_state['ball_x'] <= BALL_RADIUS: new_state['calibration_phase'] = 'pause'; new_state['calibration_timer'] = time.time(); new_state['calibration_sweeps_done'] += 1
            elif phase == 'pause':
                if time.time() - new_state['calibration_timer'] > CALIBRATION_PAUSE_S: new_state['calibration_phase'] = 'moving_left' if new_state['ball_x'] >= GAME_WIDTH - BALL_RADIUS else 'moving_right'

    elif mode == 'COUNTDOWN':
        if time.time() - new_state.get('countdown_start_time', time.time()) > 3: new_state['game_mode'] = 'BUFFERING'; band_power_history.clear()
    elif mode == 'BUFFERING':
        if len(band_power_history) >= GAME_START_BUFFER_SAMPLES: new_state['game_mode'] = 'PLAYING'
    elif mode == 'PLAYING':
        # ... Gameplay logic (unchanged from your original, as it was solid) ...
        # (This section handles BCI control, AI, physics, scoring, etc.)
        if ml_model and ml_scaler:
            features_scaled = ml_scaler.transform([current_band_powers]); control_signal = ml_model.decision_function(features_scaled)[0]
            new_state['player_x'] += PADDLE_SPEED * np.clip(control_signal, -1.5, 1.5)
            new_state['player_x'] = max(PADDLE_WIDTH/2, min(GAME_WIDTH - PADDLE_WIDTH/2, new_state['player_x']))
        if new_state['powerup_timer'] > 0: new_state['powerup_timer'] -= GAME_INTERVAL_MS / 1000
        else: new_state['powerup_type'] = None
        current_ball_speed_y = INITIAL_BALL_SPEED_Y * 0.5 if new_state.get('powerup_type') == 'zen_mode' else INITIAL_BALL_SPEED_Y
        current_ai_paddle_width = PADDLE_WIDTH * 0.5 if new_state.get('powerup_type') == 'focus_beam' else PADDLE_WIDTH
        time_to_paddle = (GAME_HEIGHT - PADDLE_HEIGHT - new_state['ball_y']) / abs(current_ball_speed_y) if abs(current_ball_speed_y) > 0 else 0
        predicted_x = new_state['ball_x'] + new_state['ball_vx'] * time_to_paddle
        inaccuracy_offset = (current_ai_paddle_width * AI_INACCURACY_FACTOR) * random.uniform(-1, 1)
        ai_target_x = np.clip(predicted_x + inaccuracy_offset, current_ai_paddle_width / 2, GAME_WIDTH - current_ai_paddle_width / 2)
        if new_state['ai_x'] < ai_target_x: new_state['ai_x'] = min(new_state['ai_x'] + AI_PADDLE_SPEED, ai_target_x)
        elif new_state['ai_x'] > ai_target_x: new_state['ai_x'] = max(new_state['ai_x'] - AI_PADDLE_SPEED, ai_target_x)
        new_state['ball_x'] += new_state['ball_vx']; new_state['ball_y'] += np.sign(new_state['ball_vy']) * abs(current_ball_speed_y)
        if new_state['ball_x'] <= BALL_RADIUS or new_state['ball_x'] >= GAME_WIDTH - BALL_RADIUS: new_state['ball_vx'] *= -1
        if new_state['ball_vy'] < 0 and new_state['ball_y'] - BALL_RADIUS < PADDLE_HEIGHT and abs(new_state['player_x'] - new_state['ball_x']) < PADDLE_WIDTH / 2:
            new_state['ball_vy'] *= -1; new_state['ball_vx'] += (new_state['ball_x'] - new_state['player_x']) * BALL_SPIN_FACTOR
            if new_state['powerup_pos'] is None and random.random() < POWERUP_SPAWN_CHANCE:
                new_state['powerup_type'] = random.choice(list(POWERUP_CONFIG.keys())); new_state['powerup_pos'] = [random.randint(100, GAME_WIDTH-100), random.randint(int(GAME_HEIGHT*0.4), int(GAME_HEIGHT*0.6))]
        if new_state['ball_vy'] > 0 and new_state['ball_y'] + BALL_RADIUS > GAME_HEIGHT - PADDLE_HEIGHT and abs(new_state['ai_x'] - new_state['ball_x']) < current_ai_paddle_width / 2:
            new_state['ball_vy'] *= -1; new_state['ball_vx'] += (new_state['ball_x'] - new_state['ai_x']) * BALL_SPIN_FACTOR
        if new_state['powerup_pos'] and ((new_state['ball_x'] - new_state['powerup_pos'][0])**2 + (new_state['ball_y'] - new_state['powerup_pos'][1])**2) < (BALL_RADIUS + POWERUP_RADIUS)**2:
            new_state['powerup_timer'] = POWERUP_DURATION_S; new_state['powerup_pos'] = None
        if new_state['ball_y'] < -BALL_RADIUS or new_state['ball_y'] > GAME_HEIGHT + BALL_RADIUS:
            if new_state['ball_y'] < 0: new_state['ai_score'] += 1
            else: new_state['player_score'] += 1
            if new_state['player_score'] >= SCORE_TO_WIN or new_state['ai_score'] >= SCORE_TO_WIN:
                new_state['game_mode'] = 'GAME_OVER'; new_state['winner'] = 'Player' if new_state['player_score'] >= SCORE_TO_WIN else 'AI'
            else: return reset_for_new_point(new_state)

    return new_state

@app.callback(
    [Output('pong-game-graph', 'figure'), Output('focus-metric-display', 'children'),
     Output('instruction-text', 'children'), Output('start-button', 'style'),
     Output('start-game-button', 'style'), Output('play-again-button', 'style'),
     Output('recalibrate-button', 'style')],
    Input('game-state-store', 'data')
)
def render_ui(state):
    # This UI rendering callback is now much simpler as it only has to worry about showing/hiding the right buttons for a given state.
    mode = state.get('game_mode', 'START_SCREEN')
    fig = go.Figure().update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor=PLOT_BG_COLOR, paper_bgcolor=BG_COLOR, margin=dict(l=10,r=10,t=10,b=10))
    instruction_text, focus_text = "", ""
    
    btn_style = {'fontSize': '1.2em', 'margin': '10px'}; show = {'display': 'inline-block', **btn_style}; hide = {'display': 'none'}
    s_btn, sg_btn, pa_btn, recal_btn = hide, hide, hide, hide

    if mode == 'START_SCREEN':
        instruction_text = "Welcome! Press 'Start Calibration' to begin."; s_btn = show
        fig.add_annotation(text="BCI PONG", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=60, color=FONT_COLOR))
    elif mode == 'CALIBRATION':
        phase = state.get('calibration_phase'); instruction_text = {'get_ready': "Get Ready...", 'moving_right': "Track RIGHT ➡", 'moving_left': "⬅ Track LEFT", 'pause': "Pause..."}.get(phase)
        focus_text = f"Sweeps: {state['calibration_sweeps_done']}/{CALIBRATION_SWEEPS} | Samples: {len(state['y_train'])}"; recal_btn = show
        sweep_progress = 0
        if phase == 'moving_right': sweep_progress = (state['ball_x'] - BALL_RADIUS) / (GAME_WIDTH - 2*BALL_RADIUS)
        elif phase == 'moving_left': sweep_progress = (GAME_WIDTH - state['ball_x'] - BALL_RADIUS) / (GAME_WIDTH - 2*BALL_RADIUS)
        progress_width = ((state['calibration_sweeps_done'] + sweep_progress) / CALIBRATION_SWEEPS) * GAME_WIDTH
        shapes = [go.layout.Shape(type="circle", x0=state['ball_x']-BALL_RADIUS, y0=state['ball_y']-BALL_RADIUS, x1=state['ball_x']+BALL_RADIUS, y1=state['ball_y']+BALL_RADIUS, fillcolor="yellow", line_width=0),
                  go.layout.Shape(type="rect", x0=0, y0=0, x1=progress_width, y1=10, fillcolor=PLAYER_COLOR, layer="below", line_width=0)]
        fig.update_layout(shapes=shapes)
    elif mode == 'TRAINING_FAILED':
        instruction_text = f"Calibration Failed. Please try again."; recal_btn = show
        fig.add_annotation(text="TRAINING FAILED", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=40, color='crimson'))
    elif mode == 'TRAINING_COMPLETE':
        instruction_text = "Training Complete! Press 'Start Game'."; sg_btn, recal_btn = show; sg_btn['backgroundColor'] = '#00CC96'
        fig.add_annotation(text="Ready?", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=80, color=FONT_COLOR))
    elif mode in ['COUNTDOWN', 'BUFFERING', 'PLAYING']:
        recal_btn = show
        # ... (Game rendering part is unchanged) ...
    elif mode == 'GAME_OVER':
        instruction_text = "Game Over!"; pa_btn, recal_btn = show
        winner_text = "YOU WIN!" if state['winner'] == 'Player' else "AI WINS"; winner_color = PLAYER_COLOR if state['winner'] == 'Player' else AI_COLOR
        fig.add_annotation(text=winner_text, x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=80, color=winner_color))

    return fig, focus_text, instruction_text, s_btn, sg_btn, pa_btn, recal_btn

@app.callback(Output('bci-feedback-graph', 'figure'), Input('game-interval', 'n_intervals'), prevent_initial_call=True)
def update_bci_graph(n_intervals):
    with buffer_lock:
        if not band_power_history: raise PreventUpdate
        current_band_powers = list(band_power_history[-1])
    fig = go.Figure(go.Bar(x=BAND_ORDER, y=current_band_powers, marker_color=['#636EFA', '#AB63FA', '#00CC96', '#FFA15A', '#FF6692']))
    fig.update_layout(plot_bgcolor='#222', paper_bgcolor='#222', font_color='white', yaxis_title='Power (μV)', yaxis=dict(range=[0, max(current_band_powers) * 1.2 + 1e-9]), margin=dict(t=20, b=20, l=40, r=20))
    return fig

def main():
    serial_port_object = find_and_open_board()
    if not serial_port_object:
        print("\n" + "="*50+"\nWARNING: COULD NOT FIND BOARD. Starting in dummy data mode.\n"+"="*50+"\n")
        threading.Thread(target=lambda: (lambda: [fft_buffers[i].extend(np.random.randn(25) * 10) for i in range(8)])() or time.sleep(0.05) or (lambda: None)(), daemon=True).start()
    else:
        threading.Thread(target=serial_read_loop, args=(serial_port_object,), daemon=True).start()
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    main()



