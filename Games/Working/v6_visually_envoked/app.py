import serial
import struct
import threading
import time
from collections import deque
import serial.tools.list_ports
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State, no_update, clientside_callback, ctx
from dash.exceptions import PreventUpdate
import logging

# --- Signal Processing & Machine Learning Libraries ---
from scipy.signal import butter, lfilter, detrend
from sklearn.cross_decomposition import CCA as SklearnCCA

# ==============================================================================
# === 1. TUNABLE CONFIGURATION ===============================================
# ==============================================================================
# --- Game Feel & Speed Adjustments ---
PADDLE_SPEED = 30
INITIAL_BALL_SPEED_Y = -4

# --- BCI & Signal Processing ---
CHANNELS_TO_USE = [1, 2, 3]
SSVEP_FREQ_LEFT = 7.5
SSVEP_FREQ_RIGHT = 12.0
FFT_WINDOW_SECONDS = 1.5
FFT_OVERLAP_PERCENT = 0.8
USE_EMA_SMOOTHING = True
EMA_SMOOTHING_FACTOR = 0.4
BCI_SCORE_AMPLIFIER = 2.5
FILTER_LOW_CUT_HZ = 5.0; FILTER_HIGH_CUT_HZ = 45.0; FILTER_ORDER = 5
CCA_NUM_HARMONICS = 3
CALIBRATION_DURATION_S = 7
CALIBRATION_THRESHOLD_STD_FACTOR = 0.8
MIN_THRESHOLD_GAP = 0.05
GAME_INTERVAL_MS = 16
AI_PADDLE_SPEED = 6
BALL_SPIN_FACTOR = 0.06
GAME_WIDTH = 800; GAME_HEIGHT = 600; PADDLE_WIDTH = 150; PADDLE_HEIGHT = 20; BALL_RADIUS = 10

# ==============================================================================
# === 2. CORE SETUP ============================================================
# ==============================================================================
INITIAL_BAUD_RATE = 9600; FINAL_BAUD_RATE = 115200; FIRMWARE_BAUD_RATE_INDEX = 0x04
SAMPLING_RATE_HZ = 250.0; STREAM_TIMEOUT_SECONDS = 5.0; DATA_PACKET_START_MARKER = 0xABCD
DATA_PACKET_END_MARKER = 0xDCBA; DATA_PACKET_TOTAL_SIZE = 37; PACKET_IDX_LENGTH = 2
PACKET_IDX_CHECKSUM = 34; ADS1299_NUM_CHANNELS = 8; ADS1299_NUM_STATUS_BYTES = 3
ADS1299_BYTES_PER_CHANNEL = 3; BOARD_USB_IDS = [{'vid': 0x1A86, 'pid': 0x7523}]
BOARD_DESCRIPTIONS = ["USB-SERIAL CH340", "CH340"]

FFT_MAXLEN = int(SAMPLING_RATE_HZ * FFT_WINDOW_SECONDS)
BCI_UPDATE_INTERVAL_MS = int((FFT_WINDOW_SECONDS * (1 - FFT_OVERLAP_PERCENT)) * 1000)
CHANNELS_IDX = [c - 1 for c in CHANNELS_TO_USE]
data_buffers = [deque(maxlen=FFT_MAXLEN) for _ in range(ADS1299_NUM_CHANNELS)]
buffer_lock = threading.Lock()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs; low = lowcut / nyq; high = highcut / nyq
    b, a = butter(order, [low, high], btype='band'); return b, a

bandpass_b, bandpass_a = butter_bandpass(FILTER_LOW_CUT_HZ, FILTER_HIGH_CUT_HZ, SAMPLING_RATE_HZ, order=FILTER_ORDER)

def preprocess_eeg_window(eeg_data):
    if eeg_data.ndim == 1: eeg_data = eeg_data.reshape(-1, 1)
    eeg_detrended = detrend(eeg_data, axis=0)
    eeg_filtered = lfilter(bandpass_b, bandpass_a, eeg_detrended, axis=0)
    return eeg_filtered

time_points = np.arange(0, FFT_WINDOW_SECONDS, 1.0 / SAMPLING_RATE_HZ)[:FFT_MAXLEN]
CCA_REFERENCE_SIGNALS = {}
for freq in [SSVEP_FREQ_LEFT, SSVEP_FREQ_RIGHT]:
    refs = []
    for h in range(1, CCA_NUM_HARMONICS + 1):
        refs.append(np.sin(2 * np.pi * h * freq * time_points))
        refs.append(np.cos(2 * np.pi * h * freq * time_points))
    CCA_REFERENCE_SIGNALS[freq] = np.array(refs).T

cca_model = SklearnCCA(n_components=1)

def get_cca_correlation(eeg_data_multi_channel, ref_signals):
    if eeg_data_multi_channel.shape[0] < eeg_data_multi_channel.shape[1] or eeg_data_multi_channel.shape[0] != ref_signals.shape[0]: return 0.0
    try:
        cca_model.fit(eeg_data_multi_channel, ref_signals)
        U, V = cca_model.transform(eeg_data_multi_channel, ref_signals)
        return np.corrcoef(U.T, V.T)[0, 1]
    except Exception: return 0.0

# ==============================================================================
# === 3. DASH APP LAYOUT =======================================================
# ==============================================================================
app = Dash(__name__, assets_folder='assets')
app.title = "BCI Pong - Working"

def get_initial_game_state():
    return { 'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2,
             'ball_y': GAME_HEIGHT / 2, 'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y,
             'player_score': 0, 'ai_score': 0 }

app.layout = html.Div(id='main-container', style={'backgroundColor': '#111', 'color': '#DDD', 'fontFamily': 'monospace', 'textAlign': 'center'}, children=[
    html.H1("BCI Pong - Working"),
    html.Div([
        html.Button('Pause / Resume', id='pause-button', n_clicks=0, style={'marginRight': '20px'}),
        html.Button('Restart Game', id='restart-button', n_clicks=0),
    ], style={'marginBottom': '10px'}),
    html.H3(id='status-display', style={'fontSize': '24px', 'color': 'yellow', 'height': '30px'}),
    html.Div(html.Canvas(id='pong-game-canvas', width=GAME_WIDTH, height=GAME_HEIGHT), style={'width': f'{GAME_WIDTH}px', 'margin': 'auto', 'border': '2px solid #555'}),
    html.Div(style={'width': '1000px', 'margin': '20px auto', 'display': 'flex', 'justifyContent': 'space-around'}, children=[
        dcc.Graph(id='psd-plot', style={'width': '60%'}),
        dcc.Graph(id='control-plot', style={'width': '35%'})
    ]),
    dcc.Store(id='game-state-store', data=get_initial_game_state()),
    dcc.Store(id='app-status-store', data={'status': 'STARTING', 'countdown': 0}),
    dcc.Store(id='calibration-store', data={'scores_left': [], 'scores_right': [], 'scores_rest': [], 'thresholds': None}),
    dcc.Store(id='bci-command-store', data={'command': 'NEUTRAL', 'raw_score': 0.0, 'smoothed_score': 0.0}),
    dcc.Store(id='key-press-store', data={'key': 'None'}),
    dcc.Interval(id='game-interval', interval=GAME_INTERVAL_MS, n_intervals=0, disabled=False),
    dcc.Interval(id='bci-interval', interval=BCI_UPDATE_INTERVAL_MS, n_intervals=0, disabled=True),
    dcc.Interval(id='status-interval', interval=500, n_intervals=0)
])

# ==============================================================================
# === 4. CLIENTSIDE CALLBACKS (CODE RESTORED) ==================================
# ==============================================================================
clientside_callback(
    """ function(n_intervals) {
        if (!window.dash_clientside) { window.dash_clientside = {}; }
        if (!window.dash_clientside.key_listener_added) {
            window.dash_clientside.key_listener_added = true;
            window.dash_clientside.current_key = 'None';
            document.addEventListener('keydown', function(event) {
                if (event.key === 'a' || event.key === 'd') { window.dash_clientside.current_key = event.key; }
            });
            document.addEventListener('keyup', function(event) {
                if (event.key === 'a' || event.key === 'd') { window.dash_clientside.current_key = 'None'; }
            });
        }
        return {key: window.dash_clientside.current_key};
    } """,
    Output('key-press-store', 'data'),
    Input('game-interval', 'n_intervals')
)

clientside_callback(
    f"""
    function(n_intervals, gameState, appStatus) {{
        if (window.dash_clientside && window.dash_clientside.renderPong) {{
            const freqLeft = {SSVEP_FREQ_LEFT};
            const freqRight = {SSVEP_FREQ_RIGHT};
            window.dash_clientside.renderPong('pong-game-canvas', gameState, appStatus, n_intervals, {GAME_INTERVAL_MS}, freqLeft, freqRight);
        }}
        return null;
    }}
    """,
    Output('pong-game-canvas', 'className'),
    Input('game-interval', 'n_intervals'),
    Input('game-state-store', 'data'),
    Input('app-status-store', 'data')
)

# ==============================================================================
# === 5. CORE BCI & GAME LOGIC CALLBACKS =======================================
# ==============================================================================

@app.callback(
    Output('bci-command-store', 'data'),
    Output('calibration-store', 'data', allow_duplicate=True),
    Input('bci-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    State('calibration-store', 'data'),
    State('bci-command-store', 'data'),
    prevent_initial_call=True
)
def update_bci_command(_, app_status, cal_data, last_bci_command): # Typo cal__data fixed to cal_data
    status = app_status.get('status', 'STARTING')
    with buffer_lock:
        if any(len(data_buffers[ch]) < FFT_MAXLEN for ch in CHANNELS_IDX): return no_update, no_update
        eeg_window = np.array([list(data_buffers[ch]) for ch in CHANNELS_IDX]).T

    processed_eeg = preprocess_eeg_window(eeg_window)
    corr_left = get_cca_correlation(processed_eeg, CCA_REFERENCE_SIGNALS[SSVEP_FREQ_LEFT])
    corr_right = get_cca_correlation(processed_eeg, CCA_REFERENCE_SIGNALS[SSVEP_FREQ_RIGHT])
    raw_score = (corr_right - corr_left) * BCI_SCORE_AMPLIFIER

    if status.startswith('CALIBRATING'):
        if 'LEFT' in status: cal_data['scores_left'].append(raw_score)
        elif 'RIGHT' in status: cal_data['scores_right'].append(raw_score)
        elif 'REST' in status: cal_data['scores_rest'].append(raw_score)
        return {'command': 'NEUTRAL', 'raw_score': raw_score, 'smoothed_score': 0.0}, cal_data

    elif status == 'PLAYING':
        smoothed_score = raw_score
        if USE_EMA_SMOOTHING:
            last_smoothed = last_bci_command.get('smoothed_score', 0.0)
            smoothed_score = (EMA_SMOOTHING_FACTOR * last_smoothed) + ((1 - EMA_SMOOTHING_FACTOR) * raw_score)
        
        thresholds = cal_data.get('thresholds')
        if not thresholds: return no_update, no_update

        if smoothed_score > thresholds['right']: command = 'RIGHT'
        elif smoothed_score < thresholds['left']: command = 'LEFT'
        else: command = 'NEUTRAL'
        return {'command': command, 'raw_score': raw_score, 'smoothed_score': smoothed_score}, no_update

    return no_update, no_update


@app.callback(
    Output('game-state-store', 'data', allow_duplicate=True),
    Input('game-interval', 'n_intervals'),
    State('game-state-store', 'data'),
    State('bci-command-store', 'data'),
    State('app-status-store', 'data'),
    State('key-press-store', 'data'),
    prevent_initial_call=True
)
def update_game_physics(_, state, bci_command, app_status, key_data):
    if app_status.get('status') != 'PLAYING':
        return no_update

    # Player (bottom paddle) movement
    key_command = key_data.get('key', 'None')
    if key_command == 'a': state['player_x'] -= PADDLE_SPEED
    elif key_command == 'd': state['player_x'] += PADDLE_SPEED
    else:
        bci_move = bci_command.get('command', 'NEUTRAL')
        if bci_move == 'LEFT': state['player_x'] -= PADDLE_SPEED
        elif bci_move == 'RIGHT': state['player_x'] += PADDLE_SPEED
    state['player_x'] = max(PADDLE_WIDTH / 2, min(GAME_WIDTH - PADDLE_WIDTH / 2, state['player_x']))
    
    # AI (top paddle) movement
    if state['ai_x'] < state['ball_x']: state['ai_x'] += AI_PADDLE_SPEED
    if state['ai_x'] > state['ball_x']: state['ai_x'] -= AI_PADDLE_SPEED
    state['ai_x'] = max(PADDLE_WIDTH / 2, min(GAME_WIDTH - PADDLE_WIDTH / 2, state['ai_x']))
    
    # Ball physics
    state['ball_x'] += state['ball_vx']; state['ball_y'] += state['ball_vy']
    if state['ball_x'] <= BALL_RADIUS or state['ball_x'] >= GAME_WIDTH - BALL_RADIUS: state['ball_vx'] *= -1
    
    # Collision with Player (bottom paddle)
    if state['ball_vy'] > 0 and state['ball_y'] + BALL_RADIUS >= GAME_HEIGHT - PADDLE_HEIGHT:
        if abs(state['player_x'] - state['ball_x']) < PADDLE_WIDTH / 2 + BALL_RADIUS:
            state['ball_vy'] *= -1
            state['ball_vx'] += (state['ball_x'] - state['player_x']) * BALL_SPIN_FACTOR
            state['ball_y'] = GAME_HEIGHT - PADDLE_HEIGHT - BALL_RADIUS

    # Collision with AI (top paddle)
    if state['ball_vy'] < 0 and state['ball_y'] - BALL_RADIUS <= PADDLE_HEIGHT:
        if abs(state['ai_x'] - state['ball_x']) < PADDLE_WIDTH / 2 + BALL_RADIUS:
            state['ball_vy'] *= -1
            state['ball_vx'] += (state['ball_x'] - state['ai_x']) * BALL_SPIN_FACTOR
            state['ball_y'] = PADDLE_HEIGHT + BALL_RADIUS

    # Scoring logic (Player is at the bottom)
    if state['ball_y'] - BALL_RADIUS > GAME_HEIGHT: # Ball passed player
        state['ai_score'] += 1
        p_score, a_score = state['player_score'], state['ai_score']
        state = get_initial_game_state(); state.update({'player_score': p_score, 'ai_score': a_score})
    elif state['ball_y'] + BALL_RADIUS < 0: # Ball passed AI
        state['player_score'] += 1
        p_score, a_score = state['player_score'], state['ai_score']
        state = get_initial_game_state(); state.update({'player_score': p_score, 'ai_score': a_score})
    
    return state

# ==============================================================================
# === 6. STATE MACHINE AND FEEDBACK PLOTS ======================================
# ==============================================================================
@app.callback(
    Output('status-display', 'children'),
    Output('app-status-store', 'data'),
    Output('calibration-store', 'data', allow_duplicate=True),
    Output('game-state-store', 'data', allow_duplicate=True),
    Output('bci-interval', 'disabled'),
    Output('game-interval', 'disabled'),
    Input('status-interval', 'n_intervals'),
    Input('pause-button', 'n_clicks'),
    Input('restart-button', 'n_clicks'),
    State('app-status-store', 'data'),
    State('calibration-store', 'data'),
    prevent_initial_call=True
)
def manage_app_flow(status_n, pause_clicks, restart_clicks, app_status, cal_data):
    triggered_id = ctx.triggered_id if ctx.triggered_id else 'status-interval'
    status = app_status.get('status', 'STARTING')
    countdown = app_status.get('countdown', 0)
    new_status = status
    new_cal_data = no_update
    new_game_state = no_update

    if triggered_id == 'restart-button' and restart_clicks > 0:
        new_status = 'STARTING'
        new_cal_data = {'scores_left': [], 'scores_right': [], 'scores_rest': [], 'thresholds': None}
        new_game_state = get_initial_game_state()
    elif triggered_id == 'pause-button' and pause_clicks > 0:
        if status != 'PAUSED': new_status = 'PAUSED'
        else: new_status = 'PLAYING'
    elif triggered_id == 'status-interval':
        if status == 'STARTING':
            new_status = 'CALIBRATING_LEFT'; countdown = CALIBRATION_DURATION_S
        elif status.startswith('CALIBRATING'):
            countdown -= 0.5
            if countdown <= 0:
                if status == 'CALIBRATING_LEFT': new_status, countdown = 'CALIBRATING_RIGHT', CALIBRATION_DURATION_S
                elif status == 'CALIBRATING_RIGHT': new_status, countdown = 'CALIBRATING_REST', CALIBRATION_DURATION_S
                elif status == 'CALIBRATING_REST': new_status = 'ANALYZING'
        elif status == 'ANALYZING':
            print("\n--- ANALYZING CALIBRATION DATA ---")
            mean_left = np.mean(cal_data['scores_left']) if cal_data['scores_left'] else 0
            std_left = np.std(cal_data['scores_left']) if cal_data['scores_left'] else 0.1
            mean_right = np.mean(cal_data['scores_right']) if cal_data['scores_right'] else 0
            std_right = np.std(cal_data['scores_right']) if cal_data['scores_right'] else 0.1
            threshold_left = mean_left - CALIBRATION_THRESHOLD_STD_FACTOR * std_left
            threshold_right = mean_right + CALIBRATION_THRESHOLD_STD_FACTOR * std_right
            cal_data['thresholds'] = {'left': min(threshold_left, -MIN_THRESHOLD_GAP/2), 'right': max(threshold_right, MIN_THRESHOLD_GAP/2)}
            new_cal_data = cal_data
            print(f"LEFT SCORES:  Mean={mean_left:.3f}, Std={std_left:.3f}")
            print(f"RIGHT SCORES: Mean={mean_right:.3f}, Std={std_right:.3f}")
            print(f"--> FINAL THRESHOLDS: Left < {threshold_left:.3f} | Right > {threshold_right:.3f}")
            new_status = 'READY'; countdown = 3
        elif status == 'READY':
            countdown -= 0.5
            if countdown <= 0: new_status = 'PLAYING'

    msg = ""
    if new_status == 'CALIBRATING_LEFT': msg = f"Focus on the LEFT flicker... {int(max(0, countdown))}"
    elif new_status == 'CALIBRATING_RIGHT': msg = f"Focus on the RIGHT flicker... {int(max(0, countdown))}"
    elif new_status == 'CALIBRATING_REST': msg = f"Look at the CENTER (rest)... {int(max(0, countdown))}"
    elif new_status == 'ANALYZING': msg = "Analyzing calibration data..."
    elif new_status == 'READY': msg = f"Get Ready! Starting in {int(max(0, countdown)) + 1}..."
    elif new_status == 'PLAYING': msg = "PLAYING! Use 'A' and 'D' to override."
    elif new_status == 'PAUSED': msg = "PAUSED"

    bci_interval_disabled = not (new_status.startswith('CALIBRATING') or new_status == 'PLAYING')
    game_interval_disabled = (new_status == 'PAUSED')
    app_status_out = {'status': new_status, 'countdown': countdown}
    return msg, app_status_out, new_cal_data, new_game_state, bci_interval_disabled, game_interval_disabled

@app.callback(
    Output('psd-plot', 'figure'), Output('control-plot', 'figure'),
    Input('bci-interval', 'n_intervals'),
    State('bci-command-store', 'data'), State('calibration-store', 'data'),
    prevent_initial_call=True
)
def update_feedback_plots(_, bci_command, cal_data):
    with buffer_lock:
        if len(data_buffers[CHANNELS_IDX[0]]) < FFT_MAXLEN: raise PreventUpdate
        y_data = np.array(list(data_buffers[CHANNELS_IDX[0]]))
    y_processed = preprocess_eeg_window(y_data).flatten(); y_win = y_processed * np.hanning(len(y_processed))
    N = len(y_win); yf = np.fft.fft(y_win); xf = np.fft.fftfreq(N, 1.0/SAMPLING_RATE_HZ)[:N//2]
    psd = 10 * np.log10(np.abs(yf[0:N//2])**2 + 1e-12)
    psd_fig = go.Figure(layout=go.Layout(title=f'Live PSD (Channel {CHANNELS_TO_USE[0]})', template='plotly_dark', xaxis_title='Frequency (Hz)', yaxis_title='Power (dB)'))
    psd_fig.add_trace(go.Scatter(x=xf, y=psd, mode='lines', name='PSD'))
    for h in range(1, CCA_NUM_HARMONICS + 1):
        opacity = 1.0 / h
        psd_fig.add_vline(x=SSVEP_FREQ_LEFT * h, line_dash="dash", line_color="cyan", opacity=opacity, annotation_text=f"{SSVEP_FREQ_LEFT*h:.1f}Hz" if h==1 else "")
        psd_fig.add_vline(x=SSVEP_FREQ_RIGHT * h, line_dash="dash", line_color="magenta", opacity=opacity, annotation_text=f"{SSVEP_FREQ_RIGHT*h:.1f}Hz" if h==1 else "")
    psd_fig.update_xaxes(range=[FILTER_LOW_CUT_HZ, FILTER_HIGH_CUT_HZ + 5])
    
    smoothed_score = (bci_command or {}).get('smoothed_score', 0.0)
    control_fig = go.Figure(layout=go.Layout(title='Live BCI Score (Smoothed)', template='plotly_dark', yaxis=dict(range=[-3.0, 3.0])))
    control_fig.add_trace(go.Bar(x=['Score'], y=[smoothed_score], marker_color=['cyan' if smoothed_score < 0 else 'magenta']))
    if cal_data and cal_data.get('thresholds'):
        thresholds = cal_data['thresholds']
        control_fig.add_hline(y=thresholds['left'], line_dash="dot", line_color="cyan", annotation_text="Left Threshold")
        control_fig.add_hline(y=thresholds['right'], line_dash="dot", line_color="magenta", annotation_text="Right Threshold")
    return psd_fig, control_fig

# ==============================================================================
# === 7. DATA ACQUISITION & MAIN EXECUTION =====================================
# ==============================================================================
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
                data_buffers[ch].append(microvolts)
    except Exception as e: print(f"Error parsing packet: {e}")
def find_and_open_board():
    print("Searching for the ADS1299 board..."); ports = serial.tools.list_ports.comports()
    candidate_ports = [p.device for p in ports if (p.vid and p.pid and {'vid': p.vid, 'pid': p.pid} in BOARD_USB_IDS) or \
                       (p.description and any(desc.lower() in p.description.lower() for desc in BOARD_DESCRIPTIONS))]
    if not candidate_ports: print("No specific candidates found. Testing all available serial ports..."); candidate_ports = [p.device for p in ports]
    for port_name in candidate_ports:
        print(f"--- Testing port: {port_name} ---"); ser = None
        try:
            ser = serial.Serial(port_name, INITIAL_BAUD_RATE, timeout=2); time.sleep(5)
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
                if start_idx == -1: buffer.clear(); break
                if len(buffer) < start_idx + DATA_PACKET_TOTAL_SIZE: break
                potential_packet = buffer[start_idx : start_idx + DATA_PACKET_TOTAL_SIZE]
                if potential_packet.endswith(end_marker):
                    payload = potential_packet[PACKET_IDX_LENGTH:PACKET_IDX_CHECKSUM]
                    if (sum(payload) & 0xFF) == potential_packet[PACKET_IDX_CHECKSUM]: parse_data_packet(potential_packet); last_data_time = time.time()
                    buffer = buffer[start_idx + DATA_PACKET_TOTAL_SIZE:]; continue
                buffer = buffer[start_idx + 1:]
    except Exception as e: print(f"Error in serial_read_loop: {e}")
    finally:
        if ser and ser.is_open: ser.close(); print("Serial port closed.")

def main():
    serial_port_object = find_and_open_board()
    if serial_port_object:
        threading.Thread(target=serial_read_loop, args=(serial_port_object,), daemon=True).start()
        log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
        print("\nDash server is running. Open http://127.0.0.1:8050/ in your browser.")
        print("The calibration routine will begin automatically.")
        app.run(debug=False, use_reloader=False)
    else:
        print("Could not start application: No board was found or data stream failed verification.")

if __name__ == "__main__":
    main()