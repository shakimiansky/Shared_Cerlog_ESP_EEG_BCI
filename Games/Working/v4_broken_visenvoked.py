import serial
import struct
import threading
import time
from collections import deque
import serial.tools.list_ports
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State, no_update, clientside_callback
from dash.exceptions import PreventUpdate
import logging

--- Signal Processing & Machine Learning Libraries ---

from scipy.signal import butter, lfilter, detrend
from sklearn.cross_decomposition import CCA as SklearnCCA

==============================================================================
=== 1. DIAGNOSTIC & CONFIGURATION ============================================
==============================================================================
--- Serial/Data Acquisition Configuration ---

INITIAL_BAUD_RATE = 9600; FINAL_BAUD_RATE = 115200; FIRMWARE_BAUD_RATE_INDEX = 0x04
SAMPLING_RATE_HZ = 250.0; STREAM_TIMEOUT_SECONDS = 5.0; DATA_PACKET_START_MARKER = 0xABCD
DATA_PACKET_END_MARKER = 0xDCBA; DATA_PACKET_TOTAL_SIZE = 37; PACKET_IDX_LENGTH = 2
PACKET_IDX_CHECKSUM = 34; ADS1299_NUM_CHANNELS = 8; ADS1299_NUM_STATUS_BYTES = 3
ADS1299_BYTES_PER_CHANNEL = 3; BOARD_USB_IDS = [{'vid': 0x1A86, 'pid': 0x7523}]
BOARD_DESCRIPTIONS = ["USB-SERIAL CH340", "CH340"]

--- Control Sensitivity ---

USE_INTENTION_ACCUMULATOR = False

--- "Last Resort" Amplifier ---

BCI_SCORE_AMPLIFIER = 3.0

--- BCI Parameters ---

CHANNELS_TO_USE = [1, 2, 3] # Occipital channels (e.g., O1, Oz, O2)
SSVEP_FREQ_LEFT = 7.5   # Target frequency for 'move left'
SSVEP_FREQ_RIGHT = 12.0  # Target frequency for 'move right'

--- Analysis Window ---

FFT_WINDOW_SECONDS = 2.5
FFT_MAXLEN = int(SAMPLING_RATE_HZ * FFT_WINDOW_SECONDS)
FFT_OVERLAP_PERCENT = 0.6
BCI_UPDATE_INTERVAL_MS = int((FFT_WINDOW_SECONDS * (1 - FFT_OVERLAP_PERCENT)) * 1000)

--- Filtering Parameters ---

FILTER_LOW_CUT_HZ = 5.0; FILTER_HIGH_CUT_HZ = 45.0; FILTER_ORDER = 5
NOTCH_FREQ_HZ = 60.0; NOTCH_QUALITY_FACTOR = 30 # Change to 50.0 for Europe/Asia

--- CCA Parameters ---

CCA_NUM_HARMONICS = 4

--- Control Logic Parameters ---

INTENTION_BUFFER_LEN = 5
INTENTION_CONFIDENCE_THRESHOLD = 3

==============================================================================
=== 2. GAME & UI CONFIGURATION ===============================================
==============================================================================

GAME_INTERVAL_MS = 33; AI_PADDLE_SPEED = 7; PADDLE_SPEED = 18
INITIAL_BALL_SPEED_Y = -5; BALL_SPIN_FACTOR = 0.05
GAME_WIDTH = 800; GAME_HEIGHT = 600; PADDLE_WIDTH = 150; PADDLE_HEIGHT = 20; BALL_RADIUS = 10

--- Calibration Parameters ---

CALIBRATION_DURATION_S = 8
CALIBRATION_THRESHOLD_STD_FACTOR = 0.7
MIN_THRESHOLD_GAP = 0.05

--- Data & Control Buffers ---

CHANNELS_IDX = [c - 1 for c in CHANNELS_TO_USE]
data_buffers = [deque(maxlen=FFT_MAXLEN) for _ in range(ADS1299_NUM_CHANNELS)]
buffer_lock = threading.Lock()
intention_buffer = deque(maxlen=INTENTION_BUFFER_LEN)

==============================================================================
=== 3. SIGNAL PROCESSING FUNCTIONS (UNCHANGED) ===============================
==============================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
nyq = 0.5 * fs; low = lowcut / nyq; high = highcut / nyq
b, a = butter(order, [low, high], btype='band'); return b, a

def iir_notch(w0, q, fs):
b, a = butter(2, [(w0-w0/(2q))/(fs/2), (w0+w0/(2q))/(fs/2)], btype='bandstop'); return b, a

bandpass_b, bandpass_a = butter_bandpass(FILTER_LOW_CUT_HZ, FILTER_HIGH_CUT_HZ, SAMPLING_RATE_HZ, order=FILTER_ORDER)
notch_b, notch_a = iir_notch(NOTCH_FREQ_HZ, NOTCH_QUALITY_FACTOR, SAMPLING_RATE_HZ)

def preprocess_eeg_window(eeg_data):
if eeg_data.ndim == 1: eeg_data = eeg_data.reshape(-1, 1)
eeg_detrended = detrend(eeg_data, axis=0)
eeg_notched = lfilter(notch_b, notch_a, eeg_detrended, axis=0)
eeg_filtered = lfilter(bandpass_b, bandpass_a, eeg_notched, axis=0)
return eeg_filtered

time_points = np.arange(0, FFT_WINDOW_SECONDS, 1.0 / SAMPLING_RATE_HZ)[:FFT_MAXLEN]
CCA_REFERENCE_SIGNALS = {}
for freq in [SSVEP_FREQ_LEFT, SSVEP_FREQ_RIGHT]:
refs = [];
for h in range(1, CCA_NUM_HARMONICS + 1):
refs.append(np.sin(2 * np.pi * h * freq * time_points))
refs.append(np.cos(2 * np.pi * h * freq * time_points))
CCA_REFERENCE_SIGNALS[freq] = np.array(refs).T

cca_model = SklearnCCA(n_components=1)

def get_cca_correlation(eeg_data_multi_channel, ref_signals):
if np.linalg.matrix_rank(eeg_data_multi_channel) < eeg_data_multi_channel.shape[1]: return 0.0
try:
if eeg_data_multi_channel.shape[0] != ref_signals.shape[0]: return 0.0
cca_model.fit(eeg_data_multi_channel, ref_signals)
U, V = cca_model.transform(eeg_data_multi_channel, ref_signals)
return np.corrcoef(U.T, V.T)[0, 1]
except Exception: return 0.0

==============================================================================
=== 4. DATA ACQUISITION & DASH APP SETUP (UNCHANGED) =========================
==============================================================================

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
candidate_ports = [p.device for p in ports if (p.vid and p.pid and {'vid': p.vid, 'pid': p.pid} in BOARD_USB_IDS) or 
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

app = Dash(name)
app.title = "Upgraded BCI Pong"

def get_initial_game_state():
return { 'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2, 'ball_y': GAME_HEIGHT / 2, 'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y, 'player_score': 0, 'ai_score': 0 }

app.layout = html.Div(id='main-container', style={'backgroundColor': '#111', 'color': '#DDD', 'fontFamily': 'monospace', 'textAlign': 'center'}, children=[
html.H1("Upgraded BCI Pong"),
html.H3("Use A/D keys to test paddle movement", style={'color': '#888'}),
html.Div(id='status-display', style={'fontSize': '24px', 'color': 'yellow', 'marginBottom': '10px'}),
html.Div(style={'width': '800px', 'margin': 'auto', 'border': '2px solid #555'}, children=[dcc.Graph(id='pong-game-graph', config={'staticPlot': True}, style={'height': '600px'})]),
html.Div(style={'width': '1000px', 'margin': '20px auto', 'display': 'flex', 'justifyContent': 'space-around'}, children=[ dcc.Graph(id='psd-plot', style={'width': '60%'}), dcc.Graph(id='control-plot', style={'width': '35%'}) ]),
dcc.Store(id='game-state-store', data=get_initial_game_state()),
dcc.Store(id='app-status-store', data={'status': 'STARTING', 'countdown': 0, 'last_update': time.time()}),
dcc.Store(id='calibration-store', data={'scores_left': [], 'scores_right': [], 'scores_rest': [], 'thresholds': None}),
dcc.Store(id='bci-command-store', data={'command': 'NEUTRAL', 'raw_score': 0.0}),
dcc.Store(id='key-press-store', data={'key': 'None', 'timestamp': 0}),
dcc.Interval(id='game-interval', interval=GAME_INTERVAL_MS, n_intervals=0, disabled=False),
dcc.Interval(id='bci-interval', interval=BCI_UPDATE_INTERVAL_MS, n_intervals=0, disabled=False),
dcc.Interval(id='status-interval', interval=500, n_intervals=0, disabled=False)
])

clientside_callback(
""" function(n_intervals) { if (!window.dash_clientside) { window.dash_clientside = {}; } if (!window.dash_clientside.key_listener_added) { window.dash_clientside.key_listener_added = true; document.addEventListener('keydown', function(event) { if (event.key === 'a' || event.key === 'd') { window.dash_clientside.last_key = {key: event.key, timestamp: new Date().getTime()}; } }); document.addEventListener('keyup', function(event) { if (event.key === 'a' || event.key === 'd') { window.dash_clientside.last_key = {key: 'None', timestamp: new Date().getTime()}; } }); } return window.dash_clientside.last_key || {key: 'None', timestamp: 0}; } """,
Output('key-press-store', 'data'),
Input('game-interval', 'n_intervals')
)

==============================================================================
=== 5. CORE BCI & CALIBRATION CALLBACKS ======================================
==============================================================================

@app.callback(
Output('app-status-store', 'data'), Output('calibration-store', 'data'), Output('bci-command-store', 'data'),
Input('bci-interval', 'n_intervals'),
State('app-status-store', 'data'), State('calibration-store', 'data'),
prevent_initial_call=True
)
def update_bci_and_state(_, app_status, cal_data):
with buffer_lock:
if any(len(data_buffers[ch]) < FFT_MAXLEN for ch in CHANNELS_IDX):
return no_update # It's safe to exit early here, as no state is being calculated
eeg_window = np.array([list(data_buffers[ch]) for ch in CHANNELS_IDX]).T

code
Code
download
content_copy
expand_less

processed_eeg = preprocess_eeg_window(eeg_window)
corr_left = get_cca_correlation(processed_eeg, CCA_REFERENCE_SIGNALS[SSVEP_FREQ_LEFT])
corr_right = get_cca_correlation(processed_eeg, CCA_REFERENCE_SIGNALS[SSVEP_FREQ_RIGHT])
raw_score = (corr_right - corr_left) * BCI_SCORE_AMPLIFIER

status = app_status['status']

if status.startswith('CALIBRATING'):
    if 'LEFT' in status: cal_data['scores_left'].append(raw_score)
    elif 'RIGHT' in status: cal_data['scores_right'].append(raw_score)
    elif 'REST' in status: cal_data['scores_rest'].append(raw_score)
    return app_status, cal_data, {'command': 'NEUTRAL', 'raw_score': raw_score}

elif status == 'PLAYING':
    thresholds = cal_data.get('thresholds')
    # This check is important. Without the fix below, `thresholds` would always be None.
    if not thresholds:
        return app_status, cal_data, {'command': 'NEUTRAL', 'raw_score': 0.0}

    if raw_score > thresholds['right']: current_decision = 'RIGHT'
    elif raw_score < thresholds['left']: current_decision = 'LEFT'
    else: current_decision = 'NEUTRAL'

    final_command = 'NEUTRAL'
    if USE_INTENTION_ACCUMULATOR:
        intention_buffer.append(current_decision)
        if intention_buffer.count('RIGHT') >= INTENTION_CONFIDENCE_THRESHOLD: final_command = 'RIGHT'
        elif intention_buffer.count('LEFT') >= INTENTION_CONFIDENCE_THRESHOLD: final_command = 'LEFT'
    else:
        final_command = current_decision

    print(f"BCI Decision: Raw Score={raw_score:.3f} -> {final_command}")
    return app_status, cal_data, {'command': final_command, 'raw_score': raw_score}

# Default case if status is not calibrating or playing (e.g., READY, ANALYZING)
return app_status, cal_data, {'command': 'NEUTRAL', 'raw_score': raw_score}
=================================================
=== BUG FIX: THIS CALLBACK IS NOW CORRECTED =====
=================================================

@app.callback(
Output('status-display', 'children'),
Output('game-interval', 'disabled'),
Output('app-status-store', 'data', allow_duplicate=True),
Output('calibration-store', 'data', allow_duplicate=True), # <-- FIX 1: ADDED THIS OUTPUT
Input('status-interval', 'n_intervals'),
State('app-status-store', 'data'),
State('calibration-store', 'data'),
prevent_initial_call=True
)
def manage_app_flow(_, app_status, cal_data):
status = app_status['status']; new_status = status; msg = ""
game_render_loop_disabled = False
current_time = time.time(); elapsed = current_time - app_status.get('last_update', current_time)

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
if status == 'STARTING':
    new_status = 'CALIBRATING_LEFT'; app_status['countdown'] = CALIBRATION_DURATION_S; app_status['last_update'] = current_time
elif status.startswith('CALIBRATING'):
    countdown = app_status['countdown'] - elapsed
    app_status['last_update'] = current_time
    if 'LEFT' in status: msg = f"CALIBRATING: Focus on the LEFT flicker... {int(max(0, countdown))}"
    elif 'RIGHT' in status: msg = f"CALIBRATING: Focus on the RIGHT flicker... {int(max(0, countdown))}"
    elif 'REST' in status: msg = f"CALIBRATING: Look at the CENTER (rest)... {int(max(0, countdown))}"
    if countdown <= 0:
        if status == 'CALIBRATING_LEFT': new_status = 'CALIBRATING_RIGHT'
        elif status == 'CALIBRATING_RIGHT': new_status = 'CALIBRATING_REST'
        elif status == 'CALIBRATING_REST': new_status = 'ANALYZING'
        app_status['countdown'] = CALIBRATION_DURATION_S
    else:
        app_status['countdown'] = countdown
elif status == 'ANALYZING':
    mean_left = np.mean(cal_data['scores_left']) if cal_data['scores_left'] else 0
    std_left = np.std(cal_data['scores_left']) if cal_data['scores_left'] else 0.1
    mean_right = np.mean(cal_data['scores_right']) if cal_data['scores_right'] else 0
    std_right = np.std(cal_data['scores_right']) if cal_data['scores_right'] else 0.1
    mean_rest = np.mean(cal_data['scores_rest']) if cal_data['scores_rest'] else 0
    threshold_left = mean_left - CALIBRATION_THRESHOLD_STD_FACTOR * std_left
    threshold_right = mean_right + CALIBRATION_THRESHOLD_STD_FACTOR * std_right

    # This line now correctly saves the thresholds to the store because of the added Output and return value
    cal_data['thresholds'] = {'left': threshold_left, 'right': threshold_right}

    print("\n--- CALIBRATION COMPLETE ---")
    print(f"LEFT SCORES:  Mean={mean_left:.3f}, Std={std_left:.3f}")
    print(f"RIGHT SCORES: Mean={mean_right:.3f}, Std={std_right:.3f}")
    print(f"REST SCORES:  Mean={mean_rest:.3f}")
    print(f"--> FINAL THRESHOLDS: Left < {threshold_left:.3f} | Right > {threshold_right:.3f}")
    if threshold_right < threshold_left + MIN_THRESHOLD_GAP:
        print("\n*** WARNING: Calibration may have failed. Thresholds are too close or inverted.")
        print("*** Check electrode contact and try to focus more intently during calibration.\n")
    new_status = 'READY'; app_status['countdown'] = 3; app_status['last_update'] = current_time
elif status == 'READY':
    countdown = app_status['countdown'] - elapsed; app_status['last_update'] = current_time
    msg = f"Get Ready! Starting in {int(max(0, countdown)) + 1}...";
    if countdown <= 0: new_status = 'PLAYING'
    app_status['countdown'] = countdown
elif status == 'PLAYING': msg = "PLAYING!"
app_status['status'] = new_status

# <-- FIX 2: RETURN THE MODIFIED cal_data TO SAVE THE THRESHOLDS
return msg, game_render_loop_disabled, app_status, cal_data
==============================================================================
=== 6. GAME LOGIC & VISUAL RENDERING CALLBACKS ===============================
==============================================================================

@app.callback(
Output('pong-game-graph', 'figure'), Output('game-state-store', 'data'),
Input('game-interval', 'n_intervals'),
State('game-state-store', 'data'),
State('bci-command-store', 'data'),
State('app-status-store', 'data'),
State('key-press-store', 'data')
)
def update_game(n, state, bci_command, app_status, key_data):
if n is None or state is None: raise PreventUpdate
new_game_state_output = no_update

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
if app_status['status'] == 'PLAYING':
    final_paddle_command = 'NEUTRAL' # Default to no movement
    key_command = key_data.get('key', 'None')

    if key_command == 'a':
        final_paddle_command = 'LEFT'
    elif key_command == 'd':
        final_paddle_command = 'RIGHT'
    else:
        # Only if no key is pressed, use the BCI command
        final_paddle_command = bci_command.get('command', 'NEUTRAL')

    # Now, act on the single source of truth for movement
    if final_paddle_command == 'LEFT':
        state['player_x'] -= PADDLE_SPEED
    elif final_paddle_command == 'RIGHT':
        state['player_x'] += PADDLE_SPEED

    state['player_x'] = max(PADDLE_WIDTH / 2, min(GAME_WIDTH - PADDLE_WIDTH / 2, state['player_x']))

    # AI and Ball physics (unchanged)
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
    new_game_state_output = state

# Rendering logic (unchanged)
fig = go.Figure()
current_time_s = n * GAME_INTERVAL_MS / 1000.0
is_left_on = np.sin(2 * np.pi * SSVEP_FREQ_LEFT * current_time_s) > 0
is_right_on = np.sin(2 * np.pi * SSVEP_FREQ_RIGHT * current_time_s) > 0
status = app_status['status']
left_color = '#003333'; right_color = '#330033'
if is_left_on: left_color = 'cyan'
if is_right_on: right_color = 'magenta'
if 'CALIBRATING_LEFT' in status: right_color = '#1a1a1a'
if 'CALIBRATING_RIGHT' in status: left_color = '#1a1a1a'
fig.add_shape(type="rect", x0=0, y0=0, x1=GAME_WIDTH*0.25, y1=GAME_HEIGHT, fillcolor=left_color, line_width=0, layer='below')
fig.add_shape(type="rect", x0=GAME_WIDTH*0.75, y0=0, x1=GAME_WIDTH, y1=GAME_HEIGHT, fillcolor=right_color, line_width=0, layer='below')
fig.add_shape(type="rect", x0=state['player_x']-PADDLE_WIDTH/2, y0=0, x1=state['player_x']+PADDLE_WIDTH/2, y1=PADDLE_HEIGHT, fillcolor="cyan", line=dict(width=0))
fig.add_shape(type="rect", x0=state['ai_x']-PADDLE_WIDTH/2, y0=GAME_HEIGHT-PADDLE_HEIGHT, x1=state['ai_x']+PADDLE_WIDTH/2, y1=GAME_HEIGHT, fillcolor="magenta", line=dict(width=0))
fig.add_shape(type="circle", x0=state['ball_x']-BALL_RADIUS, y0=state['ball_y']-BALL_RADIUS, x1=state['ball_x']+BALL_RADIUS, y1=state['ball_y']+BALL_RADIUS, fillcolor="white", line=dict(width=0))
fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#1a1a1a', paper_bgcolor='#111', margin=dict(l=10, r=10, t=10, b=10),
                  annotations=[dict(text=str(state['player_score']), x=30, y=GAME_HEIGHT/2 - 30, showarrow=False, font=dict(size=40, color='cyan')),
                               dict(text=str(state['ai_score']), x=30, y=GAME_HEIGHT/2 + 30, showarrow=False, font=dict(size=40, color='magenta'))])
return fig, new_game_state_output

@app.callback(
Output('psd-plot', 'figure'), Output('control-plot', 'figure'),
Input('bci-interval', 'n_intervals'),
State('bci-command-store', 'data'), State('calibration-store', 'data')
)
def update_feedback_plots(_, bci_command, cal_data):
# This function remains unchanged
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
psd_fig.add_vline(x=SSVEP_FREQ_LEFT * h, line_dash="dash", line_color="cyan", opacity=opacity, annotation_text=f"{SSVEP_FREQ_LEFTh:.1f}Hz" if h==1 else "")
psd_fig.add_vline(x=SSVEP_FREQ_RIGHT * h, line_dash="dash", line_color="magenta", opacity=opacity, annotation_text=f"{SSVEP_FREQ_RIGHTh:.1f}Hz" if h==1 else "")
psd_fig.update_xaxes(range=[FILTER_LOW_CUT_HZ, FILTER_HIGH_CUT_HZ + 5])
raw_score = (bci_command or {}).get('raw_score', 0.0)
control_fig = go.Figure(layout=go.Layout(title='Live BCI Score (Right - Left)', template='plotly_dark', yaxis=dict(range=[-1,1])))
control_fig.add_trace(go.Bar(x=['Score'], y=[raw_score], marker_color=['cyan' if raw_score < 0 else 'magenta']))
if cal_data and cal_data.get('thresholds'):
thresholds = cal_data['thresholds']
control_fig.add_hline(y=thresholds['left'], line_dash="dot", line_color="cyan", annotation_text="Left Threshold")
control_fig.add_hline(y=thresholds['right'], line_dash="dot", line_color="magenta", annotation_text="Right Threshold")
return psd_fig, control_fig

==============================================================================
=== 7. MAIN EXECUTION (UNCHANGED) ============================================
==============================================================================

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

if name == "main":
main()