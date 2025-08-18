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

# <<< CHANGE #1: New configuration for the subtle gradient stimulus >>>
# --- SSVEP BCI Configuration ---
SSVEP_FREQ_LEFT = 12.0
SSVEP_FREQ_RIGHT = 17.0
CONTROL_SMOOTHING_WINDOW = 10
CONTROL_SENSITIVITY = 0.5

# --- Visual Gradient Stimulus Configuration ---
GRADIENT_RESOLUTION = 100  # How many cells in the gradient bar
GRADIENT_HEIGHT_PX = 120   # Height of the flicker bar in pixels
GRADIENT_Y_POS = 300       # Vertical position (center) of the bar
FLICKER_INTENSITY = 0.8    # How strong the pulse is (0.0 to 1.0). 0.8 means brightness ranges from 10% to 90%.
GRADIENT_COLOR_MIN = 'hsl(210, 15%, 20%)' # Color for minimum brightness (dark blue-grey)
GRADIENT_COLOR_MAX = 'hsl(180, 100%, 90%)'# Color for maximum brightness (very light cyan)

# --- Game Configuration ---
GAME_INTERVAL_MS = 50
PADDLE_SPEED = 35
AI_PADDLE_SPEED = 8
INITIAL_BALL_SPEED_Y = -4
BALL_SPIN_FACTOR = 0.05
GAME_WIDTH = 800
GAME_HEIGHT = 600
PADDLE_WIDTH = 150
PADDLE_HEIGHT = 20
BALL_RADIUS = 10

# --- Data Buffers ---
fft_buffers = [deque(maxlen=int(SAMPLING_RATE_HZ * 2)) for _ in range(ADS1299_NUM_CHANNELS)]
control_metric_history = deque(maxlen=CONTROL_SMOOTHING_WINDOW)
buffer_lock = threading.Lock()

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
app.title = "SSVEP BCI Pong (Gradient)"

def get_initial_game_state():
    return {'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2, 'ball_y': GAME_HEIGHT / 2, 'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y, 'player_score': 0, 'ai_score': 0, 'best_channel': 0}

app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#DDDDDD', 'textAlign': 'center', 'fontFamily': 'monospace'}, children=[
    html.H1("SSVEP BCI Pong (Gradient)"),
    html.P(id='instruction-text', children="Look towards the side of the screen you want to move to."),
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
    if n is None: raise PreventUpdate

    # BCI Logic (Steps 1, 2, 3) are unchanged from previous "All-Channel" version
    with buffer_lock:
        if any(len(buf) < SAMPLING_RATE_HZ for buf in fft_buffers):
            fig = go.Figure(); fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', annotations=[dict(text="Waiting for EEG Data...", x=GAME_WIDTH/2, y=GAME_HEIGHT/2, showarrow=False, font=dict(size=24, color='white'))])
            return fig, state, "Status: Waiting for data...", "Connecting to your brain..."

    best_control_metric = 0; best_channel_idx = -1
    with buffer_lock:
        for ch in range(ADS1299_NUM_CHANNELS):
            eeg_data = list(fft_buffers[ch])
            if not eeg_data: continue
            y_data = np.array(eeg_data) - np.mean(eeg_data); N = len(y_data)
            win = np.hanning(N); y_win = y_data * win
            yf = np.fft.fft(y_win); xf = np.fft.fftfreq(N, 1.0 / SAMPLING_RATE_HZ)[:N//2]
            psd = (2 / (SAMPLING_RATE_HZ * np.sum(win**2))) * np.abs(yf[0:N//2])**2
            idx_left = np.argmin(np.abs(xf - SSVEP_FREQ_LEFT)); idx_right = np.argmin(np.abs(xf - SSVEP_FREQ_RIGHT))
            current_metric = psd[idx_right] - psd[idx_left]
            if abs(current_metric) > abs(best_control_metric):
                best_control_metric = current_metric; best_channel_idx = ch
    
    state['best_channel'] = best_channel_idx + 1
    control_metric_history.append(best_control_metric); smoothed_metric = np.mean(control_metric_history)
    max_abs_val = max(abs(m) for m in control_metric_history) if control_metric_history else 1.0
    control_signal = np.clip(smoothed_metric / (max_abs_val + 1e-9), -1.0, 1.0) * CONTROL_SENSITIVITY
    
    # Game State Update (Step 4) is unchanged
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
        p_score, a_score, b_chan = state['player_score'], state['ai_score'], state['best_channel']
        state = get_initial_game_state()
        state.update({'player_score': p_score, 'ai_score': a_score, 'best_channel': b_chan})

    # <<< CHANGE #2: Overhauled drawing logic to use a subtle gradient heatmap >>>
    # --- Step 5: Draw Everything ---
    fig = go.Figure()
    
    # Create the flickering gradient stimulus
    current_time_s = n * GAME_INTERVAL_MS / 1000.0
    
    # Sinusoidal flicker values between [0, 1]
    flicker_val_left = 0.5 + (np.sin(2 * np.pi * SSVEP_FREQ_LEFT * current_time_s) * (FLICKER_INTENSITY / 2))
    flicker_val_right = 0.5 + (np.sin(2 * np.pi * SSVEP_FREQ_RIGHT * current_time_s) * (FLICKER_INTENSITY / 2))
    
    # Create weights for blending the two flickers across the screen
    left_weights = np.linspace(1, 0, GRADIENT_RESOLUTION)
    right_weights = 1 - left_weights
    
    # Combine the flickers and weights to get the final brightness values for the gradient
    gradient_z_values = (flicker_val_left * left_weights) + (flicker_val_right * right_weights)

    # Add the heatmap as the background stimulus
    fig.add_trace(go.Heatmap(
        z=[gradient_z_values],
        x=np.linspace(0, GAME_WIDTH, GRADIENT_RESOLUTION),
        y=[GRADIENT_Y_POS - GRADIENT_HEIGHT_PX/2, GRADIENT_Y_POS + GRADIENT_HEIGHT_PX/2],
        colorscale=[[0, GRADIENT_COLOR_MIN], [1, GRADIENT_COLOR_MAX]],
        showscale=False,
        zmin=0,
        zmax=1
    ))
    
    # Add game elements on top of the gradient
    fig.add_shape(type="rect", x0=state['player_x']-PADDLE_WIDTH/2, y0=0, x1=state['player_x']+PADDLE_WIDTH/2, y1=PADDLE_HEIGHT, fillcolor="cyan", line=dict(width=0))
    fig.add_shape(type="rect", x0=state['ai_x']-PADDLE_WIDTH/2, y0=GAME_HEIGHT-PADDLE_HEIGHT, x1=state['ai_x']+PADDLE_WIDTH/2, y1=GAME_HEIGHT, fillcolor="magenta", line=dict(width=0))
    fig.add_shape(type="circle", x0=state['ball_x']-BALL_RADIUS, y0=state['ball_y']-BALL_RADIUS, x1=state['ball_x']+BALL_RADIUS, y1=state['ball_y']+BALL_RADIUS, fillcolor="white", line=dict(width=0))
    
    # Layout and Scores
    fig.update_layout(xaxis=dict(range=[0, GAME_WIDTH], visible=False), yaxis=dict(range=[0, GAME_HEIGHT], visible=False), plot_bgcolor='#000', paper_bgcolor='#111', margin=dict(l=10, r=10, t=10, b=10),
                      annotations=[dict(text=str(state['player_score']), x=30, y=GAME_HEIGHT/2 - 30, showarrow=False, font=dict(size=40, color='cyan')),
                                   dict(text=str(state['ai_score']), x=30, y=GAME_HEIGHT/2 + 30, showarrow=False, font=dict(size=40, color='magenta'))])
    
    focus_text = f"Best Channel: {state.get('best_channel', 'N/A')} | Control Signal: {control_signal:.2f}"
    instruction_text = f"Look Left ({SSVEP_FREQ_LEFT}Hz) to Move Left | Look Right ({SSVEP_FREQ_RIGHT}Hz) to Move Right"
    return fig, state, focus_text, instruction_text

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