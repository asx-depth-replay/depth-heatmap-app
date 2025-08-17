import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz
import google.generativeai as genai

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Market Replay VCR")

# --- 2. Application Title ---
st.title("📈 Market Replay VCR")
st.markdown("Select a session to load and replay the market heatmap with AI commentary.")

# --- 3. API & Data Functions ---
BATCH_SIZE = 1000
API_URLS = {
    'LIST_SESSIONS': 'https://list-sessions-897370608024.australia-southeast1.run.app',
    'GET_SESSION_DETAILS': 'https://get-session-details-897370608024.australia-southeast1.run.app',
    'GET_SNAPSHOTS': 'https://get-snapshots-897370608024.australia-southeast1.run.app'
}

# --- 4. AI Agent Functions ---
# Configure the AI model using the secret key
try:
    genai.configure(api_key=st.secrets["AIzaSyACTSjkzHGdjGboFxpugt5HH2b6U3HFeN0"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    st.error("AI model could not be configured. Please check your GOOGLE_API_KEY in secrets.toml.")
    model = None

def get_ai_commentary(summary):
    """Sends the latest data to the AI and gets a comment."""
    if not model:
        return "AI model not available."
    
    prompt = f"""
    You are a veteran day trader with 20 years of experience watching market depth.
    Your task is to provide a single, brief, insightful comment (25 words or less)
    on the following market data snapshot. Focus on shifts in liquidity, potential
    support/resistance, or overall market sentiment. Be concise and professional.

    Current Data:
    {summary}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Could not generate AI commentary at this time."

def create_latest_data_summary(current_df, full_df):
    """Analyzes the latest data and creates a text summary for the AI."""
    if current_df.empty:
        return "No data available."

    latest_time = current_df['datetime'].max()
    latest_snapshot = current_df[current_df['datetime'] == latest_time]
    
    latest_bids = latest_snapshot[latest_snapshot['Type'] == 'BUY']
    latest_asks = latest_snapshot[latest_snapshot['Type'] == 'SELL']
    
    if latest_bids.empty or latest_asks.empty:
        return "Waiting for full market data."

    best_bid_row = latest_bids.loc[latest_bids['Price'].idxmax()]
    best_ask_row = latest_asks.loc[latest_asks['Price'].idxmin()]
    mid_point = (best_bid_row['Price'] + best_ask_row['Price']) / 2
    spread = best_ask_row['Price'] - best_bid_row['Price']

    five_mins_ago = latest_time - pd.Timedelta(minutes=5)
    past_snapshot_df = full_df[full_df['datetime'] <= five_mins_ago]
    if not past_snapshot_df.empty:
        past_mid_point_df = calculate_price_lines(past_snapshot_df)
        past_mid_point = past_mid_point_df.iloc[-1]['mid_point']
        trend_delta = mid_point - past_mid_point
        trend_summary = f"Trending Up (+${trend_delta:.2f} in last 5 mins)" if trend_delta > 0 else f"Trending Down (-${abs(trend_delta):.2f} in last 5 mins)"
    else:
        trend_summary = "Not enough data for a 5-minute trend."
        
    summary = f"""Timestamp: {latest_time.strftime('%H:%M:%S')} AEST
Mid-Point: ${mid_point:.2f}
Spread: ${spread:.2f}
5-Minute Trend: {trend_summary}"""
    return summary

# --- The rest of the data functions remain the same ---
@st.cache_data(ttl=300)
def fetch_available_sessions():
    try:
        response = requests.get(API_URLS['LIST_SESSIONS'])
        response.raise_for_status()
        return sorted(response.json(), key=lambda x: x.get('id', ''), reverse=True)
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not fetch sessions. {e}")
        return []

def get_snapshots(session_id, timestamps):
    if not timestamps: return []
    timestamps_str = ",".join(map(str, timestamps))
    snapshot_url = f"{API_URLS['GET_SNAPSHOTS']}?id={session_id}&timestamps={timestamps_str}"
    response = requests.get(snapshot_url)
    return response.json() if response.ok else []

def parse_rows(batch_data):
    all_rows = []
    for snapshot in batch_data:
        dt = pd.to_datetime(snapshot.get('timestamp'), unit='ms')
        for order in snapshot.get('bids', []):
            all_rows.append({'datetime': dt, 'Type': 'BUY', 'Price': order.get('price'), 'Volume': order.get('size')})
        for order in snapshot.get('asks', []):
            all_rows.append({'datetime': dt, 'Type': 'SELL', 'Price': order.get('price'), 'Volume': order.get('size')})
    return all_rows

def process_dataframe(df):
    df.dropna(inplace=True)
    for col in ['Price', 'Volume']:
        df[col] = pd.to_numeric(df[col])
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
    df.sort_values('datetime', inplace=True)
    return df

@st.cache_data
def load_historical_data(session_id):
    details_url = f"{API_URLS['GET_SESSION_DETAILS']}?id={session_id}"
    details_response = requests.get(details_url)
    details_response.raise_for_status()
    timestamps = details_response.json().get('timestamps', [])
    if not timestamps: return None
    all_rows = []
    for i in range(0, len(timestamps), BATCH_SIZE):
        batch_data = get_snapshots(session_id, timestamps[i:i + BATCH_SIZE])
        all_rows.extend(parse_rows(batch_data))
    if not all_rows: return None
    return process_dataframe(pd.DataFrame(all_rows))

def calculate_price_lines(df):
    if df.empty: return pd.DataFrame(columns=['datetime', 'best_bid', 'best_bid_volume', 'best_ask', 'best_ask_volume', 'mid_point'])
    bids = df[df['Type'] == 'BUY'].groupby('datetime')['Price'].max().rename('best_bid')
    asks = df[df['Type'] == 'SELL'].groupby('datetime')['Price'].min().rename('best_ask')
    price_lines_df = pd.concat([bids, asks], axis=1).dropna()
    price_lines_df['mid_point'] = (price_lines_df['best_bid'] + price_lines_df['best_ask']) / 2
    return price_lines_df.reset_index()


# --- Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.header("1. Select Session")
sessions = fetch_available_sessions()
session_names = [s.get('name', s.get('id')) for s in sessions]
placeholder = "--- Select a session ---"
options = [placeholder] + session_names
selected_session_name = st.sidebar.selectbox("Choose a session to replay", options=options)
selected_session_id = None
if selected_session_name != placeholder:
    selected_session_id = next((s.get('id') for s in sessions if s.get('name', s.get('id')) == selected_session_name), None)

# --- Main Application Logic ---
if not selected_session_id:
    st.info("👋 Welcome! Please select a session to begin.")
else:
    with st.spinner("Loading session data... This may take a moment on first load."):
        full_depth_df = load_historical_data(selected_session_id)

    if full_depth_df is None or full_depth_df.empty:
        st.error("Could not load data for the selected session.")
    else:
        # Filter to session hours
        trade_date = full_depth_df['datetime'].iloc[0].date()
        SESSION_START = pd.to_datetime(f"{trade_date} 10:00:00").tz_localize('Australia/Melbourne')
        SESSION_END = pd.to_datetime(f"{trade_date} 16:00:00").tz_localize('Australia/Melbourne')
        full_depth_df = full_depth_df[(full_depth_df['datetime'] >= SESSION_START) & (full_depth_df['datetime'] < SESSION_END)]

        if full_depth_df.empty:
            st.warning("No data found within the 10:00 AM - 4:00 PM trading session.")
        else:
            unique_timestamps = full_depth_df['datetime'].unique()
            st.sidebar.header("2. VCR Controls")

            # Initialize session state for VCR and AI
            if 'playhead_index' not in st.session_state or st.session_state.get('session_id') != selected_session_id:
                st.session_state.is_playing = False
                st.session_state.playhead_index = 0
                st.session_state.session_id = selected_session_id
                st.session_state.last_comment = "AI Commentary will appear here. Press Play to begin."
                st.session_state.last_comment_time = None

            vcr_cols = st.sidebar.columns(2)
            if vcr_cols[0].button("▶️ Play / ⏸️ Pause"):
                st.session_state.is_playing = not st.session_state.is_playing
            if vcr_cols[1].button("⏮️ Reset"):
                st.session_state.is_playing = False
                st.session_state.playhead_index = 0
                st.session_state.last_comment = "AI Commentary will appear here. Press Play to begin."
                st.session_state.last_comment_time = None

            replay_speed = st.sidebar.select_slider("Replay Speed", options=[1, 5, 10, 20, 50, 100, 200], value=20)
            scrubber_val = st.sidebar.slider("Timeline", 0, len(unique_timestamps) - 1, st.session_state.playhead_index)
            if scrubber_val != st.session_state.playhead_index:
                st.session_state.playhead_index = scrubber_val
                st.session_state.is_playing = False

            if st.session_state.is_playing:
                st_autorefresh(interval=100, key="vcr_refresher")
                avg_time_delta = pd.to_timedelta(np.diff(unique_timestamps).mean()).total_seconds()
                snaps_per_sec = 1 / avg_time_delta if avg_time_delta > 0 else 1
                increment = max(1, int(snaps_per_sec * replay_speed * 0.1))
                st.session_state.playhead_index = min(st.session_state.playhead_index + increment, len(unique_timestamps) - 1)
                if st.session_state.playhead_index == len(unique_timestamps) - 1:
                    st.session_state.is_playing = False

            current_timestamp = unique_timestamps[st.session_state.playhead_index]
            display_df = full_depth_df[full_depth_df['datetime'] <= current_timestamp]
            
            st.header(f"Replaying Session: {selected_session_name}")
            st.subheader(f"Time: {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

            # --- AI Commentary Logic ---
            time_since_last_comment = (current_timestamp - st.session_state.last_comment_time).total_seconds() if st.session_state.last_comment_time else float('inf')
            # Get new comment if playing and 60 seconds of session time have passed
            if st.session_state.is_playing and time_since_last_comment > 60:
                summary = create_latest_data_summary(display_df, full_depth_df)
                st.session_state.last_comment = get_ai_commentary(summary)
                st.session_state.last_comment_time = current_timestamp
            
            with st.expander("👨‍💻 Veteran Trader AI Commentary", expanded=True):
                st.write(st.session_state.last_comment)
            
            # --- Charting Logic ---
            price_lines_df = calculate_price_lines(display_df)
            
            if display_df.empty or price_lines_df.empty:
                st.warning("No data to display for the current time in the replay.")
            else:
                bin_size = st.sidebar.number_input("Set Price Bin Size ($)", 0.01, 0.20, 0.05, 0.01, "%.2f")
                display_df['SignedVolume'] = np.where(display_df['Type'] == 'BUY', display_df['Volume'], -display_df['Volume'])
                full_price_lines_df = calculate_price_lines(full_depth_df)
                min_price = full_price_lines_df['best_bid'].min() - 0.50
                max_price = full_price_lines_df['best_ask'].max() + 0.50
                price_bins = np.arange(np.floor(min_price), np.ceil(max_price) + bin_size, bin_size)
                
                display_df['price_bin'] = pd.cut(display_df['Price'], bins=price_bins, right=False)
                binned_heatmap = display_df.pivot_table(index='price_bin', columns='datetime', values='SignedVolume', aggfunc='sum', observed=True).fillna(0)
                
                fig = go.Figure()
                non_zero_values = binned_heatmap.values[binned_heatmap.values != 0]
                clip_level = np.percentile(np.abs(non_zero_values), 95) if non_zero_values.size > 0 else 1
                
                fig.add_trace(go.Heatmap(
                    x=binned_heatmap.columns, y=[b.left for b in binned_heatmap.index], z=binned_heatmap.values,
                    colorscale='RdBu', zmid=0, zmin=-clip_level, zmax=clip_level,
                    name='Net Liquidity', hoverinfo='none', colorbar=dict(x=1.0, title='Net Liquidity')
                ))
                fig.add_trace(go.Scatter(
                    x=price_lines_df['datetime'], y=price_lines_df['best_bid'],
                    mode='lines', name='Best Bid', line=dict(color='green', width=2, dash='dash'),
                    hovertemplate='<b>Time:</b> %{x|%H:%M:%S}<br><b>Best Bid:</b> $%{y:.3f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=price_lines_df['datetime'], y=price_lines_df['best_ask'],
                    mode='lines', name='Best Ask', line=dict(color='black', width=2, dash='dash'),
                    hovertemplate='<b>Time:</b> %{x|%H:%M:%S}<br><b>Best Ask:</b> $%{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(height=650, title_text='Market Heatmap Replay', yaxis_title='Price Level')
                st.plotly_chart(fig, use_container_width=True)
