import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Live Market Heatmap")

# --- 2. Application Title ---
st.title("📈 Live Market Heatmap")
st.markdown("Click the button in the sidebar to start streaming today's live session.")

# --- 3. API & Data Functions ---
API_URLS = {
    'LIST_SESSIONS': 'https://list-sessions-897370608024.australia-southeast1.run.app',
    'GET_SESSION_DETAILS': 'https://get-session-details-897370608024.australia-southeast1.run.app',
    'GET_SNAPSHOTS': 'https://get-snapshots-897370608024.australia-southeast1.run.app'
}

@st.cache_data(ttl=300)
def fetch_available_sessions():
    """Fetches the list of available sessions from the Firestore API."""
    try:
        response = requests.get(API_URLS['LIST_SESSIONS'])
        response.raise_for_status()
        return sorted(response.json(), key=lambda x: x.get('id', ''), reverse=True)
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not fetch sessions. {e}")
        return []

def get_snapshots(session_id, timestamps):
    """Helper function to fetch a batch of snapshots."""
    if not timestamps: return []
    timestamps_str = ",".join(map(str, timestamps))
    snapshot_url = f"{API_URLS['GET_SNAPSHOTS']}?id={session_id}&timestamps={timestamps_str}"
    response = requests.get(snapshot_url)
    if response.ok:
        return response.json()
    return []

def parse_rows(batch_data):
    """Helper function to parse JSON data into a list of dicts."""
    all_rows = []
    for snapshot in batch_data:
        dt = pd.to_datetime(snapshot.get('timestamp'), unit='ms')
        for order in snapshot.get('bids', []):
            all_rows.append({'datetime': dt, 'Type': 'BUY', 'Price': order.get('price'), 'Volume': order.get('size')})
        for order in snapshot.get('asks', []):
            all_rows.append({'datetime': dt, 'Type': 'SELL', 'Price': order.get('price'), 'Volume': order.get('size')})
    return all_rows

def update_live_data(session_id):
    """Fetches only new data since the last update."""
    try:
        details_url = f"{API_URLS['GET_SESSION_DETAILS']}?id={session_id}"
        details_response = requests.get(details_url)
        details_response.raise_for_status()
        all_timestamps = details_response.json().get('timestamps', [])
        
        last_timestamp = st.session_state.get('last_timestamp', 0)
        new_timestamps = [ts for ts in all_timestamps if ts > last_timestamp]

        if new_timestamps:
            new_rows = parse_rows(get_snapshots(session_id, new_timestamps))
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                
                # --- FIX PART 1: ADD TIMEZONE LOGIC HERE ---
                new_df['datetime'] = new_df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')

                if st.session_state.depth_df_raw is not None:
                    st.session_state.depth_df_raw = pd.concat([st.session_state.depth_df_raw, new_df]).drop_duplicates()
                else:
                    st.session_state.depth_df_raw = new_df
                
                st.session_state.last_timestamp = max(new_timestamps)
    except requests.exceptions.RequestException as e:
        st.error(f"API Error during update: {e}")

def calculate_mid_point(df):
    """Calculates the mid-point for each timestamp in a vectorized way."""
    bids = df[df['Type'] == 'BUY'].groupby('datetime')['Price'].max().rename('best_bid')
    asks = df[df['Type'] == 'SELL'].groupby('datetime')['Price'].min().rename('best_ask')
    merged_df = pd.concat([bids, asks], axis=1).dropna()
    merged_df['mid_point'] = (merged_df['best_bid'] + merged_df['best_ask']) / 2
    return merged_df.reset_index()

# --- 4. Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.header("1. Live Session")

if 'live_mode_on' not in st.session_state:
    st.session_state.live_mode_on = False
    st.session_state.depth_df_raw = None
    st.session_state.last_timestamp = 0

if st.sidebar.button("▶️ Start Live Session"):
    st.session_state.live_mode_on = True
    st.session_state.depth_df_raw = None
    st.session_state.last_timestamp = 0
    st.rerun()

if st.sidebar.button("⏹️ Stop Live Session"):
    st.session_state.live_mode_on = False
    st.rerun()

st.sidebar.header("2. Chart Controls")
bin_size = st.sidebar.number_input(
    "Set Price Bin Size ($)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f"
)

# --- 5. Main Application Logic ---
if not st.session_state.live_mode_on:
    st.info("👋 Welcome! Click 'Start Live Session' in the sidebar to begin.")
else:
    st_autorefresh(interval=5000, key="data_refresher")

    sessions = fetch_available_sessions()
    au_tz = pytz.timezone('Australia/Melbourne')
    today_str = datetime.now(au_tz).strftime('%Y%m%d')
    
    todays_session_id = next((s['id'] for s in sessions if s.get('id', '').startswith(today_str)), None)
    
    if not todays_session_id:
        st.error(f"Could not find a live session for today ({today_str}). Please check the API.")
    else:
        update_live_data(todays_session_id)
        depth_df_raw = st.session_state.get('depth_df_raw')

        if depth_df_raw is None or depth_df_raw.empty:
            st.warning("Waiting for the first data snapshot...")
        else:
            # --- FIX PART 2: REMOVE THE INCORRECT LINE FROM HERE ---
            # This line is no longer needed as conversion is done in update_live_data
            # depth_df_raw['datetime'] = pd.to_datetime(depth_df_raw['datetime']).dt.tz_convert('Australia/Melbourne')
            
            # Ensure dtypes are correct after potential concat operations
            depth_df_raw['Price'] = pd.to_numeric(depth_df_raw['Price'])
            depth_df_raw['Volume'] = pd.to_numeric(depth_df_raw['Volume'])

            trade_date = depth_df_raw['datetime'].iloc[0].date()
            price_df = calculate_mid_point(depth_df_raw)

            SESSION_START = pd.to_datetime(f"{trade_date} 10:00:00").tz_localize('Australia/Melbourne')
            SESSION_END = pd.to_datetime(f"{trade_date} 16:00:00").tz_localize('Australia/Melbourne')
            
            depth_df = depth_df_raw[(depth_df_raw['datetime'] >= SESSION_START) & (depth_df_raw['datetime'] < SESSION_END)]
            price_df = price_df[(price_df['datetime'] >= SESSION_START) & (price_df['datetime'] < SESSION_END)]

            last_update_time = depth_df_raw['datetime'].max().strftime('%H:%M:%S')
            st.header(f"Session Liquidity Heatmap (Live - Last Update: {last_update_time})")
            
            if depth_df.empty or price_df.empty:
                st.warning("Waiting for data within the main trading session (10:00 AM - 4:00 PM)...")
            else:
                depth_df['SignedVolume'] = np.where(depth_df['Type'] == 'BUY', depth_df['Volume'], -depth_df['Volume'])
                min_price = price_df['mid_point'].min() - 0.50
                max_price = price_df['mid_point'].max() + 0.50
                price_bins = np.arange(np.floor(min_price), np.ceil(max_price) + bin_size, bin_size)
                
                depth_df['price_bin'] = pd.cut(depth_df['Price'], bins=price_bins, right=False)
                binned_heatmap = depth_df.pivot_table(
                    index='price_bin', columns='datetime', values='SignedVolume', aggfunc='sum', observed=True
                ).fillna(0)

                fig = go.Figure()
                non_zero_values = binned_heatmap.values[binned_heatmap.values != 0]
                clip_level = np.percentile(np.abs(non_zero_values), 95) if non_zero_values.size > 0 else 1
                
                fig.add_trace(go.Heatmap(
                    x=binned_heatmap.columns, y=[interval.left for interval in binned_heatmap.index], z=binned_heatmap.values,
                    colorscale='RdBu', zmid=0, zmin=-clip_level, zmax=clip_level,
                    name='Net Liquidity', hoverinfo='none', colorbar=dict(x=1.0, title='Net Liquidity')
                ))
                
                fig.add_trace(go.Scatter(
                    x=price_df['datetime'], y=price_df['mid_point'],
                    mode='lines', name='Mid-Point', line=dict(color='rgba(0, 0, 0, 0.8)', width=2, dash='dash'),
                    hovertemplate='<b>Time:</b> %{x|%H:%M:%S}<br><b>Mid-Point:</b> $%{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(height=650, title_text='Market Heatmap with Price Overlay', yaxis_title='Price Level')
                st.plotly_chart(fig, use_container_width=True)
