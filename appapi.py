import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_autorefresh import st_autorefresh

# --- 1. Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Market Heatmap Dashboard"
)

# --- 2. Application Title ---
st.title("📈 Live Market Heatmap")
st.markdown("Select a session to begin streaming live market depth. Uploading a sales file is optional.")

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

def load_depth_data_from_api(session_id):
    """Fetches and processes live market depth data from the API."""
    if not session_id: return None
    try:
        details_url = f"{API_URLS['GET_SESSION_DETAILS']}?id={session_id}"
        details_response = requests.get(details_url)
        details_response.raise_for_status()
        timestamps = details_response.json().get('timestamps', [])
        if not timestamps:
            st.warning(f"No timestamps found for session: {session_id}")
            return None

        all_rows = []
        BATCH_SIZE = 100
        # Spinner is helpful for the initial, potentially large, load
        with st.spinner(f"Fetching {len(timestamps)} snapshots..."):
            for i in range(0, len(timestamps), BATCH_SIZE):
                batch_timestamps = timestamps[i:i + BATCH_SIZE]
                timestamps_str = ",".join(map(str, batch_timestamps))
                snapshot_url = f"{API_URLS['GET_SNAPSHOTS']}?id={session_id}&timestamps={timestamps_str}"
                snapshot_response = requests.get(snapshot_url)
                if snapshot_response.ok:
                    for snapshot in snapshot_response.json():
                        dt = pd.to_datetime(snapshot.get('timestamp'), unit='ms')
                        for order in snapshot.get('bids', []):
                            all_rows.append({'datetime': dt, 'Type': 'BUY', 'Price': order.get('price'), 'Volume': order.get('size')})
                        for order in snapshot.get('asks', []):
                            all_rows.append({'datetime': dt, 'Type': 'SELL', 'Price': order.get('price'), 'Volume': order.get('size')})
        if not all_rows:
            st.warning("API returned no valid depth data.")
            return None

        depth_df = pd.DataFrame(all_rows)
        depth_df.dropna(inplace=True)
        for col in ['Price', 'Volume']:
            depth_df[col] = pd.to_numeric(depth_df[col])
        depth_df['datetime'] = depth_df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
        depth_df.sort_values('datetime', inplace=True)
        return depth_df
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not load session data. {e}")
        return None

def calculate_mid_point(df):
    """Calculates the best bid, best ask, and mid-point for each timestamp."""
    mid_points = []
    for timestamp, group in df.groupby('datetime'):
        bids = group[group['Type'] == 'BUY']
        asks = group[group['Type'] == 'SELL']
        if not bids.empty and not asks.empty:
            best_bid = bids['Price'].max()
            best_ask = asks['Price'].min()
            mid_point = (best_bid + best_ask) / 2
            mid_points.append({'datetime': timestamp, 'mid_point': mid_point})
    return pd.DataFrame(mid_points)

@st.cache_data
def process_sales_data(sales_upload, trade_date):
    """Loads and prepares the manually uploaded course of sales data."""
    if not sales_upload: return None
    try:
        sales_df = pd.read_csv(sales_upload)
        sales_df['datetime'] = pd.to_datetime(sales_df['Time'].str.strip(), format='%I:%M:%S %p').apply(
            lambda t: pd.to_datetime(f"{trade_date} {t.time()}") if pd.notnull(t) else pd.NaT
        ).dt.tz_localize('Australia/Melbourne')
        sales_df['Price'] = pd.to_numeric(sales_df['Price $'], errors='coerce')
        sales_df['Volume'] = pd.to_numeric(sales_df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
        sales_df.dropna(subset=['datetime', 'Price', 'Volume'], inplace=True)
        return sales_df
    except Exception as e:
        st.error(f"Error reading or processing sales CSV file: {e}")
        return None

# --- 4. Sidebar Controls ---
st.sidebar.title("Controls")
st_autorefresh(interval=5000, key="data_refresher")

st.sidebar.header("1. Select Session")
sessions = fetch_available_sessions()
session_names = [s.get('name', s.get('id')) for s in sessions]

# --- MODIFIED: Add a placeholder to prevent auto-loading ---
placeholder = "--- Select a session ---"
options = [placeholder] + session_names
selected_session_name = st.sidebar.selectbox("Choose a session", options=options)

selected_session_id = None
# Only find an ID if the user has selected a real session
if selected_session_name != placeholder:
    selected_session_id = next((s.get('id') for s in sessions if s.get('name', s.get('id')) == selected_session_name), None)


st.sidebar.header("2. Upload Sales File (Optional)")
uploaded_sales_file = st.sidebar.file_uploader("Upload Course of Sales CSV", type="csv")

st.sidebar.header("3. Chart Controls")
bin_size = st.sidebar.number_input(
    "Set Price Bin Size ($)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f"
)

# --- 5. Main Application Logic ---
if selected_session_id:
    depth_df = load_depth_data_from_api(selected_session_id)

    if depth_df is not None and not depth_df.empty:
        trade_date = depth_df['datetime'].iloc[0].date()
        
        mid_point_df = calculate_mid_point(depth_df)
        sales_df = process_sales_data(uploaded_sales_file, trade_date)
        
        price_line_source = "Calculated Mid-Point"
        if sales_df is not None and not sales_df.empty:
            price_line_source = st.sidebar.radio(
                "Price Line Source",
                ("Actual Traded Price", "Calculated Mid-Point"),
                index=0
            )

        trade_date_str = trade_date.strftime('%Y-%m-%d')
        SESSION_START = pd.to_datetime(f"{trade_date_str} 10:00:00").tz_localize('Australia/Melbourne')
        SESSION_END = pd.to_datetime(f"{trade_date_str} 16:00:00").tz_localize('Australia/Melbourne')
        
        depth_df = depth_df[(depth_df['datetime'] >= SESSION_START) & (depth_df['datetime'] < SESSION_END)]
        
        if price_line_source == "Actual Traded Price" and sales_df is not None:
            price_df = sales_df[(sales_df['datetime'] >= SESSION_START) & (sales_df['datetime'] < SESSION_END)]
            price_line_data = {'x': price_df['datetime'], 'y': price_df['Price'], 'name': 'Trade Price', 'dash': 'solid'}
            custom_data = np.stack((price_df['Volume'],), axis=-1)
            hover_template = '<b>Time:</b> %{x|%H:%M:%S}<br><b>Price:</b> $%{y:.3f}<br><b>Volume:</b> %{customdata[0]:,}<extra></extra>'
        else:
            price_df = mid_point_df[(mid_point_df['datetime'] >= SESSION_START) & (mid_point_df['datetime'] < SESSION_END)]
            price_line_data = {'x': price_df['datetime'], 'y': price_df['mid_point'], 'name': 'Mid-Point', 'dash': 'dash'}
            custom_data = None
            hover_template = '<b>Time:</b> %{x|%H:%M:%S}<br><b>Mid-Point:</b> $%{y:.3f}<extra></extra>'

        if depth_df.empty or price_df.empty:
            st.warning("Waiting for data within the main trading session (10:00 AM - 4:00 PM)...")
        else:
            last_update_time = depth_df['datetime'].max().strftime('%H:%M:%S')
            st.header(f"Session Liquidity Heatmap (Live - Last Update: {last_update_time})")

            depth_df['SignedVolume'] = np.where(depth_df['Type'] == 'BUY', depth_df['Volume'], -depth_df['Volume'])
            heatmap_pivot = depth_df.pivot_table(index='Price', columns='datetime', values='SignedVolume', aggfunc='sum').fillna(0)
            
            min_price = price_df[price_line_data['y'].name].min() - 0.50
            max_price = price_df[price_line_data['y'].name].max() + 0.50
            price_bins = np.arange(np.floor(min_price), np.ceil(max_price) + bin_size, bin_size)
            binned_heatmap = heatmap_pivot.groupby(pd.cut(heatmap_pivot.index, bins=price_bins, right=False), observed=False).sum()

            fig = go.Figure()
            non_zero_values = binned_heatmap.values[binned_heatmap.values != 0]
            clip_level = np.percentile(np.abs(non_zero_values), 95) if non_zero_values.size > 0 else 1
            
            fig.add_trace(go.Heatmap(
                x=binned_heatmap.columns, y=[interval.left for interval in binned_heatmap.index], z=binned_heatmap.values,
                colorscale='RdBu', zmid=0, zmin=-clip_level, zmax=clip_level,
                name='Net Liquidity', hoverinfo='none', colorbar=dict(x=1.0, title='Net Liquidity')
            ))
            
            fig.add_trace(go.Scatter(
                x=price_line_data['x'], y=price_line_data['y'], customdata=custom_data,
                mode='lines', name=price_line_data['name'], 
                line=dict(color='rgba(0, 0, 0, 0.8)', width=2, dash=price_line_data['dash']),
                hovertemplate=hover_template
            ))
            
            fig.update_layout(height=650, title_text='Market Heatmap with Price Overlay', yaxis_title='Price Level')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Welcome! Please select a session to begin streaming.")
