import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# --- 1. Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Market Heatmap Dashboard"
)

# --- 2. Application Title ---
st.title("📈 Market Heatmap with Price Overlay")
st.markdown("Select a session and upload the corresponding sales file to generate the heatmap.")


# --- 3. API Configuration and Data Fetching Functions ---

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
        sessions = response.json()
        return sorted(sessions, key=lambda x: x.get('id', ''), reverse=True)
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not fetch sessions. {e}")
        return []

def load_depth_data_from_api(session_id):
    """
    Fetches market depth data from the API and standardizes it to the
    'Australia/Melbourne' timezone.
    """
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
        BATCH_SIZE = 1000
        total_timestamps = len(timestamps)
        progress_bar = st.progress(0, text=f"Preparing to fetch {total_timestamps} snapshots...")
        with st.spinner("Downloading data..."):
            for i in range(0, total_timestamps, BATCH_SIZE):
                batch_timestamps = timestamps[i:i + BATCH_SIZE]
                timestamps_str = ",".join(map(str, batch_timestamps))
                snapshot_url = f"{API_URLS['GET_SNAPSHOTS']}?id={session_id}&timestamps={timestamps_str}"
                snapshot_response = requests.get(snapshot_url)
                if snapshot_response.ok:
                    batch_data = snapshot_response.json()
                    for snapshot in batch_data:
                        # Interpret the timestamp as milliseconds ('ms')
                        dt = pd.to_datetime(snapshot.get('timestamp'), unit='ms')
                        for order in snapshot.get('bids', []):
                            all_rows.append({'datetime': dt, 'Type': 'BUY', 'Price': order.get('price'), 'Volume': order.get('size')})
                        for order in snapshot.get('asks', []):
                            all_rows.append({'datetime': dt, 'Type': 'SELL', 'Price': order.get('price'), 'Volume': order.get('size')})

                percent_complete = min((i + BATCH_SIZE) / total_timestamps, 1.0)
                progress_text = f"Fetched {min(i + BATCH_SIZE, total_timestamps)} / {total_timestamps} snapshots"
                progress_bar.progress(percent_complete, text=progress_text)
        progress_bar.empty()
        if not all_rows:
            st.warning("API returned no valid depth data.")
            return None

        depth_df = pd.DataFrame(all_rows)
        depth_df.dropna(inplace=True)
        depth_df['Price'] = pd.to_numeric(depth_df['Price'])
        depth_df['Volume'] = pd.to_numeric(depth_df['Volume'])
        depth_df['datetime'] = depth_df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
        depth_df.sort_values('datetime', inplace=True)
        return depth_df
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not load session data. {e}")
        return None

# --- 4. Data Processing for Sales File ---
@st.cache_data
def process_sales_data(sales_upload, trade_date):
    """Loads and prepares the manually uploaded course of sales data."""
    if sales_upload is None: return None
    try:
        sales_df = pd.read_csv(sales_upload)
        sales_df['datetime'] = pd.to_datetime(sales_df['Time'].str.strip(), format='%I:%M:%S %p').apply(
            lambda t: pd.to_datetime(f"{trade_date} {t.time()}") if pd.notnull(t) else pd.NaT
        )
        sales_df['datetime'] = sales_df['datetime'].dt.tz_localize('Australia/Melbourne')
        sales_df['Price'] = pd.to_numeric(sales_df['Price $'], errors='coerce')
        sales_df['Volume'] = pd.to_numeric(sales_df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
        sales_df.dropna(subset=['datetime', 'Price', 'Volume'], inplace=True)
        return sales_df
    except Exception as e:
        st.error(f"Error reading or processing sales CSV file: {e}")
        return None

# --- 5. Sidebar for Controls ---
st.sidebar.title("Controls")
st.sidebar.header("1. Select Session")
sessions = fetch_available_sessions()
session_names = [s.get('name', s.get('id')) for s in sessions]
selected_session_name = st.sidebar.selectbox("Choose a session", options=session_names)

selected_session_id = None
if selected_session_name:
    for session in sessions:
        if session.get('name', session.get('id')) == selected_session_name:
            selected_session_id = session.get('id')
            break
st.sidebar.header("2. Upload Sales File")
uploaded_sales_file = st.sidebar.file_uploader("Upload Course of Sales CSV", type="csv")
st.sidebar.header("3. Chart Controls")
# CORRECTED: Variable name now has one underscore
bin_size = st.sidebar.number_input(
    label="Set Price Bin Size ($)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f"
)

# --- 6. Main Application Logic ---
if selected_session_id and uploaded_sales_file:
    depth_df = load_depth_data_from_api(selected_session_id)
    if depth_df is not None and not depth_df.empty:
        trade_date = depth_df['datetime'].iloc[0].date()
        sales_df = process_sales_data(uploaded_sales_file, trade_date)
        if sales_df is not None and not sales_df.empty:
            st.header("Session Liquidity Heatmap")

            trade_date_str = trade_date.strftime('%Y-%m-%d')
            SESSION_START = pd.to_datetime(f"{trade_date_str} 10:00:00").tz_localize('Australia/Melbourne')
            SESSION_END = pd.to_datetime(f"{trade_date_str} 16:00:00").tz_localize('Australia/Melbourne')

            depth_df = depth_df[(depth_df['datetime'] >= SESSION_START) & (depth_df['datetime'] < SESSION_END)]
            sales_df = sales_df[(sales_df['datetime'] >= SESSION_START) & (sales_df['datetime'] < SESSION_END)]

            if depth_df.empty or sales_df.empty:
                 st.warning("No data found within the main trading session (10:00 AM - 4:00 PM).")
            else:
                # --- Prepare Data for Heatmap ---
                depth_df['SignedVolume'] = np.where(depth_df['Type'] == 'BUY', depth_df['Volume'], -depth_df['Volume'])
                heatmap_pivot = depth_df.pivot_table(index='Price', columns='datetime', values='SignedVolume', aggfunc='sum').fillna(0)
                min_price = sales_df['Price'].min() - 0.50
                max_price = sales_df['Price'].max() + 0.50
                price_bins = np.arange(np.floor(min_price), np.ceil(max_price) + bin_size, bin_size)
                binned_heatmap = heatmap_pivot.groupby(pd.cut(heatmap_pivot.index, bins=price_bins, right=False), observed=False).sum()

                # --- Create Plotly Figure ---
                fig = go.Figure()
                non_zero_values = binned_heatmap.values[binned_heatmap.values != 0]
                clip_level = np.percentile(np.abs(non_zero_values), 95) if non_zero_values.size > 0 else 1
                fig.add_trace(go.Heatmap(
                    x=binned_heatmap.columns, y=[interval.left for interval in binned_heatmap.index], z=binned_heatmap.values,
                    colorscale='RdBu', zmid=0, zmin=-clip_level, zmax=clip_level,
                    name='Net Liquidity', hoverinfo='none', colorbar=dict(x=1.0, title='Net Liquidity')
                ))
                fig.add_trace(go.Scatter(
                    x=sales_df['datetime'], y=sales_df['Price'],
                    customdata=np.stack((sales_df['Volume'],), axis=-1),
                    mode='lines', name='Trade Price', line=dict(color='rgba(0, 0, 0, 0.8)', width=2),
                    hovertemplate='<b>Time:</b> %{x|%H:%M:%S}<br><b>Price:</b> $%{y:.3f}<br><b>Volume:</b> %{customdata[0]:,}<extra></extra>'
                ))
                fig.update_layout(height=650, title_text='Market Heatmap with Price Overlay', yaxis_title='Price Level')
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Welcome! Please select a session and upload the corresponding sales file to begin.")
