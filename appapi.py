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
st.header("📈 Live Market Heatmap")
st.markdown("Click the button in the sidebar to start streaming today's live session.")

# --- 3. API & Data Functions ---
BATCH_SIZE = 100
API_URLS = {
    'LIST_SESSIONS': 'https://list-sessions-897370608024.australia-southeast1.run.app',
    'GET_SESSION_DETAILS': 'https://get-session-details-897370608024.australia-southeast1.run.app',
    'GET_SNAPSHOTS': 'https://get-snapshots-897370608024.australia-southeast1.run.app',
    # --- CHANGE 1: ADDING THE NEW, POWERFUL HEATMAP API ---
    'GET_HEATMAP_DATA': 'https://get-heatmap-data-897370608024.australia-southeast1.run.app'
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
    """Note: Original API only accepted one timestamp per call, so we loop."""
    if not timestamps: return []
    all_snapshots = []
    for ts in timestamps:
        snapshot_url = f"{API_URLS['GET_SNAPSHOTS']}?id={session_id}&timestamps={ts}"
        try:
            response = requests.get(snapshot_url)
            response.raise_for_status()
            all_snapshots.extend(response.json())
        except requests.exceptions.RequestException:
            continue
    return all_snapshots

def parse_rows(batch_data):
    """Helper function to parse JSON data for the metrics dashboard."""
    all_rows = []
    for snapshot in batch_data:
        dt = pd.to_datetime(snapshot.get('timestamp'), unit='ms')
        last_price = snapshot.get('lastPrice')
        total_traded_volume = snapshot.get('totalTradedVolume')
        volume_change = snapshot.get('volumeChange')
        vwap = snapshot.get('VWAP')
        for order in snapshot.get('bids', []):
            all_rows.append({'datetime': dt, 'Type': 'BUY', 'Price': order.get('price'), 'Volume': order.get('size'), 'lastPrice': last_price, 'totalTradedVolume': total_traded_volume, 'volumeChange': volume_change, 'VWAP': vwap})
        for order in snapshot.get('asks', []):
            all_rows.append({'datetime': dt, 'Type': 'SELL', 'Price': order.get('price'), 'Volume': order.get('size'), 'lastPrice': last_price, 'totalTradedVolume': total_traded_volume, 'volumeChange': volume_change, 'VWAP': vwap})
    return all_rows

def initial_load_with_progress(session_id):
    """Performs the initial load of raw data using the original, robust batching method."""
    try:
        details_url = f"{API_URLS['GET_SESSION_DETAILS']}?id={session_id}"
        details_response = requests.get(details_url)
        details_response.raise_for_status()
        timestamps = details_response.json().get('timestamps', [])
        if not timestamps:
            st.warning("Waiting for the first data snapshot...")
            return

        total_timestamps = len(timestamps)
        progress_bar = st.progress(0, text=f"Catching up with live session (0/{total_timestamps})...")
        all_rows = []
        for i in range(0, total_timestamps, BATCH_SIZE):
            batch_timestamps = timestamps[i:i + BATCH_SIZE]
            batch_data = get_snapshots(session_id, batch_timestamps)
            all_rows.extend(parse_rows(batch_data))
            percent_complete = min((i + BATCH_SIZE) / total_timestamps, 1.0)
            progress_text = f"Catching up ({min(i + BATCH_SIZE, total_timestamps)}/{total_timestamps})..."
            progress_bar.progress(percent_complete, text=progress_text)
        progress_bar.empty()
        
        if all_rows:
            df = pd.DataFrame(all_rows)
            df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
            st.session_state.depth_df_raw = df
            st.session_state.last_timestamp = max(timestamps)
    except requests.exceptions.RequestException as e:
        st.error(f"API Error during initial load: {e}")

def incremental_update(session_id):
    """Fetches new raw snapshots for the metrics dashboard."""
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
                new_df['datetime'] = new_df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
                st.session_state.depth_df_raw = pd.concat([st.session_state.depth_df_raw, new_df]).drop_duplicates()
                st.session_state.last_timestamp = max(new_timestamps)
    except requests.exceptions.RequestException:
        pass

def calculate_dashboard_metrics(current_snapshot, prev_metrics):
    """Calculates a full dashboard of metrics from the latest snapshot."""
    metrics = {}
    bids = current_snapshot[current_snapshot['Type'] == 'BUY']
    asks = current_snapshot[current_snapshot['Type'] == 'SELL']
    if bids.empty or asks.empty: return prev_metrics
    if 'lastPrice' in current_snapshot.columns and pd.notna(current_snapshot['lastPrice'].iloc[0]):
        metrics['Last Price'] = current_snapshot['lastPrice'].iloc[0]
        metrics['Total Traded Volume'] = current_snapshot['totalTradedVolume'].iloc[0]
        metrics['Volume Change'] = current_snapshot['volumeChange'].iloc[0]
        metrics['VWAP'] = current_snapshot['VWAP'].iloc[0]
    metrics['Best Bid'] = bids['Price'].max()
    metrics['Best Ask'] = asks['Price'].min()
    metrics['Spread'] = metrics['Best Ask'] - metrics['Best Bid']
    metrics['Best Bid Volume'] = bids[bids['Price'] == metrics['Best Bid']]['Volume'].sum()
    metrics['Best Ask Volume'] = asks[asks['Price'] == metrics['Best Ask']]['Volume'].sum()
    top_10_bids, top_10_asks = bids.nlargest(10, 'Price'), asks.nsmallest(10, 'Price')
    metrics['Top-10 Buy Volume'], metrics['Top-10 Sell Volume'] = top_10_bids['Volume'].sum(), top_10_asks['Volume'].sum()
    metrics['Total Buy Volume'], metrics['Total Sell Volume'] = bids['Volume'].sum(), asks['Volume'].sum()
    metrics['Buy/Sell Delta'] = metrics['Total Buy Volume'] - metrics['Total Sell Volume']
    total_top_10_vol = metrics['Top-10 Buy Volume'] + metrics['Top-10 Sell Volume']
    metrics['Imbalance Ratio'] = (metrics['Top-10 Buy Volume'] / total_top_10_vol) if total_top_10_vol > 0 else 0.5
    metrics['Last Price Change'] = metrics.get('Last Price', 0) - prev_metrics.get('Last Price', metrics.get('Last Price', 0))
    metrics['Best Bid Volume Change'] = metrics.get('Best Bid Volume', 0) - prev_metrics.get('Best Bid Volume', 0)
    metrics['Best Ask Volume Change'] = metrics.get('Best Ask Volume', 0) - prev_metrics.get('Best Ask Volume', 0)
    metrics['Top-10 Buy Volume Change'] = metrics.get('Top-10 Buy Volume', 0) - prev_metrics.get('Top-10 Buy Volume', 0)
    metrics['Top-10 Sell Volume Change'] = metrics.get('Top-10 Sell Volume', 0) - prev_metrics.get('Top-10 Sell Volume', 0)
    metrics['Buy/Sell Delta Change'] = metrics.get('Buy/Sell Delta', 0) - prev_metrics.get('Buy/Sell Delta', 0)
    return metrics

def create_depth_chart(snapshot_df):
    bids = snapshot_df[snapshot_df['Type'] == 'BUY'].sort_values('Price', ascending=False)
    asks = snapshot_df[snapshot_df['Type'] == 'SELL'].sort_values('Price', ascending=True)
    bids['Accumulated'] = bids['Volume'].cumsum()
    asks['Accumulated'] = asks['Volume'].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bids['Price'], y=bids['Accumulated'], mode='lines', name='Bids', line=dict(color='green'), fill='tozeroy'))
    fig.add_trace(go.Scatter(x=asks['Price'], y=asks['Accumulated'], mode='lines', name='Asks', line=dict(color='red'), fill='tozeroy'))
    fig.update_layout(title_text='Market Depth Chart', xaxis_title='Price', yaxis_title='Accumulated Volume', height=400)
    return fig
    
# --- CHANGE 2: ADDING A NEW HELPER FUNCTION TO CALL THE HEATMAP API ---
@st.cache_data(ttl=5) # Cache the result for 5s to avoid re-fetching on simple UI interactions
def fetch_chart_data(session_id, bin_size):
    """Fetches pre-calculated data for the heatmap and overlays from the server."""
    if not API_URLS.get('GET_HEATMAP_DATA') or '<' in API_URLS['GET_HEATMAP_DATA']:
        return None
    try:
        url = f"{API_URLS['GET_HEATMAP_DATA']}?id={session_id}&bin_size={bin_size}"
        response = requests.get(url, timeout=45) # Use a long timeout for the heavy calculation
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

# --- 4. Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.header("1. Live Session")
if 'live_mode_on' not in st.session_state:
    st.session_state.live_mode_on = False
    st.session_state.prev_metrics = {}
    st.session_state.is_paused = False
if st.sidebar.button("▶️ Start Live Session"):
    st.session_state.live_mode_on, st.session_state.is_paused = True, False
    st.session_state.depth_df_raw, st.session_state.last_timestamp, st.session_state.prev_metrics = None, 0, {}
    st.rerun()
if st.session_state.live_mode_on:
    if st.sidebar.button("⏸️ Pause / ▶️ Continue"):
        st.session_state.is_paused = not st.session_state.is_paused
        st.rerun()
    st.sidebar.caption(f"Status: {'PAUSED' if st.session_state.is_paused else 'RUNNING'}")
st.sidebar.header("2. Chart Controls")
# --- CHANGE 3: IMPROVING THE BIN SIZE WIDGET FOR BETTER UX ---
bin_size_options = [0.01, 0.02, 0.05, 0.10, 0.20]
bin_size = st.sidebar.select_slider(
    "Set Price Bin Size ($)",
    options=bin_size_options,
    value=0.05
)

# --- 5. Main Application Logic ---
if not st.session_state.live_mode_on:
    st.info("👋 Welcome! Click 'Start Live Session' in the sidebar to begin.")
else:
    sessions = fetch_available_sessions()
    today_str = datetime.now(pytz.timezone('Australia/Melbourne')).strftime('%Ym%d')
    todays_session_id = next((s['id'] for s in sessions if s.get('id', '').startswith(today_str)), None)
    
    if not todays_session_id:
        st.error(f"Could not find a live session for today ({today_str}). Please check the API.")
    else:
        is_initial_load = st.session_state.get('depth_df_raw') is None
        if is_initial_load:
            initial_load_with_progress(todays_session_id)
        if not st.session_state.get('is_paused', False) and not is_initial_load:
            st_autorefresh(interval=5000, key="data_refresher")
            incremental_update(todays_session_id)
        
        depth_df_raw = st.session_state.get('depth_df_raw')
        if depth_df_raw is None or depth_df_raw.empty:
            st.warning("Waiting for session data...")
        else:
            depth_df_raw['Price'] = pd.to_numeric(depth_df_raw['Price'])
            depth_df_raw['Volume'] = pd.to_numeric(depth_df_raw['Volume'])
            last_update_time = depth_df_raw['datetime'].max()
            st.header(f"Session Liquidity Heatmap (Live - Last Update: {last_update_time.strftime('%H:%M:%S')})")
            
            with st.expander("Show Key Indicators", expanded=True):
                latest_snapshot = depth_df_raw[depth_df_raw['datetime'] == last_update_time]
                metrics = calculate_dashboard_metrics(latest_snapshot, st.session_state.prev_metrics)
                st.session_state.prev_metrics = metrics
                if metrics:
                    top_cols = st.columns(4)
                    top_cols[0].metric("Last Price", f"${metrics.get('Last Price', 0):,.2f}", f"${metrics.get('Last Price Change', 0):.2f}")
                    top_cols[1].metric("VWAP", f"${metrics.get('VWAP', 0):,.4f}")
                    top_cols[2].metric("Total Traded Volume", f"${metrics.get('Total Traded Volume', 0):,}")
                    top_cols[3].metric("Volume Change (Last Tick)", f"${metrics.get('Volume Change', 0):+,.0f}")
                    st.markdown("<hr style='margin-top: -0.5em; margin-bottom: 1em;'>", unsafe_allow_html=True)
                    cols = st.columns(3)
                    cols[0].markdown(f"""**Best Bid / Ask**<br>{'${:,.2f}'.format(metrics.get('Best Bid', 0))} / {'${:,.2f}'.format(metrics.get('Best Ask', 0))}<br><small>(Spread: {'${:,.2f}'.format(metrics.get('Spread',0))})</small>""", unsafe_allow_html=True)
                    cols[0].markdown(f"""**Best Vol (Buy/Sell)**<br>{metrics.get('Best Bid Volume', 0):,} / {metrics.get('Best Ask Volume', 0):,}<br><small>(Δ: {metrics.get('Best Bid Volume Change', 0):+,.0f} / {metrics.get('Best Ask Volume Change', 0):+,.0f})</small>""", unsafe_allow_html=True)
                    cols[1].markdown(f"""**Top 10 Depth (Buy/Sell)**<br>{metrics.get('Top-10 Buy Volume', 0):,} / {metrics.get('Top-10 Sell Volume', 0):,}<br><small>(Δ: {metrics.get('Top-10 Buy Volume Change', 0):+,.0f} / {metrics.get('Top-10 Sell Volume Change', 0):+,.0f})</small>""", unsafe_allow_html=True)
                    cols[1].markdown(f"""**Total Visible Depth (Buy/Sell)**<br>{metrics.get('Total Buy Volume', 0):,} / {metrics.get('Total Sell Volume', 0):,}""", unsafe_allow_html=True)
                    cols[2].markdown(f"""**Top 10 Imbalance**<br>{metrics.get('Imbalance Ratio', 0):.1%} <small>Buy-Side</small>""", unsafe_allow_html=True)
                    cols[2].markdown(f"""**Buy/Sell Delta**<br>{metrics.get('Buy/Sell Delta', 0):+,.0f}""", unsafe_allow_html=True)
            
            st.markdown("---")

            # --- CHANGE 4: REPLACING THE HEAVY CLIENT-SIDE CALCULATION ---
            # Instead of the slow pivot_table logic, we now call our new helper function
            chart_data = fetch_chart_data(todays_session_id, bin_size)

            if not chart_data:
                st.warning("Generating heatmap data on the server...")
            else:
                # The plotting logic now just displays the pre-calculated data
                heatmap = chart_data['heatmap']
                overlays = chart_data['overlays']
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    x=heatmap['x'], y=heatmap['y'], z=heatmap['z'],
                    colorscale='RdBu', zmid=0, name='Net Liquidity', 
                    hoverinfo='none', colorbar=dict(x=1.0, title='Net Liquidity')
                ))
                fig.add_trace(go.Scatter(
                    x=overlays['mid_point']['x'], y=overlays['mid_point']['y'],
                    mode='lines', name='Mid-Point',
                    line=dict(color='rgba(0, 0, 0, 0.8)', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=overlays['vwap']['x'], y=overlays['vwap']['y'],
                    mode='lines', name='VWAP',
                    line=dict(color='rgba(0, 0, 0, 1)', width=2, dash='solid')
                ))
                fig.update_layout(height=650, title_text='Market Heatmap with Price Overlay', yaxis_title='Price Level')
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show Market Depth Chart"):
                current_snapshot_df = depth_df_raw[depth_df_raw['datetime'] == last_update_time]
                if not current_snapshot_df.empty:
                    depth_fig = create_depth_chart(current_snapshot_df)
                    st.plotly_chart(depth_fig, use_container_width=True)
                else:
                    st.write("No snapshot data to display.")

