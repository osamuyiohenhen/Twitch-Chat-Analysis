import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import time
import subprocess
import sys
import os

st.set_page_config(layout="wide", page_title="Twitch Sentiment")
st.title("Twitch Sentiment Engine")

# Config
# from primary import DB_PATH
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twitch_data.db")
WINDOW_SECONDS = 2  # window of time to look at chat messages

# Session State
# This remembers if we are connected even if the graph refreshes
if "process" not in st.session_state:
    st.session_state.process = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "current_channel" not in st.session_state:
    st.session_state.current_channel = ""

# Sidebar (Connection Area)
with st.sidebar:
    st.header("Connection")
    with st.form("connect_form"):
        channel_input = st.text_input("Channel Name", placeholder="xQc")
        submitted = st.form_submit_button("Connect")
        if submitted:
            if st.session_state.process is None:
                # Clear old CSV data so graph starts fresh
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)

                # Launch 'run.py' in the background
                # sys.executable guarantees we use the same Python venv
                cmd = [sys.executable, "run.py", "--channel", channel_input]
                p = subprocess.Popen(cmd)

                # 3. Save state
                st.session_state.process = p
                st.session_state.connected = True
                st.session_state.current_channel = channel_input
                st.success(f"Connected to {channel_input}!")
            else:
                st.warning("Model already running. Disconnect first.")

    # Disconnect Button
    if st.button("Stop / Disconnect"):
        if st.session_state.process:
            st.session_state.process.terminate()
            st.session_state.process = None
            st.session_state.connected = False
            st.session_state.current_channel = ""

    # Disclaimer
    st.sidebar.warning("Note: Multiple simultaneous users may cause mixed results.")


# Main display
status_area = st.empty()
col1, col2 = st.columns([3, 1])

with col1:
    chart_placeholder = st.empty()
    recent_msg_placeholder = st.empty()
with col2:
    metric_placeholder = st.empty()


# This fragment only runs update_dashboard every 0.5s
# This prevents the whole page from lagging or jumping.
@st.fragment(run_every=0.5)
def update_dashboard():
    if st.session_state.connected:
        status_area.subheader(f"Monitoring: {st.session_state.current_channel}")
            
        if os.path.exists(DB_PATH):
            try:
                # 1. Ask SQLite for ONLY the last 2 seconds of data
                cutoff_time = time.time() - WINDOW_SECONDS
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query(
                    "SELECT * FROM chat_log WHERE timestamp > ?", 
                    conn, 
                    params=(cutoff_time,)
                )
                conn.close()
            except Exception as e:
                return  # Skip frame if DB is temporarily locked

            # 2. Filter Data (Last 2 Seconds)
            if not df.empty:
                pos_totals = []
                neg_totals = []

                for _, row in df.iterrows():
                    score = float(row["score"])
                    label = row["label"]

                    if label == "positive":
                        pos_totals.append(score)
                        neg_totals.append(1.0 - score)
                    elif label == "negative":
                        pos_totals.append(1.0 - score)
                        neg_totals.append(score)
                    else:  # neutral
                        pos_totals.append(0.5)
                        neg_totals.append(0.5)

                avg_pos = sum(pos_totals) / len(pos_totals) if pos_totals else 0
                avg_neg = sum(neg_totals) / len(neg_totals) if neg_totals else 0

                # Update Metrics
                count = len(df)
                metric_placeholder.metric(
                    label=f"Avg Sentiment ({count} msgs)",
                    value=f"{'Positive' if avg_pos > avg_neg else 'Negative'}",
                )
                
                # Build Graph (Plotly for colors, fixed axis, and smooth transitions)
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=["Positive", "Negative"],
                            y=[avg_pos, avg_neg],
                            marker_color=["#00CC96", "#EF553B"], # Green and Red
                        )
                    ]
                )

                fig.update_layout(
                    yaxis_range=[0, 1],
                    height=400,
                    margin=dict(t=10, b=10),
                    transition={"duration": 500, "easing": "linear"}, # Matches the session fragment refresh
                )

                # Render
                chart_placeholder.plotly_chart(fig, width="stretch")

                # Display Most Recent Message
                latest_row = df.iloc[-1]
                channel = latest_row["channel"]
                message = latest_row["message"]
                label = latest_row["label"]

                # Color code based on sentiment
                sentiment_color = (
                    "🟢" if label.lower() == "positive"
                    else "🔴" if label.lower() == "negative"
                    else "⚪"
                )

                recent_msg_placeholder.markdown(
                    f"**Latest Message** {sentiment_color}\n\n"
                    f"**{channel}**: _{message}_"
                )

            else:
                metric_placeholder.info("Waiting for chat...")
    else:
        status_area.info("👈 Enter a channel and click Connect to start.")


# Run the fragment
update_dashboard()
