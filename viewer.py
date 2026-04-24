import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import subprocess
import sys
import os

st.set_page_config(layout="wide", page_title="Twitch Sentiment")
st.title("Twitch Sentiment Engine")

# Config
FILE_PATH = "live_data.csv"
WINDOW_SECONDS = 0.5 # window of time to look at chat messages

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
    channel_input = st.text_input("Channel Name", placeholder="xQc")

    # 'Connect' Button
    if st.button("Connect"):
        if st.session_state.process is None:
            # Clear old CSV data so graph starts fresh
            if os.path.exists(FILE_PATH):
                open(FILE_PATH, "w").close()

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
            st.warning("Already running! Disconnect first.")

    # BUTTON: DISCONNECT
    if st.button("Stop / Disconnect"):
        if st.session_state.process:
            # Kill the background process
            st.session_state.process.kill()
            st.session_state.process = None
            st.session_state.connected = False
            st.info("Stopped.")

# --- MAIN DISPLAY ---
status_area = st.empty()
col1, col2 = st.columns([3, 1])

with col1:
    chart_placeholder = st.empty()
    recent_msg_placeholder = st.empty()
with col2:
    metric_placeholder = st.empty()

# This fragment only runs update_dashboard every 0.1s
# This prevents the whole page from lagging or jumping.
@st.fragment(run_every=0.1)
def update_dashboard():
    if st.session_state.connected:
        status_area.subheader(f"Monitoring: {st.session_state.current_channel}")

        # 1. Read CSV (Handle busy file errors)
        if os.path.exists(FILE_PATH):
            try:
                df = pd.read_csv(FILE_PATH)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            except Exception:
                return # Try again next fragment tick

            # 2. Filter Data (Last 2 Seconds)
            if not df.empty and "timestamp" in df.columns:
                current_time = time.time()
                recent_df = df[df["timestamp"] > (current_time - WINDOW_SECONDS)]

                if not recent_df.empty:
                    # 3. Logic: Calculate Averages (Split)
                    pos_totals = []
                    neg_totals = []

                    for _, row in recent_df.iterrows():
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

                    avg_pos = sum(pos_totals) / len(pos_totals)
                    avg_neg = sum(neg_totals) / len(neg_totals)

                    # Update Metrics
                    count = len(recent_df)
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
                        yaxis_range=[0, 1], # KEEP CONSTANT
                        height=400,
                        margin=dict(t=10, b=10),
                        # 100ms transition makes the bars update smoothly 
                        transition={"duration": 100, "easing": "cubic-in-out"},
                    )

                    # Render
                    chart_placeholder.plotly_chart(
                        fig, width='stretch'
                    )

                    # Display Most Recent Message
                    if not df.empty:
                        latest_row = df.iloc[-1]
                        channel = latest_row["channel"]
                        message = latest_row["message"]
                        label = latest_row["label"]
                        
                        # Color code based on sentiment
                        sentiment_color = "🟢" if label.lower() == "positive" else "🔴" if label.lower() == "negative" else "⚪"
                        
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
