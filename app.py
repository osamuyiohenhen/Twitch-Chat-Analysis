import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import subprocess
import sys
import os

st.set_page_config(layout="wide", page_title="Twitch Sentiment")
st.title("Twitch Sentiment Engine")

# Config
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twitch_data.db")
LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session.lock")
WINDOW_SECONDS = 1  # window of time to look at chat messages

# Session State
if "vod_offset" not in st.session_state:
    st.session_state.vod_offset = None
if "vod_id" not in st.session_state:
    st.session_state.vod_id = None
if "smoothed_pos" not in st.session_state:
    st.session_state.smoothed_pos = 0.5
if "smoothed_neg" not in st.session_state:
    st.session_state.smoothed_neg = 0.5
if "process" not in st.session_state:
    st.session_state.process = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "current_channel" not in st.session_state:
    st.session_state.current_channel = ""
if "pending_vod_url" not in st.session_state:
    st.session_state.pending_vod_url = None
if "pending_vod_time" not in st.session_state:
    st.session_state.pending_vod_time = None
if "seen_intro" not in st.session_state:
    st.session_state.seen_intro = False

if not st.session_state.seen_intro:
    with st.expander("ℹ️ Instructions", expanded=True):
        st.markdown("""
        **Getting Started:**
        1. Enter a Twitch channel name and click Connect.
        2. You'll see 2 live sentiment bars (Positive and Negative) every moment as chats come in.
        3. After about 30 seconds, a timeline will appear below showing chat activity broken down by positive, neutral, and negative messages per minute.
        4. Double-click any bar on the timeline to open a link to that moment in the VOD.
        
        ⚠️ **Note:** If the page becomes unresponsive or turns white: 1) Reload the page, 2) Click "Force Disconnect" in the sidebar, then reconnect.
        """)
        if st.button("Got it"):
            st.session_state.seen_intro = True
            st.rerun()


# Sidebar (Connection Area)
with st.sidebar:
    st.header("Connection")
    with st.form("connect_form"):
        channel_input = st.text_input("Channel Name", placeholder="xQc")
        submitted = st.form_submit_button("Connect")
        if submitted:
            if os.path.exists(LOCK_FILE):
                st.warning("A session appears to already be running (possibly from a crashed page).Click 'Force Disconnect' below if this is incorrect.")
            elif st.session_state.process is None:
                # Clear old CSV data so graph starts fresh
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)

                # Launch 'run.py' in the background
                # sys.executable guarantees we use the same Python venv
                cmd = [sys.executable, "run.py", "--channel", channel_input]
                p = subprocess.Popen(cmd)

                with open(LOCK_FILE, "w") as f:
                    f.write(str(p.pid))

                # Save state
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
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

    if os.path.exists(LOCK_FILE):
        if st.button("⚠️ Force Disconnect (clears stuck session)"):
            try:
                with open(LOCK_FILE) as f:
                    old_pid = int(f.read().strip())
                import signal
                os.kill(old_pid, signal.SIGTERM)
            except Exception:
                pass  # process might already be dead
            os.remove(LOCK_FILE)
            st.session_state.process = None
            st.session_state.connected = False
            st.rerun()

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


# This fragment only runs update_dashboard every 0.2s
@st.fragment(run_every=0.2)
def update_dashboard():
    if st.session_state.connected:
        status_area.subheader(
            f"Monitoring: {st.session_state.current_channel.capitalize()}"
        )

        if os.path.exists(DB_PATH):
            try:
                # Query database for the last 2 seconds of data
                cutoff_time = time.time() - WINDOW_SECONDS
                conn = sqlite3.connect(DB_PATH)
                df_live = pd.read_sql_query(
                    "SELECT * FROM chat_log WHERE timestamp > ?",
                    conn,
                    params=(cutoff_time,),
                )
                conn.close()
            except Exception:
                return  # Skip frame if DB is temporarily locked

            # Filter Data (Last 2 Seconds)
            if not df_live.empty:
                pos_totals = []
                neg_totals = []

                for _, row in df_live.iterrows():
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

                avg_pos = sum(pos_totals) / len(pos_totals) if pos_totals else 0.5
                avg_neg = sum(neg_totals) / len(neg_totals) if neg_totals else 0.5

                # Smooth bar and blend with previous values (0.4 = 40% new, 60% old)
                smoothing_factor = 0.4
                st.session_state.smoothed_pos = (
                    smoothing_factor * avg_pos
                    + (1 - smoothing_factor) * st.session_state.smoothed_pos
                )
                st.session_state.smoothed_neg = (
                    smoothing_factor * avg_neg
                    + (1 - smoothing_factor) * st.session_state.smoothed_neg
                )

                # Use smoothed values for display
                pos_percent = int(st.session_state.smoothed_pos * 100)
                neg_percent = int(st.session_state.smoothed_neg * 100)

                # Update Metrics
                count = len(df_live)
                metric_placeholder.metric(
                    label=f"Avg Sentiment ({count} msgs)",
                    value=f"{'Positive' if avg_pos > avg_neg else 'Negative'}",
                )

                # Target the chart placeholder container with styled HTML/JS to avoid flickers
                with chart_placeholder.container():
                    st.html(f"""
                        <div style="
                            display: flex; 
                            justify-content: center; 
                            align-items: flex-end; 
                            height: 400px; 
                            width: 100%; 
                            background-color: var(--background-color); 
                            padding: 20px 20px 40px 20px; 
                            box-sizing: border-box; 
                            font-family: sans-serif; 
                            gap: 60px;
                        ">
                            <div style="display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: flex-end; width: 250px;">
                                <div style="color: var(--text-color); font-weight: bold; font-size: 18px; margin-bottom: 10px;">{pos_percent}%</div>
                                <div style="background-color: #00CC96; width: 100%; height: {pos_percent}%; border-radius: 6px 6px 0 0; transition: height 0.2s ease-in-out; box-shadow: 0 4px 16px rgba(0,204,150,0.2);"></div>
                                <div style="color: var(--text-color); font-size: 15px; font-weight: 500; margin-top: 14px;">Positive</div>
                            </div>
                            
                            <div style="display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: flex-end; width: 250px;">
                                <div style="color: var(--text-color); font-weight: bold; font-size: 18px; margin-bottom: 10px;">{neg_percent}%</div>
                                <div style="background-color: #EF553B; width: 100%; height: {neg_percent}%; border-radius: 6px 6px 0 0; transition: height 0.2s ease-in-out; box-shadow: 0 4px 16px rgba(239,85,59,0.2);"></div>
                                <div style="color: var(--text-color); font-size: 15px; font-weight: 500; margin-top: 14px;">Negative</div>
                            </div>
                        </div>
                    """)

                # Display Most Recent Message
                latest_row = df_live.iloc[-1]
                message = latest_row["message"]
                label = latest_row["label"]

                # Color code based on sentiment
                sentiment_color = (
                    "🟢"
                    if label.lower() == "positive"
                    else "🔴"
                    if label.lower() == "negative"
                    else "⚪"
                )

                recent_msg_placeholder.markdown(
                    f"**Latest Message** {sentiment_color}\n\n"
                    f"**{st.session_state.current_channel.capitalize()}**: _{message}_"
                )

            else:
                metric_placeholder.info("Waiting for chat...")
    else:
        status_area.info("👈 Enter a channel and click Connect to start.")


def handle_click():
    selection = st.session_state.timeline_chart["selection"]
    points = selection.get("points", [])
    if points and st.session_state.vod_id:
        clicked_minute = points[0].get("x")
        if clicked_minute and ":" in str(clicked_minute):
            h, m = str(clicked_minute).split(":")
            total_minutes = int(h) * 60 + int(m)
            st.session_state.pending_vod_url = f"https://twitch.tv/videos/{st.session_state.vod_id}?t={total_minutes}m0s"
            st.session_state.pending_vod_time = clicked_minute


@st.fragment(run_every=30)
def session_timeline():
    existing_selection = st.session_state.get("timeline_chart", {}).get("selection", {})
    existing_points = existing_selection.get("points", [])

    # Check if a channel name is valid in the session memory
    if st.session_state.get("connected") and st.session_state.current_channel:
        title_text = (
            f"{st.session_state.current_channel.capitalize()}'s Chat Activity Timeline"
        )
    else:
        title_text = "Chat Activity Timeline"

    if st.session_state.connected and os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df_timeline = pd.read_sql_query(
                """
                SELECT 
                    CAST(timestamp / 60 AS INT) * 60 as bucket,
                    SUM(CASE WHEN label='positive' THEN 1 ELSE 0 END) as pos_count,
                    SUM(CASE WHEN label='negative' THEN 1 ELSE 0 END) as neg_count,
                    SUM(CASE WHEN label='neutral' THEN 1 ELSE 0 END) as neu_count
                FROM chat_log
                GROUP BY bucket
                ORDER BY bucket
            """,
                conn,
            )
            conn.close()
        except Exception:
            return

        if not df_timeline.empty:
            # Fetch VOD offset once per session
            if st.session_state.vod_offset is None:
                conn = sqlite3.connect(DB_PATH)
                session_row = pd.read_sql_query(
                    "SELECT * FROM session_info LIMIT 1", conn
                )
                conn.close()
                if not session_row.empty and session_row["stream_start_time"].iloc[0]:
                    gap = (
                        session_row["monitor_start_time"].iloc[0]
                        - session_row["stream_start_time"].iloc[0]
                    )
                    st.session_state.vod_offset = gap
                    st.session_state.vod_id = session_row["vod_id"].iloc[0]
                else:
                    st.session_state.vod_offset = 0

            # Convert to true VOD-relative minutes
            start_bucket = df_timeline["bucket"].iloc[0]
            df_timeline["minute"] = (
                (df_timeline["bucket"] - start_bucket) / 60
                + (st.session_state.vod_offset / 60)
            ).astype(int)

            # Format as H:MM
            df_timeline["time_label"] = df_timeline["minute"].apply(
                lambda m: f"{m // 60}:{m % 60:02d}"
            )

            fig = go.Figure(
                data=[
                    go.Bar(
                        name="Positive",
                        x=df_timeline["time_label"],
                        y=df_timeline["pos_count"],
                        marker_color="#00CC96",
                        hovertemplate="Timestamp: %{x}<br>Positive: %{y} msgs<extra></extra>",
                    ),
                    go.Bar(
                        name="Neutral",
                        x=df_timeline["time_label"],
                        y=df_timeline["neu_count"],
                        marker_color="#636EFA",
                        hovertemplate="Timestamp: %{x}<br>Neutral: %{y} msgs<extra></extra>",
                    ),
                    go.Bar(
                        name="Negative",
                        x=df_timeline["time_label"],
                        y=df_timeline["neg_count"],
                        marker_color="#EF553B",
                        hovertemplate="Timestamp: %{x}<br>Negative: %{y} msgs<extra></extra>",
                    ),
                ]
            )

            totals = df_timeline[["pos_count", "neu_count", "neg_count"]].sum(axis=1)
            y_cap = np.percentile(totals, 95) * 1.15  # cap slightly above the 95th percentile, not the absolute max


            fig.update_layout(
                yaxis=dict(fixedrange=True),
                barmode="stack",
                bargap=0,
                barcornerradius=3,
                title=title_text,
                xaxis_title="Stream Time (VOD timestamp)",
                yaxis_title="Messages",
                height=300,
                margin=dict(t=40, b=10),
                xaxis=dict(
                tickmode="auto",
                nticks=20,
                ),
            )

            fig.update_traces(marker_line_color="black", marker_line_width=0.5)
            st.plotly_chart(
                fig,
                width="stretch",
                on_select=handle_click,
                selection_mode="points",
                key="timeline_chart",
                config={
                "modeBarButtonsToRemove": ["zoomIn2d", "zoomOut2d", "resetScale2d", ],
                "scrollZoom": False,
                }
            )

            # Show VOD link if one was set by the callback
            if st.session_state.get("pending_vod_url"):
                st.link_button(f"🎬 Open VOD at {st.session_state.pending_vod_time}", st.session_state.pending_vod_url) # noqa


# Run the fragments
update_dashboard()
session_timeline()
