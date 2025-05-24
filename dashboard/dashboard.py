# dashboard.py – Auto-Refreshing Streamlit Dashboard for XAlgo Logs

import pandas as pd
import streamlit as st
from datetime import datetime
import os
import time

st.set_page_config(page_title="XAlgo Signal Dashboard", layout="wide")
st.title("📊 XAlgo Core – Live Signal & Execution Monitor")

# Toggle for auto-refresh
auto_refresh = st.sidebar.checkbox("🔁 Auto Refresh Every 2 Seconds", value=True)
refresh_button = st.sidebar.button("🔄 Manual Refresh")

# Refresh logic
if auto_refresh:
    time.sleep(2)
    try:
        st.rerun()  # Use st.experimental_rerun() if on older version
    except AttributeError:
        st.experimental_rerun()
    st.stop()

# Load Data Safely
signal_path = "logs/signal_log.csv"
exec_path = "logs/execution_log.csv"

col1, col2 = st.columns(2)

# Signals
col1.subheader("📡 Signals")
if os.path.exists(signal_path):
    try:
        signal_df = pd.read_csv(signal_path)
        signal_df["timestamp"] = pd.to_datetime(signal_df["timestamp"], errors='coerce')
        signal_df = signal_df.sort_values("timestamp", ascending=False)
        col1.dataframe(signal_df.head(100), use_container_width=True)
    except Exception as e:
        col1.error(f"Failed to load signal_log.csv: {e}")
else:
    col1.warning("No signal_log.csv found.")

# Executions
col2.subheader("🚀 Executions")
if os.path.exists(exec_path):
    try:
        exec_df = pd.read_csv(exec_path)
        exec_df["timestamp"] = pd.to_datetime(exec_df["timestamp"], errors='coerce')
        exec_df = exec_df.sort_values("timestamp", ascending=False)
        col2.dataframe(exec_df.head(100), use_container_width=True)
    except Exception as e:
        col2.error(f"Failed to load execution_log.csv: {e}")
else:
    col2.warning("No execution_log.csv found.")
