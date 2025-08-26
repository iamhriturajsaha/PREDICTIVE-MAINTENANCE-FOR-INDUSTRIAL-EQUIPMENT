
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go

# ----------------- LOAD MODEL & SCALER ----------------- #
model = tf.keras.models.load_model("final_predictive_maintenance_lstm.keras", compile=False, safe_mode=False)
scaler = joblib.load("scaler.pkl")

# Features used in training (must match exactly!)
features = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3"
] + [f"sensor_measurement_{i}" for i in range(1, 22)]   # 21 sensors

seq_len = 30

# ----------------- SAFE PREDICT WRAPPER ----------------- #
def safe_predict(model, X):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    preds = model(X, training=False)   # forward pass only
    return preds.numpy()

# ----------------- STREAMLIT LAYOUT ----------------- #
st.set_page_config(page_title="⚙️ Equipment Health Dashboard", layout="wide")

st.title("⚙️ Equipment Health Monitoring Dashboard")
st.markdown("A real-time predictive maintenance dashboard powered by **LSTM Deep Learning**.")

# Sidebar settings
st.sidebar.header("⚙️ Controls")
threshold = st.sidebar.slider("Failure Alert Threshold (RUL)", 5, 50, 20)
last_n = st.sidebar.slider("Show Last N Cycles", 10, 100, 30)
engine_ids = st.sidebar.multiselect("Select Engines to Monitor", [1, 2, 3, 4, 5], default=[1])

# Auto-refresh every 5 seconds
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return 0
count = st_autorefresh(interval=5000, key="refresh_counter")

# ----------------- SIMULATED TEST DATA ----------------- #
def generate_engine_data(engine_id, n_cycles=200):
    np.random.seed(engine_id)
    cycles = np.arange(1, n_cycles + 1)

    df = pd.DataFrame({
        "unit_number": engine_id,
        "time_in_cycles": cycles,
        "operational_setting_1": np.random.uniform(-1, 1, n_cycles),
        "operational_setting_2": np.random.uniform(0, 1, n_cycles),
        "operational_setting_3": np.random.uniform(0, 1, n_cycles),
    })

    # Add 21 sensor signals
    for i in range(1, 22):
        df[f"sensor_measurement_{i}"] = np.sin(cycles / (10+i)) + np.random.normal(0, 0.1, n_cycles)

    # Scale features
    df[features] = scaler.transform(df[features])
    return df

# ----------------- GLOBAL ALERT TRACKING ----------------- #
global_status = "✅ All Engines Healthy"
global_color = "green"
engine_status = {}

for engine_id in engine_ids:
    df = generate_engine_data(engine_id)
    max_cycle = min(seq_len + count, len(df))
    stream_df = df.iloc[:max_cycle]

    preds, rul_cycles = [], []
    for i in range(seq_len, len(stream_df)):
        X_live = np.expand_dims(stream_df[features].iloc[i-seq_len:i].values, axis=0)
        predicted_rul = max(0, safe_predict(model, X_live)[0][0])  # <-- FIXED
        preds.append(predicted_rul)
        rul_cycles.append(stream_df.iloc[i]["time_in_cycles"])

    if len(preds) > 0:
        latest_rul = preds[-1]
        status = "✅ Healthy"
        color = "green"
        if latest_rul < threshold/2:
            status, color = "🔴 Critical", "red"
        elif latest_rul < threshold:
            status, color = "🟠 Warning", "orange"
        
        engine_status[engine_id] = (status, color, rul_cycles, preds)

        # Update global status
        if color == "red":
            global_status, global_color = "🚨 CRITICAL ALERT: Immediate Maintenance Required!", "red"
        elif color == "orange" and global_color != "red":
            global_status, global_color = "⚠️ Warning: Some Engines Need Attention", "orange"

# ----------------- DISPLAY GLOBAL ALERT BANNER ----------------- #
st.markdown(f""" 
<div style='padding:15px; border-radius:10px; background-color:{global_color}; 
color:white; font-size:20px; font-weight:bold; text-align:center;'>
{global_status}
</div>
""", unsafe_allow_html=True)

# ----------------- PER-ENGINE VISUALIZATION ----------------- #
for engine_id, (status, color, rul_cycles, preds) in engine_status.items():
    st.markdown(f"### Engine {engine_id} Status: <span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rul_cycles[-last_n:], 
        y=preds[-last_n:], 
        mode="lines+markers", 
        name="Predicted RUL",
        line=dict(color="blue", width=3),
        marker=dict(size=8)
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold", annotation_position="top right")
    fig.update_layout(
        title=f"📉 Predicted Remaining Useful Life (Engine {engine_id})",
        xaxis_title="Cycle",
        yaxis_title="Predicted RUL",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Recent Predictions (Engine {engine_id})")
    recent_df = pd.DataFrame({
        "Cycle": rul_cycles[-last_n:],
        "Predicted_RUL": preds[-last_n:]
    })
    st.dataframe(recent_df, use_container_width=True)
