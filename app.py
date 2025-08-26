import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title='Predictive Maintenance Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown(
    """
    <h1 style='text-align: center; color: darkblue;'>🛠 Predictive Maintenance Dashboard</h1>
    <p style='text-align: center; color: darkgreen;'>Monitor equipment health, sensor data trends, and potential failures in real-time</p>
    """, unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Failure Probability Threshold", 0.0, 1.0, 0.5, 0.01)

# -----------------------------
# Load Model, Scaler, and Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset.csv")
    df['Type'] = df['Type'].map({'M':0, 'L':1, 'H':2})
    return df

@st.cache_resource
def load_models():
    models = joblib.load("models.pkl")  # classical ML models
    scaler = joblib.load("scaler.pkl")
    lstm_model = load_model("model.keras")  # LSTM model
    return models, scaler, lstm_model

df = load_data()
models, scaler, lstm_model = load_models()

# -----------------------------
# Dashboard Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Sensor Trends", "⚡ Machine Failure Probabilities", "🚨 Real-Time Monitoring"])

# -----------------------------
# Tab 1: Sensor Trends
# -----------------------------
with tab1:
    st.subheader("Sensor Trends Over Time")
    sensors = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    selected_sensor = st.selectbox("Select Sensor to Visualize", sensors)
    
    plt.figure(figsize=(12,4))
    sns.lineplot(data=df[selected_sensor], color='purple')
    plt.title(f"{selected_sensor} Trend")
    plt.xlabel("Samples")
    plt.ylabel(selected_sensor)
    st.pyplot(plt.gcf())
    plt.clf()

# -----------------------------
# Tab 2: Machine Failure Probabilities
# -----------------------------
with tab2:
    st.subheader("Predicted Machine Failure Probabilities")
    model_name = st.selectbox("Select Model", list(models.keys()) + ["LSTM"])
    
    # Prepare features for classical ML
    X = df.drop(['Machine failure', 'UDI', 'Product ID'], axis=1)
    
    if model_name != "LSTM":
        X_scaled = scaler.transform(X)
        probs = models[model_name].predict_proba(X_scaled)[:,1]
    else:
        # Create sequences for LSTM (timesteps=5)
        timesteps = 5
        X_scaled = scaler.transform(X)
        X_seq = []
        for i in range(len(X_scaled)-timesteps):
            X_seq.append(X_scaled[i:i+timesteps])
        X_seq = np.array(X_seq)
        probs = lstm_model.predict(X_seq)
        # Pad with zeros for first timesteps
        probs = np.concatenate((np.zeros(timesteps), probs.flatten()))
    
    df_probs = pd.DataFrame({
        'Machine': df['UDI'],
        'Failure Probability': probs,
        'Status': ['⚠️ At Risk' if p > threshold else '✅ Normal' for p in probs]
    })
    
    st.dataframe(df_probs.style.applymap(lambda x: 'color: red;' if x=='⚠️ At Risk' else 'color: green;', subset=['Status']))
    
    # Plot probabilities
    plt.figure(figsize=(12,5))
    sns.barplot(x='Machine', y='Failure Probability', data=df_probs, palette='coolwarm')
    plt.xticks(rotation=90)
    plt.axhline(y=threshold, color='red', linestyle='--', label='Alert Threshold')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# -----------------------------
# Tab 3: Real-Time Monitoring Simulation
# -----------------------------
with tab3:
    st.subheader("Simulate Real-Time Sensor Stream")
    num_samples = st.slider("Number of Samples to Stream", 1, 50, 10)
    model_name_rt = st.selectbox("Select Model for Simulation", list(models.keys()) + ["LSTM"], key='sim_model')
    simulate_button = st.button("Start Simulation")
    
    if simulate_button:
        st.info("Starting real-time monitoring simulation...")
        for i in range(num_samples):
            row = X.sample(1)
            if model_name_rt != "LSTM":
                prob = models[model_name_rt].predict_proba(scaler.transform(row))[0][1]
            else:
                # LSTM requires sequence, replicate last timesteps
                timesteps = 5
                last_rows = X_scaled[-timesteps:]  # last timesteps
                seq = np.vstack([last_rows, scaler.transform(row)])
                seq = seq[-timesteps:].reshape(1, timesteps, X.shape[1])
                prob = lstm_model.predict(seq)[0][0]
            
            status = "⚠️ At Risk" if prob > threshold else "✅ Normal"
            st.write(f"Sample {i+1}: Failure Probability = {prob:.2f} → {status}")
            time.sleep(0.5)
