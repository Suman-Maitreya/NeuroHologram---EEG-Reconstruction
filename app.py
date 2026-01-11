import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mne
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE CONFIG & THEME
# ==========================================
st.set_page_config(
    page_title="NeuroHologram | Sparse EEG Reconstruction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clinical/Medical CSS
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    div.stButton > button {
        background-color: #007AFF; color: white; border-radius: 8px; border: none;
        padding: 10px 24px; transition: all 0.3s ease 0s;
        box-shadow: 0px 4px 6px rgba(0, 122, 255, 0.2);
    }
    div.stButton > button:hover {
        background-color: #0056b3; transform: translateY(-2px);
        box-shadow: 0px 4px 12px rgba(0,122,255,0.4);
    }
    .main-header { font-size: 2.5rem; color: #2C3E50; text-align: center; font-weight: 800; }
    .subtitle { font-size: 1.2rem; color: #7F8C8D; text-align: center; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">🧠 NeuroHologram</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sparse EEG Reconstruction System</p>', unsafe_allow_html=True)

# ==========================================
# 2. LOADER
# ==========================================
@st.cache_resource
def load_system():
    try:
        model = tf.keras.models.load_model('brain_reconstructor.keras')
        X_test = np.load('X_train_sparse.npy')
        Y_test = np.load('Y_train_full.npy')
        return model, X_test, Y_test
    except:
        return None, None, None

model, X_test, Y_test = load_system()

if model is None:
    st.error("🚨 System Offline: Model files missing. Please run 'train.py' first.")
    st.stop()

# ==========================================
# 3. MATH ENGINE (Exact ADMM from Demo)
# ==========================================
def admm_total_variation(signal, weight=2.0, rho=1.0, max_iter=20):
    """
    True ADMM Optimization (Matches live_demo.py)
    """
    n = len(signal)
    D = np.eye(n) * -1
    D[np.arange(n-1), np.arange(n-1)+1] = 1
    D = D[:-1, :]
    
    x = np.zeros(n); z = np.zeros(n-1); u = np.zeros(n-1)
    I = np.eye(n); DtD = np.dot(D.T, D)
    inv_matrix = np.linalg.inv(I + rho * DtD)
    
    for k in range(max_iter):
        v = signal + rho * np.dot(D.T, z - u)
        x = np.dot(inv_matrix, v)
        Dx = np.dot(D, x); v_z = Dx + u
        kappa = weight / rho
        z = np.sign(v_z) * np.maximum(np.abs(v_z) - kappa, 0)
        u = u + Dx - z
    return x

# ==========================================
# 4. CONTROL PANEL
# ==========================================
st.sidebar.header("🎛️ Bio-Signal Controls")

# Patient Randomizer
if st.sidebar.button("🎲 Generate New Patient Data"):
    st.session_state['current_idx'] = np.random.randint(0, X_test.shape[0] - 128)
elif 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 5000

# Sensor Kill Switch (Only list the 4 Active Sensors)
# Indices: 0=Fp1, 16=Fp2, 13=O1, 31=O2
st.sidebar.subheader("📉 Sensor Failure Simulation")
kill_option = st.sidebar.selectbox(
    "Disable an Active Sensor:", 
    options=[None, 0, 16, 13, 31],
    format_func=lambda x: "None (All Active)" if x is None else f"Channel {x} (Corner Sensor)"
)

# Math Controls
use_admm = st.sidebar.toggle("Enable ADMM Optimization", value=True)
admm_strength = st.sidebar.slider("Optimization Strength (λ)", 0.1, 5.0, 0.5)

# ==========================================
# 5. PROCESSING PIPELINE
# ==========================================
tab1, tab2 = st.tabs(["🖥️ Live Dashboard", "📘 Project Documentation"])

start_idx = st.session_state['current_idx']

# 1. Get Ground Truth
chunk_truth = Y_test[start_idx:start_idx+128, :]

# 2. Get Input (Sparse) and Apply Kill Switch
# We start with the Standard Sparse Input (4 channels) from X_test
chunk_input = X_test[start_idx:start_idx+128, :].copy()

# If user kills a sensor, zero it out in the input
if kill_option is not None:
    chunk_input[:, kill_option] = 0

# 3. Run AI
ai_input = chunk_input.reshape(1, 128, 32)
reconstruction_raw = model.predict(ai_input, verbose=0)[0]

# 4. Run Math (ADMM)
reconstruction_final = np.zeros_like(reconstruction_raw)
if use_admm:
    # Optimize only the missing channels (Indices where input is 0)
    # We check the first sample to see which channels are dead
    dead_indices = np.where(chunk_input[0, :] == 0)[0]
    
    for i in range(32):
        if i in dead_indices:
            reconstruction_final[:, i] = admm_total_variation(reconstruction_raw[:, i], weight=admm_strength)
        else:
            reconstruction_final[:, i] = reconstruction_raw[:, i]
else:
    reconstruction_final = reconstruction_raw

# ==========================================
# TAB 1: VISUALIZATION
# ==========================================
with tab1:
    # MNE Setup
    ch_names = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # Find peak activity moment
    peak_idx = np.argmax(np.max(np.abs(chunk_truth), axis=1))

    def plot_head(data, title):
        fig, ax = plt.subplots(figsize=(4, 4))
        # Plotting full 32-channel array (with zeros) forces MNE to show gaps
        mne.viz.plot_topomap(data, info, axes=ax, show=False, cmap='RdBu_r', vlim=(-20, 20), contours=0, sensors=True)
        ax.set_title(title, fontsize=12, fontweight='bold', color='#2C3E50')
        return fig

    # 3 Columns Layout
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.pyplot(plot_head(chunk_truth[peak_idx], "1. GROUND TRUTH (32-Ch)"), use_container_width=True)
    with c2:
        # Plot the actual sparse array (with 28 zeros) -> Shows white gaps
        st.pyplot(plot_head(chunk_input[peak_idx], "2. INPUT (4-Ch Sparse)"), use_container_width=True)
    with c3:
        st.pyplot(plot_head(reconstruction_final[peak_idx], "3. RECONSTRUCTION (AI+Math)"), use_container_width=True)

    st.divider()
    
    # Interactive Signal Plot (Plotly)
    # Pick a Missing Sensor to visualize (e.g., AF3 - Index 1)
    target_sensor = 1 
    if kill_option is not None: target_sensor = kill_option # If user killed one, show that one
        
    st.subheader(f"📉 Signal Recovery: {ch_names[target_sensor]}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=chunk_truth[:, target_sensor], name='True Signal', line=dict(color='#00CC44', width=2)))
    fig.add_trace(go.Scatter(y=chunk_input[:, target_sensor], name='Input (Sensor Reading)', line=dict(color='#FF4444', width=2)))
    fig.add_trace(go.Scatter(y=reconstruction_final[:, target_sensor], name='Reconstruction', line=dict(color='#007AFF', width=2, dash='dot')))
    
    fig.update_layout(
        template="plotly_white", 
        height=350, 
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Time (ms)",
        yaxis_title="Amplitude (uV)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: DOCUMENTATION
# ==========================================
with tab2:
    st.markdown("### 📘 Project Documentation")
    st.info("Objective: Reconstruct high-resolution medical EEG (32-channels) from sparse consumer headsets (4-channels).")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🧬 1. Biological Logic")
        st.write("The brain is a volume conductor. Signals from the center of the brain propagate to the edges. By capturing data at the 4 corners (Fp1, Fp2, O1, O2), we can mathematically interpolate the missing center signals using learned biological correlations.")
    
    with colB:
        st.markdown("#### 📐 2. Mathematical Logic")
        st.write("We solve the **Inverse Problem** ($Y = AX$) using:")
        st.latex(r'\min_x \frac{1}{2}\|x - x_{AI}\|^2 + \lambda \|\nabla x\|_1')
        st.write("**ADMM (Alternating Direction Method of Multipliers)** minimizes the Total Variation (TV) to ensure the reconstructed signals are smooth and biologically plausible.")

    st.markdown("#### 🤖 3. AI Architecture")
    st.write("A **Convolutional Denoising Autoencoder** acts as the prior. It compresses the 4-channel input into a Low-Rank Latent Space (8 dimensions) and expands it back to 32 channels.")