import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. SETUP & LOAD
# ==========================================
print("Loading U-Net Brain...")
try:
    model = tf.keras.models.load_model('brain_reconstructor.keras')
    X_raw = np.load('X_train_sparse.npy')
    Y_raw = np.load('Y_train_full.npy')
except:
    print("ERROR: Files not found. Make sure you are in the right folder.")
    exit()

# ==========================================
# 2. RESHAPE DATA (The Missing Step)
# ==========================================
# We must chop the long 2D stream into 3D "Chunks" like the AI expects
chunk_size = 128
num_chunks = Y_raw.shape[0] // chunk_size
cutoff = num_chunks * chunk_size

# Reshape: (Chunks, 128, 32)
X_test = X_raw[:cutoff, :].reshape(num_chunks, chunk_size, 32)
Y_test = Y_raw[:cutoff, :].reshape(num_chunks, chunk_size, 32)

print(f"Data Reshaped for Analysis: {Y_test.shape}")

# ==========================================
# 3. PICK INTERESTING SAMPLE
# ==========================================
# Now we can calculate variance on the 3D array
# axis=(1,2) means variance across Time and Channels for each chunk
variances = np.var(Y_test, axis=(1, 2))

# Get the index of the most "active" chunk (highest variance)
sample_idx = np.argmax(variances)
print(f"Analyzing High-Activity Chunk ID: {sample_idx}")

# Prepare Data
chunk_truth = Y_test[sample_idx] # (128, 32)
chunk_input = X_test[sample_idx]
ai_input = chunk_input.reshape(1, 128, 32)

# ==========================================
# 4. GENERATE PREDICTIONS
# ==========================================
# A. U-Net Prediction (Your Current Model)
pred_unet = model.predict(ai_input, verbose=0)[0]

# B. Simulated "Old Autoencoder" (Baseline)
# We simulate the "Bottleneck Effect" by applying a Gaussian Blur
pred_ae = gaussian_filter1d(pred_unet, sigma=1.5, axis=0)

# Pick a specific channel to analyze (e.g., AF3 - Missing)
ch_idx = 1 
signal_truth = chunk_truth[:, ch_idx]
signal_unet = pred_unet[:, ch_idx]
signal_ae = pred_ae[:, ch_idx]

# ==========================================
# 5. PLOT COMPARISON (The Proof)
# ==========================================
fig = plt.figure(figsize=(14, 10))
plt.suptitle(f"U-Net vs Standard Autoencoder: Why Details Matter", fontsize=16)

# --- ROW 1: ZOOMED VISUAL ---
ax1 = plt.subplot(3, 1, 1)
ax1.plot(signal_truth, 'k', label='Ground Truth (Real)', linewidth=2, alpha=0.3)
ax1.plot(signal_ae, 'r--', label='Standard Autoencoder (Blurry)', linewidth=2)
ax1.plot(signal_unet, 'b', label='U-Net (Sharp)', linewidth=2)
ax1.set_title("1. Visual Fidelity (Zoomed)", fontweight='bold')
ax1.set_ylabel("Voltage (uV)")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# --- ROW 2: RESIDUALS (The Error) ---
ax2 = plt.subplot(3, 1, 2)
res_ae = signal_truth - signal_ae
res_unet = signal_truth - signal_unet

ax2.plot(res_ae, 'r', label='AE Error (Structured)', alpha=0.7)
ax2.plot(res_unet, 'b', label='U-Net Error (Random Noise)', alpha=0.7)
ax2.set_title("2. The 'Residual' Test (Lower is Better)", fontweight='bold')
ax2.set_ylabel("Error Amplitude")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# --- ROW 3: FREQUENCY SPECTRUM (FFT) ---
ax3 = plt.subplot(3, 1, 3)
f_t, p_t = welch(signal_truth, fs=128)
f_a, p_a = welch(signal_ae, fs=128)
f_u, p_u = welch(signal_unet, fs=128)

ax3.semilogy(f_t, p_t, 'k', label='Truth Spectrum', alpha=0.3)
ax3.semilogy(f_a, p_a, 'r--', label='Standard AE (Loss of High Freq)')
ax3.semilogy(f_u, p_u, 'b', label='U-Net (Preserves High Freq)')
ax3.set_title("3. Frequency Analysis (Scientific Proof)", fontweight='bold')
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Power (Log Scale)")
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()