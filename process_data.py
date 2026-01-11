import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
FILENAME = 's01.dat'
# We keep 4 corners: Fp1(0), Fp2(16), O1(13), O2(31) - Indices based on DEAP
KEEP_INDICES = [0, 16, 13, 31] 

# ==========================================
# 1. LOAD DATA
# ==========================================
print(f"Loading {FILENAME}...")
with open(FILENAME, 'rb') as f:
    content = pickle.load(f, encoding='latin1')

raw_data = content['data'] # Shape: (40, 40, 8064)

# ==========================================
# 2. PRE-PROCESSING
# ==========================================
print("Processing data...")

# A. Keep only first 32 channels (EEG only)
eeg_data = raw_data[:, :32, :] 

# B. Stack all 40 videos into one long timeline
# Swap axes to be (Videos, Time, Channels) -> (40, 8064, 32)
eeg_data = np.transpose(eeg_data, (0, 2, 1))

# Flatten the videos: (322560, 32)
full_brain_data = eeg_data.reshape(-1, 32)

print(f"Full Dataset Shape: {full_brain_data.shape}")

# ==========================================
# 3. THE LOBOTOMY (MASKING)
# ==========================================
print("Applying Mask (Deleting 28 Channels)...")

# Create a mask of Zeros
mask = np.zeros((32,))
# Set our 4 active sensors to Ones
mask[KEEP_INDICES] = 1

# Apply the mask (Broadcasting)
sparse_brain_data = full_brain_data * mask

# ==========================================
# 4. SAVE THE FILES
# ==========================================
np.save('X_train_sparse.npy', sparse_brain_data) # The Input (Problem)
np.save('Y_train_full.npy', full_brain_data)     # The Target (Answer)

print("SUCCESS: Data processed and saved as .npy files.")