import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_autoencoder
import os

# Create a folder to save checkpoints so they don't clutter your main folder
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# 1. LOAD DATA
print("Loading Training Data...")
try:
    X_train = np.load('X_train_sparse.npy')
    Y_train = np.load('Y_train_full.npy')
except FileNotFoundError:
    print("ERROR: .npy files not found. Run 'process_data.py' first.")
    exit()

# 2. RESHAPE
chunk_size = 128
num_chunks = X_train.shape[0] // chunk_size
cutoff = num_chunks * chunk_size

X_train = X_train[:cutoff, :].reshape(num_chunks, chunk_size, 32)
Y_train = Y_train[:cutoff, :].reshape(num_chunks, chunk_size, 32)

# 3. BUILD MODEL
print("Building Model...")
model = build_autoencoder(time_steps=chunk_size, num_channels=32)

# 4. DEFINE CHECKPOINT (Saves model every epoch)
checkpoint_callback = ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}.keras', # Saves as model_epoch_01.keras, etc.
    save_freq='epoch',
    verbose=1
)

# 5. TRAIN
print("Starting Training...")
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    callbacks=[checkpoint_callback] # Add the callback here
)

# 6. SAVE FINAL MODEL
model.save('brain_reconstructor.keras')
print("SUCCESS: Final model saved. Check 'checkpoints/' folder for epoch backups.")

# Plot Learning Curve
plt.plot(history.history['loss'], label='Training Error')
plt.plot(history.history['val_loss'], label='Validation Error')
plt.title('AI Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Error (MSE)')
plt.legend()
plt.show()