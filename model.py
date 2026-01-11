import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Activation, Concatenate

def build_autoencoder(time_steps=128, num_channels=32):
    """
    Builds a 1D U-Net for EEG Reconstruction.
    Features 'Skip Connections' to preserve high-frequency medical details.
    """
    input_signal = Input(shape=(time_steps, num_channels))

    # ======================
    # ENCODER (Downsampling)
    # ======================
    # Level 1
    c1 = Conv1D(32, 3, padding='same')(input_signal)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling1D(2, padding='same')(c1) # 128 -> 64

    # Level 2
    c2 = Conv1D(64, 3, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling1D(2, padding='same')(c2) # 64 -> 32

    # Level 3
    c3 = Conv1D(128, 3, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling1D(2, padding='same')(c3) # 32 -> 16

    # ======================
    # BOTTLENECK
    # ======================
    b = Conv1D(256, 3, padding='same')(p3)
    b = BatchNormalization()(b)
    b = Activation('relu')(b)

    # ======================
    # DECODER (Upsampling + Skip Connections)
    # ======================
    # Level 3 Upsample
    u3 = UpSampling1D(2)(b) # 16 -> 32
    u3 = Conv1D(128, 3, padding='same')(u3)
    # SKIP CONNECTION: Merge with Encoder Level 3
    u3 = Concatenate()([u3, c3]) 
    c4 = Conv1D(128, 3, padding='same')(u3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    # Level 2 Upsample
    u2 = UpSampling1D(2)(c4) # 32 -> 64
    u2 = Conv1D(64, 3, padding='same')(u2)
    # SKIP CONNECTION: Merge with Encoder Level 2
    u2 = Concatenate()([u2, c2])
    c5 = Conv1D(64, 3, padding='same')(u2)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    # Level 1 Upsample
    u1 = UpSampling1D(2)(c5) # 64 -> 128
    u1 = Conv1D(32, 3, padding='same')(u1)
    # SKIP CONNECTION: Merge with Encoder Level 1
    u1 = Concatenate()([u1, c1])
    c6 = Conv1D(32, 3, padding='same')(u1)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    # ======================
    # OUTPUT
    # ======================
    decoded = Conv1D(num_channels, 3, activation='linear', padding='same')(c6)

    # Compile
    autoencoder = Model(input_signal, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

if __name__ == "__main__":
    model = build_autoencoder()
    model.summary()
    print("SUCCESS: U-Net Architecture Built.")