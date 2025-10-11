# model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout

def build_tinycnn_model(input_shape, num_classes):
    """
    Defines the final, optimized TinyCNN architecture used in the paper.
    
    This is a lightweight, end-to-end convolutional neural network designed 
    for efficiency on low-resource devices.
    
    Args:
        input_shape (tuple): The shape of the input spectrograms (height, width, channels).
        num_classes (int): The number of target emotion classes.
        
    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential([
        Input(shape=input_shape, name="Input_Spectrogram"),
        
        # --- Feature Extraction Blocks ---
        # Block 1
        Conv2D(16, (3, 3), activation='relu', padding='same', name="Conv1"),
        BatchNormalization(name="BatchNorm1"),
        MaxPooling2D((2, 2), name="MaxPool1"),
        
        # Block 2
        Conv2D(32, (3, 3), activation='relu', padding='same', name="Conv2"),
        BatchNormalization(name="BatchNorm2"),
        MaxPooling2D((2, 2), name="MaxPool2"),
        
        # Block 3
        Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv3"),
        BatchNormalization(name="BatchNorm3"),
        
        # --- Classifier Head ---
        GlobalAveragePooling2D(name="GlobalAvgPool"),
        Dropout(0.3, name="Dropout"),
        Dense(num_classes, activation='softmax', name="Output_Layer")
    ])
    
    # Compile the model with the optimized hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    # This block allows you to run this script directly to see the model summary
    
    # Example parameters
    INPUT_SHAPE = (64, 251, 1) # (Mel bands, Time frames, Channels)
    NUM_CLASSES = 8 # e.g., for RAVDESS
    
    print("Building and summarizing the TinyCNN model...")
    tiny_cnn = build_tinycnn_model(INPUT_SHAPE, NUM_CLASSES)
    tiny_cnn.summary()