# train_evaluate.py

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import the model architecture from our separate file
from model import build_tinycnn_model

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)


def run_evaluation_pipeline(dataset_name):
    """
    Loads preprocessed data for a specific dataset, then runs the full
    speaker-independent cross-validation protocol to train and evaluate
    the TinyCNN model.
    
    Args:
        dataset_name (str): The name of the dataset to evaluate (e.g., 'RAVDESS').
    """
    
    # ===================================================================
    # Step 1: Load Pre-processed Data
    # ===================================================================
    PROCESSED_DATA_DIR = "processed_data"
    print(f"Loading pre-processed data for {dataset_name.upper()}...")
    
    try:
        # Load spectrograms, labels, and speaker IDs
        X_data = np.load(os.path.join(PROCESSED_DATA_DIR, f'X_{dataset_name.lower()}_spec.npy'))
        y_data = np.load(os.path.join(PROCESSED_DATA_DIR, f'y_{dataset_name.lower()}.npy'))
        speakers_data = np.load(os.path.join(PROCESSED_DATA_DIR, f'speakers_{dataset_name.lower()}.npy'))
    except FileNotFoundError:
        print(f"Error: Processed files for {dataset_name} not found in '{PROCESSED_DATA_DIR}'.")
        print("Please run the data preprocessing script first.")
        return

    print("Data loaded successfully.")
    print(f"Spectrograms shape: {X_data.shape}")
    print(f"Number of samples: {len(y_data)}")
    print(f"Number of unique speakers: {len(np.unique(speakers_data))}")

    # ===================================================================
    # Step 2: Prepare Data for TensorFlow
    # ===================================================================
    # Add a channel dimension for the CNN input
    X_data_cnn = X_data[..., np.newaxis]

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    # Convert integer labels to one-hot encoding for categorical_crossentropy
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    print(f"Found {num_classes} classes: {class_names}")

    # ===================================================================
    # Step 3: Run Speaker-Independent Cross-Validation
    # ===================================================================
    N_FOLDS = 5 if dataset_name.upper() in ['RAVDESS', 'CREMA-D'] else len(np.unique(speakers_data))
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_accuracies, fold_f1_scores = [], []
    all_y_true, all_y_pred = [], []

    # Main evaluation loop
    for fold, (train_indices, test_indices) in enumerate(sgkf.split(X_data, y_encoded, groups=speakers_data)):
        print(f"\n===== FOLD {fold + 1}/{N_FOLDS} =====")
        
        # Split data into training and testing sets for this fold
        X_train, X_test = X_data_cnn[train_indices], X_data_cnn[test_indices]
        y_train, y_test = y_categorical[train_indices], y_categorical[test_indices]
        y_test_labels = y_encoded[test_indices] # For metrics calculation

        # Build a new, untrained model for each fold
        model = build_tinycnn_model(input_shape=X_train.shape[1:], num_classes=num_classes)
        
        if fold == 0:
            print("Model architecture summary:")
            model.summary()
        
        # Define early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=15, # As per your optimized parameters
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train, 
            epochs=100,       # As per your optimized parameters
            batch_size=32,    # As per your optimized parameters
            validation_split=0.1, 
            callbacks=[early_stopping], 
            verbose=2 # Cleaner log during training
        )
        
        # Evaluate the model on the unseen test set
        y_pred_probs = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        
        # Store results for final aggregation
        all_y_true.extend(y_test_labels)
        all_y_pred.extend(y_pred_labels)
        
        acc = accuracy_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels, average='macro')
        
        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        
        print(f"Fold {fold + 1} Results -> Accuracy: {acc:.4f}, Macro F1-Score: {f1:.4f}")

    # ===================================================================
    # Step 4: Display Final Aggregated Results
    # ===================================================================
    print(f"\n\n{'='*40}\n FINAL CROSS-VALIDATION RESULTS FOR {dataset_name.upper()} \n{'='*40}")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} (± {np.std(fold_accuracies):.4f})")
    print(f"Average Macro F1-Score: {np.mean(fold_f1_scores):.4f} (± {np.std(fold_f1_scores):.4f})")
    
    print("\n--- Overall Classification Report (Aggregated over all folds) ---")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))

    # --- Plot and save the confusion matrix ---
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Aggregated Confusion Matrix for {dataset_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure to a file
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{dataset_name.lower()}.png'))
    print(f"\nConfusion matrix saved to '{output_dir}/confusion_matrix_{dataset_name.lower()}.png'")
    # plt.show() # Optional: uncomment to display the plot directly when running the script


if __name__ == '__main__':
    # Set up argument parser to run the script from the command line
    parser = argparse.ArgumentParser(description='Train and evaluate the TinyCNN model on a specific dataset.')
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        choices=['RAVDESS', 'TESS', 'CREMA-D'], 
        help='The dataset to run the evaluation on. Must be one of: RAVDESS, TESS, CREMA-D'
    )
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(args.dataset.upper())