# preprocessing.py

import numpy as np
import librosa

# ===================================================================
# 1. Core Constants for Preprocessing
# ===================================================================

SAMPLE_RATE = 16000
FIXED_DURATION_S = 2.5
FIXED_DURATION_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION_S)

N_MELS = 64
WINDOW_SIZE_MS = 25
HOP_SIZE_MS = 10
N_FFT = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
HOP_LENGTH = int(SAMPLE_RATE * HOP_SIZE_MS / 1000)
MAX_FRAMES = int(FIXED_DURATION_SAMPLES / HOP_LENGTH) + 1


# ===================================================================
# 2. Main Preprocessing and Spectrogram Extraction Function
# ===================================================================

def audio_to_spectrogram(file_path):
    """
    Takes an audio file path, preprocesses the audio, and converts it
    into a log-Mel spectrogram ready for the TinyCNN model.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        np.ndarray: The resulting log-Mel spectrogram, padded to a fixed length.
    """
    try:
        # 1. Load and resample audio
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')

        # 2. Remove silence
        signal, _ = librosa.effects.trim(signal, top_db=20)

        # 3. Pad or truncate to a fixed length
        if len(signal) > FIXED_DURATION_SAMPLES:
            # Take the center of the clip for longer files
            start = (len(signal) - FIXED_DURATION_SAMPLES) // 2
            signal = signal[start : start + FIXED_DURATION_SAMPLES]
        else:
            # Pad shorter files with silence
            padding = FIXED_DURATION_SAMPLES - len(signal)
            signal = np.pad(signal, (0, padding), 'constant')
            
        # 4. Normalize amplitude
        signal = librosa.util.normalize(signal)

        # 5. Compute log-Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=N_FFT, n_mels=N_MELS
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)

        # 6. Pad the spectrogram to ensure fixed width (MAX_FRAMES)
        if log_mel_spectrogram.shape[1] > MAX_FRAMES:
            log_mel_spectrogram = log_mel_spectrogram[:, :MAX_FRAMES]
        else:
            pad_width = MAX_FRAMES - log_mel_spectrogram.shape[1]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

        return log_mel_spectrogram

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None