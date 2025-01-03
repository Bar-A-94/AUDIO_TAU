import librosa
import numpy as np
import soundfile as sf
import os


def moving_average_ignore_zeros(arr, window_size):
        cumsum = np.cumsum(np.insert(arr, 0, 0))
        nonzero_count = np.cumsum(np.insert(arr != 0, 0, 0))  # Count of non-zero elements
        
        # Compute sums of the windows
        window_sums = cumsum[window_size:] - cumsum[:-window_size]
        # Compute the number of non-zero elements in each window
        effective_window_sizes = nonzero_count[window_size:] - nonzero_count[:-window_size]
        
        # Avoid division by zero by setting window sizes to 1 where all values are zero
        effective_window_sizes[effective_window_sizes == 0] = 1
        moving_avg = window_sums / effective_window_sizes
        
        # Pad the result to match the original array size
        pad_size = len(arr) - len(moving_avg)
        return np.pad(moving_avg, (pad_size, 0), mode='reflect')


def apply_agc(name, digit, desired_RMS=0.035):
    # Paths
    input_path = os.path.join("A2", "resources", "audio_files", "segmented", name + f"_{digit}.wav")
    output_path = os.path.join("A2", "resources", "audio_files", "agc_segmented", name + f"_{digit}.wav")

    # Load audio with librosa
    target_sr = 16000
    audio, sr = librosa.load(input_path, sr=target_sr)  # Automatically resamples to `target_sr`


    # Frame parameters
    window_size = 0.025  # 25 ms
    hop_size = 0.01      # 10 ms
    hop_length = int(hop_size * target_sr)
    n_fft = int(window_size * target_sr)

    # Calculate RMS and gain
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    moving_avg_rms = moving_average_ignore_zeros(rms, 100)
    gain_smoothed = desired_RMS / moving_avg_rms
    gain_smoothed = np.clip(gain_smoothed, 0.1, 10)  # Prevent excessive gain

    # Initialize overlap-add buffers
    audio_agc = np.zeros_like(audio)
    overlap_count = np.zeros_like(audio)

    # Apply gain with overlap-add
    for i, g in enumerate(gain_smoothed):
        start = i * hop_length
        end = min(start + n_fft, len(audio))
        window = np.hanning(end - start)  # Apply a window function
        audio_agc[start:end] += audio[start:end] * g * window
        overlap_count[start:end] += window

    # Normalize by overlap count
    nonzero_indices = overlap_count > 0
    audio_agc[nonzero_indices] /= overlap_count[nonzero_indices]

    # Avoid overflow
    audio_agc = np.clip(audio_agc, -1.0, 1.0)

    # Trim silence
    audio_agc, _ = librosa.effects.trim(audio_agc, top_db=20)

    # Save the processed audio
    sf.write(output_path, audio_agc, target_sr)
    print(f"AGC saved: {output_path}")    

def agc_for_all():
    for name in ["bar","neta","avital","yaron", "guy", "roni", "nirit", "rom", "ohad"]:
        for digit in range(10):
            apply_agc(name, digit)

