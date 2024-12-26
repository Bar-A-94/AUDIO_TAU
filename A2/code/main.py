import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

def q1c_mel_spec(input_dir, output_dir, window_size=0.025, hop_size=0.01, n_mels=80):
    """
    Question 1.c - Calculate the Mel spectrogram for each audio file in the input directory.

    Args:
        input_dir (str): Directory containing input audio files.
        output_dir (str): Directory to save the spectrograms.
        sample_rate (int): Target sample rate for the audio files.
        n_fft (int): Number of FFT components (calculated from window size).
        hop_length (int): Hop length in samples.
        n_mels (int): Number of Mel filter banks.
    """
    # Basic calculations
    n_fft=int(window_size * 16000)
    hop_length=int(hop_size * 16000)
    mel_dict = {}

    # Iterate through all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            # Load the audio file
            file_path = os.path.join(input_dir, file_name)
            audio, sr = librosa.load(file_path, sr=16000)

            # Calculate the Mel spectrogram
            mel_dict[file_name] = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

            # Save the spectrogram as an image
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_mel.png")
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_dict[file_name],
                sr=sr,
                hop_length=hop_length,
                x_axis="time",
                y_axis="mel",
                fmax=8000
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Mel Spectrogram: {file_name}")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            print(f"Spectrogram saved: {output_path}")
    
    return mel_dict

def q3b_DTW(mel_1, mel_2):
    """
    Perform Dynamic Time Warping (DTW) to calculate the alignment cost between two Mel spectrograms.

    Args:
        mel_1 (np.ndarray): Mel spectrogram of the first audio file (shape: [n_mels, time_steps_1]).
        mel_2 (np.ndarray): Mel spectrogram of the second audio file (shape: [n_mels, time_steps_2]).

    Returns:
        float: The total alignment cost (DTW distance) between the two Mel spectrograms.
    """
    distance_matrix = np.zeros((mel_1.shape[1], mel_2.shape[1]))
    distance_matrix[0][0] = 0

    # Fill the first row
    for j in range(1, mel_2.shape[1]):
        distance_matrix[0, j] = distance_matrix[0, j - 1] + np.linalg.norm(mel_1[:, 0] - mel_2[:, j])

    # Fill the first column
    for i in range(1, mel_1.shape[1]):
        distance_matrix[i, 0] = distance_matrix[i - 1, 0] + np.linalg.norm(mel_1[:, i] - mel_2[:, 0])

    for i in range(1, mel_1.shape[1]):
        for j in range(1, mel_2.shape[1]):
            cost = np.linalg.norm(mel_1[:, i] - mel_2[:, j])  # Euclidean distance
            distance_matrix[i, j] = cost + min(distance_matrix[i - 1, j - 1], # Match
                                                distance_matrix[i - 1, j],    # Insertion
                                                distance_matrix[i, j - 1])    # Deletion
            
    return distance_matrix[mel_1.shape[1] - 1, mel_2.shape[1] - 1]
                                          
def q5a_collapse_B(string):
    collapsed = ""
    for i in range(len(string)):
        if i == 0:
            if string[i]!="^":
                collapsed = collapsed + string[i]
        else:
            if string[i] != string[i-1] and string[i] != "^":
                collapsed = collapsed + string[i]
    return collapsed


if __name__ == "__main__":

    # Question 1.c - mel spectrogram for each audio file
    input_dir = os.path.join("A2", "resources", "audio_files", "segmented")
    output_dir = os.path.join("A2", "resources", "mel_spectrogram")
    mel_dict = q1c_mel_spec(input_dir, output_dir)

    # Class representative - bar, Training set - Neta + ? + ? + ?, Evaluation Set - ? + ? + ? + ?

    # Question 2 - DTW
    for file_name_1 in os.listdir(input_dir):
        if file_name_1.startswith("bar") and file_name_1.endswith(".wav"):
            for file_name_2 in os.listdir(input_dir):
                if file_name_2.startswith("neta") and file_name_2.endswith(".wav"):
                    print(file_name_1 + " from " + file_name_2 + " is " + str(q3b_DTW(mel_dict[file_name_1], mel_dict[file_name_2])))

    # Question 5 - CTC
    # Define pred by 5.b
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00
                    

