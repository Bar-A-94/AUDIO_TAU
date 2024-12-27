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
            mel_dict[os.path.splitext(file_name)[0]] = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

            # Save the spectrogram as an image
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_mel.png")
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_dict[os.path.splitext(file_name)[0]],
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

def q3_distance_matrix(mel_dict):
    distance_matrix = np.zeros((4,10,10))
    for x, name in enumerate(["neta", "avital","yaron","guy"]):
        for i in range(10):
            for j in range(10):
                distance_matrix[x, i, j]=q3b_DTW(mel_dict[name +"_" + str(i)], mel_dict["bar_" + str(j)])

        # Plot heatmaps for each table with values inside
        plt.figure(figsize=(10, 8))
        heatmap = plt.imshow(distance_matrix[x], cmap='viridis', interpolation='nearest')
        plt.colorbar(label='DTW Distance')
        plt.title(f"Heatmap of DTW Distances for {name}", fontsize=14)
        plt.xlabel("Bar Index")
        plt.ylabel(f"{name} Index")
        plt.xticks(ticks=np.arange(10), labels=[f"Bar_{i}" for i in range(10)])
        plt.yticks(ticks=np.arange(10), labels=[f"{name}_{i}" for i in range(10)])
        for i in range(10):
            for j in range(10):
                plt.text(j, i, f"{distance_matrix[x][i, j]:.2f}",
                     ha="center", va="center", color="white" if heatmap.norm(distance_matrix[x][i, j]) > 0.5 else "black")
    plt.tight_layout()
    plt.show()

def q4_collapse_B(string):
    collapsed = ""
    for i in range(len(string)):
        if i == 0:
            if string[i]!="^":
                collapsed = collapsed + string[i]
        else:
            if string[i] != string[i-1] and string[i] != "^":
                collapsed = collapsed + string[i]
    return collapsed

def q5a_initialize_pred():
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
    labels = {'a': 0,'b': 1, '^':2}

    return pred, labels

def q5_ctc_forward(string, labels, matrix):
    # Extend ground truth with blanks
    blank = "^"
    extended_string="^"
    for c in string:
        extended_string += c + blank
    forward_matrix = np.zeros((matrix.shape[0], len(extended_string)))
    forward_matrix[0, 0] = matrix[0, labels["^"]]
    forward_matrix[0, 1] = matrix[0, labels[extended_string[1]]]
    for t in range(1, matrix.shape[0]):
        for l in range(0,len(extended_string)):
            prob_l = matrix[t, labels[extended_string[l]]]
            # Stay at current level
            forward_matrix[t, l] = prob_l * forward_matrix[t - 1, l]
            # Transition from previous label
            if l > 0:
                forward_matrix[t, l] += prob_l * forward_matrix[t - 1, l - 1]
            # Transition with skiping over blank
            if l > 1 and extended_string[l] != extended_string[l - 2]:
                forward_matrix[t, l] += prob_l * forward_matrix[t - 1, l - 2]
    plot_pred(forward_matrix, labels)
    return forward_matrix[-1, -1] + forward_matrix[-1, -2]

def plot_pred(pred, labels):
    """
    Plot the prediction matrix with labels for the y-axis and include values in each cell.

    Args:
        pred (np.ndarray): Prediction matrix (shape: T x |labels|).
        labels (dict): Dictionary mapping labels to indices.
    """
    # Reverse the labels dictionary to map indices to label names
    reversed_labels = {v: k for k, v in labels.items()}
    y_ticks = [reversed_labels[i] for i in range(pred.shape[1])]

    # Create the plot
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(pred.T, cmap='viridis', aspect='auto')  # Transpose to align y-axis with labels
    plt.colorbar(label='Prediction Value')
    
    # Set axis labels
    plt.xlabel('Time Step')
    plt.ylabel('Labels')
    plt.yticks(ticks=np.arange(len(y_ticks)), labels=y_ticks)
    plt.title('Prediction Heatmap')
    
    # Add values to each cell
    for i in range(pred.shape[1]):  # Iterate over rows (labels)
        for j in range(pred.shape[0]):  # Iterate over columns (time steps)
            value = pred[j, i]
            color = "white" if heatmap.norm(value) > 0.5 else "black"  # Contrast for better visibility
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=color)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Question 1.c - mel spectrogram for each audio file
    input_dir = os.path.join("A2", "resources", "audio_files", "segmented")
    output_dir = os.path.join("A2", "resources", "mel_spectrogram")
    # mel_dict = q1c_mel_spec(input_dir, output_dir)

    # Class representative - bar, Training set - neta + avital + yaron + guy, Evaluation Set - roni + nirit + ? + ?

    # Question 2 - DTW
    # q3_distance_matrix(mel_dict)

    # Question 5 - CTC
    pred, labels = q5a_initialize_pred()
    # plot_pred(pred, labels)

    # log_pred = np.log(pred)
    # plot_pred(log_pred, labels)

    # print(q5_ctc_forward("abb",labels, pred)) # 0.00
    # print(q5_ctc_forward("ab",labels, pred)) # 0.548
    print(q5_ctc_forward("aba",labels, pred)) # 0.088









    


