import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import seaborn as sns
from pathlib import Path


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
    win_length = int(window_size * 16000)   
    mel_dict = {}

    # Iterate through all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            # Load the audio file
            file_path = os.path.join(input_dir, file_name)
            audio, sr = librosa.load(file_path, sr=16000)

            # Calculate the Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_dict[os.path.splitext(file_name)[0]] = mel_spec_db

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
    dtw_cost = np.zeros((mel_1.shape[1], mel_2.shape[1]))
    dtw_cost[0][0] = 0

    # Fill the first row
    for j in range(1, mel_2.shape[1]):
        dtw_cost[0, j] = dtw_cost[0, j - 1] + np.linalg.norm(mel_1[:, 0] - mel_2[:, j])

    # Fill the first column
    for i in range(1, mel_1.shape[1]):
        dtw_cost[i, 0] = dtw_cost[i - 1, 0] + np.linalg.norm(mel_1[:, i] - mel_2[:, 0])

    for i in range(1, mel_1.shape[1]):
        for j in range(1, mel_2.shape[1]):
            cost = np.linalg.norm(mel_1[:, i] - mel_2[:, j])  # Euclidean distance
            dtw_cost[i, j] = cost + min(dtw_cost[i - 1, j - 1], # Match
                                                dtw_cost[i - 1, j],    # Insertion
                                                dtw_cost[i, j - 1])    # Deletion
            
    # Compute the alignment path length by backtracking
    i, j = mel_1.shape[1] - 1, mel_2.shape[1] - 1
    path_length = 0

    while i > 0 or j > 0:
        path_length += 1
        if i > 0 and j > 0 and dtw_cost[i - 1, j - 1] <= dtw_cost[i - 1, j] and dtw_cost[i - 1, j - 1] <= dtw_cost[i, j - 1]:
            i, j = i - 1, j - 1  # Match
        elif i > 0 and (j == 0 or dtw_cost[i - 1, j] <= dtw_cost[i, j - 1]):
            i -= 1  # Insertion
        else:
            j -= 1  # Deletion

    path_length += 1  # Include the starting point (0, 0)
            
    return dtw_cost[mel_1.shape[1] - 1, mel_2.shape[1] - 1], path_length

def q3_distance_matrix(mel_dict, db, target, normalize=False):
    distance_matrix = np.zeros((4, 10, 10))
    db = db
    for x, name in enumerate(target):
        for i in range(10):
            for j in range(10):
                distance_matrix[x, i, j], path_length = q3b_DTW(mel_dict[name + "_" + str(i)], mel_dict[db + "_" + str(j)])

                if normalize:
                    distance_matrix[x,i,j] = distance_matrix[x,i,j]/(len(mel_dict[name + "_" + str(i)][1]) + len(mel_dict[db + "_" + str(j)][1]))


        # Plot the heatmap
        plt.figure(figsize=(12, 10))
        heatmap = plt.imshow(distance_matrix[x], cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Normalized DTW Distance')
        plt.title(f"Heatmap of DTW Distances for {name}", fontsize=16)
        plt.xlabel("Bar Index", fontsize=12)
        plt.ylabel(f"{name} Index", fontsize=12)
        plt.xticks(ticks=np.arange(10), labels=[f"Bar_{i}" for i in range(10)], fontsize=10)
        plt.yticks(ticks=np.arange(10), labels=[f"{name}_{i}" for i in range(10)], fontsize=10)

        count = 0
        # Add text and rectangles for minimum values
        for i in range(10):
            # Find minimum value and its column index for this row
            min_col = np.argmin(distance_matrix[x][i])
            if min_col == i:
                count += 1
            
            # Add text for all values in this row
            for j in range(10):
                plt.text(j, i, f"{distance_matrix[x][i, j]:.4f}",
                        ha="center", va="center",
                        zorder=10)
            
            # Add rectangle around minimum value
            plt.gca().add_patch(
                plt.Rectangle(
                    (min_col - 0.5, i - 0.5),  # Bottom-left corner
                    1,  # Width
                    1,  # Height
                    edgecolor="black",
                    linewidth=2,
                    facecolor="none",  # No fill
                    zorder=11
                )
            )
                
        # Save the plot
        plt.tight_layout()
        os.makedirs(os.path.join("plots"), exist_ok=True)
        filename = f"DTW_distance_matrix{'_agc' if normalize else ''}_{name}.png"
        plt.savefig(os.path.join(grandparent_dir,"A2", "resources","plots", filename))
        plt.close()
        print(f"Heatmap saved for {name}: {filename}")
    return distance_matrix

def find_max_precision_threshold(distance_mat, agc=False):
    best_threshold = 0
    best_precision = 0

    thresh_range = np.arange(1000, 40000) if not agc else np.arange(1,150,0.05)

    for mid_threshold in thresh_range:

        # Threshold the matrix
        thresholded_matrix = (distance_mat <= mid_threshold).astype(int)

        # Calculate recall against stacked identity matrix
        #Note that we chose precision because accuracy is a bad metric - Works best when all zeros are chosen
        diagonal_sum = 0
        for i in range(4):  # Iterate through the 4 matrices in the tensor
            diagonal_sum += np.trace(thresholded_matrix[i])
        if diagonal_sum < 10:
            continue ## I dont want places where only the smallest value is entered, and precision is 1.0
        total_ones = np.sum(thresholded_matrix)
        if total_ones == 0:
            precision = 0
        else:
            precision = diagonal_sum / total_ones

        if precision > best_precision:
                best_precision = precision
                best_threshold = mid_threshold

    return best_threshold, best_precision

def calc_precision_over_mat(matrix, threshold):
    thresholded_matrix = (matrix <= threshold).astype(int)
    diagonal_sum = 0
    for i in range(4):  # Iterate through the 4 matrices in the tensor
        diagonal_sum += np.trace(thresholded_matrix[i])
    total_ones = np.sum(thresholded_matrix)
    if total_ones == 0:
        precision = 0
    else:
        precision = diagonal_sum / total_ones
    return precision

def build_confusion_matrices(tensor, threshold, name):
    confusion_matrix = np.sum((tensor <= threshold).astype(int), axis=0)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Validation Set Confusion Matrix")
    plt.savefig(os.path.join(grandparent_dir,"A2", "resources","plots", f"q3g_validation_{name}.png"))
    plt.close()
    return confusion_matrix




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
    extended_labels = {idx: char for idx, char in enumerate(extended_string)}
    #plot_pred_forward("q5d_forward_matrix", forward_matrix, extended_labels)
    plot_pred_forward("q5d_pred_matrix", matrix, {v: k for k, v in labels.items()})
    return forward_matrix[-1, -1] + forward_matrix[-1, -2]

def q6_ctc_align(question_number, string, labels, matrix):
    """
    Perform forced alignment using the max operator and find the most probable path.

    Args:
        string (str): text to align string.
        labels (dict): Dictionary mapping labels (including blank) to indices.
        matrix (np.ndarray): Prediction probabilities matrix (T x |labels|).

    Returns:
        tuple: (most_probable_labels, best_score)
            - most_probable_labels: The most probable path as a list of labels.
            - best_score: The score of the most probable path.
    """
    # Extend ground truth with blanks
    blank = "^"
    extended_string="^"
    for c in string:
        extended_string += c + blank
    most_probable_path = []


    # Initialize forward and backtrace matrices
    forward_matrix = np.zeros((matrix.shape[0], len(extended_string)))
    backtrace_matrix = np.full((matrix.shape[0], len(extended_string)), -1, dtype=int)

    # Initialization at t=0
    forward_matrix[0, 0] = matrix[0, labels["^"]]
    forward_matrix[0, 1] = matrix[0, labels[extended_string[1]]]
    backtrace_matrix[0, 0] = 0
    backtrace_matrix[0, 1] = 0

    for t in range(1, matrix.shape[0]):
        for l in range(len(extended_string)):
            prob_l = matrix[t, labels[extended_string[l]]]
            
            # Stay at current level
            forward_matrix[t, l] = prob_l * forward_matrix[t - 1, l]
            backtrace_matrix[t, l] = l
            # Transition from previous label
            if l > 0:
                if forward_matrix[t, l] <  prob_l * forward_matrix[t - 1, l - 1]:
                    forward_matrix[t, l] = prob_l * forward_matrix[t - 1, l - 1]
                    backtrace_matrix[t, l] = l - 1
            # Transition with skiping over blank
            if l > 1 and extended_string[l] != extended_string[l - 2]:
                if forward_matrix[t, l] < prob_l * forward_matrix[t - 1, l - 2]:
                    forward_matrix[t, l] = prob_l * forward_matrix[t - 1, l - 2]
                    backtrace_matrix[t, l] = l - 2
    
    # Backtrace to find the most probable path
    if forward_matrix[-1, -1] > forward_matrix[-1, -2]:
        current = len(extended_string) - 1
        best_score = forward_matrix[-1, -1]
    else:
        current = len(extended_string) - 2
        best_score = forward_matrix[-1, -2]
    
    # Backtrace to find the most probable path
    for t in range(matrix.shape[0] - 1, -1, -1):
        most_probable_path.append(int(current))
        current = backtrace_matrix[t, current]

    most_probable_path.reverse()
    most_probable_labels = [extended_string[i] for i in most_probable_path]
    
    # Plot the forward and backward matrix
    extended_labels = {idx: char for idx, char in enumerate(extended_string)}
    plot_pred_forward(question_number + "_forward_matrix", forward_matrix, extended_labels, aligned=most_probable_labels)
    plot_pred_backward(question_number +"_backtrace_matrix", backtrace_matrix, extended_labels, most_probable_path)

    return most_probable_labels, best_score

def plot_pred_forward(name, pred, labels, aligned=None):
    """
    Plot the prediction matrix with labels for the y-axis and include values in each cell.
    Colors are normalized by column (time step).

    Args:
        pred (np.ndarray): Prediction matrix (shape: T x |labels|).
        labels (dict): Dictionary mapping labels to indices.
    """
    y_ticks = [labels[i] for i in range(pred.shape[1])]

    # Create the plot
    plt.figure(figsize=(pred.T.shape[0]*1.6, pred.T.shape[0]*1.6))
    
    # Normalize each column separately
    normalized_pred = np.zeros_like(pred)
    for t in range(pred.shape[1]):  # For each time step
        col_min = pred[:, t].min()
        col_max = pred[:, t].max()
        if col_max > col_min:  # Avoid division by zero
            normalized_pred[:, t] = (pred[:, t] - col_min) / (col_max - col_min)
        else:
            normalized_pred[:, t] = 0

    heatmap = plt.imshow(normalized_pred.T, cmap='viridis', aspect='auto')
    plt.colorbar(label='Normalized Prediction Value')
    
    # Set axis labels
    if aligned:
        plt.xlabel('Aligned sequence')
        plt.xticks(ticks=np.arange(len(aligned)), labels=aligned)
    else:
        plt.xlabel('Time Step')
    plt.ylabel('Labels')
    plt.yticks(ticks=np.arange(len(y_ticks)), labels=y_ticks)
    plt.title('Pred Matrix')
    
    # Add original values to each cell
    for i in range(pred.shape[1]):  # Iterate over rows (labels)
        for j in range(pred.shape[0]):  # Iterate over columns (time steps)
            value = pred[j, i]
            # Use normalized value for determining text color
            norm_value = normalized_pred[j, i]
            color = "white" if norm_value < 0.5 else "black"
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", name))
    plt.close()
    print(f"{name} plot saved: {os.path.join(output_dir, 'plots')}")

def plot_pred_backward(name, backtrace_matrix, labels, most_probable_path):
    """
    Plot the backtrace matrix with selected path highlighted in yellow.

    Args:
        name (str): Name of the plot for saving.
        backtrace_matrix (np.ndarray): Backtrace matrix (shape: T x |labels|).
        labels (dict): Dictionary mapping label indices to names.
        most_probable_labels (list): Most probable labels for the path.
        output_dir (str): Directory to save the plot.
    """
    
    # Prepare table data and highlight cells
    table_data = backtrace_matrix.T
    string_table_data = np.empty(table_data.shape, dtype=object)  # Create a new array for strings
    for i in range(table_data.shape[0]):
        for j in range(table_data.shape[1]):
            if table_data[i][j] == -1:
                string_table_data [i][j] = "NP"
            elif i == 0 and j == 0 or i ==1 and j==0:
                string_table_data [i][j] = "Start"
            else:
                string_table_data [i][j] = labels[table_data[i][j]]
    label_names = [labels[i] for i in range(len(labels))]
    col_labels = [f"t={t}" for t in range(backtrace_matrix.shape[0])]
    row_labels = label_names
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(len(col_labels)* 0.6, len(row_labels)*0.3))
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=string_table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Highlight cells in the most probable path
    for t, label in enumerate(most_probable_path):
        cell = table[(label + 1, t)]  # Offset due to row/column labels
        cell.set_facecolor("yellow")

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Save and close the plot
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plots", name + ".png"))
    plt.close()
    print(f"{name} plot saved: {os.path.join(output_dir, 'plots')}")


if __name__ == "__main__":
    # Initialize
    grandparent_dir = Path(__file__).resolve().parent.parent.parent
    input_dir = os.path.join(grandparent_dir,"A2", "resources", "audio_files", "segmented")
    output_dir = os.path.join(grandparent_dir, "A2", "resources")
    agc_dir = os.path.join(grandparent_dir,"A2", "resources", "audio_files", "agc_segmented")
    
    db = "bar"
    training_set = ["neta", "avital", "yaron", "guy"]
    evaluation_set = ["roni", "nirit" ,"rom", "ohad"]
    
    # Question 2 - Mel-spectrogram
    mel_dict = q1c_mel_spec(input_dir, os.path.join(output_dir,"mel_spectrogram", "raw"))
    agc_mel_dict = q1c_mel_spec(agc_dir, os.path.join(output_dir,"mel_spectrogram", "agc"))
    
    
    # Question 3 - DTW
    training_distance_mat = q3_distance_matrix(mel_dict, db, training_set)
    thresh, precision = find_max_precision_threshold(training_distance_mat)
    print("Clean try - no agc, no path norm")
    print("For threshold:", thresh, "precision is:", precision)
    
    eval_distance_matrix = q3_distance_matrix(mel_dict, db, evaluation_set)
    conf_matrices = build_confusion_matrices(eval_distance_matrix, thresh, "normal")
    eval_precision = calc_precision_over_mat(eval_distance_matrix, thresh)
    print("Eval precision", eval_precision)


    training_distance_mat = q3_distance_matrix(agc_mel_dict, db, training_set, normalize=True)
    thresh, precision = find_max_precision_threshold(training_distance_mat, True)
    print("AGC + Path norm")
    print("For threshold:", thresh, "precision is:", precision)
    
    eval_distance_matrix_agc_norm = q3_distance_matrix(agc_mel_dict, db, evaluation_set, normalize=True)
    conf_matrices_agc_norm = build_confusion_matrices(eval_distance_matrix_agc_norm, thresh, "agc & norm")
    eval_precision = calc_precision_over_mat(eval_distance_matrix_agc_norm, thresh)
    print("Eval presicion", eval_precision)


    training_distance_mat = q3_distance_matrix(mel_dict, db, training_set, normalize=True)
    thresh, precision = find_max_precision_threshold(training_distance_mat, True)
    print("Path norm")
    print("For threshold:", thresh, "precision is:", precision)

    eval_distance_matrix_norm = q3_distance_matrix(mel_dict, db, evaluation_set, normalize=True)
    conf_matrices_norm = build_confusion_matrices(eval_distance_matrix_norm, thresh, "norm")
    eval_precision = calc_precision_over_mat(eval_distance_matrix_norm, thresh)
    print("Eval presicion", eval_precision)

    training_distance_mat = q3_distance_matrix(agc_mel_dict, db, training_set)
    thresh, precision = find_max_precision_threshold(training_distance_mat)
    print("AGC")
    print("For threshold:", thresh, "precision is:", precision)
    
    eval_distance_matrix_agc = q3_distance_matrix(agc_mel_dict, db, evaluation_set)
    conf_matrices_agc = build_confusion_matrices(eval_distance_matrix_agc, thresh, "agc")
    eval_precision = calc_precision_over_mat(eval_distance_matrix_agc, thresh)
    print("Eval presicion", eval_precision)


    # Question 5 - CTC
    pred, labels = q5a_initialize_pred()
    print(q5_ctc_forward("aba",labels, pred)) # 0.088

    # Question 6 - CTC align
    print(q6_ctc_align("q6d&e","aba", labels, pred)) # (['^', 'a', 'b', 'a', '^'], np.float64(0.04032000211715699))
    
    # Question 7
    data = pkl.load(open('A2/supplied_files/force_align.pkl', 'rb'))
    labels = {v: k for k, v in data['label_mapping'].items()}
    print(q6_ctc_align("q7d&e",data['text_to_align'], labels, data['acoustic_model_out_probs'])) # (['^', '^', '^', '^', '^', '^', '^', 't', 'h', 'e', '^', 'n', 'n', '^', ' ', ' ', '^', 'g', '^', 'o', '^', 'o', 'd', '^', '^', '^', '^', 'b', 'y', '^', 'e', '^', '^', '^', '^', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '^', 's', '^', 'a', 'i', 'd', ' ', ' ', ' ', ' ', ' ', ' ', 'r', 'r', 'a', 't', 't', '^', '^', 's', '^', '^', '^', '^', '^', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 't', 'h', 'e', 'y', ' ', ' ', ' ', ' ', ' ', ' ', '^', 'w', '^', 'a', 'n', 'n', 't', 't', '^', '^', '^', '^', '^', '^', '^', ' ', ' ', '^', 'h', 'o', 'm', '^', 'e', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^'], np.float64(1.5022820406499606e-30))