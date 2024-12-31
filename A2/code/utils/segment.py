from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import librosa
import numpy as np
import scipy

def segment_audio_by_silence(file_name, min_silence_len=200, silence_thresh=-40, buffer_ms=500):
    """
    Segments an audio file into multiple files based on silence detection.

    Args:
        file_name (str): Name of the input audio file (without extension).
        min_silence_len (int): Minimum length of silence in milliseconds to consider as a separator.
        silence_thresh (int): Silence threshold in dBFS.
        buffer_ms (int): Additional buffer to include before and after segments, in milliseconds.
    """
    # Define input and output directories
    input_path = os.path.join("A2", "resources", "audio_files", "raw", f"{file_name}.wav")
    output_dir = os.path.join("A2", "resources", "audio_files", "segmented")
    os.makedirs(output_dir, exist_ok=True)
    
    #  Load audio and resample
    audio = AudioSegment.from_file(input_path)
    original_sr = audio.frame_rate
    audio = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
    
    if original_sr != 16000:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
        print("what")

    if file_name == "nirit" or file_name == "yaron":
        audio = audio[1000:]
    
    # Detect non-silent segments
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    # Filter out non-silent segments that are less than 20 milliseconds long
    nonsilent_ranges = [(start, end) for start, end in nonsilent_ranges if (end - start) >= 20]

    if file_name=="guy":
        nonsilent_ranges = nonsilent_ranges[0:8] + nonsilent_ranges[9:]

    if file_name=="rom":
        nonsilent_ranges = nonsilent_ranges[0:5] + nonsilent_ranges[6:]

    if file_name=="ohad":
        nonsilent_ranges = nonsilent_ranges[0:4] + nonsilent_ranges[5:]
        nonsilent_ranges[6] = (nonsilent_ranges[6][0]+250, nonsilent_ranges[6][1])

    # Add a buffer to each segment
    buffered_segments = [(max(start - buffer_ms, 0), min(end + buffer_ms, len(audio)))
                            for start, end in nonsilent_ranges]
        
    # Save each segment as a separate file
    for i, (start, end) in enumerate(buffered_segments):
        segment = audio[start:end]
        output_path = os.path.join(output_dir, file_name + f"_{i}.wav")
        segment.export(output_path, format="wav")
        print(f"Segment saved: {output_path}")

if __name__ == "__main__":
    input_dir = os.path.join("A2", "resources", "audio_files", "raw")
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            name = os.path.splitext(file_name)[0]
            segment_audio_by_silence(name)

