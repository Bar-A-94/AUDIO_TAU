from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import scipy

def segment_audio_by_silence(file_name, min_silence_len=200, silence_thresh=-40, buffer_ms=200):
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
    
    # Load the audio file
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000)

    
    # Detect non-silent segments
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    # Add a buffer to each segment
    buffered_segments = [(max(start - buffer_ms, 0), min(end + buffer_ms, len(audio)))
                            for start, end in nonsilent_ranges]
        
    # Save each segment as a separate file
    for i, (start, end) in enumerate(buffered_segments):
        segment = audio[start:end]
        output_path = os.path.join(output_dir, file_name + f"_segment_{i}.wav")
        segment.export(output_path, format="wav")
        print(f"Segment saved: {output_path}")

if __name__ == "__main__":
    segment_audio_by_silence("bar")
    segment_audio_by_silence("neta")