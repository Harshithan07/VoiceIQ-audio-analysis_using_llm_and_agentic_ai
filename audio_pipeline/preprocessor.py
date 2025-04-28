import librosa
import soundfile as sf

def load_audio(audio_path, sr=16000):
    """
    Loads audio file and returns waveform and sample rate.
    """
    waveform, sr = librosa.load(audio_path, sr=sr)
    return waveform, sr

def preprocess_audio(audio_path, sr=16000):
    """
    Loads, trims silence, and normalizes audio.
    """
    y, sr = librosa.load(audio_path, sr=sr)  # Resample to 16kHz
    y_trimmed, _ = librosa.effects.trim(y)
    y_normalized = librosa.util.normalize(y_trimmed)
    return y_normalized, sr

def save_audio(audio, sr, path):
    """
    Saves waveform to specified path.
    """
    sf.write(path, audio, sr)
    return path

def segment_audio(audio, sr, segment_duration=30):
    """
    Splits audio into fixed-length segments (in seconds).
    """
    segment_length = segment_duration * sr
    segments = [
        audio[i:i+segment_length] for i in range(0, len(audio), segment_length)
    ]
    return segments
