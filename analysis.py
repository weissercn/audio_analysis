import librosa
import numpy as np
from scipy import signal
from scipy.signal import istft, stft


def reduce_noise_from_stft_first_second(y, sr):
    # Parameters for STFT
    frame_length = 2048
    hop_length = 512
    
    # Compute STFT: Short-time Fourier Transform
    f, t, Zxx = stft(y, fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)
    
    # Estimate noise profile from a relatively quiet segment
    # Using the first 1000ms of audio or a known quiet segment
    noise_segment = Zxx[:, :int(sr/hop_length)]  # First second
    noise_profile = np.mean(np.abs(noise_segment), axis=1)
    
    # Apply spectral subtraction
    # Subtract the noise profile from each frame
    gain_factor = 2.0   # Adjust this to control noise reduction strength
    threshold = 0.01   # Minimum amplitude threshold
    
    # Calculate the magnitude of the STFT
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Subtract noise profile from magnitude
    mag_reduced = np.maximum(mag - gain_factor * noise_profile[:, np.newaxis], threshold * mag)
    
    # Reconstruct signal
    Zxx_reduced = mag_reduced * np.exp(1j * phase)
    _, y_reduced = istft(Zxx_reduced, fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)
    
    return y_reduced

def reduce_noise(y, sr, type="wiener"):
    if type == "first_second":
        y_clean = reduce_noise_from_stft_first_second(y, sr)
    elif type == "wiener":
        y_clean = signal.wiener(y, mysize=1001)
    elif type == "medfilt":
        y_clean = signal.medfilt(y, kernel_size=51)
    elif type =="nmf":
        # Compute spectrogram
        S = np.abs(librosa.stft(y))**2

        # Perform NMF
        W, H = librosa.decompose.decompose(S, n_components=2, sort=True)

        # The first component is typically the dominant sound
        # Reconstruct using only the first component
        S_clean = W[:, 0:1] @ H[0:1, :]

        # Convert back to time domain
        y_clean = librosa.griffinlim(np.sqrt(S_clean))
    elif type == "adaptive":
        y_preemphasized = librosa.effects.preemphasis(y)

        y_clean = signal.wiener(y_preemphasized, mysize=1001)

    else:
        raise ValueError(f"Invalid noise reduction type: {type}")
    return y_clean


def compute_pause_boundaries(y, sr):
    # Calculate root mean square energy
    rms = librosa.feature.rms(y=y)[0]

    # Define threshold (you may need to adjust this)
    threshold = np.mean(rms) * 0.1  # 10% of mean energy

    # Find segments below threshold
    is_pause = rms < threshold

    # Convert to time segments
    frame_length = 512  # default frame length
    hop_length = 512   # default hop length
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Find pause boundaries
    pause_starts = []
    pause_ends = []
    in_pause = False

    for i, is_pause_frame in enumerate(is_pause):
        if is_pause_frame and not in_pause:
            pause_starts.append(times[i])
            in_pause = True
        elif not is_pause_frame and in_pause:
            pause_ends.append(times[i])
            in_pause = False

    # Handle if audio ends in a pause
    if in_pause:
        pause_ends.append(times[-1])
    return pause_starts, pause_ends

def split_audio_at_pause_boundaries(y, sr):
    pause_starts, pause_ends = compute_pause_boundaries(y, sr)
    
    # Split audio at pause boundaries
    segments = []
    current_start = 0
    for start, end in zip(pause_starts, pause_ends):
        segment = y[current_start:int(start * sr)]
        if len(segment) > 0:
            segments.append(segment)
        current_start = int(end * sr)
    
    # Add final segment after last pause
    final_segment = y[current_start:]
    if len(final_segment) > 0:
        segments.append(final_segment)
        
    return segments

