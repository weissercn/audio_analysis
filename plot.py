import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Play the audio file using IPython.display
from IPython.display import Audio, display
from scipy.signal import istft, stft

from analysis import compute_pause_boundaries


def plot_waveform(y, sr, spec="",save_spec=None):
    # --- Plot 1: Waveform ---
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title(f'Waveform {spec}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Save the plot (optional)
    if save_spec:
        plt.savefig(f'plots/waveform_plot_{spec}.png')
        print(f"Saved waveform_plot_{spec}.png")

def plot_spectrogram(y, sr, spec="", save_spec=None):
    # --- Plot 2: Spectrogram (Log Frequency Scale) ---
    # Calculate the Short-Time Fourier Transform (STFT)
    S = librosa.stft(y)
    # Convert amplitude spectrogram to dB-scaled spectrogram
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    plt.figure(figsize=(12, 4))
    # Display the spectrogram
    # Using y_axis='log' emphasizes lower frequencies, common for voice
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(img, format='%+2.0f dB') # Add a color bar to show dB scale
    plt.title(f'Spectrogram (Log Frequency Scale) {spec}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    # Save the plot (optional)
    if save_spec:
        plt.savefig(f'plots/spectrogram_plot_{spec}.png')
        print(f"Saved spectrogram_plot_{spec}.png")

def plot_spectrogram_with_pause_boundaries(y, sr):
    pause_starts, pause_ends = compute_pause_boundaries(y, sr)
    plot_spectrogram(y, sr)
    plt.vlines(pause_starts, 0, sr/2, color='white', linestyle=':', alpha=0.7, label='Pause starts')
    plt.vlines(pause_ends, 0, sr/2, color='white', linestyle='-', alpha=0.7, label='Pause ends')
    plt.legend(loc='upper right')

def plot_spectrogram_mel(y, sr):
    # Plot Mel Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # Convert to decibels for better visualization
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Display the mel spectrogram
    img1 = librosa.display.specshow(spectrogram_db, 
                                    x_axis='time', 
                                    y_axis='mel', 
                                    sr=sr)
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    
def plot_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')

def plot_rms(y, sr):
    rms = librosa.feature.rms(y=y)
    plt.figure(figsize=(12, 4))
    frames = np.arange(len(rms[0]))
    time = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    plt.title('RMS Energy Over Time')
    plt.plot(time, rms[0])
    plt.show()

def plot_zero_crossing_rate(y, sr):
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    plt.figure(figsize=(12, 4))
    frames = np.arange(len(zero_crossings[0]))
    time = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    plt.title('Zero Crossing Rate Over Time')
    plt.plot(time, zero_crossings[0])
    plt.show()