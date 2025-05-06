from typing import List, Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Play the audio file using IPython.display
from IPython.display import Audio, display
from scipy.signal import istft, stft

from analysis import compute_pause_boundaries


def plot_waveform(y: np.ndarray, sr: int, spec: str = "", save_plot: bool = False) -> None:
    """Plot the waveform of an audio signal.
    
    Args:
        y: Audio time series as a numpy array
        sr: Sampling rate of the audio signal
        spec: Optional specification string to add to the plot title
        save_plot: If True, saves the plot to 'plots/waveform_plot_{spec}.png'
    """
    # --- Plot 1: Waveform ---
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title(f'Waveform {spec}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Save the plot (optional)
    if save_plot:
        plt.savefig(f'plots/waveform_plot_{spec}.png')
        print(f"Saved waveform_plot_{spec}.png")

def plot_spectrogram(y: np.ndarray, sr: int, spec: str = "", save_plot: bool = False) -> None:
    """Plot the spectrogram of an audio signal using log frequency scale.
    
    Args:
        y: Audio time series as a numpy array
        sr: Sampling rate of the audio signal
        spec: Optional specification string to add to the plot title
        save_plot: If True, saves the plot to 'plots/spectrogram_plot_{spec}.png'
    """
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
    if save_plot:
        plt.savefig(f'plots/spectrogram_plot_{spec}.png')
        print(f"Saved spectrogram_plot_{spec}.png")

def plot_spectrogram_with_pause_boundaries(y: np.ndarray, sr: int, spec: str = "", save_plot: bool = False) -> None:
    """Plot the spectrogram with pause boundaries marked.
    
    Args:
        y: Audio time series as a numpy array
        sr: Sampling rate of the audio signal
        spec: Optional specification string to add to the plot title
        save_plot: If True, saves the plot to 'plots/spectrogram_pause_boundaries_{spec}.png'
    """
    pause_starts, pause_ends = compute_pause_boundaries(y, sr)
    plot_spectrogram(y, sr, spec=spec)
    plt.vlines(pause_starts, 0, sr/2, color='white', linestyle=':', alpha=0.7, label='Pause starts')
    plt.vlines(pause_ends, 0, sr/2, color='white', linestyle='-', alpha=0.7, label='Pause ends')
    plt.legend(loc='upper right')
    if save_plot:
        plt.savefig(f'plots/spectrogram_pause_boundaries_{spec}.png')
        print(f"Saved spectrogram_pause_boundaries_{spec}.png")

def plot_spectrogram_mel(y: np.ndarray, sr: int, spec: str = "", save_plot: bool = False) -> None:
    """Plot the mel spectrogram of an audio signal.
    
    Args:
        y: Audio time series as a numpy array
        sr: Sampling rate of the audio signal
        spec: Optional specification string to add to the plot title
        save_plot: If True, saves the plot to 'plots/mel_spectrogram_plot_{spec}.png'
    """
    # Plot Mel Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # Convert to decibels for better visualization
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(12, 4))
    # Display the mel spectrogram
    img = librosa.display.specshow(spectrogram_db, 
                                    x_axis='time', 
                                    y_axis='mel', 
                                    sr=sr)
    plt.colorbar(img, format='%+2.0f dB') # Add a color bar to show dB scale
    plt.title(f'Mel Spectrogram {spec}')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    if save_plot:
        plt.savefig(f'plots/mel_spectrogram_plot_{spec}.png')
        print(f"Saved mel_spectrogram_plot_{spec}.png")

def plot_mfcc(y: np.ndarray, sr: int, spec: str = "", save_plot: bool = False) -> None:
    """Plot the Mel-frequency cepstral coefficients (MFCC) of an audio signal.
    
    Args:
        y: Audio time series as a numpy array
        sr: Sampling rate of the audio signal
        spec: Optional specification string to add to the plot title
        save_plot: If True, saves the plot to 'plots/mfcc_plot_{spec}.png'
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC {spec}')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    if save_plot:
        plt.savefig(f'plots/mfcc_plot_{spec}.png')
        print(f"Saved mfcc_plot_{spec}.png")

def plot_rms(y: np.ndarray, sr: int, spec: str = "", save_plot: bool = False) -> None:
    """Plot the Root Mean Square (RMS) energy over time.
    
    Args:
        y: Audio time series as a numpy array
        sr: Sampling rate of the audio signal
        spec: Optional specification string to add to the plot title
        save_plot: If True, saves the plot to 'plots/rms_plot_{spec}.png'
    """
    rms = librosa.feature.rms(y=y)
    plt.figure(figsize=(12, 4))
    frames = np.arange(len(rms[0]))
    time = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    plt.title(f'RMS Energy Over Time {spec}')
    plt.plot(time, rms[0])
    if save_plot:
        plt.savefig(f'plots/rms_plot_{spec}.png')
        print(f"Saved rms_plot_{spec}.png")
    plt.show()

def plot_zero_crossing_rate(y: np.ndarray, sr: int, spec: str = "", save_plot: bool = False) -> None:
    """Plot the zero crossing rate over time.
    
    Args:
        y: Audio time series as a numpy array
        sr: Sampling rate of the audio signal
        spec: Optional specification string to add to the plot title
        save_plot: If True, saves the plot to 'plots/zero_crossing_rate_plot_{spec}.png'
    """
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    plt.figure(figsize=(12, 4))
    frames = np.arange(len(zero_crossings[0]))
    time = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    plt.title(f'Zero Crossing Rate Over Time {spec}')
    plt.plot(time, zero_crossings[0])
    if save_plot:
        plt.savefig(f'plots/zero_crossing_rate_plot_{spec}.png')
        print(f"Saved zero_crossing_rate_plot_{spec}.png")
    plt.show()