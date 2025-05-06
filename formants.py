from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from IPython.display import Audio, display
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks, freqz, lfilter


def estimate_resonance_bandwidth(w: np.ndarray, magnitude_db: np.ndarray, peak_idx: int) -> float:
    """
    Estimate the bandwidth of a resonance peak at -3dB from its maximum.

    Args:
        w: Frequency array in Hz
        magnitude_db: Magnitude spectrum in dB
        peak_idx: Index of the peak in the arrays

    Returns:
        float: Bandwidth in Hz at -3dB from peak
    """
    peak_mag = magnitude_db[peak_idx]
    threshold = peak_mag - 3  # -3dB bandwidth
    
    # Find bandwidth
    left_idx = peak_idx
    while left_idx > 0 and magnitude_db[left_idx] > threshold:
        left_idx -= 1
    
    right_idx = peak_idx
    while right_idx < len(magnitude_db) - 1 and magnitude_db[right_idx] > threshold:
        right_idx += 1
    
    if right_idx > left_idx:
        bandwidth = w[right_idx] - w[left_idx]
    else:
        bandwidth = 100
    
    return bandwidth

def synthesize_harmonics(f0_track: np.ndarray, times: np.ndarray, sr: int, 
                        n_harmonics: int = 5, audio_length: Optional[int] = None) -> np.ndarray:
    """
    Synthesize audio using only harmonics of F0.

    Args:
        f0_track: Array of fundamental frequency values over time
        times: Time points corresponding to f0_track values
        sr: Sample rate in Hz
        n_harmonics: Number of harmonics to synthesize
        audio_length: Length of output audio in samples. If None, uses times[-1] * sr

    Returns:
        np.ndarray: Synthesized audio signal containing only harmonics
    """
    # Create time array for full audio
    if audio_length is None:
        audio_length = int(times[-1] * sr)
    t = np.arange(audio_length) / sr
    
    # Interpolate F0 to audio sample rate
    f0_interp = interp1d(times, f0_track, kind='linear', 
                        bounds_error=False, fill_value=0)
    f0_at_samples = f0_interp(t)
    
    # Initialize audio
    y_harmonics = np.zeros_like(t)
    
    # Add each harmonic
    for h in range(1, n_harmonics + 1):
        # Calculate instantaneous phase
        phase = np.cumsum(2 * np.pi * f0_at_samples * h / sr)
        
        # Add harmonic with decreasing amplitude
        amplitude = 1.0 / h  # Simple amplitude envelope
        y_harmonics += amplitude * np.sin(phase)
    
    # Normalize
    y_harmonics = y_harmonics / np.max(np.abs(y_harmonics)) * 0.8
    
    return y_harmonics


def filter_by_formant_tracks(y: np.ndarray, sr: int, formants: np.ndarray, 
                           times: np.ndarray, bandwidths: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    Filter original audio to isolate formants using time-varying filters.

    Args:
        y: Input audio signal
        sr: Sample rate in Hz
        formants: Array of formant frequencies over time (n_formants x n_frames)
        times: Time points corresponding to formant values
        bandwidths: List of bandwidths for each formant in Hz. If None, uses default values

    Returns:
        List[np.ndarray]: List of filtered audio signals, one for each formant
    """
    if bandwidths is None:
        bandwidths = [200, 300, 400, 500]  # Default bandwidths for F1-F4
    
    n_formants = formants.shape[0]
    y_formants = []
    
    for i in range(n_formants):
        # Create time-varying filter for this formant
        y_filtered = np.zeros_like(y)
        
        # Interpolate formant track to audio sample rate
        f_interp = interp1d(times, formants[i], kind='linear', 
                           bounds_error=False, fill_value=0)
        t_audio = np.arange(len(y)) / sr
        formant_at_samples = f_interp(t_audio)
        
        # Process in chunks
        chunk_size = 1024
        for j in range(0, len(y), chunk_size):
            chunk_end = min(j + chunk_size, len(y))
            chunk = y[j:chunk_end]
            
            # Get center frequency for this chunk
            f_center = np.mean(formant_at_samples[j:chunk_end])
            
            if f_center > 0:
                # Design bandpass filter
                f_low = max(f_center - bandwidths[i]/2, 50)
                # Nyquist: the highest frequency component that can be accurately represented in a sampled signal 
                #without aliasing.
                f_high = min(f_center + bandwidths[i]/2, sr/2 - 100)
                
                # For stability, use a second-order section (SOS) filter.
                # Butterworth filter: a type of filter that is known for its flat frequency response in the passband and stopband.
                sos = butter(4, [f_low, f_high], btype='bandpass', fs=sr, output='sos')

                # Apply filter
                chunk_filtered = scipy.signal.sosfiltfilt(sos, chunk)
                
                # Apply window for smooth transitions
                window = scipy.signal.windows.hann(len(chunk))
                y_filtered[j:chunk_end] += chunk_filtered * window
        
        # Normalize, 0.8 to avoid clipping
        y_filtered = y_filtered / (np.max(np.abs(y_filtered)) + 1e-8) * 0.8 
        y_formants.append(y_filtered)
    
    return y_formants

def extract_formants_from_envelope(y: np.ndarray, sr: int, n_formants: int = 4, 
                                 f0_times: Optional[np.ndarray] = None,
                                 frame_length: float = 0.025, 
                                 hop_length: float = 0.010) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract formants using spectral envelope analysis to avoid harmonics confusion.

    Args:
        y: Input audio signal
        sr: Sample rate in Hz
        n_formants: Number of formants to extract
        f0_times: Optional time points for F0 track. If provided, formants will be interpolated to match
        frame_length: Frame length in seconds
        hop_length: Hop length in seconds

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Array of formant frequencies (n_formants x n_frames)
            - Time points corresponding to formant values
    """
    frame_samples = int(frame_length * sr)
    hop_samples = int(hop_length * sr)
    
    # Frame the signal, hop_samples is for overlap
    frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=hop_samples)
    n_frames = frames.shape[1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_samples)
    
    # Initialize formant tracks
    formants = np.zeros((n_formants, n_frames))
    
    for i in range(n_frames):
        frame = frames[:, i]
        
        # Apply window
        windowed = frame * scipy.signal.windows.hamming(len(frame))
        
        # Skip if frame energy is too low
        if np.sum(windowed**2) < 1e-6:
            continue
        
        try:
            # Pre-emphasis
            pre_emphasized = lfilter([1, -0.97], [1], windowed)
            
            # LPC analysis for spectral envelope
            lpc_order = 14  # Higher order for better envelope estimation
            lpc_coeffs = librosa.lpc(pre_emphasized, order=lpc_order)
            
            # Get frequency response (spectral envelope)
            w, h = freqz([1], lpc_coeffs, worN=1024, fs=sr)
            magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
            
            # Smooth the envelope to remove harmonic ripples
            magnitude_smooth = gaussian_filter1d(magnitude_db, sigma=5)
            
            # Find peaks in the smoothed envelope
            peaks, properties = find_peaks(magnitude_smooth, 
                                         prominence=3,
                                         distance=int(200 * 1024 / (sr/2)))  # Min 200 Hz between formants
            
            # Get actual formant candidates
            formant_candidates = []
            for peak_idx in peaks:
                freq = w[peak_idx]
                amp = magnitude_smooth[peak_idx]
                
                # Look for broad resonances, not narrow harmonics
                bandwidth = estimate_resonance_bandwidth(w, magnitude_smooth, peak_idx)
                
                if bandwidth > 50:  # Only keep broad resonances
                    formant_candidates.append({
                        'freq': freq,
                        'amp': amp,
                        'bw': bandwidth
                    })
            
            # Sort by amplitude and assign to formant tracks
            formant_candidates.sort(key=lambda x: x['amp'], reverse=True)
            
            # Typical formant ranges
            formant_ranges = [
                (250, 800),    # F1
                (800, 2500),   # F2  
                (2000, 3500),  # F3
                (3000, 4500),  # F4
            ]
            
            # Assign formants based on ranges
            for j, (f_min, f_max) in enumerate(formant_ranges[:n_formants]):
                for candidate in formant_candidates:
                    if f_min <= candidate['freq'] <= f_max:
                        formants[j, i] = candidate['freq']
                        break
            
        except Exception as e:
            if i > 0:
                formants[:, i] = formants[:, i-1]
    
    # Post-processing
    for j in range(n_formants):
        valid_mask = formants[j] > 0
        if np.any(valid_mask):
            # Interpolate
            formants[j] = np.interp(times, times[valid_mask], formants[j][valid_mask])
            # Apply median filter
            formants[j] = scipy.signal.medfilt(formants[j], kernel_size=5)
            # Smooth
            formants[j] = gaussian_filter1d(formants[j], sigma=3)

    if f0_times is not None:
        # Ensure same time axis
        if len(f0_times) != len(times):
            formants_interp = np.zeros((4, len(f0_times)))
            for i in range(4):
                formants_interp[i] = np.interp(f0_times, times, formants[i])
            formants = formants_interp
    
    return formants, times

def get_f0_track(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fundamental frequency (F0) track from audio signal.

    Args:
        y: Input audio signal
        sr: Sample rate in Hz

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Array of F0 values over time
            - Time points corresponding to F0 values
    """
    # Extract F0
    f0_track, _, _ = librosa.pyin(y, fmin=80, fmax=300, sr=sr)
    times = librosa.times_like(f0_track, sr=sr)
    
    # Fill NaN values
    valid_f0 = ~np.isnan(f0_track)
    if np.any(valid_f0):
        f0_track = np.interp(times, times[valid_f0], f0_track[valid_f0])
    return f0_track, times

def plot_formant_harmonics_comparison(y: np.ndarray, y_harmonics: np.ndarray, 
                                    y_all_formants: np.ndarray, sr: int) -> None:
    """
    Plot comparison of original audio, harmonics, and formants.

    Args:
        y: Original audio signal
        y_harmonics: Synthesized harmonics signal
        y_all_formants: Combined formants signal
        sr: Sample rate in Hz
    """
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Time axis
    t_audio = np.arange(len(y)) / sr
    
    # Original
    axes[0].plot(t_audio, y, alpha=0.7)
    axes[0].set_title('Original Audio')
    axes[0].set_ylabel('Amplitude')
    
    # Harmonics
    axes[1].plot(t_audio[:len(y_harmonics)], y_harmonics, alpha=0.7, color='red')
    axes[1].set_title('Harmonics Only (Synthesized)')
    axes[1].set_ylabel('Amplitude')
    
    # Formants
    axes[2].plot(t_audio, y_all_formants, alpha=0.7, color='green')
    axes[2].set_title('All Formants Combined (Filtered)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

def plot_spectrogram_with_harmonics_and_formants_labeled(y: np.ndarray, sr: int, 
                                                       f0_track: np.ndarray,
                                                       formants: np.ndarray, 
                                                       times: np.ndarray) -> None:
    """
    Plot spectrogram with labeled harmonics and formants.

    Args:
        y: Input audio signal
        sr: Sample rate in Hz
        f0_track: Array of F0 values over time
        formants: Array of formant frequencies (n_formants x n_frames)
        times: Time points corresponding to F0 and formant values
    """
    # Additional spectrogram with harmonics and formants
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', 
                                ax=ax, cmap='viridis')
    
    # Plot F0 track
    ax.plot(times, f0_track, 'w-', linewidth=2, label='F0')
    
    # Plot harmonics (dotted lines)
    harmonics = [f0_track * (i+1) for i in range(5)]  # Calculate harmonics
    for i in range(5):
        ax.plot(times, harmonics[i], 'r:', linewidth=1.5, 
                label=f'H{i+1}' if i < 3 else None)
    
    # Plot formants (solid lines)
    formant_colors = ['orange', 'yellow', 'cyan', 'magenta']
    for i in range(4):
        ax.plot(times, formants[i], color=formant_colors[i], 
                linewidth=2.5, label=f'F{i+1}')
    
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Harmonics (dotted) vs Formants (solid)')
    ax.set_ylim(0, 5000)
    ax.legend(loc='upper right', ncol=2)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def plot_spectrogram_with_harmonics_and_formants_separated(y: np.ndarray,
                                                         y_harmonics: np.ndarray, 
                                                         y_all_formants: np.ndarray,
                                                         y_harmonics_plus_formants: np.ndarray,
                                                         sr: int) -> None:
    """
    Plot separate spectrograms comparing original audio, harmonics, formants, and their combination.

    Args:
        y: Original audio signal
        y_harmonics: Synthesized harmonics signal
        y_all_formants: Combined formants signal
        y_harmonics_plus_formants: Combined harmonics and formants signal
        sr: Sample rate in Hz
    """
    # Create a spectrogram comparison
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Original spectrogram
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='log', ax=axes[0])
    axes[0].set_title('Original Audio')
    axes[0].set_ylim(0, 5000)

    # Harmonics spectrogram
    D_harm = librosa.amplitude_to_db(np.abs(librosa.stft(y_harmonics)), ref=np.max)
    librosa.display.specshow(D_harm, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title('Harmonics Only')
    axes[1].set_ylim(0, 5000)

    # Formants spectrogram
    D_form = librosa.amplitude_to_db(np.abs(librosa.stft(y_all_formants)), ref=np.max)
    librosa.display.specshow(D_form, sr=sr, x_axis='time', y_axis='log', ax=axes[2])
    axes[2].set_title('All Formants Combined')
    axes[2].set_ylim(0, 5000)

    # Combined spectrogram
    D_comb = librosa.amplitude_to_db(np.abs(librosa.stft(y_harmonics_plus_formants)), ref=np.max)
    librosa.display.specshow(D_comb, sr=sr, x_axis='time', y_axis='log', ax=axes[3])
    axes[3].set_title('Harmonics + Formants Combined')
    axes[3].set_ylim(0, 5000)

    plt.tight_layout()
    plt.show()

def compute_and_plot_formants_harmonics(y: np.ndarray, sr: int) -> None:
    """
    Compute and visualize formants and harmonics from audio signal.

    This function performs a complete analysis of the input audio signal:
    1. Extracts F0 track
    2. Extracts formants
    3. Synthesizes harmonics
    4. Filters formants
    5. Creates visualizations
    6. Plays audio samples

    Args:
        y: Input audio signal
        sr: Sample rate in Hz
    """
    f0_track, times = get_f0_track(y, sr)

    # Extract formants
    formants, formant_times = extract_formants_from_envelope(y, sr, n_formants=4, f0_times=times)
    # Synthesize harmonics
    y_harmonics = synthesize_harmonics(f0_track, times, sr, n_harmonics=5, audio_length=len(y))

    # Filter formants from original audio
    y_formants = filter_by_formant_tracks(y, sr, formants, times)

    # Create combined versions
    # Weights are chosen empirically to balance the harmonics and formants
    y_all_formants = np.sum(y_formants, axis=0) * 0.5  # Sum all formants
    y_harmonics_plus_formants = y_harmonics * 0.3 + y_all_formants * 0.7

    # Print average frequencies
    print("Average frequencies:")
    print(f"F0: {np.mean(f0_track):.1f} Hz")

    # Calculate and print harmonic frequencies
    for i in range(5):  # For 5 harmonics
        harmonic_freq = np.mean(f0_track) * (i + 1)
        print(f"H{i+1}: {harmonic_freq:.1f} Hz")

    # Print formant frequencies
    for i in range(4):
        valid_formants = formants[i][formants[i] > 0]
        if len(valid_formants) > 0:
            print(f"F{i+1}: {np.mean(valid_formants):.1f} Hz")
    print()

    plot_spectrogram_with_harmonics_and_formants_labeled(y, sr, f0_track, formants, times)
    plot_spectrogram_with_harmonics_and_formants_separated(y, y_harmonics, y_all_formants, y_harmonics_plus_formants, sr)

    # Play all versions
    print("1. Original audio:")
    display(Audio(data=y, rate=sr))

    print("\n2. Harmonics only (buzzy/robotic sound):")
    display(Audio(data=y_harmonics, rate=sr))

    print("\n3. Formant 1 only:")
    display(Audio(data=y_formants[0], rate=sr))

    print("\n4. Formant 2 only:")
    display(Audio(data=y_formants[1], rate=sr))

    print("\n5. Formant 3 only:")
    display(Audio(data=y_formants[2], rate=sr))

    print("\n6. Formant 4 only:")
    display(Audio(data=y_formants[3], rate=sr))

    print("\n7. All formants combined:")
    display(Audio(data=y_all_formants, rate=sr))

    print("\n8. Harmonics + Formants combined:")
    display(Audio(data=y_harmonics_plus_formants, rate=sr))



