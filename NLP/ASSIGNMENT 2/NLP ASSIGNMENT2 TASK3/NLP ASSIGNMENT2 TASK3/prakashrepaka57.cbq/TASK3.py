import librosa # type: ignore
import librosa.display # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import warnings

# Suppress warnings if librosa's pitch algorithm (YIN) has low confidence
warnings.filterwarnings('ignore', 'Trying to estimate tuning from a non-pitch-tracked signal')

# --- CONFIGURATION ---
# IMPORTANT: Replace this with the path to your own WAV file from Task-1
# As a placeholder, I'm using a built-in librosa example file.

# --- USE YOUR FILE HERE ---
FILE_PATH = r'name.wav'
y, sr = librosa.load(FILE_PATH, sr=None)


if y.size > 0:
    
    # --- ANALYSIS PARAMETERS ---
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    # --- FEATURE EXTRACTION ---
    # 1. Energy (RMS)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    
    # 2. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    
    # 3. Pitch (Fundamental Frequency) using the YIN algorithm
    # 'f0' will contain frequency values, or 'np.nan' for unvoiced frames
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )
    
    # Get time points for each frame for plotting
    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

    
    # --- TASK 1 & 2: Time-Domain Plot with Voiced/Unvoiced & Pitch ---
    
    print("\n--- Tasks 1 & 2: Generating Time-Domain Plot ---")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot the waveform
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.5, color='b')
    ax.set_title('Time-Domain Plot with Voiced/Unvoiced Regions and Pitch')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    # Task 1: Mark voiced regions
    # We shade the background where 'voiced_flag' is True
    ax.fill_between(times, -1, 1, where=voiced_flag, alpha=0.2, color='g', label='Voiced Region')

    # Task 2: Mark highest and lowest pitch
    # We only consider valid (non-NaN) pitch values
    valid_f0 = f0[voiced_flag]
    
    if len(valid_f0) > 0:
        # Find highest pitch
        idx_max_pitch = np.nanargmax(f0)
        time_max_pitch = times[idx_max_pitch]
        freq_max_pitch = f0[idx_max_pitch]
        
        # Find lowest pitch
        # We must ignore NaNs and zeros, so we use the 'valid_f0' array
        idx_min_pitch_in_valid = np.nanargmin(valid_f0)
        # Now find the corresponding original index
        # This is a bit complex, but finds the *first* occurrence
        time_min_pitch = times[voiced_flag][idx_min_pitch_in_valid]
        freq_min_pitch = valid_f0[idx_min_pitch_in_valid]

        ax.axvline(x=time_max_pitch, color='r', linestyle='--', label=f'Highest Pitch: {freq_max_pitch:.2f} Hz')
        ax.axvline(x=time_min_pitch, color='m', linestyle='--', label=f'Lowest Pitch: {freq_min_pitch:.2f} Hz')
        
        print(f"Highest Pitch Found: {freq_max_pitch:.2f} Hz at {time_max_pitch:.2f}s")
        print(f"Lowest Pitch Found:  {freq_min_pitch:.2f} Hz at {time_min_pitch:.2f}s")

    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig('time_domain_plot.png')
    plt.close(fig)
    print("Time-domain plot saved as 'time_domain_plot.png'")

    
    # --- TASK 3: Fundamental Frequency (Pitch) ---
    
    print("\n--- Task 3: Overall Fundamental Frequency ---")
    if len(valid_f0) > 0:
        mean_f0 = np.nanmean(valid_f0)
        median_f0 = np.nanmedian(valid_f0)
        print(f"The mean fundamental frequency (pitch) for voiced sections is: {mean_f0:.2f} Hz")
        print(f"The median fundamental frequency (pitch) for voiced sections is: {median_f0:.2f} Hz")
    else:
        print("No valid pitch (voiced sections) detected in the file.")

        
    # --- TASK 4: Frame Energy (Voiced vs. Unvoiced) ---
    
    print("\n--- Task 4: Frame Energy Analysis ---")
    
    # Find a good example of a voiced frame (highest energy)
    idx_voiced_frame = np.argmax(rms)
    time_voiced_frame = times[idx_voiced_frame]
    
    # Find a good example of an unvoiced frame (lowest energy, but not pure silence)
    # Let's find a frame with low energy but *some* ZCR
    rms_low = np.percentile(rms[rms > 0], 10) # 10th percentile of non-zero energy
    zcr_high = np.percentile(zcr, 90) # 90th percentile ZCR
    
    # Find frames that match this "unvoiced" criteria
    unvoiced_candidates = np.where((rms <= rms_low) & (zcr >= zcr_high))[0]
    
    if len(unvoiced_candidates) > 0:
        idx_unvoiced_frame = unvoiced_candidates[0] # Pick the first one
    else:
        # Fallback: just pick the frame with the lowest (non-zero) energy
        idx_unvoiced_frame = np.argmin(rms[rms > 0]) 
        
    time_unvoiced_frame = times[idx_unvoiced_frame]

    # Get the raw audio samples for these specific frames
    start_voiced = idx_voiced_frame * HOP_LENGTH
    end_voiced = start_voiced + FRAME_LENGTH
    frame_voiced = y[start_voiced:end_voiced]

    start_unvoiced = idx_unvoiced_frame * HOP_LENGTH
    end_unvoiced = start_unvoiced + FRAME_LENGTH
    frame_unvoiced = y[start_unvoiced:end_unvoiced]

    # Compute frame energy (Sum of squares)
    energy_voiced = np.sum(frame_voiced**2)
    energy_unvoiced = np.sum(frame_unvoiced**2)

    print(f"Voiced Frame selected around {time_voiced_frame:.2f}s")
    print(f"Computed Energy (Sum of Squares): {energy_voiced:.4f}")
    
    print(f"Unvoiced Frame selected around {time_unvoiced_frame:.2f}s")
    print(f"Computed Energy (Sum of Squares): {energy_unvoiced:.4f}")
    
    # Comment on Frame Energy
    print("\n* Energy Comment:")
    print(f"The energy of the voiced frame ({energy_voiced:.4f}) is significantly higher "
          f"than the energy of the unvoiced frame ({energy_unvoiced:.4f}). "
          "This is expected, as voiced sounds (like vowels or sung notes) involve "
          "vocal cord vibration and carry much more power than unvoiced sounds "
          "(like whispers, 's', or 'f' sounds), which are more noise-like.")


    # --- TASK 5: Zero-Crossings ---
    
    print("\n--- Task 5: Zero-Crossing Rate (ZCR) ---")
    
    # Plot ZCR vs. Waveform
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 8), sharex=True)
    
    librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.7, color='b')
    ax1.set_title('Waveform')
    ax1.set_ylabel('Amplitude')
    
    ax2.plot(times, zcr, color='r')
    ax2.set_title('Zero-Crossing Rate (ZCR)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('ZCR')
    ax2.set_ylim(0, 1) # ZCR is normalized between 0 and 1
    
    plt.tight_layout()
    fig.savefig('zcr_plot.png')
    plt.close(fig)
    print("ZCR plot saved as 'zcr_plot.png'")

    print("\n* ZCR Comment:")
    print("The plot above shows the Zero-Crossing Rate (ZCR) over time. "
          "ZCR is the rate at which the signal changes sign (from positive to negative or vice-versa).")
    print("- **Low ZCR** (e.g., in the middle section) corresponds to the **voiced** regions. "
          "This is because periodic, low-frequency sounds (like a trumpet note) cross the zero-axis relatively slowly.")
    print("- **High ZCR** (e.g., at the beginning/end) corresponds to the **unvoiced** regions. "
          "This is characteristic of noise-like sounds (like the breathy attack of the trumpet), "
          "which have high-frequency components and cross the zero-axis very frequently.")
    

    # --- TASK 6: Autocorrelation ---
    
    print("\n--- Task 6: Autocorrelation ---")
    
    # We re-use the voiced and unvoiced frames from Task 4
    
    # Autocorrelation of the VOICED frame
    # We use 'full' mode to see the correlation at all lags
    autocorr_voiced = np.correlate(frame_voiced, frame_voiced, mode='full')
    # We only care about the second half (positive lags)
    autocorr_voiced = autocorr_voiced[len(autocorr_voiced)//2:]
    
    # Autocorrelation of the UNVOICED frame
    autocorr_unvoiced = np.correlate(frame_unvoiced, frame_unvoiced, mode='full')
    autocorr_unvoiced = autocorr_unvoiced[len(autocorr_unvoiced)//2:]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 8), sharex=True)
    
    # Voiced Plot
    ax1.plot(autocorr_voiced)
    ax1.set_title(f'Autocorrelation of VOICED Frame (around {time_voiced_frame:.2f}s)')
    ax1.set_xlabel('Lag (samples)')
    ax1.set_ylabel('Correlation')

    # Find the first significant peak after the 0-lag peak
    # This peak corresponds to the fundamental period (T0)
    # We search *after* an initial dip (e.g., after first 50 samples) to avoid the main lobe
    lag_offset = 50 
    peak_index = np.argmax(autocorr_voiced[lag_offset:]) + lag_offset
    T0_samples = peak_index
    F0_hz = sr / T0_samples
    
    ax1.axvline(x=T0_samples, color='r', linestyle='--', label=f'Peak (Fundamental Period)\nLag = {T0_samples} samples\nF0 = {F0_hz:.2f} Hz')
    ax1.legend()
    
    # Unvoiced Plot
    ax2.plot(autocorr_unvoiced)
    ax2.set_title(f'Autocorrelation of UNVOICED Frame (around {time_unvoiced_frame:.2f}s)')
    ax2.set_xlabel('Lag (samples)')
    ax2.set_ylabel('Correlation')
    
    plt.tight_layout()
    fig.savefig('autocorrelation_plot.png')
    plt.close(fig)
    print("Autocorrelation plot saved as 'autocorrelation_plot.png'")

    print("\n* Autocorrelation Comment:")
    print("Autocorrelation measures the similarity of a signal with a delayed copy of itself. "
          "It is a key method for finding pitch in voiced sounds.")
    print(f"- **Voiced Frame Plot:** The top plot shows a strong, clear set of peaks. The first major peak (marked in red) "
          f"after the initial one at lag 0 represents the **fundamental period ($T_0$)** of the signal. "
          f"In this frame, the peak is at lag ${T0_samples}$, which corresponds to a fundamental frequency ($F_0 = {sr} / T_0$) "
          f"of **{F0_hz:.2f} Hz**. This clear periodicity is the defining characteristic of a voiced sound.")
    print("- **Unvoiced Frame Plot:** The bottom plot shows the autocorrelation of an unvoiced, noisy frame. "
          "It has a single strong peak at lag 0 and then decays to (or hovers around) zero very quickly. "
          "There are no other significant, repeating peaks. This indicates a lack of periodicity and is characteristic of noise.")

else:
    print("Audio file was not loaded correctly. Please check the file path and format.")