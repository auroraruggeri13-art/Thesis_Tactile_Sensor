import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import find_peaks

# Close any plots from previous runs
plt.close('all')

# --- CONFIGURATION ---
DATA_DIRECTORY_finger = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Sensor-Logs"
DATA_DIRECTORY_ATI = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\ATI-FT-Data-Logs"
BAROMETER_FILENAME = "datalog_2025-10-10_09-56-47.csv"
FORCETORQUE_FILENAME = "2025-10-10_09-56-57_FT29602.csv"
n_experiment = 40
X_COORDINATES_PATTERN = [40]
Y_COORDINATE = [15]
Time_shift_FT = 0.13  # Time shift in seconds to apply to Force/Torque data

# --- PEAK FINDING PARAMETERS ---
PEAK_PARAMS = {
    'barometer': { 'window_size': 1, 'prominence_factor': 0.2, 'distance': 5},
    'fz_pos':    { 'window_size': 50, 'prominence_factor': 0.3, 'distance': 500, 'absolute_threshold': True}, 
    'fz_neg':    { 'window_size': 50, 'prominence_factor': 0.3, 'distance': 500, 'absolute_threshold': True},
    # Add parameters for Tx and Ty (copied from Fz, can be tuned later)
    'tx_pos':    { 'window_size': 100, 'prominence_factor': 0.3, 'distance': 500, 'absolute_threshold': True}, 
    'tx_neg':    { 'window_size': 100, 'prominence_factor': 0.3, 'distance': 500, 'absolute_threshold': True},
    'ty_pos':    { 'window_size': 100, 'prominence_factor': 0.3, 'distance': 500, 'absolute_threshold': True}, 
    'ty_neg':    { 'window_size': 100, 'prominence_factor': 0.3, 'distance': 500, 'absolute_threshold': True},
    # Define distance for merging peaks from different signals (in number of data points)
    'merge_distance': 50 
}


def find_signal_peaks(data, find_maxima=True, **kwargs):
    """
    Finds peaks in a signal relative to the signal's mean value.
    """
    if data is None or len(data) < 3: return np.array([], dtype=int)
    
    # Get parameters
    window_size = kwargs.get('window_size', 20)
    distance = kwargs.get('distance', 50)
    prominence_factor = kwargs.get('prominence_factor', 0.5)
    use_absolute_threshold = kwargs.get('absolute_threshold', False)
    
    # Smooth the data
    smoothed_data = moving_average(data, window_size)
    
    # This makes the signal's baseline effectively zero for peak finding.
    signal_mean = np.mean(smoothed_data)
    centered_data = smoothed_data - signal_mean

    find_peaks_params = {'distance': distance}
    
    if use_absolute_threshold:
        # Calculate threshold based on the maximum deviation from the mean
        max_deviation = np.max(np.abs(centered_data))
        threshold = max(1e-6, max_deviation * prominence_factor)
        find_peaks_params['height'] = threshold
    else:
        # Prominence is already a relative measure, but this logic is kept
        signal_std = np.std(smoothed_data)
        find_peaks_params['prominence'] = max(1e-6, signal_std * prominence_factor)

    if find_maxima:
        # Find peaks on the new mean-centered data
        peaks, _ = find_peaks(centered_data, **find_peaks_params)
    else:
        # Find troughs (negative peaks) on the new mean-centered data
        peaks, _ = find_peaks(-centered_data, **find_peaks_params)
        
    return peaks

def moving_average(data, window_size):
    if window_size <= 1: return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

def read_barometer_csv(filepath):
    """Reads the barometer CSV logged by the Python script."""
    print(f"Reading barometer file: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath)
        df.columns = [col.strip() for col in df.columns]
        file_date_str = os.path.basename(filepath).split('_')[1]
        
        df['datetime'] = pd.to_datetime(file_date_str + ' ' + df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        
        df.dropna(subset=['datetime'], inplace=True)
        df.set_index('datetime', inplace=True)
        
        # ADD THESE LINES:
        print(f"Barometer timestamps:")
        print(f"  First: {df.index.min()}")
        print(f"  Last:  {df.index.max()}")
        print(f"  Total samples: {len(df)}")
        
        barometer_cols = [f'barometer {i}' for i in range(1, 7)]
        return (df, *[df[col].to_numpy() for col in barometer_cols])
    except Exception as e:
        print(f"Error reading barometer file: {e}"); return None

def read_forcetorque_csv(filepath):
    """
    Reads the force/torque CSV file, auto-detecting comma or whitespace delimiters.
    """
    print(f"Reading force/torque file: {os.path.basename(filepath)}")
    header_row_index = None
    delimiter = ','  # Default to comma

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Find the header row - look for the line with Status, Fx, Fz, and Time
                if 'Status' in line and 'Fx' in line and 'Fz' in line and 'Time' in line:
                    header_row_index = i                    
                    # Check the header to detect the delimiter
                    if ',' not in line:
                        delimiter = r'\s+'  # Use regex for one or more spaces/tabs
                    break
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}"); return None
    
    if header_row_index is None:
        print("Error: Could not find a valid header row in force/torque file."); return None
        
    try:
        # Read the CSV starting from the header row
        # Handle the trailing comma issue by specifying exact column names
        expected_columns = ['Status (hex)', 'RDTSequence', 'F/T Sequence', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'Time']
        
        df = pd.read_csv(filepath, skiprows=header_row_index, sep=',', engine='python', 
                        names=expected_columns, skipinitialspace=True, na_filter=False,
                        usecols=range(10))  # Only read first 10 columns, ignore trailing comma column
                   
        # Remove rows where Time is truly empty or NaN
        initial_rows = len(df)
        df = df[df['Time'].notna()]
        df = df[df['Time'].astype(str).str.strip() != '']
        df = df[df['Time'].astype(str).str.strip() != 'nan']
        final_rows = len(df)
        
        if df.empty:
            print("DataFrame is empty after removing invalid Time values")
            return None
        
        # Clean the Time column - remove trailing commas and spaces
        df['Time'] = df['Time'].astype(str).str.strip()
        
        # Extract date from filename (e.g., "2025-10-08_11-09-51_FT29602.csv")
        file_date_str = os.path.basename(filepath).split('_')[0]  # Gets "2025-10-08"
                
        # Simple datetime parsing approach
        try:
            # Combine date and time, handling the 3-decimal format directly
            datetime_strings = file_date_str + ' ' + df['Time']
                        
            # Try parsing with milliseconds format first
            df['datetime'] = pd.to_datetime(datetime_strings, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            
            # Count successful vs failed parsing
            successful_parses = df['datetime'].notna().sum()
            total_rows = len(df)
            
            if successful_parses == 0:
                # If all failed, try a different approach
                print("All datetime parsing failed, trying alternative format...")
                df['datetime'] = pd.to_datetime(datetime_strings, errors='coerce', infer_datetime_format=True)
                successful_parses = df['datetime'].notna().sum()
                print(f"Alternative parsing: {successful_parses}/{total_rows} successful")
                
        except Exception as e:
            print(f"Datetime parsing error: {e}")
            return None
        
        # Remove rows where datetime parsing failed
        df.dropna(subset=['datetime'], inplace=True)
        
        if df.empty:
            print("DataFrame is empty after datetime parsing")
            return None
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)

        # Print timestamps
        print(f"Force/Torque timestamps:")
        print(f"  First: {df.index.min()}")
        print(f"  Last:  {df.index.max()}")
        print(f"  Total samples: {len(df)}")

        # Convert force/torque columns to numeric
        for col in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN in force/torque columns
        df.dropna(subset=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'], inplace=True)
        
        if df.empty:
            print("DataFrame is empty after numeric conversion")
            return None

        # Extract force and torque arrays
        forces = (df['Fx'].to_numpy(), df['Fy'].to_numpy(), df['Fz'].to_numpy())
        torques = (df['Tx'].to_numpy(), df['Ty'].to_numpy(), df['Tz'].to_numpy())
        
        return (df, *forces, *torques)
            
    except Exception as e:
        print(f"Error parsing force/torque file: {e}")
        import traceback
        traceback.print_exc()
        return None

def synchronize_data(barometer_df, forcetorque_df):
    baro_start, baro_end = barometer_df.index.min(), barometer_df.index.max()
    ft_start, ft_end = forcetorque_df.index.min(), forcetorque_df.index.max()
    overlap_start, overlap_end = max(baro_start, ft_start), min(baro_end, ft_end)
    if overlap_start >= overlap_end:
        print("Synchronization failed: No overlapping time range found.")
        print(f"  Barometer: {baro_start} to {baro_end}")
        print(f"  Force/Torque: {ft_start} to {ft_end}")
        return None
    print(f"Data overlap found from {overlap_start} to {overlap_end}")
    synced_baro_df = barometer_df.loc[overlap_start:overlap_end].copy()
    synced_ft_df = forcetorque_df.loc[overlap_start:overlap_end].copy()
    synced_baro_df['elapsed_time'] = (synced_baro_df.index - overlap_start).total_seconds()
    synced_ft_df['elapsed_time'] = (synced_ft_df.index - overlap_start).total_seconds()
    return synced_baro_df, synced_ft_df

def plot_barometer_data(barometer_df):
    """
    Plots original barometer data with detected peaks for all 6 barometers.
    """
    if barometer_df is None or barometer_df.empty:
        print("No barometer data to plot")
        return

    fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle('Original Barometer Data (Unsynchronized)', fontsize=16)
    
    # Convert datetime index to elapsed time in seconds for plotting
    start_time = barometer_df.index.min()
    elapsed_time = (barometer_df.index - start_time).total_seconds().values  # Convert to numpy array
    
    for i in range(6):
        row, col = i // 2, i % 2
        baro_col = f'barometer {i+1}'
        
        if baro_col in barometer_df.columns:
            current_baro_data = barometer_df[baro_col].values
            
            # Plot the original data
            ax[row, col].plot(elapsed_time, current_baro_data, label='Data', zorder=1, alpha=0.9)
            ax[row, col].set_title(f'Barometer {i+1}')
            
            # Find and plot peaks for the current barometer
            pos_peaks = find_signal_peaks(current_baro_data, find_maxima=True, **PEAK_PARAMS['barometer'])
            neg_peaks = find_signal_peaks(current_baro_data, find_maxima=False, **PEAK_PARAMS['barometer'])
            all_peak_indices = np.concatenate((pos_peaks, neg_peaks)) if len(pos_peaks) > 0 or len(neg_peaks) > 0 else []
            
            if len(all_peak_indices) > 0:
                peak_times = elapsed_time[all_peak_indices]  # Now works with numpy array
                peak_values = current_baro_data[all_peak_indices]
                ax[row, col].plot(peak_times, peak_values, 'ro', markersize=5, label='Detected Peaks', zorder=3)
        
        if row == 2: ax[row, col].set_xlabel('Elapsed Time (s)')
        if col == 0: ax[row, col].set_ylabel('Pressure (hPa)')
        ax[row, col].grid(True, linestyle='--', alpha=0.6)
        ax[row, col].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_forcetorque_data(time_seconds, forces, torques, filtered_peak_indices=None, is_synchronized=False):
    """
    Plots force and torque data in two separate figures.
    - Figure 1: Fx, Fy, Fz on separate subplots.
    - Figure 2: Tx, Ty, Tz on separate subplots.
    - Red dots are shown on all subplots corresponding to the Fz peak times.
    """
    # Only plot if synchronized and has peaks
    if not is_synchronized or filtered_peak_indices is None or len(filtered_peak_indices) == 0:
        return
    
    title_prefix = "Synchronized"
    force_labels, torque_labels = ['Fx', 'Fy', 'Fz'], ['Tx', 'Ty', 'Tz']
    
    peak_times = time_seconds[filtered_peak_indices]

    # --- Create Force Plot ---
    fig_f, ax_f = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_f.suptitle(f'{title_prefix} Force Data with Fz Peaks', fontsize=16)

    for i in range(3):
        # Plot the continuous force data
        ax_f[i].plot(time_seconds, forces[i], label=f'{force_labels[i]} Data', alpha=0.9)
        
        # If it's the Fz plot, also show the moving average for clarity
        if i == 2:
             ax_f[i].plot(time_seconds, moving_average(forces[i], PEAK_PARAMS['fz_neg']['window_size']), 
                         'g-', alpha=0.7, label='Moving Avg', linewidth=1.2)

        # Plot red dots at peak locations on ALL force subplots
        peak_values = forces[i][filtered_peak_indices]
        ax_f[i].plot(peak_times, peak_values, 'ro', markersize=6, label='Fz Peak Events')

        ax_f[i].set_ylabel(f'{force_labels[i]} (N)')
        ax_f[i].grid(True, linestyle='--', alpha=0.6)
        ax_f[i].legend()

    ax_f[-1].set_xlabel('Elapsed Time (s)')
    fig_f.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Create Torque Plot ---
    fig_t, ax_t = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_t.suptitle(f'{title_prefix} Torque Data with Fz Peaks', fontsize=16)

    for i in range(3):
        # Plot the continuous torque data
        ax_t[i].plot(time_seconds, torques[i], label=f'{torque_labels[i]} Data')

        # Plot red dots at peak locations on ALL torque subplots
        peak_values = torques[i][filtered_peak_indices]
        ax_t[i].plot(peak_times, peak_values, 'ro', markersize=6, label='Fz Peak Events')
        
        ax_t[i].set_ylabel(f'{torque_labels[i]} (Nmm)')
        ax_t[i].grid(True, linestyle='--', alpha=0.6)
        ax_t[i].legend()

    ax_t[-1].set_xlabel('Elapsed Time (s)')
    fig_t.tight_layout(rect=[0, 0.03, 1, 0.95])

def extract_merged_events(synced_baro_data, synced_ft_data):
    """
    Finds peaks in Fz, Tx, and Ty, merges close peaks into single events,
    and extracts sensor data at those event times.
    """
    print("\nExtracting sensor data at merged peak points...")
    s_baro_time, *all_s_barometers = synced_baro_data
    s_ft_time, s_fx, s_fy, s_fz, s_tx, s_ty, s_tz = synced_ft_data
    
    # --- NEW MERGED PEAK FINDING LOGIC ---
    # 1. Find all peaks for Fz, Tx, and Ty individually
    fz_peaks = np.concatenate((
        find_signal_peaks(s_fz, find_maxima=False, **PEAK_PARAMS['fz_neg']),
        find_signal_peaks(s_fz, find_maxima=True, **PEAK_PARAMS['fz_pos'])
    ))
    tx_peaks = np.concatenate((
        find_signal_peaks(s_tx, find_maxima=False, **PEAK_PARAMS['tx_neg']),
        find_signal_peaks(s_tx, find_maxima=True, **PEAK_PARAMS['tx_pos'])
    ))
    ty_peaks = np.concatenate((
        find_signal_peaks(s_ty, find_maxima=False, **PEAK_PARAMS['ty_neg']),
        find_signal_peaks(s_ty, find_maxima=True, **PEAK_PARAMS['ty_pos'])
    ))

    # 2. Combine all found peaks into one sorted list of unique candidates
    candidate_indices = np.unique(np.concatenate((fz_peaks, tx_peaks, ty_peaks)))
    candidate_indices.sort()
    
    if len(candidate_indices) == 0:
        print("No candidate peaks found in Fz, Tx, or Ty.")
        return None, None, None

    # 3. Merge peaks that are closer than the 'merge_distance'
    merge_dist = PEAK_PARAMS.get('merge_distance', 50)
    merged_indices = [candidate_indices[0]] # Always keep the first peak

    for current_peak_idx in candidate_indices[1:]:
        # If the current peak is far enough from the last accepted one, add it
        if current_peak_idx - merged_indices[-1] > merge_dist:
            merged_indices.append(current_peak_idx)

    final_peak_indices = np.array(merged_indices)
    # --- END OF NEW LOGIC ---

    print(f"Found {len(fz_peaks)} Fz peaks, {len(tx_peaks)} Tx peaks, {len(ty_peaks)} Ty peaks.")
    print(f"Combined and merged into {len(final_peak_indices)} unique events.")
    
    # Data extraction logic remains the same, using the final merged indices
    peak_ft_times = s_ft_time[final_peak_indices]
    ft_event_data = {'Fx': np.array([peak_ft_times, s_fx[final_peak_indices]]).T,'Fy': np.array([peak_ft_times, s_fy[final_peak_indices]]).T,
                     'Fz': np.array([peak_ft_times, s_fz[final_peak_indices]]).T,'Tx': np.array([peak_ft_times, s_tx[final_peak_indices]]).T,
                     'Ty': np.array([peak_ft_times, s_ty[final_peak_indices]]).T,'Tz': np.array([peak_ft_times, s_tz[final_peak_indices]]).T}
    
    indices_right = np.searchsorted(s_baro_time, peak_ft_times, side='left')
    indices_right = np.clip(indices_right, 0, len(s_baro_time) - 1)
    indices_left = np.clip(indices_right - 1, 0, len(s_baro_time) - 1)
    
    dist_left = np.abs(s_baro_time[indices_left] - peak_ft_times)
    dist_right = np.abs(s_baro_time[indices_right] - peak_ft_times)
    corresponding_baro_indices = np.where(dist_left < dist_right, indices_left, indices_right)
    
    peak_baro_times = s_baro_time[corresponding_baro_indices]
    baro_event_data = {}
    for i, b_data in enumerate(all_s_barometers):
        baro_event_data[f'Barometer {i+1}'] = np.array([peak_baro_times, b_data[corresponding_baro_indices]]).T
        
    return ft_event_data, baro_event_data, final_peak_indices

def plot_synchronized_data_with_events(synced_baro_data, baro_events):
    s_baro_time, *all_s_barometers = synced_baro_data
    fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle('Synchronized Barometer Data with Fz Peak Events', fontsize=16)
    for i in range(6):
        row, col = i // 2, i % 2
        baro_label = f'Barometer {i+1}'
        ax[row, col].plot(s_baro_time, all_s_barometers[i], label='Synced Data', zorder=1, alpha=0.8)
        if baro_label in baro_events:
            event_times, event_values = baro_events[baro_label][:, 0], baro_events[baro_label][:, 1]
            ax[row, col].plot(event_times, event_values, 'ro', markersize=6, label='Fz Peak Event', zorder=3)
        ax[row, col].set_title(baro_label)
        if row == 2: ax[row, col].set_xlabel('Synchronized Time (s)')
        if col == 0: ax[row, col].set_ylabel('Pressure (hPa)')
        ax[row, col].grid(True, linestyle='--', alpha=0.6); ax[row, col].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_sync_verification(baro_orig, ft_orig, baro_synced, ft_synced):
    """
    Creates a "Before & After" plot to visually verify data synchronization.
    """
    print("\nPlotting synchronization verification chart...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    fig.suptitle('Synchronization Verification', fontsize=16, y=0.98)

    # --- Subplot 1: Before Synchronization (Raw Datetime Index) ---
    ax1.set_title('Before Synchronization')
    color1 = 'tab:blue'
    ax1.set_ylabel('Barometer 1 (hPa)', color=color1)
    ax1.plot(baro_orig.index, baro_orig['barometer 1'], color=color1, label='Barometer 1')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.tick_params(axis='x', rotation=30)
    
    ax1b = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color2 = 'tab:red'
    ax1b.set_ylabel('Fz (N)', color=color2)
    ax1b.plot(ft_orig.index, ft_orig['Fz'], color=color2, alpha=0.7, label='Fz')
    ax1b.tick_params(axis='y', labelcolor=color2)
    ax1.grid(True, linestyle=':')
    ax1.set_xlabel('Raw Timestamp')

    # --- Subplot 2: After Synchronization (Elapsed Time) ---
    ax2.set_title('After Synchronization')
    ax2.set_ylabel('Barometer 1 (hPa)', color=color1)
    ax2.plot(baro_synced['elapsed_time'], baro_synced['barometer 1'], color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)

    ax2b = ax2.twinx()
    ax2b.set_ylabel('Fz (N)', color=color2)
    ax2b.plot(ft_synced['elapsed_time'], ft_synced['Fz'], color=color2, alpha=0.7)
    ax2b.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, linestyle=':')
    ax2.set_xlabel('Synchronized Elapsed Time (s)')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

if __name__ == "__main__":
    baro_read_result = read_barometer_csv(os.path.join(DATA_DIRECTORY_finger, BAROMETER_FILENAME))
    ft_read_result = read_forcetorque_csv(os.path.join(DATA_DIRECTORY_ATI, FORCETORQUE_FILENAME))

    if baro_read_result and ft_read_result:
        barometer_df_orig, *_ = baro_read_result
        forcetorque_df_orig, *_ = ft_read_result

        print(f"\nApplying a +{Time_shift_FT} second shift to the force/torque timestamps...")
        forcetorque_df_orig.index += pd.to_timedelta(Time_shift_FT, unit='s')

        # Plot original data before synchronization
        print("\nPlotting original data...")
        plot_barometer_data(barometer_df_orig)
        
        # Plot original force/torque data
        ft_time_elapsed = (forcetorque_df_orig.index - forcetorque_df_orig.index.min()).total_seconds().values
        ft_forces = (
            np.asarray(forcetorque_df_orig['Fx'].values), 
            np.asarray(forcetorque_df_orig['Fy'].values), 
            np.asarray(forcetorque_df_orig['Fz'].values)
        )
        ft_torques = (
            np.asarray(forcetorque_df_orig['Tx'].values), 
            np.asarray(forcetorque_df_orig['Ty'].values), 
            np.asarray(forcetorque_df_orig['Tz'].values)
        )
        # plot_forcetorque_data(ft_time_elapsed, ft_forces, ft_torques)

        synced_data = synchronize_data(barometer_df_orig, forcetorque_df_orig)
        
        if synced_data:
            synced_baro_df, synced_ft_df = synced_data
            
            plot_sync_verification(barometer_df_orig, forcetorque_df_orig, synced_baro_df, synced_ft_df)
            
            s_baro_time = synced_baro_df['elapsed_time'].to_numpy()
            s_b_all = [synced_baro_df[f'barometer {i}'].to_numpy() for i in range(1, 7)]
            s_ft_time = synced_ft_df['elapsed_time'].to_numpy()
            s_fx, s_fy, s_fz = synced_ft_df['Fx'].to_numpy(), synced_ft_df['Fy'].to_numpy(), synced_ft_df['Fz'].to_numpy()
            s_tx, s_ty, s_tz = synced_ft_df['Tx'].to_numpy(), synced_ft_df['Ty'].to_numpy(), synced_ft_df['Tz'].to_numpy()
            
            synced_baro_data = (s_baro_time, *s_b_all)
            synced_ft_data = (s_ft_time, s_fx, s_fy, s_fz, s_tx, s_ty, s_tz)
            
            analysis_results = extract_merged_events(synced_baro_data, synced_ft_data)

            if analysis_results and analysis_results[0] is not None:
                ft_events, baro_events, final_peak_indices = analysis_results

                plot_forcetorque_data(s_ft_time, (s_fx, s_fy, s_fz), (s_tx, s_ty, s_tz), final_peak_indices, is_synchronized=True)                        
                plot_synchronized_data_with_events(synced_baro_data, baro_events)

                data_for_csv = {
                    't': ft_events['Fz'][:, 0], 'fx': ft_events['Fx'][:, 1], 'fy': ft_events['Fy'][:, 1], 'fz': ft_events['Fz'][:, 1],
                    'tx': ft_events['Tx'][:, 1], 'ty': ft_events['Ty'][:, 1], 'tz': ft_events['Tz'][:, 1],
                    'b1': baro_events['Barometer 1'][:, 1], 'b2': baro_events['Barometer 2'][:, 1], 'b3': baro_events['Barometer 3'][:, 1],
                    'b4': baro_events['Barometer 4'][:, 1], 'b5': baro_events['Barometer 5'][:, 1], 'b6': baro_events['Barometer 6'][:, 1],
                }
                events_df = pd.DataFrame(data_for_csv)
                
                # Use the configuration variables to assign coordinates
                pattern_len = len(X_COORDINATES_PATTERN)
                num_events = len(events_df)
                
                # Apply the pattern cyclically to the detected events
                events_df['-x (mm)'] = X_COORDINATES_PATTERN * (num_events // pattern_len) + X_COORDINATES_PATTERN[:num_events % pattern_len]
                events_df['-y (mm)'] = Y_COORDINATE * (num_events // pattern_len) + X_COORDINATES_PATTERN[:num_events % pattern_len]
                
                new_column_order = [
                    't', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', '-x (mm)', '-y (mm)',
                    'fx', 'fy', 'fz', 'tx', 'ty', 'tz'
                ]
                events_df = events_df[new_column_order]

                SAVING_FOLDER = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\synchronized sensor and ATI"
                output_filepath = os.path.join(SAVING_FOLDER, f"synchronized_events_{n_experiment}.csv")
                events_df.to_csv(output_filepath, index=False, float_format='%.4f')
                print(f"Data saved to: {output_filepath}")

    plt.show()