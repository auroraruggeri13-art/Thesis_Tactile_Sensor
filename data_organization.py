import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Close any plots from previous runs
plt.close('all')

# --- CONFIGURATION ---
DATA_DIRECTORY_finger = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Sensor-Logs"
DATA_DIRECTORY_ATI = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\ATI-FT-Data-Logs"
BAROMETER_FILENAME = "datalog_2025-10-10_10-07-53.csv"
FORCETORQUE_FILENAME = "2025-10-10_10-08-00_FT29602.csv"
n_experiment = 40
X_COORDINATES_PATTERN = [20]
Y_COORDINATE = [10]
Time_shift_FT = 0.13  # Time shift in seconds to apply to Force/Torque data

# --- DATA PROCESSING FUNCTIONS ---

def read_barometer_csv(filepath):
    """Reads the barometer CSV and removes any duplicate timestamps."""
    print(f"Reading barometer file: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath)
        df.columns = [col.strip() for col in df.columns]
        file_date_str = os.path.basename(filepath).split('_')[1]
        
        df['datetime'] = pd.to_datetime(file_date_str + ' ' + df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        
        df.dropna(subset=['datetime'], inplace=True)
        df.set_index('datetime', inplace=True)
        
        # **CRITICAL FIX**: Remove rows with duplicate timestamps, keeping the first one
        df = df[~df.index.duplicated(keep='first')]
        
        print(f"Barometer timestamps (unique): {df.index.min()} to {df.index.max()} ({len(df)} samples)")
        return df
    except Exception as e:
        print(f"Error reading barometer file: {e}")
        return None

def read_forcetorque_csv(filepath):
    """Reads the force/torque CSV file, skipping header and parsing timestamps."""
    print(f"Reading force/torque file: {os.path.basename(filepath)}")
    header_row_index = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if 'Status' in line and 'Fx' in line and 'Time' in line:
                    header_row_index = i
                    break
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}"); return None
        
    if header_row_index is None:
        print("Error: Could not find a valid header row in force/torque file."); return None
        
    try:
        expected_columns = ['Status (hex)', 'RDTSequence', 'F/T Sequence', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'Time']
        df = pd.read_csv(filepath, skiprows=header_row_index, sep=',', engine='python', 
                         names=expected_columns, usecols=range(10), skipinitialspace=True, na_filter=False)
        
        df = df[df['Time'].astype(str).str.strip().ne('')].copy()
        df['Time'] = df['Time'].astype(str).str.strip()
        file_date_str = os.path.basename(filepath).split('_')[0]
        
        df['datetime'] = pd.to_datetime(file_date_str + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        
        # Convert force/torque columns to numeric before duplicate handling
        ft_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        for col in ft_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where any force/torque measurement is NA
        df.dropna(subset=ft_cols, inplace=True)
        
        # Handle duplicate timestamps
        df.set_index('datetime', inplace=True)
        if df.index.duplicated().any():
            print("Note: Found duplicate timestamps in force/torque data. Averaging numeric values...")
            # Average only the numeric columns (forces and torques)
            ft_means = df[ft_cols].groupby(level=0).mean()
            
            # For non-numeric columns, keep the first occurrence
            non_ft_cols = [col for col in df.columns if col not in ft_cols]
            non_ft_first = df[non_ft_cols].groupby(level=0).first()
            
            # Combine the averaged numeric data with the first occurrence of non-numeric data
            df = pd.concat([ft_means, non_ft_first], axis=1)
            df = df.reindex(columns=ft_cols + non_ft_cols)  # Restore original column order
            
        print(f"Force/Torque timestamps: {df.index.min()} to {df.index.max()} ({len(df)} samples)")
        
        return df
    except Exception as e:
        print(f"Error parsing force/torque file: {e}"); return None

def find_active_barometer_range(barometer_df, window_size=50, std_threshold=0.01):
    """Identifies the time range where barometer signals show activity (are not flat)."""
    print("Finding active barometer range...")
    barometer_cols = [f'barometer {i}' for i in range(1, 7)]
    rolling_std = barometer_df[barometer_cols].rolling(window=window_size, min_periods=1).std()
    
    is_active = (rolling_std > std_threshold).any(axis=1)
    active_indices = is_active[is_active].index
    
    if active_indices.empty:
        print("Warning: No active barometer range found. Using the full time range.")
        return barometer_df.index.min(), barometer_df.index.max()
        
    start_time, end_time = active_indices.min(), active_indices.max()
    print(f"  -> Active range identified: {start_time} to {end_time}")
    return start_time, end_time

def plot_forces_and_torques(df):
    """Creates a plot with subplots for all forces and torques."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle('Synchronized Forces and Torques', fontsize=16)
    
    force_labels = ['Fx', 'Fy', 'Fz']
    torque_labels = ['Tx', 'Ty', 'Tz']
    time_data = df['elapsed_time']

    # Plot Forces on the left column
    for i in range(3):
        axes[i, 0].plot(time_data, df[force_labels[i]], color='tab:blue')
        axes[i, 0].set_ylabel(f'{force_labels[i]} (N)')
        axes[i, 0].grid(True, linestyle=':')
    axes[2, 0].set_xlabel('Elapsed Time (s)')
    
    # Plot Torques on the right column
    for i in range(3):
        axes[i, 1].plot(time_data, df[torque_labels[i]], color='tab:orange')
        axes[i, 1].set_ylabel(f'{torque_labels[i]} (Nmm)')
        axes[i, 1].grid(True, linestyle=':')
    axes[2, 1].set_xlabel('Elapsed Time (s)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_pressures(df):
    """Creates a plot with subplots for all 6 interpolated pressures."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle('Synchronized and Interpolated Pressures', fontsize=16)
    
    time_data = df['elapsed_time']

    for i in range(6):
        row, col = i // 2, i % 2
        baro_label = f'barometer {i+1}'
        axes[row, col].plot(time_data, df[baro_label])
        axes[row, col].set_title(baro_label)
        axes[row, col].grid(True, linestyle=':')
        
        if col == 0:
            axes[row, col].set_ylabel('Pressure (hPa)')
        if row == 2:
            axes[row, col].set_xlabel('Elapsed Time (s)')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_alignment_check(df):
    """Creates a plot showing barometer 3 and Fz on the same plot to check time alignment."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot barometer 3 on left y-axis
    time_data = df['elapsed_time']
    line1 = ax1.plot(time_data, df['barometer 3'], 'b-', label='Barometer 3')
    ax1.set_xlabel('Elapsed Time (s)')
    ax1.set_ylabel('Pressure (hPa)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot Fz on right y-axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(time_data, df['Fz'], 'r-', label='Fz')
    ax2.set_ylabel('Force Fz (N)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('Time Alignment Check: Barometer 3 vs Fz')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Step 1: Read raw data from CSV files
    barometer_df_orig = read_barometer_csv(os.path.join(DATA_DIRECTORY_finger, BAROMETER_FILENAME))
    forcetorque_df_orig = read_forcetorque_csv(os.path.join(DATA_DIRECTORY_ATI, FORCETORQUE_FILENAME))

    if barometer_df_orig is not None and forcetorque_df_orig is not None:
        
        # Step 2: Apply manual time shift to align data
        print(f"\nApplying a +{Time_shift_FT} second shift to the force/torque timestamps...")
        forcetorque_df_orig.index += pd.to_timedelta(Time_shift_FT, unit='s')

        # Step 3: Find the active time range from barometer signals
        active_start, active_end = find_active_barometer_range(barometer_df_orig, std_threshold=0.01)
        
        # Step 4: Trim both datasets to this common active range
        buffer = pd.Timedelta(seconds=0.1) # Add buffer to avoid interpolation errors at edges
        baro_active = barometer_df_orig.loc[active_start - buffer : active_end + buffer]
        ft_active = forcetorque_df_orig.loc[active_start - buffer : active_end + buffer]
        
        if ft_active.empty or baro_active.empty:
            print("Synchronization failed: No data found in the determined active range.")
        else:
            print(f"\nData trimmed to active range. FT samples: {len(ft_active)}, Baro samples: {len(baro_active)}")
            
            # Step 5: Interpolate barometer data to exactly match force/torque timestamps
            print("Interpolating barometer data to match all force/torque timestamps...")
            print(f"Force/Torque data points: {len(ft_active)}")
            print(f"Original barometer points: {len(baro_active)}")
            
            baro_cols = [f'barometer {i}' for i in range(1, 7)]
            
            # Ensure barometer data is numeric and sorted by time
            for col in baro_cols:
                baro_active[col] = pd.to_numeric(baro_active[col], errors='coerce')
            baro_active = baro_active.sort_index()
            
            # Get precise timestamps in nanoseconds to maintain exact F/T temporal resolution
            baro_times = baro_active.index.astype(np.int64)
            ft_times = ft_active.index.astype(np.int64)
            
            # Create DataFrame with exact F/T timestamps
            baro_interpolated = pd.DataFrame(index=ft_active.index)
            
            # Interpolate each barometer column to match every F/T timestamp using simple linear interpolation
            for col in baro_cols:
                # Linear interpolation guarantees values stay within the original range
                baro_interpolated[col] = np.interp(
                    ft_times,  # x-coordinates of the interpolated values
                    baro_times,  # x-coordinates of the data points
                    baro_active[col].values  # y-coordinates of the data points
                )
            
            # Verify interpolation maintained exact number of points
            print(f"\nInterpolated barometer points: {len(baro_interpolated)} (should match F/T points: {len(ft_active)})")
            if len(baro_interpolated) != len(ft_active):
                raise ValueError("Interpolation failed to maintain exact number of force/torque points")

            # Step 6: Combine into a single synchronized DataFrame
            synced_df = ft_active.copy()
            barometer_cols = [f'barometer {i}' for i in range(1, 7)]
            for col in barometer_cols:
                synced_df[col] = baro_interpolated[col]
            synced_df.dropna(inplace=True) # Drop rows that couldn't be interpolated (e.g., edges)
            synced_df['elapsed_time'] = (synced_df.index - synced_df.index.min()).total_seconds()
            print(f"Created final synchronized dataframe with {len(synced_df)} samples.")
            
            # Step 7: PLOT THE SYNCHRONIZED DATA
            plot_forces_and_torques(synced_df)
            plot_pressures(synced_df)
            plot_alignment_check(synced_df)  # Add alignment check plot

            # Step 8: Prepare the final data for export to CSV
            final_df = synced_df.copy()
            
            # Rename columns for the final CSV file
            rename_dict = {f'barometer {i}': f'b{i}' for i in range(1, 7)}
            rename_dict.update({'Fx':'fx', 'Fy':'fy', 'Fz':'fz', 'Tx':'tx', 'Ty':'ty', 'Tz':'tz'})
            final_df.rename(columns=rename_dict, inplace=True)
            final_df.rename(columns={'elapsed_time': 't'}, inplace=True)

            # Assign X and Y coordinates cyclically to every row
            num_rows = len(final_df)
            final_df['-x (mm)'] = X_COORDINATES_PATTERN * (num_rows // len(X_COORDINATES_PATTERN)) + X_COORDINATES_PATTERN[:num_rows % len(X_COORDINATES_PATTERN)]
            final_df['-y (mm)'] = Y_COORDINATE * (num_rows // len(Y_COORDINATE)) + Y_COORDINATE[:num_rows % len(Y_COORDINATE)]

            # Define and apply final column order
            final_column_order = [
                't', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', '-x (mm)', '-y (mm)',
                'fx', 'fy', 'fz', 'tx', 'ty', 'tz'
            ]
            final_df = final_df[final_column_order]

            # Step 9: Save the processed data
            SAVING_FOLDER = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\synchronized sensor and ATI"
            os.makedirs(SAVING_FOLDER, exist_ok=True) # Ensure the save directory exists
            output_filename = f"synchronized_events_{n_experiment}.csv"
            output_filepath = os.path.join(SAVING_FOLDER, output_filename)
            
            final_df.to_csv(output_filepath, index=False, float_format='%.4f')
            print(f"\n✅ Successfully saved continuous synchronized data to: {output_filepath}")

            # Step 10: Display all generated plots
            plt.show()