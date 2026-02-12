import pandas as pd
import numpy as np

df = pd.read_csv('realtime_data.csv')

print('Data shape:', df.shape)
print('\n=== Barometer Statistics ===')
print(df[['b1','b2','b3','b4','b5','b6']].describe())

print('\n=== Force Statistics ===')
print(f"Fz range: {df['fz_pred'].min():.3f} to {df['fz_pred'].max():.3f}")

# Find contact periods
contact_mask = df['fz_pred'] < -1.0
print(f'\nSamples in contact (fz<-1): {contact_mask.sum()} / {len(df)}')

if contact_mask.sum() > 0:
    contact_start = df[contact_mask].index[0]
    contact_end = df[contact_mask].index[-1]
    
    print(f'\n=== Contact Event Analysis ===')
    print(f"Contact starts at index: {contact_start}")
    print(f"Contact ends at index: {contact_end}")
    
    # Before contact (50 samples before)
    before_start = max(0, contact_start - 50)
    before_data = df.iloc[before_start:contact_start][['b1','b2','b3','b4','b5','b6']]
    
    # During contact (first 50 samples)
    during_data = df.iloc[contact_start:min(contact_start+50, contact_end)][['b1','b2','b3','b4','b5','b6']]
    
    # After contact (50 samples after)
    after_end = min(len(df), contact_end + 50)
    after_data = df.iloc[contact_end:after_end][['b1','b2','b3','b4','b5','b6']]
    
    print('\nMean barometer values:')
    print(f"Before contact: {before_data.mean().values}")
    print(f"During contact: {during_data.mean().values}")
    print(f"After contact:  {after_data.mean().values}")
    print(f"\nDrift (after - before): {(after_data.mean() - before_data.mean()).values}")
