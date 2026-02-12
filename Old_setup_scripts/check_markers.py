import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test 51000 - sensor v5\atracsys_trial51000.txt')

for mid in [1, 2]:
    sub = df[df['marker_id'] == mid]
    pos = sub[['field.position0', 'field.position1', 'field.position2']].values
    
    print(f'\nMarker {mid}:')
    print(f'  X range: {pos[:,0].min():.2f} to {pos[:,0].max():.2f} mm (span: {pos[:,0].max()-pos[:,0].min():.2f})')
    print(f'  Y range: {pos[:,1].min():.2f} to {pos[:,1].max():.2f} mm (span: {pos[:,1].max()-pos[:,1].min():.2f})')
    print(f'  Z range: {pos[:,2].min():.2f} to {pos[:,2].max():.2f} mm (span: {pos[:,2].max()-pos[:,2].min():.2f})')
    print(f'  Total variance: {np.var(pos, axis=0).sum():.4f}')
