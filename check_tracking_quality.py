import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load tag2 (probe) data
df = pd.read_csv(r'C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test 51000 - sensor v5\51000tag2_pose_trial.txt')

# Extract positions
px = df['field.pose.position.x'].values
py = df['field.pose.position.y'].values
pz = df['field.pose.position.z'].values

# Calculate step-to-step changes
dx = np.diff(px)
dy = np.diff(py)
dz = np.diff(pz)
step_dist = np.sqrt(dx**2 + dy**2 + dz**2)

print("Tag2 (Probe) Tracking Quality Check:")
print(f"Total samples: {len(df)}")
print(f"\nStep distances (frame-to-frame movement):")
print(f"  Mean: {step_dist.mean()*1000:.3f} mm")
print(f"  Median: {np.median(step_dist)*1000:.3f} mm")
print(f"  Max: {step_dist.max()*1000:.3f} mm")
print(f"  95th percentile: {np.percentile(step_dist, 95)*1000:.3f} mm")

# Find large jumps (>5mm between consecutive frames)
large_jumps = np.where(step_dist > 0.005)[0]
print(f"\nLarge jumps (>5mm): {len(large_jumps)} out of {len(step_dist)} steps")
if len(large_jumps) > 0:
    print(f"  Jump indices: {large_jumps[:10]}..." if len(large_jumps) > 10 else f"  Jump indices: {large_jumps}")
    print(f"  Jump sizes: {step_dist[large_jumps[:5]]*1000} mm (first 5)")

# Check quaternion consistency
qx = df['field.pose.orientation.x'].values
qy = df['field.pose.orientation.y'].values
qz = df['field.pose.orientation.z'].values
qw = df['field.pose.orientation.w'].values

# Quaternion magnitude (should be 1.0)
qmag = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
print(f"\nQuaternion magnitude check:")
print(f"  Mean: {qmag.mean():.10f}")
print(f"  Std: {qmag.std():.2e}")
print(f"  Min: {qmag.min():.10f}")
print(f"  Max: {qmag.max():.10f}")

# Plot trajectory
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# XY projection
axes[0,0].plot(px*1000, py*1000, 'b-', linewidth=0.5, alpha=0.7)
axes[0,0].scatter(px[0]*1000, py[0]*1000, c='g', s=100, marker='o', label='Start', zorder=5)
axes[0,0].scatter(px[-1]*1000, py[-1]*1000, c='r', s=100, marker='x', label='End', zorder=5)
axes[0,0].set_xlabel('X [mm]')
axes[0,0].set_ylabel('Y [mm]')
axes[0,0].set_title('Tag2 (Probe) XY Trajectory')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()
axes[0,0].axis('equal')

# XZ projection
axes[0,1].plot(px*1000, pz*1000, 'b-', linewidth=0.5, alpha=0.7)
axes[0,1].scatter(px[0]*1000, pz[0]*1000, c='g', s=100, marker='o', label='Start', zorder=5)
axes[0,1].scatter(px[-1]*1000, pz[-1]*1000, c='r', s=100, marker='x', label='End', zorder=5)
axes[0,1].set_xlabel('X [mm]')
axes[0,1].set_ylabel('Z [mm]')
axes[0,1].set_title('Tag2 (Probe) XZ Trajectory')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend()

# Step distance over time
axes[1,0].plot(step_dist*1000, 'b-', linewidth=0.5)
axes[1,0].axhline(5, color='r', linestyle='--', label='5mm threshold')
axes[1,0].set_xlabel('Sample index')
axes[1,0].set_ylabel('Step distance [mm]')
axes[1,0].set_title('Frame-to-Frame Movement')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].legend()

# Position over time
axes[1,1].plot(px*1000, label='X', linewidth=0.5)
axes[1,1].plot(py*1000, label='Y', linewidth=0.5)
axes[1,1].plot(pz*1000, label='Z', linewidth=0.5)
axes[1,1].set_xlabel('Sample index')
axes[1,1].set_ylabel('Position [mm]')
axes[1,1].set_title('Position vs Time')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend()

plt.tight_layout()
plt.show()
