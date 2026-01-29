import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load tag1 (box) data
df = pd.read_csv(r'C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test 51000 - sensor v5\51000tag1_pose_trial.txt')

# Extract positions
px = df['field.pose.position.x'].values
py = df['field.pose.position.y'].values
pz = df['field.pose.position.z'].values

# Calculate step-to-step changes
dx = np.diff(px)
dy = np.diff(py)
dz = np.diff(pz)
step_dist = np.sqrt(dx**2 + dy**2 + dz**2)

print("Tag1 (Box) Movement Check:")
print(f"Total samples: {len(df)}")
print(f"\nPosition ranges:")
print(f"  X: {px.min()*1000:.2f} to {px.max()*1000:.2f} mm (span: {(px.max()-px.min())*1000:.2f} mm)")
print(f"  Y: {py.min()*1000:.2f} to {py.max()*1000:.2f} mm (span: {(py.max()-py.min())*1000:.2f} mm)")
print(f"  Z: {pz.min()*1000:.2f} to {pz.max()*1000:.2f} mm (span: {(pz.max()-pz.min())*1000:.2f} mm)")

print(f"\nStep distances (frame-to-frame movement):")
print(f"  Mean: {step_dist.mean()*1000:.3f} mm")
print(f"  Median: {np.median(step_dist)*1000:.3f} mm")
print(f"  Max: {step_dist.max()*1000:.3f} mm")
print(f"  Std: {step_dist.std()*1000:.3f} mm")

# Check for drift over time
print(f"\nDrift analysis (first to last position):")
total_drift = np.sqrt((px[-1]-px[0])**2 + (py[-1]-py[0])**2 + (pz[-1]-pz[0])**2)
print(f"  Total drift: {total_drift*1000:.2f} mm")
print(f"  X drift: {(px[-1]-px[0])*1000:.2f} mm")
print(f"  Y drift: {(py[-1]-py[0])*1000:.2f} mm")
print(f"  Z drift: {(pz[-1]-pz[0])*1000:.2f} mm")

# Check orientation stability
qx = df['field.pose.orientation.x'].values
qy = df['field.pose.orientation.y'].values
qz = df['field.pose.orientation.z'].values
qw = df['field.pose.orientation.w'].values

# Calculate rotation angle changes between consecutive frames
q_dot = qx[:-1]*qx[1:] + qy[:-1]*qy[1:] + qz[:-1]*qz[1:] + qw[:-1]*qw[1:]
q_dot = np.clip(q_dot, -1, 1)  # Numerical safety
angle_change = 2 * np.arccos(np.abs(q_dot)) * 180/np.pi

print(f"\nOrientation stability:")
print(f"  Mean angle change: {angle_change.mean():.3f} deg")
print(f"  Max angle change: {angle_change.max():.3f} deg")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Position over time
axes[0,0].plot(px*1000, label='X', linewidth=0.8)
axes[0,0].plot(py*1000, label='Y', linewidth=0.8)
axes[0,0].plot(pz*1000, label='Z', linewidth=0.8)
axes[0,0].set_xlabel('Sample index')
axes[0,0].set_ylabel('Position [mm]')
axes[0,0].set_title('Tag1 (Box) Position vs Time')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()

# XY trajectory
axes[0,1].plot(px*1000, py*1000, 'b-', linewidth=0.8)
axes[0,1].scatter(px[0]*1000, py[0]*1000, c='g', s=100, marker='o', label='Start', zorder=5)
axes[0,1].scatter(px[-1]*1000, py[-1]*1000, c='r', s=100, marker='x', label='End', zorder=5)
axes[0,1].set_xlabel('X [mm]')
axes[0,1].set_ylabel('Y [mm]')
axes[0,1].set_title('Tag1 (Box) XY Movement')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].axis('equal')
axes[0,1].legend()

# Step distances
axes[1,0].plot(step_dist*1000, 'b-', linewidth=0.5)
axes[1,0].set_xlabel('Sample index')
axes[1,0].set_ylabel('Step distance [mm]')
axes[1,0].set_title('Frame-to-Frame Movement')
axes[1,0].grid(True, alpha=0.3)

# Angle changes
axes[1,1].plot(angle_change, 'r-', linewidth=0.5)
axes[1,1].set_xlabel('Sample index')
axes[1,1].set_ylabel('Angle change [deg]')
axes[1,1].set_title('Frame-to-Frame Rotation')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
