
import matplotlib
matplotlib.use('TkAgg')  # Using the robust Qt backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 1. CONFIGURATION
# ==========================================
APRILTAG_CSV = '../RotationTest_raw.csv'
IMU_CSV = '../mekf_log.csv'

# Set to True to multiply AprilTag Yaw by -1 to match IMU
FLIP_APRILTAG_YAW = True 

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading AprilTag data...")
at_df = pd.read_csv(APRILTAG_CSV)
at_df = at_df[at_df['pose_valid'] == 1]

# Zero out AprilTag time so it starts exactly at 0.0
t_at = at_df['time_s'].values
t_at = t_at - t_at[0]

roll_at = at_df['roll_deg'].values
pitch_at = at_df['pitch_deg'].values
yaw_at = at_df['yaw_deg'].values * (-1.0 if FLIP_APRILTAG_YAW else 1.0)

print("Loading IMU data...")
imu_df = pd.read_csv(IMU_CSV)

t_imu = imu_df['time'].values
t_imu = t_imu - t_imu[0]

# =======================================================
# AXIS MAPPING FIX
# Swap 'roll' and 'pitch' inside the brackets below to fix the mismatch.
# Multiply by -1.0 if the graph is moving in the exact opposite direction.
# =======================================================
roll_imu  = np.degrees(imu_df['pitch'].values) * 1.0   # <-- Swapped with pitch
pitch_imu = np.degrees(imu_df['roll'].values) * 1.0    # <-- Swapped with roll
yaw_imu   = np.degrees(imu_df['yaw'].values) * 1.0

# Zero out the massive IMU timestamp so it also starts at 0.0
t_imu = imu_df['time'].values
t_imu = t_imu - t_imu[0]

# ==========================================
# 3. SETUP PLOT
# ==========================================
fig, (ax_r, ax_p, ax_y) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(bottom=0.2) 

# Plot IMU as the fixed blue reference
ax_r.plot(t_imu, roll_imu, label='IMU (deg)', color='blue', alpha=0.7)
ax_p.plot(t_imu, pitch_imu, label='IMU (deg)', color='blue', alpha=0.7)
ax_y.plot(t_imu, yaw_imu, label='IMU (deg)', color='blue', alpha=0.7)

# Plot AprilTag as the shiftable red line
line_r, = ax_r.plot(t_at, roll_at, label='AprilTag (deg)', color='red', alpha=0.8, linewidth=1.5)
line_p, = ax_p.plot(t_at, pitch_at, label='AprilTag (deg)', color='red', alpha=0.8, linewidth=1.5)
line_y, = ax_y.plot(t_at, yaw_at, label='AprilTag (deg)', color='red', alpha=0.8, linewidth=1.5)

# Formatting
ax_r.set_ylabel('Roll (deg)')
ax_p.set_ylabel('Pitch (deg)')
ax_y.set_ylabel('Yaw (deg)')
ax_y.set_xlabel('Normalized Time (s)')

for ax in [ax_r, ax_p, ax_y]:
    ax.legend(loc='upper right')
    ax.grid(True)

# ==========================================
# 4. INTERACTIVE SLIDER
# ==========================================
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
time_slider = Slider(
    ax=ax_slider,
    label='Shift AprilTag Time (s)',
    valmin=-60.0,  # Expanded range to +/- 60 seconds just in case
    valmax=60.0,
    valinit=0.0,
    valstep=0.01 
)

def update(val):
    offset = time_slider.val
    line_r.set_xdata(t_at + offset)
    line_p.set_xdata(t_at + offset)
    line_y.set_xdata(t_at + offset)
    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.suptitle("Interactive Sensor Alignment: Drag slider to match 'Start of Movement'")
plt.show()

print(f"Final Time Offset Selected: {time_slider.val:.3f} seconds")
