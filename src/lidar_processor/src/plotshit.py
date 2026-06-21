import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import numpy as np

def plot_covariance_ellipse(pos, cov, n_std=1.0, ax=None, **kwargs):
    """
    Plots an ellipse to represent the covariance bound.
    """
    if ax is None:
        ax = plt.gca()

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 3 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

fig, ax = plt.subplots()

def animate(i):
    try:
        data = np.loadtxt("fuck_gemini.csv", delimiter=",")
        data = np.atleast_2d(data)
        
        if data.size == 0 or data.shape[1] < 15:
            return
            
    except (OSError, ValueError, IndexError):
        return

    ax.cla()

    # --- 1. Extract Robot State ---
    pos_x, pos_y = data[0, 6], data[0, 7]
    yaw = data[0, 11]
    
    # Twist velocity is typically body-relative (x is forward). 
    # We will rotate it to the world frame for plotting.
    vel_body = np.array([data[0, 12], data[0, 13]])

    # --- 2. Create 2D Rotation Matrix ---
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [c, -s],
        [s,  c]
    ])

    # --- 3. Transform Obstacles to Inertial/World Frame ---
    # Center transformation: P_world = R * P_body + T
    centers_body = data[:, 0:2]
    centers_world = (R @ centers_body.T).T + np.array([pos_x, pos_y])

    # Covariance transformation: Cov_world = R * Cov_body * R^T
    covs_body = np.reshape(data[:, 2:6], (-1, 2, 2))
    covs_world = np.zeros_like(covs_body)
    for j in range(len(covs_body)):
        covs_world[j] = R @ covs_body[j] @ R.T

    # --- 4. Plot Obstacles (Now in World Frame) ---
    ax.scatter(centers_world[:, 0], centers_world[:, 1], c='blue', zorder=2, label="Obstacle Centers")

    for j in range(len(covs_world)):
        label = "Covariance" if j == 0 else ""
        plot_covariance_ellipse(centers_world[j], covs_world[j], ax=ax, edgecolor='red', facecolor='none', label=label)

    # --- 5. Plot Robot State ---
    ax.scatter(pos_x, pos_y, c='green', marker='X', s=100, zorder=3, label="Robot Position")

    # Orientation (Yaw is already world-relative)
    ax.quiver(pos_x, pos_y, c, s, color='green', angles='xy', scale_units='xy', scale=1, width=0.008, label="Orientation")

    # Velocity (Rotated into world frame)
    vel_world = R @ vel_body
    ax.quiver(pos_x, pos_y, vel_world[0], vel_world[1], color='purple', angles='xy', scale_units='xy', scale=1, width=0.008, label="Velocity")

    # --- 6. Formatting ---
    # Dynamically track the robot
    window_size = 6
    ax.set_xlim([pos_x - window_size, pos_x + window_size])
    ax.set_ylim([pos_y - window_size, pos_y + window_size])
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05))

ani = FuncAnimation(fig, animate, interval=250, cache_frame_data=False)
plt.tight_layout()
plt.show()