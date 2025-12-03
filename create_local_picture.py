import numpy as np
import matplotlib.pyplot as plt

W= [
            np.array([10, -20]) / np.linalg.norm([10, -20]),  # sensor x1
            np.array([5, -20]) / np.linalg.norm([5, -20]),    # sensor x2
            np.array([0, -10]) / np.linalg.norm([0, -10]),    # sensor x3
            np.array([-5, -20]) / np.linalg.norm([-5, -20]),  # sensor x4
            np.array([-10, -20]) / np.linalg.norm([-10, -20]),# sensor x5
            np.array([15, 15]) / np.linalg.norm([15, 15]),    # sensor x6
            np.array([-15, 15]) / np.linalg.norm([-15, 15]),  # sensor x7
        ]

# Create 2D plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
X = [W[i][0] for i in range(len(W))]
Y = [W[i][1] for i in range(len(W))]
plt.plot(X, Y, 'o')
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')

ax.add_artist(circle)
# Add text labels to show the corresponding sensor
for i, (x, y) in enumerate(zip(X, Y)):
    ax.text(x * 1.1, y * 1.1, f'Sensor {i+1}', fontsize=10, ha='center', va='center')
plt.grid(True)
plt.savefig('sensor_weights.png')

