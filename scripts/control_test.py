import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Constants & Gains ---
# Tuning these gains is critical for performance
K_RHO = 9.0    # Linear speed gain
K_ALPHA = 15.0 # Steering gain (must be > K_RHO for stability)
K_BETA = -10.0  # Orientation gain (must be < 0)

DT = 0.01       # Time step
SIM_TIME = 15.0 # Total simulation time

class DifferentialDriveRobot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.path_x = [x]
        self.path_y = [y]

    def update(self, v, omega, dt):
        """ Update robot state based on kinematic model """
        # Limit velocities for realism
        v = np.clip(v, -5.0, 5.0)
        omega = np.clip(omega, -5.0, 5.0)

        self.theta += omega * dt
        # Normalize theta to [-pi, pi]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        
        self.path_x.append(self.x)
        self.path_y.append(self.y)

def normalize_angle(angle):
    """ Normalize an angle to [-pi, pi] """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def run_control_loop(robot, goal_x, goal_y, goal_theta):
    """ Calculates control inputs (v, omega) to reach goal """
    
    # 1. Calculate errors in global frame
    dx = goal_x - robot.x
    dy = goal_y - robot.y
    
    # 2. Convert to Polar Coordinates (rho, alpha, beta)
    rho = np.sqrt(dx**2 + dy**2)
    alpha = normalize_angle(np.arctan2(dy, dx) - robot.theta)
    beta = normalize_angle(goal_theta - robot.theta - alpha)
    
    # 3. Apply Control Law
    # Logic to handle reverse driving if target is behind
    if abs(alpha) > np.pi / 2:
        alpha = normalize_angle(alpha - np.pi)
        beta = normalize_angle(beta - np.pi)
        v = -K_RHO * rho
    else:
        v = K_RHO * rho

    omega = K_ALPHA * alpha + K_BETA * beta

    # Stop if close enough
    if rho < 0.05:
        v = 0.0
        omega = 0.0

    return v, omega

# --- Main Simulation ---
robot_pos = (0.0, 0.0, -np.pi/2) # Start at origin facing East
robot = DifferentialDriveRobot(x=robot_pos[0], y=robot_pos[1], theta=robot_pos[2])
goal = (5.0, 5.0, np.pi/2) # Target: (5, 5) facing North

history_x = []
history_y = []

# Simulate
for t in np.arange(0, SIM_TIME, DT):
    v, omega = run_control_loop(robot, *goal)
    robot.update(v, omega, DT)
    history_x.append(robot.x)
    history_y.append(robot.y)

# --- Plotting ---
plt.figure(figsize=(8, 8))
plt.plot(history_x, history_y, '-b', label='Path')
plt.arrow(goal[0], goal[1], np.cos(goal[2]), np.sin(goal[2]), 
          head_width=0.3, color='r', label='Goal')
plt.arrow(history_x[0], history_y[0], np.cos(robot_pos[2]), np.sin(robot_pos[2]), 
          head_width=0.3, color='g', label='Start')
plt.grid(True)
plt.legend()
plt.title(f"Position Control: Start{robot_pos} -> Goal{goal}")
plt.axis('equal')
plt.show()