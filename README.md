# Mobile Robotics Project: Autonomous Navigation and Control

## Overview
This project was developed as part of the **Basics of Mobile Robotics (MICRO-452)** course at EPFL, under the supervision of Professor Francesco Mondada from the [MOBOTS Laboratory](https://www.epfl.ch/labs/mobots/).

## Objectives
This project aims to develop an autonomous navigation system for the Thymio robot that can:

- Navigate through a predefined environment with static obstacles using **global navigation** (A* path planning)
- Dynamically avoid unexpected obstacles using **local navigation** (Braitenberg-inspired controller)
- Maintain accurate position estimation through **Extended Kalman Filtering**
- Reliably reach arbitrary target positions while handling **kidnapping scenarios**
- Perform real-time **visual localization** using ArUco markers

## Hardware
| Peripheral | Model |
|-----------|-------|
| Robot | Thymio II (MOBSYA) |
| Camera | Overhead webcam for bird's-eye view |

## Team Members
**Group T41** - Master's students in Robotics and Electrical Engineering at EPFL:

- **Linus Crugnola** - Filtering (Extended Kalman Filter)
- **Alexis Bollack** - Local & Global Navigation
- **Maxime Luyet** - Vision Processing

## Demonstration
Below is a demonstration of our autonomous navigation system in action:

<p align="center">
  <video width="800" controls>
    <source src="img/Mobile_robotic_project_T41_demo.mp4" type="video/mp4">
  </video>
  <br>
  <em>Thymio robot autonomously navigating from start to goal position with obstacle avoidance</em>
</p>

**Key Features Demonstrated:**
- **Yellow Point**: Current robot position (EKF-filtered)
- **Pink Point**: Projection onto planned path
- **Orange Point**: Lookahead target point (~10 cm ahead)
- **Gray Shadows**: Cost map with safety margins around obstacles

## Project Structure
```
Mobile-robotic/
├── code/                      # Source code modules
│   ├── filtering.py           # Extended Kalman Filter implementation
│   ├── global_navigation.py   # A* path planning and smoothing
│   ├── local_navigation.py    # Braitenberg obstacle avoidance
│   └── vision.py              # ArUco detection and occupancy grid
├── scripts/                   # Utility scripts
│   ├── aruco_markers/         # ArUco marker generation
│   └── cam_calibration/       # Camera calibration tools
├── img/                       # Images and videos
├── Group_T41_report.ipynb     # Complete project report (Jupyter)
├── REPORT.md                  # Technical documentation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup Guide

### Prerequisites
- **Anaconda** or **Miniconda** installed on your system
- **Git** (for cloning the repository)
- **VSCode** with Jupyter extension (recommended for best experience)

### Installation Steps

#### 1. Environment Setup
Create and activate a new Conda environment with Python 3.11+:

```bash
# Create new environment
conda create --name mobile-robotics python=3.11

# Activate the environment
conda activate mobile-robotics
```

#### 2. Project Setup
```bash
# Clone the repository
git clone https://github.com/Emixm4/Mobile-robotic.git

# Navigate to the project directory
cd Mobile-robotic

# Install dependencies
pip install -r requirements.txt
```

#### 3. Camera Calibration (Optional)
If using a different camera, perform calibration:

```bash
cd scripts/cam_calibration
python calibration.py
```

Follow the on-screen instructions to capture chessboard images and generate calibration parameters.

#### 4. Open and Run the Notebook

**Option 1 (Recommended):**
1. Open the project in VSCode: `code .`
2. Open `Group_T41_report.ipynb`
3. Select the `mobile-robotics` kernel
4. Follow the step-by-step instructions in the notebook
5. Execute cells in sequential order as indicated

**Option 2:**
1. Launch Jupyter Notebook: `jupyter notebook`
2. Open `Group_T41_report.ipynb`
3. Note: LaTeX formatting may not render correctly in classic Jupyter
4. For better LaTeX rendering, use **VSCode** or **JupyterLab**



## License
This project is developed for educational purposes as part of the MICRO-452 course at EPFL.

## Acknowledgments
Special thanks to:
- Professor Francesco Mondada and the MOBOTS Laboratory team
- EPFL's Basics of Mobile Robotics course staff
- The Thymio robotics platform development team

---

**Course**: MICRO-452 - Basics of Mobile Robotics  
**Institution**: École Polytechnique Fédérale de Lausanne (EPFL)  
**Academic Year**: 2024-2025
