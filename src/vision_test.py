import numpy as np
import cv2
import os
import sys
import utils 

# Global constants 

CAMERA_MATRIX = np.array([[1484.14935, 0, 964.220337],
                       [0, 1490.50225, 539.531044],
                       [0, 0, 1]])

CAMERA_COEFFICIENTS = np.array([0.154056967, -1.04865202, -0.000689137957, -0.00132248869, 1.48905183])

CAMERA_RESOLUTION = (1920, 1080)

# ArUco marker IDs configuration
# Generated markers: 23, 30, 42, 53, 69, 78
MARKER_CORNERS = [42, 30, 69, 23]  # IDs for the 4 corners: [TL, TR, BL, BR]
MARKER_ROBOT = 78                  # ID for the robot marker
MARKER_GOAL = 53                   # ID for the goal marker
ARUCO_SIZE_CM = 5.0                # Real size of ArUco markers in cm (5cm x 5cm at 300 DPI)

class Vision():
    def __init__(self, cam_id):
        #initialize camera ID and capture object
        self.CAM_ID = cam_id
        self.cap = None

        # Initialize ArUco dictionary and detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.param = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.param)
        
        self.world_size = None
        self.aruko_position = None 
        
        # Map configuration (will be calculated after detecting corners)
        self.perspective_matrix = None
        self.pixels_per_cm = None
        self.map_size_pixels = None        
        self.map_size_cm = None
    
    def _undistort_image(self, map):
        """
        Undistort image using camera calibration parameters.
        Maintains original resolution (1920x1080) without cropping.
        
        Args:
            map: Input image to undistort
            
        Returns:
            numpy.ndarray: Undistorted image with same resolution as input
        """
        # Calculate optimal camera matrix for undistortion
        h, w = map.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, CAMERA_COEFFICIENTS, (w, h), 1, (w, h))
        
        # Apply undistortion using calibration parameters
        frame = cv2.undistort(map, CAMERA_MATRIX, CAMERA_COEFFICIENTS, None, newcameramtx)
        
        # Crop to valid pixel region after undistortion
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
        return frame
    
    # ============================================
    # 0. INITIALIZATION METHODS
    # ============================================
    
    def webcam_connextion(self):
        """
        Connect to the webcam and verify the connection status.
    
        Returns:
            bool: True if connection is successful, False otherwise
        """
        # Attempt to open the webcam
        self.cap = cv2.VideoCapture(self.CAM_ID)
        
        # Check if the webcam is opened
        if not self.cap.isOpened():
            print(f"Error: Unable to connect to webcam with ID {self.CAM_ID}")
            self.cap = None
            return False
        
        # Configure the defined resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
        
        print(f"Webcam connected successfully!")      
        return True
                
    def disconnect_webcam(self):
        """
        Disconnect the webcam and release resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Webcam deconnected")
    
    # ============================================
    # 1. FRAME EXTRACTION & PERSPECTIVE CORRECTION
    # ============================================
    
    def _get_frame(self):
        """
        Capture a single frame from the webcam and apply undistortion.
        
        Returns:
            numpy.ndarray: Undistorted frame at 1920x1080 resolution, or None if failed
        """
        # Capture and undistort a single frame from camera
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                return None
            frame = self._undistort_image(frame)
            return frame
    
    def _detect_aruco_markers(self, frame):
        """
        Detect all ArUco markers in the frame and return structured position array.
        
        Args:
            frame: Input image
            
        Returns:
            numpy.ndarray: Array with shape (6, 2) containing [x, y] positions:
                          [[TL_x, TL_y],    # Marker 42 (Top-Left corner) - Origin position
                           [TR_x, TR_y],    # Marker 30 (Top-Right corner) - Origin position
                           [BL_x, BL_y],    # Marker 69 (Bottom-Left corner) - Origin position
                           [BR_x, BR_y],    # Marker 23 (Bottom-Right corner) - Origin position
                           [Robot_x, Robot_y],  # Marker 78 (Robot) - Origin position
                           [Goal_x, Goal_y]]    # Marker 53 (Goal) - Origin position
                          Origin: Top-Left corner (corner 0) of each marker
                          Frame origin: Top-Left (0,0), x increases right, y increases down
                          Returns None if any marker is missing.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # Build dictionary of detected markers
        markers = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # Get origin position (Top-Left corner = corner 0) of marker
                marker_corners = corners[i][0]
                origin_x = marker_corners[0][0]  # Corner 0, x coordinate
                origin_y = marker_corners[0][1]  # Corner 0, y coordinate
                markers[marker_id] = (origin_x, origin_y)
                
                # Set self.pixels_per_cm knowing the size of an ArUco Marker
                if marker_id == 42:  # Calculate on first detected marker
                    # Calculate marker width and height in pixels
                    width = np.linalg.norm(marker_corners[1] - marker_corners[0])
                    height = np.linalg.norm(marker_corners[2] - marker_corners[1])
                    avg_size_pixels = (width + height) / 2.0
                    
                    # ArUco marker is ARUCO_SIZE_CM (5cm) in real world
                    self.pixels_per_cm = avg_size_pixels / ARUCO_SIZE_CM
                    print(f"Calibrated: {avg_size_pixels} pixels for both side, {self.pixels_per_cm:.2f} pixels/cm")
                    
        
        # Define expected marker IDs in order: TL, TR, BL, BR, Robot, Goal
        expected_ids = [42, 30, 69, 23, 78, 53]
        
        # Check if all required markers are detected
        missing = [mid for mid in expected_ids if mid not in markers]
        if missing:
            print(f"Warning: Missing markers {missing}")
            return None
        
        # Build aruko_pos array in the required order
        aruko_pos = np.array([
            [markers[42][0], markers[42][1]],  # TL (Top-Left) - Origin
            [markers[30][0], markers[30][1]],  # TR (Top-Right) - Origin
            [markers[69][0], markers[69][1]],  # BL (Bottom-Left) - Origin
            [markers[23][0], markers[23][1]],  # BR (Bottom-Right) - Origin
            [markers[78][0], markers[78][1]],  # Robot - Origin
            [markers[53][0], markers[53][1]]   # Goal - Origin
        ], dtype=np.float32)
        
        return aruko_pos

    def _compute_perspective_transform(self, markers):
        """
        Calculate perspective transformation matrix from 4 corner markers.
        
        Args:
            markers: Array of 4 corner positions [[TL_x, TL_y], [TR_x, TR_y], [BL_x, BL_y], [BR_x, BR_y]]
            
        Returns:
            tuple: (perspective_matrix, output_size) or (None, None) if failed
        """
        # markers is array with shape (4, 2): [TL, TR, BL, BR]
        src_points = markers.astype(np.float32)
        
        # Calculate map dimensions based on corner distances in pixels
        width_pixels = np.linalg.norm(src_points[1] - src_points[0])   # Distance TR - TL
        height_pixels = np.linalg.norm(src_points[2] - src_points[0])  # Distance BL - TL
        
        # Calculate pixels per cm if map_size_cm is known
        if self.map_size_cm is not None:
            self.pixels_per_cm = width_pixels / self.map_size_cm[0]
            print(f"Conversion factor: self.pixels_per_cm")
        else:
            print("Conversion factor pxl to cm has not been set")
            exit(1)
        
        # Define destination points for bird's eye view (rectangular map)
        # Order must match src_points: [TL, TR, BL, BR]
        dst_points = np.array([
            [0, 0],                          # TL -> top-left
            [width_pixels, 0],               # TR -> top-right
            [0, height_pixels],              # BL -> bottom-left
            [width_pixels, height_pixels]    # BR -> bottom-right
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        output_size = (int(width_pixels), int(height_pixels))
        
        return matrix, output_size
    
    def get_birds_eye_view(self, frame=None):
        """
        Extract bird's eye view (top-down perspective) from frame.
        
        Args:
            frame: Input frame (if None, captures new frame)
            
        Returns:
            numpy.ndarray: Warped bird's eye view image, or None if failed
        """
        # Capture frame if not provided
        if frame is None:
            frame = self._get_frame()
            #Test
            if frame is None:
                print("Failed to get frame.")
                return None
        
        # Detect ArUco markers to get corner positions
        self.aruko_position = self._detect_aruco_markers(frame)
        #Test
        if self.aruko_position is None:
            print("Failed to detect ArUco markers.")
            return None
        
        # Extract the 4 corner markers (first 4 positions: TL, TR, BL, BR)
        corners = self.aruko_position[:4]
        
        # Calculate perspective transform if not already done
        if self.perspective_matrix is None:
            self.perspective_matrix, self.map_size_pixels = \
                self._compute_perspective_transform(corners)
            
            if self.perspective_matrix is None:
                print("Failed to compute perspective transform.")
                return None
        
        # Apply perspective transformation to get bird's eye view
        warped = cv2.warpPerspective(frame, self.perspective_matrix, self.map_size_pixels)
        
        return warped
    
    # ============================================
    # 2. MAP GENERATION
    # ============================================
    
    def cm_to_pixel(self, cm):
        """
        Convert centimeters to pixels.
        
        Args:
            cm: Distance in centimeters
            
        Returns:
            int: Distance in pixels
        """
        if self.pixels_per_cm is None:
            raise ValueError("Pixels per cm not calibrated. Call get_birds_eye_view first.")
        return int(cm * self.pixels_per_cm)
    
    def pixel_to_cm(self, pixels):
        """
        Convert pixels to centimeters.
        
        Args:
            pixels: Distance in pixels
            
        Returns:
            float: Distance in centimeters
        """
        if self.pixels_per_cm is None:
            raise ValueError("Pixels per cm not calibrated. Call get_birds_eye_view first.")
        return pixels / self.pixels_per_cm
    
    def generate_map(self, frame=None, grid_size_cm=1):
        """
        Generate binary occupancy grid map from bird's eye view.
        
        Args:
            frame: Input frame (if None, captures new frame)
            grid_size_cm: Size of each grid cell in cm (default: 1cm x 1cm)
            
        Returns:
            numpy.ndarray: Binary map where 1=obstacle, 0=free space
                          Shape: (height_cells, width_cells)
        """
        # Get bird's eye view
        warped = self.get_birds_eye_view(frame)
        if warped is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to detect obstacles
        # You may need to adjust threshold values based on your setup
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate grid dimensions
        grid_size_pixels = self.cm_to_pixel(grid_size_cm)
        height_cells = binary.shape[0] // grid_size_pixels
        width_cells = binary.shape[1] // grid_size_pixels
        
        # Create occupancy grid
        occupancy_map = np.zeros((height_cells, width_cells), dtype=np.uint8)
        
        # Fill grid cells
        for i in range(height_cells):
            for j in range(width_cells):
                # Extract cell region
                y_start = i * grid_size_pixels
                y_end = (i + 1) * grid_size_pixels
                x_start = j * grid_size_pixels
                x_end = (j + 1) * grid_size_pixels
                
                cell_region = binary[y_start:y_end, x_start:x_end]
                
                # If majority of pixels are white (obstacle), mark as occupied
                if np.mean(cell_region) > 127:
                    occupancy_map[i, j] = 1
        
        return occupancy_map
    
    # ============================================
    # 3. ROBOT & GOAL LOCALIZATION
    # ============================================
    
    def localize_robot(self, frame=None):
        """
        Extract robot position and orientation from frame.
        
        Args:
            frame: Input frame (if None, captures new frame)
            
        Returns:
            dict: {'position': (x_cm, y_cm), 'orientation': angle_degrees}
                  or None if robot marker not detected
        """
        # Get frame
        if frame is None:
            frame = self._get_frame()
            if frame is None:
                return None
        
        # Undistort frame
        frame = self._undistort_image(frame)
        
        # Detect markers (use dict version for internal processing)
        markers = self._detect_aruco_markers_dict(frame)
        
        # Check if robot marker is detected
        if MARKER_ROBOT not in markers:
            print(f"Robot marker (ID {MARKER_ROBOT}) not detected")
            return None
        
        # Get robot marker corners
        corners = markers[MARKER_ROBOT]
        
        # Calculate center position
        center = np.mean(corners, axis=0)
        
        # Calculate orientation (angle from top edge of marker)
        top_edge = corners[1] - corners[0]
        angle = np.degrees(np.arctan2(top_edge[1], top_edge[0]))
        
        # Transform position to bird's eye view coordinates
        if self.perspective_matrix is not None:
            center_homogeneous = np.array([[center[0], center[1], 1]], dtype=np.float32).T
            transformed = self.perspective_matrix @ center_homogeneous
            transformed /= transformed[2]
            center = transformed[:2].flatten()
        
        # Convert to cm
        x_cm = self.pixel_to_cm(center[0])
        y_cm = self.pixel_to_cm(center[1])
        
        return {
            'position': (x_cm, y_cm),
            'orientation': angle
        }
    
    def localize_goal(self, frame=None):
        """
        Extract goal position from frame.
        
        Args:
            frame: Input frame (if None, captures new frame)
            
        Returns:
            tuple: (x_cm, y_cm) or None if goal marker not detected
        """
        # Get frame
        if frame is None:
            frame = self._get_frame()
            if frame is None:
                return None
        
        # Undistort frame
        frame = self._undistort_image(frame)
        
        # Detect markers
        markers = self._detect_aruco_markers_dict(frame)
        
        # Check if goal marker is detected
        if MARKER_GOAL not in markers:
            print(f"Goal marker (ID {MARKER_GOAL}) not detected")
            return None
        
        # Get goal marker center
        corners = markers[MARKER_GOAL]
        center = np.mean(corners, axis=0)
        
        # Transform position to bird's eye view coordinates
        if self.perspective_matrix is not None:
            center_homogeneous = np.array([[center[0], center[1], 1]], dtype=np.float32).T
            transformed = self.perspective_matrix @ center_homogeneous
            transformed /= transformed[2]
            center = transformed[:2].flatten()
        
        # Convert to cm
        x_cm = self.pixel_to_cm(center[0])
        y_cm = self.pixel_to_cm(center[1])
        
        return (x_cm, y_cm)
    
    # ============================================
    # 4. MAIN PROCESSING METHOD
    # ============================================
    
    def process_frame(self, grid_size_cm=1):
        """
        Complete vision processing pipeline.
        Captures frame, generates map, localizes robot and goal.
        
        Args:
            grid_size_cm: Size of grid cells in cm
            
        Returns:
            dict: {
                'map': Binary occupancy grid (numpy array),
                'robot': {'position': (x, y), 'orientation': angle},
                'goal': (x, y),
                'frame': Original frame (optional, for visualization)
            }
            or None if processing failed
        """
        # Capture frame
        frame = self._get_frame()
        if frame is None:
            return None
        
        # Generate map
        occupancy_map = self.generate_map(frame, grid_size_cm)
        if occupancy_map is None:
            return None
        
        # Localize robot
        robot_info = self.localize_robot(frame)
        
        # Localize goal
        goal_pos = self.localize_goal(frame)
        
        return {
            'map': occupancy_map,
            'robot': robot_info,
            'goal': goal_pos,
            'frame': frame
        }
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    
    

    #Test functions
    def message_from_vision(self):
        print("Vision module is correctly linked.")