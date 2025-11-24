import numpy as np
import cv2
import os
import sys
import utils 

# Global constants 
ROBOT_WIDTH_CM = 11.0
MAP_SIZE_CM = (71.0, 71.0)  # Width x Height in cm

MAP_MARKERS = [69, 42, 23, 30]  # IDs for the 4 corners: [TL, TR, BL, BR]

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
        self.corners_position = None 
        
        # Map configuration (will be calculated after detecting corners)
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

    def _detect_map_markers(self, frame):
        """
        Detect map ArUco markers in the frame and return dictionary of marker positions.
        
        Args:
            frame: Input image (will be modified with drawn markers)
            
        Returns:
            tuple: (frame, markers_dict)
                frame: Image with detected markers drawn
                markers_dict: Dictionary {marker_id: (x, y)} where:
                    - 69: TL (Top-Left corner)
                    - 42: TR (Top-Right corner)
                    - 23: BL (Bottom-Left corner)
                    - 30: BR (Bottom-Right corner)
                Origin: Top-Left corner (corner 0) of each marker
                Frame origin: Top-Left (0,0), x increases right, y increases down
                Returns (frame, None) if markers cannot be detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Build dictionary of detected markers (only MAP_MARKERS)
        markers = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # Only keep markers that define the map (filter out robot/goal)
                if marker_id in MAP_MARKERS:
                    # Get origin position (Top-Left corner = corner 0) of marker
                    marker_corners = corners[i][0]
                    origin_x = marker_corners[0][0]  # Corner 0, x coordinate
                    origin_y = marker_corners[0][1]  # Corner 0, y coordinate
                    markers[int(marker_id)] = (float(origin_x), float(origin_y))
        else:
            print("Impossible to capture markers")
            return frame, None
        
        # Verify all expected markers are present
        missing = [mid for mid in MAP_MARKERS if mid not in markers]
        if missing:
            print(f"Warning: Missing markers {missing}")
        
        return frame, markers 
    
    def _get_transform_matrix(self, corners_dict, width_px, height_px):
        """
        Compute perspective transform matrix from detected ArUco corners.
        
        Args:
            corners_dict: Dictionary of detected markers {marker_id: (x, y)}
                        Expected IDs: 69=TL, 42=TR, 23=BL, 30=BR
            width_px: Width in pixels for output image
            height_px: Height in pixels for output image
            
        Returns:
            numpy.ndarray: 3x3 perspective transformation matrix
        """
        # Extract corner positions from dictionary in correct order
        # Source points are the actual detected marker positions
        src_pts = np.array([
            corners_dict[69],  # Top-Left (ID 69)
            corners_dict[42],  # Top-Right (ID 42)
            corners_dict[23],  # Bottom-Left (ID 23)
            corners_dict[30]   # Bottom-Right (ID 30)
        ], dtype="float32")
        
        # Define destination points for bird's eye view (perfect rectangle)
        dst_pts = np.array([
            [0, 0],                    # Top-Left
            [width_px, 0],             # Top-Right
            [0, height_px],            # Bottom-Left
            [width_px, height_px]      # Bottom-Right
        ], dtype="float32")
    
        # Compute perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return matrix

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
            if frame is None:
                print("Failed to get frame.")
                return None
            
            frame, corner_pos = self._detect_map_markers(frame)
            
            # Check if all required corner markers were detected
            required_ids = [69, 42, 23, 30]
            if corner_pos is None or not all(marker_id in corner_pos for marker_id in required_ids):
                print(f"Failed to detect all required ArUco corner markers: {required_ids}")
                return None
        else:
            # If frame provided, still need to detect markers
            frame, corner_pos = self._detect_map_markers(frame)
            required_ids = [69, 42, 23, 30]
            if corner_pos is None or not all(marker_id in corner_pos for marker_id in required_ids):
                print(f"Failed to detect all required ArUco corner markers: {required_ids}")
                return None
        
        # Compute map dimensions in pixels from detected corners
        # Width: horizontal distance between bottom corners
        width_px = corner_pos[30][0] - corner_pos[23][0]  # BR.x - BL.x
        if self.pixels_per_cm is None:
            self.pixels_per_cm = MAP_SIZE_CM[0] / width_px  # pixels/cm based on known map size
        
        # Height: vertical distance (Y increases downward, so BL.y > TL.y)
        height_px = corner_pos[23][1] - corner_pos[69][1]  # BL.y - TL.y
        
        # Get perspective transformation matrix
        M = self._get_transform_matrix(corner_pos, width_px, height_px)
        
        # Apply perspective transformation to get bird's eye view
        output_size = (int(width_px), int(height_px))
        warped = cv2.warpPerspective(frame, M, output_size)
        
        return warped
    
    def get_robot_position(self, birdseye_view):
        """
        Detect the robot's ArUco marker in the bird's eye view image.
        
        Args:
            birdseye_view: Bird's eye view image (BGR format)
            
        Returns:
            tuple: (x, y) position of robot marker origin (corner 0) in pixels,
                   or None if not found
        """
        gray = cv2.cvtColor(birdseye_view, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(birdseye_view, corners, ids)
        robot_pos = None
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == MARKER_ROBOT:
                    # Get origin position (Top-Left corner = corner 0) of robot marker
                    marker_corners = corners[i][0]
                    robot_pos = (float(marker_corners[0][0]), float(marker_corners[0][1]))
                    return robot_pos
        
        print("Robot marker not found.")
        return None

    
    def detect_obstacles(self, birdseye_view, pixels_per_cm=None, robot_width_cm=ROBOT_WIDTH_CM):
        """
        Detect black obstacles on white background and create binary occupancy grid.
        Obstacles are dilated to account for robot width for safe path planning.
        
        Args:
            birdseye_view: Bird's eye view image (BGR format)
            pixels_per_cm: Pixels per centimeter scale (if None, uses self.pixels_per_cm)
            robot_width_cm: Width of robot in cm for dilation safety margin
            
        Returns:
            numpy.ndarray: Binary occupancy grid where:
                - 1 = obstacle (including safety margin)
                - 0 = free space
                Returns None if detection fails.
        """
        if birdseye_view is None:
            print("Error: No bird's eye view image provided")
            return None
        
        # Use class attribute if pixels_per_cm not provided
        if pixels_per_cm is None:
            if self.pixels_per_cm is None:
                print("Warning: pixels_per_cm not set. Using default threshold method.")
                # Fallback: use image dimensions directly
                pixels_per_cm = 10  # Default assumption
            else:
                pixels_per_cm = self.pixels_per_cm
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(birdseye_view, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Binary thresholding (Otsu's method for automatic threshold)
        # Black obstacles on white background: threshold inverted
        # After threshold: obstacles=255 (white), free space=0 (black)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Alternative method using adaptive threshold for varying lighting:
        # binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                                cv2.THRESH_BINARY_INV, 11, 2)
        
        # Step 4: Morphological operations to clean up noise
        # Remove small noise with opening (erosion followed by dilation)
        kernel_noise = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise, iterations=1)
        
        # Fill small holes in obstacles with closing (dilation followed by erosion)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_noise, iterations=1)
        
        # Step 5: Dilate obstacles to account for robot width (safety margin)
        # Calculate dilation kernel size based on robot width
        # Dilation radius = half robot width (to expand obstacle on all sides)
        dilation_radius_px = int((robot_width_cm / 2.0) * pixels_per_cm)
        
        # Ensure minimum dilation of at least 1 pixel
        dilation_radius_px = max(1, dilation_radius_px)
        
        # Create rectangular kernel to preserve straight edges
        # This maintains the original obstacle shape better than circular kernel
        kernel_size = 2 * dilation_radius_px + 1
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                (kernel_size, kernel_size))
        
        dilated = cv2.dilate(cleaned, kernel_dilate, iterations=1)
        
        # Step 6: Convert to binary occupancy grid (0 and 1)
        # 255 -> 1 (obstacle), 0 -> 0 (free space)
        occupancy_grid = (dilated > 0).astype(np.uint8)
        
        return occupancy_grid


    # Fonction utilitaire pour visualiser la grille d'occupation
    def visualize_occupancy_grid(self,occupancy_grid, window_name="Occupancy Grid"):
        """
        Visualize the binary occupancy grid.
        
        Args:
            occupancy_grid: Binary grid (0=free, 1=obstacle)
            window_name: Name of the display window
        """
        # Convert to viewable image: obstacles=black, free=white
        visualization = (1 - occupancy_grid) * 255
        visualization = visualization.astype(np.uint8)
        
        cv2.imshow(window_name, visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
            print("Webcam disconnected")
    
    def message_from_vision(self):
        print("Vision module is correctly linked.")