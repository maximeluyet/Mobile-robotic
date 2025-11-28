import numpy as np
import cv2
import threading

# Global constants

# TODO: detail: in the end maybe add to vision class since they are only used there...
ROBOT_WIDTH_CM = 11.0
MAP_SIZE_CM = (95.0, 60.0)  # Width x Height in cm

MAP_MARKERS_OLD = [69, 42, 23, 30]  # IDs for the 4 corners: [TL, TR, BL, BR]
MAP_MARKERS = [42, 30, 69, 23]  # IDs for the 4 corners: [TL, TR, BL, BR]

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
ARUCO_SIZE_CM = 10.0                # Real size of ArUco markers in cm (5cm x 5cm at 300 DPI)

class Vision():
    def __init__(self):

        # Initialize ArUco dictionary and detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.param = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.param)
        
        self.world_size = None
        self.corners_position = None 
        
        # Map configuration (will be calculated after detecting corners)
        self.M = None                   # perspective warp matrix
        self.map_size_pixels = None     # map size in pixels 
        self.pixels_per_cm = None       # 
    

    def set_camera_parameters(self, camera_parameters):

        pass


    def setup(self, frame):
        # Function that detects the map, cuts and undistorts image to the map and stores operations
        # returns undistorted frame
        # After the setup the function _cut_frame_to_world can be used to do this operation efficiently

        ## TODO: clean this up

        # Get undistortion matrix, undistort and crop
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, CAMERA_COEFFICIENTS, (w, h), 1, (w, h))

        self.distortion = newcameramtx
        self.roi = roi

        frame = cv2.undistort(frame, CAMERA_MATRIX, CAMERA_COEFFICIENTS, None, self.distortion)

        x, y, w, h = self.roi
        frame = frame[y:y+h, x:x+w]

        # Detect the map markers
        markers = self._detect_map_markers(frame)
        print("Detected markers:", markers)
        required_ids = [69, 42, 23, 30]
        if markers is None or not all(marker_id in markers for marker_id in required_ids):
            print(f"Failed to detect all required ArUco corner markers: {required_ids}")
            return None
        
        # Compute map dimensions in pixels from detected corners
        # Width: horizontal distance between bottom corners
        width_px = markers[MAP_MARKERS[1]][0] - markers[MAP_MARKERS[0]][0]  # BR.x - BL.x
        if self.pixels_per_cm is None:
            self.pixels_per_cm = width_px / MAP_SIZE_CM[0]  # pixels/cm based on known map size
        
        # Height: vertical distance (Y increases downward, so BL.y > TL.y)
        height_px = markers[MAP_MARKERS[3]][1] - markers[MAP_MARKERS[1]][1]  # BL.y - TL.y
        print("Map size in pixels: width =", width_px, ", height =", height_px)
        print("Pixels per cm:", self.pixels_per_cm)
        # Get perspective transformation matrix
        self.M = self._get_transform_matrix(markers, width_px, height_px)
        
        # Apply perspective transformation to get bird's eye view
        self.output_size = (int(width_px), int(height_px))

        warped = cv2.warpPerspective(frame, self.M, self.output_size)

        return warped


    def get_occupancy_grid(self, frame, is_world_frame=False):
        # Function to detect obstacles in frame and generate the occupancy grid
        # TODO: clean up and put less comments
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

        # Crop the frame to world
        if not is_world_frame:
            frame = self._cut_frame_to_world(frame)
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: detect and erase robot and goal aruco markers
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Binary thresholding (Otsu's method for automatic threshold)
        # Black obstacles on white background: threshold inverted
        # After threshold: obstacles=255 (white), free space=0 (black)
        # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Alternative method using adaptive threshold for varying lighting:
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Step 4: Morphological operations to clean up noise
        # Remove small noise with opening (erosion followed by dilation)
        kernel_noise = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise, iterations=1)
        
        # Fill small holes in obstacles with closing (dilation followed by erosion)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_noise, iterations=1)
        
        # Step 5: Dilate obstacles to account for robot width (safety margin)
        # Calculate dilation kernel size based on robot width
        # Dilation radius = half robot width (to expand obstacle on all sides)
        dilation_radius_px = int((ROBOT_WIDTH_CM / 2.0) * self.pixels_per_cm)
        
        # Ensure minimum dilation of at least 1 pixel
        dilation_radius_px = max(1, dilation_radius_px)
        
        # Create rectangular kernel to preserve straight edges
        # This maintains the original obstacle shape better than circular kernel
        kernel_size = 2 * dilation_radius_px + 1
        print("Dilation kernel size (pixels):", kernel_size)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (kernel_size, kernel_size))
        
        dilated = cv2.dilate(cleaned, kernel_dilate, iterations=1)
        
        # Step 6: Convert to binary occupancy grid (0 and 1)
        # 255 -> 1 (obstacle), 0 -> 0 (free space)
        occupancy_grid = (dilated > 0).astype(np.int8)
        occupancy_grid[occupancy_grid==1] = -1
        
        return occupancy_grid


    def get_goal_position(self, frame, is_world_frame=False):
        # TODO: clean up
        # Gets the goal position and returns it as tuple (x, y)
        """
        Detect the goal's ArUco marker in the bird's eye view image.
        
        Args:
            birdseye_view: Bird's eye view image (BGR format)
            
        Returns:
            tuple: (birdseye_view, goal_pos) where:
                - birdseye_view: Image with goal marker drawn
                - goal_pos: numpy array [x, y] where:
                    * x, y: center position in pixels (origin = top-left)
                Returns (birdseye_view, None) if goal not found
        """
        goal_detected = False

        # Crop the frame to world
        if not is_world_frame:
            frame = self._cut_frame_to_world(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        goal_pos = np.array([0., 0.])  # Initialize as NumPy array
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                coin = corners[i][0]
                if marker_id == MARKER_GOAL:
                    center = np.mean(coin, axis=0)
                    goal_pos[0] = center[0]  # x position
                    goal_pos[1] = center[1]  # y position
                    goal_detected = True

        return goal_detected, goal_pos
    

    def localize_robot(self, frame, is_world_frame=False):
        # TODO: clean up
        # Gets the robot pose on the map and returns (x, y, theta)
        # also returns the undistorted frame
        """
        Detect the robot's ArUco marker in the bird's eye view image.
        
        Args:
            birdseye_view: Bird's eye view image (BGR format)
            
        Returns:
            tuple: (birdseye_view, robot_pos) where:
                - birdseye_view: Image with robot marker drawn
                - robot_pos: numpy array [x, y, angle] where:
                    * x, y: center position in pixels (origin = top-left)
                    * angle: orientation in radians [-π, π]
                Returns (birdseye_view, None) if robot not found
        """

        robot_detected = False

        # Crop the frame to world
        if not is_world_frame:
            frame = self._cut_frame_to_world(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        robot_pos = np.array([0., 0., 0.])  # Initialize as NumPy array
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                coin = corners[i][0]
                if marker_id == MARKER_ROBOT:
                    center = np.mean(coin, axis=0)
                    robot_pos[0] = center[0]  # x position
                    robot_pos[1] = center[1]  # y position
                    
                    # Calculate orientation from marker front edge vector
                    front_edge = coin[1] - coin[0]
                    orientation = np.arctan2(front_edge[1], front_edge[0])
                    
                    # Normalize angle to [-π, π]
                    robot_pos[2] = np.arctan2(np.sin(orientation), np.cos(orientation))
                    
                    # # Draw orientation arrow and center point
                    # start_point = (int(center[0]), int(center[1]))
                    # end_point = (int(center[0] + 50 * np.cos(orientation)), 
                    #            int(center[1] + 50 * np.sin(orientation)))
                    
                    robot_detected = True

        return robot_detected, robot_pos, frame

    
    # TODO: Detail: think about making this public to explicitly use in robot loop...
    def _cut_frame_to_world(self, frame):
        # Undistorts and cuts frame to the world (map)

        # TODO: check if setup was already done

        frame = cv2.undistort(frame, CAMERA_MATRIX, CAMERA_COEFFICIENTS, None, self.distortion)
        frame = frame[self.roi[1] : self.roi[1] + self.roi[3], self.roi[0] : self.roi[0] + self.roi[2]]
        frame = cv2.warpPerspective(frame, self.M, self.output_size)

        return frame
    
    
    def _detect_map_markers(self, frame):
        # TODO: clean up and put less comments
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
        #cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
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
        
        return markers 
    
    
    def _get_transform_matrix(self, corners_dict, width_px, height_px):
        # TODO: clean up and put less comments
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
            corners_dict[MAP_MARKERS[0]],  # Top-Left (ID 69)
            corners_dict[MAP_MARKERS[1]],  # Top-Right (ID 42)
            corners_dict[MAP_MARKERS[2]],  # Bottom-Left (ID 23)
            corners_dict[MAP_MARKERS[3]]   # Bottom-Right (ID 30)
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


class Camera:
    # Camera class to handle webcam in a separate thread

    def __init__(self, cam_id=0):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(self.cam_id)

        if not self.cap.isOpened():
            print(f"Error: Unable to open camera with ID {self.cam_id}")
            self.cap = None
            self.running = False
            return
        
        else:
            print(f"Camera with ID {self.cam_id} opened successfully.")

        self.ret, self.frame = self.cap.read()

        self.running = True

        # launch thread
        threading.Thread(target=self.update, args=(), daemon=True).start()

        print("Camera thread launched")


    def get_frame(self):
        # Get the latest frame from the camera
        return self.ret, self.frame
    
    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def stop(self):
        self.running = False

        if self.cap is not None:
            self.cap.release()
            print(f"Camera with ID {self.cam_id} released, thread stopped")