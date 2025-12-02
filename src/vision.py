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


    def get_occupancy_grid(self, frame, is_world_frame=False, is_marker_in_corners=False, debug=False):
        if not is_world_frame:
            frame = self._cut_frame_to_world(frame)
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Step 3: Adaptive Thresholding
        # This might create the "halo" artifact around markers, but we will fix it next
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        binary_copy = binary.copy()
        
        # --- FIXED: MASKING AFTER THRESHOLDING ---
        if is_marker_in_corners:
            # Calculate mask size
            # ARUCO_SIZE_CM + Margin (e.g., 2cm)
            mask_size_px = int((ARUCO_SIZE_CM + 2.0) * self.pixels_per_cm)
            
            # IMPORTANT: Paint with 0 (Black). 
            # In the 'binary' image, 0 means Free Space, 255 means Obstacle.
            # This wipes out the markers AND any "halo" artifacts cleanly.
            color = 0 
            
            h, w = binary.shape
            
            # Top-Left
            cv2.rectangle(binary, (0, 0), (mask_size_px, mask_size_px), color, -1)
            # Top-Right
            cv2.rectangle(binary, (w - mask_size_px, 0), (w, mask_size_px), color, -1)
            # Bottom-Left
            cv2.rectangle(binary, (0, h - mask_size_px), (mask_size_px, h), color, -1)
            # Bottom-Right
            cv2.rectangle(binary, (w - mask_size_px, h - mask_size_px), (w, h), color, -1)
        # -----------------------------------------

        # Step 4: Morphological operations (Clean up noise)
        kernel_noise = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_noise, iterations=1)

        # --- NEW STEP: FILL HOLES ---
        # Find the external contours (boundaries) of the cleaned shapes
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw over the original 'cleaned' image
        # -1 index: Draw all contours
        # 255: Color White
        # thickness=-1: Fill the shape
        cv2.drawContours(cleaned, contours, -1, 255, thickness=-1)
        # ----------------------------
        
        # Step 5: Dilate obstacles (Safety Margin)
        dilation_radius_px = int((ROBOT_WIDTH_CM / 2.0) * self.pixels_per_cm)
        dilation_radius_px = max(1, dilation_radius_px)
        
        kernel_size = 2 * dilation_radius_px + 1
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        dilated = cv2.dilate(cleaned, kernel_dilate, iterations=1)

        # --- STEP 6: ADD FADING BORDER (COST MAP) ---
        
        # 1. Invert the image so obstacles are 0 (Black) and free space is 255 (White).
        # cv2.distanceTransform calculates distance to the nearest ZERO pixel.
        inv_dilated = cv2.bitwise_not(dilated)
        
        # 2. Calculate Distance Transform
        # 'dist_img' contains the distance (in pixels) from every point to the nearest obstacle
        dist_img = cv2.distanceTransform(inv_dilated, cv2.DIST_L2, 5)
        
        # 3. Define how big the fading zone should be (e.g., 15 cm extra buffer)
        fade_zone_cm = 7.0
        fade_zone_px = fade_zone_cm * self.pixels_per_cm
        
        # 4. Create the linear gradient
        # Formula: Intensity = 1.0 - (CurrentDistance / MaxFadeDistance)
        # Pixels closer to obstacle -> Higher value (Whiter)
        # Pixels far away -> 0 (Black)
        fading_border = np.clip((1.0 - dist_img / fade_zone_px), 0.0, 1.0)
        fading_border = (fading_border * 255).astype(np.uint8)
        
        # 5. Combine the Hard Obstacle with the Fading Border
        # We use cv2.max to overlay the fade on top of the original hard obstacles
        cost_map = cv2.max(dilated, fading_border)
        
        # Return the Cost Map (0-255) instead of Binary (-1/0)
        # Note: You will need to update your path planner to understand that
        # 255 = Wall, 100-200 = Risky, 0 = Free

        # Step 7 : add obstacle on the edge of the map
        border_thickness = int(ROBOT_WIDTH_CM * self.pixels_per_cm/2)
        cv2.rectangle(cost_map, (0, 0), (cost_map.shape[1]-1, cost_map.shape[0]-1), 255, border_thickness)

        if debug == True:
            cv2.imwrite("Occupancy Grid - Grayscale.png", gray)
            cv2.imwrite("Occupancy Grid - Blurred.png", blurred)
            cv2.imwrite("Occupancy Grid - Binary.png", binary_copy)
            cv2.imwrite("Occupancy Grid - Cleaned.png", cleaned)
            cv2.imwrite("Occupancy Grid - Dilated.png", dilated)
            cv2.imwrite("Occupancy Grid - Fading.png", cost_map)
            cv2.imwrite("Occupancy Grid - Fading Border.png", fading_border)
        
        return cost_map

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
                    # Corner 2 is BR
                    marker_corners = corners[i][0]
                    origin_x = marker_corners[2][0]  # Corner 0, x coordinate
                    origin_y = marker_corners[2][1]  # Corner 0, y coordinate
                    # add little offset to have full marker inside the map
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