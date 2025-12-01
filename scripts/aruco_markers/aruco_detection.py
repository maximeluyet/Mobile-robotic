import cv2 as cv
import numpy as np

# Configuration
CAM_ID = 0  # Change this if using external camera (try 1, 2, etc.)
CAMERA_RESOLUTION = (1920, 1080)
DISPLAY_WIDTH = 960  # Display window width (half of 1920)

# Setup ArUco detector (4x4 dictionary with 100 markers)
detectorParams = cv.aruco.DetectorParameters()
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
detector = cv.aruco.ArucoDetector(dictionary, detectorParams)

# Open camera
cap = cv.VideoCapture(CAM_ID)
cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

if not cap.isOpened():
    print(f"Error: Unable to open camera {CAM_ID}")
    exit(1)

print("ArUco Marker Detection - Real-time")
print("Expected markers: 23, 30, 42, 53, 69, 78")
print("  Corners: 42 (TL), 30 (TR), 69 (BR), 23 (BL)")
print("  Robot: 78 | Goal: 53")
print("Press 'q' to quit")
print("-" * 50)

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Convert to grayscale for detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
        
        # Expected markers
        expected_markers = {23: 'Corner BL', 30: 'Corner TR', 42: 'Corner TL', 
                           53: 'Goal', 69: 'Corner BR', 78: 'Robot'}
        detected_ids = []
        
        # Draw detected markers
        if markerIds is not None and len(markerIds) > 0:
            # Draw marker boundaries and IDs
            cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            detected_ids = [id[0] for id in markerIds]
            
            # Display information for each detected marker
            for i, marker_id in enumerate(markerIds):
                # Get marker corners
                corners = markerCorners[i][0]
                
                # Calculate center
                center_x = int(np.mean(corners[:, 0]))
                center_y = int(np.mean(corners[:, 1]))
                
                # Display marker ID and role at center
                marker_label = f"ID: {marker_id[0]} ({expected_markers.get(marker_id[0], 'Unknown')})"
                cv.putText(frame, marker_label, 
                          (center_x - 60, center_y - 10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display position
                cv.putText(frame, f"({center_x}, {center_y})", 
                          (center_x - 40, center_y + 20),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display detection status at top
            status_text = f"Detected: {len(markerIds)}/6 markers"
            cv.putText(frame, status_text, 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display marker status list at bottom left
            y_offset = frame.shape[0] - 160
            for marker_id, role in expected_markers.items():
                status = "✓" if marker_id in detected_ids else "✗"
                color = (0, 255, 0) if marker_id in detected_ids else (0, 0, 255)
                status_line = f"{status} ID {marker_id}: {role}"
                cv.putText(frame, status_line, (10, y_offset), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
        else:
            # No markers detected
            cv.putText(frame, "No markers detected", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display instructions
        cv.putText(frame, "Press 'q' to quit", 
                  (10, frame.shape[0] - 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Resize frame for display
        display_height = int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])
        display_frame = cv.resize(frame, (DISPLAY_WIDTH, display_height))
        
        # Show frame
        cv.imshow('ArUco Marker Detection', display_frame)
        
        # Exit on 'q' press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    print("Camera released and windows closed")