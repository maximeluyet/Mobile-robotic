import numpy as np
import cv2
import os
import sys

# Global constants 

CAMERA_MATRIX = np.array([[1044.075255, 0.000000, 521.590362],
                          [0.000000, 1051.837388, 381.637697],
                          [0.000000, 0.000000, 1.000000]])

CAMERA_COEFFICIENTS = np.array([0.155759, -1.196364, 0.000317, -0.001282, 2.565589])

CAMERA_RESOLUTION = (1920, 1080)

class Vision():
    def __init__(self, cam_id):
        self.CAM_ID = cam_id
        self.cap = None
    
    def _undistort_image(self, map):
        
        h,  w = map.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, CAMERA_COEFFICIENTS, (w,h), 1, (w,h))

        # Undistort
        dst = cv2.undistort(map, CAMERA_MATRIX, CAMERA_COEFFICIENTS, None, newcameramtx)
        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        #cv2.imwrite('calibresult.png', dst)
        
        return map

    def generate_map(self, img_data):

        pass

    def localize_robot(self, img_data):

        pass

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
    

    #Test functions
    def message_from_vision(self):
        print("Vision module is correctly linked.")