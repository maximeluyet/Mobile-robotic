import cv2 as cv
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(script_dir, "img")

# Create img directory if it doesn't exist
os.makedirs(img_dir, exist_ok=True)

# Get ArUco dictionary
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Generate first marker (ID 23)
markerImage1 = None
markerImage1 = cv.aruco.generateImageMarker(dictionary, 23, 200, markerImage1, 1)
marker1_path = os.path.join(img_dir, "marker23.png")
cv.imwrite(marker1_path, markerImage1)

# Generate second marker (ID 42)
markerImage2 = None
markerImage2 = cv.aruco.generateImageMarker(dictionary, 42, 200, markerImage2, 1)
marker2_path = os.path.join(img_dir, "marker42.png")
cv.imwrite(marker2_path, markerImage2)

print("Generated markers:")
print(f"- {marker1_path}")
print(f"- {marker2_path}")