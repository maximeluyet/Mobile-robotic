import cv2 as cv
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(script_dir, "img")

# Create img directory if it doesn't exist
os.makedirs(img_dir, exist_ok=True)

# Get ArUco dictionary (4x4 with 100 markers to support IDs up to 99)
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)

# Configuration for 5cm markers at 300 DPI
# Formula: pixels = (size_cm / 2.54) * DPI
# For 5cm at 300 DPI: (5 / 2.54) * 300 ≈ 591 pixels
MARKER_SIZE_PIXELS = 591
DPI = 300

# Generate first marker (ID 23)
markerImage1 = None
markerImage1 = cv.aruco.generateImageMarker(dictionary, 23, MARKER_SIZE_PIXELS, markerImage1, 1)
marker1_path = os.path.join(img_dir, "marker23.png")
cv.imwrite(marker1_path, markerImage1, [cv.IMWRITE_PNG_COMPRESSION, 0])

# Generate first marker (ID 30)
markerImage2 = None
markerImage2 = cv.aruco.generateImageMarker(dictionary, 30, MARKER_SIZE_PIXELS, markerImage2, 1)
marker2_path = os.path.join(img_dir, "marker30.png")
cv.imwrite(marker2_path, markerImage2, [cv.IMWRITE_PNG_COMPRESSION, 0])

# Generate second marker (ID 42)
markerImage3 = None
markerImage3 = cv.aruco.generateImageMarker(dictionary, 42, MARKER_SIZE_PIXELS, markerImage3, 1)
marker3_path = os.path.join(img_dir, "marker42.png")
cv.imwrite(marker3_path, markerImage3, [cv.IMWRITE_PNG_COMPRESSION, 0])

# Generate marker (ID 53)
markerImage4 = None
markerImage4 = cv.aruco.generateImageMarker(dictionary, 53, MARKER_SIZE_PIXELS, markerImage4, 1)
marker4_path = os.path.join(img_dir, "marker53.png")
cv.imwrite(marker4_path, markerImage4, [cv.IMWRITE_PNG_COMPRESSION, 0]) 

# Generate marker (ID 69)
markerImage5 = None
markerImage5 = cv.aruco.generateImageMarker(dictionary, 69, MARKER_SIZE_PIXELS, markerImage5, 1)
marker5_path = os.path.join(img_dir, "marker69.png")
cv.imwrite(marker5_path, markerImage5, [cv.IMWRITE_PNG_COMPRESSION, 0]) 

# Generate marker (ID 78)
markerImage6 = None
markerImage6 = cv.aruco.generateImageMarker(dictionary, 78, MARKER_SIZE_PIXELS, markerImage6, 1)
marker6_path = os.path.join(img_dir, "marker78.png")
cv.imwrite(marker6_path, markerImage6, [cv.IMWRITE_PNG_COMPRESSION, 0]) 

print("Generated markers (591x591 pixels for 5cm at 300 DPI):")
print(f"- {marker1_path}")
print(f"- {marker2_path}")
print(f"- {marker3_path}")
print(f"- {marker4_path}")
print(f"- {marker5_path}")
print(f"- {marker6_path}")
print(f"\nIMPORTANT: When printing, ensure your printer is set to 300 DPI")
print(f"and 'Actual Size' (no scaling) for exact 5cm output.")