import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output, Image, display


def draw_red_line_in_image(corners, image):
    """
    Draw red lines connecting the 4 corner markers.
    
    Args:
        corners: Array with shape (4, 2) containing [x, y] of corners [TL, TR, BL, BR]
        image: Input image
    
    Returns:
        Image with red border lines drawn
    """
    # Extract corner positions (convert to integer for drawing)
    TL = (int(corners[0][0]), int(corners[0][1]))  # Top-Left
    TR = (int(corners[1][0]), int(corners[1][1]))  # Top-Right
    BL = (int(corners[2][0]), int(corners[2][1]))  # Bottom-Left
    BR = (int(corners[3][0]), int(corners[3][1]))  # Bottom-Right
    
    # Draw red lines (thickness=3, color=red in BGR)
    cv2.line(image, TL, TR, (0, 0, 255), 3)  # Top edge
    cv2.line(image, TR, BR, (0, 0, 255), 3)  # Right edge
    cv2.line(image, BR, BL, (0, 0, 255), 3)  # Bottom edge
    cv2.line(image, BL, TL, (0, 0, 255), 3)  # Left edge
    
    # # Draw circles at corners for better visibility
    # cv2.circle(image, TL, 8, (0, 0, 255), -1)
    # cv2.circle(image, TR, 8, (0, 0, 255), -1)
    # cv2.circle(image, BL, 8, (0, 0, 255), -1)
    # cv2.circle(image, BR, 8, (0, 0, 255), -1)
    
    # Add labels (corrected IDs based on actual marker positions)
    cv2.putText(image, "TL(69)", (TL[0]-40, TL[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "TR(42)", (TR[0]+10, TR[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "BL(23)", (BL[0]-40, BL[1]+25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "BR(30)", (BR[0]+10, BR[1]+25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return image