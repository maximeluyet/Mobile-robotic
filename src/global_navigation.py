# imports
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt

class GlobalNavigation:
    
    def __init__(self):

        pass

    def set_map(self, map_data):
        ''' Set the environment map for navigation.'''
        
        pass

    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

    def grid_search_astar(self, Map, S, G):
        came_from = {}
        g_costs = {S: 0.0}
        explored = set()
        operation_count = 0

        open_set = [(self.heuristic(S, G), 0.0, S)]  # (f, g, pos)

        while open_set:
            current_f, current_g, current_pos = heappop(open_set)

            # Skip if we've already expanded this cell with a better cost
            if current_pos in explored:
                continue
            explored.add(current_pos)

            if current_pos == G:
                break

            r, c = current_pos
            # 8-connected neighbors (N, S, W, E, NW, NE, SW, SE)
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < Map.shape[0] and 0 <= nc < Map.shape[1]:
                    if Map[nr, nc] != -1:  # not an obstacle
                        step_cost = np.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                        tentative_g = current_g + step_cost + Map[nr, nc]
                        neighbor = (nr, nc)

                        if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                            g_costs[neighbor] = tentative_g
                            came_from[neighbor] = current_pos
                            f_cost = tentative_g + self.heuristic(neighbor, G)
                            heappush(open_set, (f_cost, tentative_g, neighbor))
                            operation_count += 1

        # Reconstruct path if goal reached
        if current_pos == G:
            path = []
            while current_pos != S:
                path.append(current_pos)
                current_pos = came_from[current_pos]
            path.append(S)
            path.reverse()
            return path, explored, operation_count

        return None, explored, operation_count

    def find_closest_point(self, robot_pose, path):
        ''' Find the closest point on the path to the robot's current position.'''
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(path):
            dist = np.sqrt((point[0] - robot_pose[0])**2 + (point[1] - robot_pose[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx

    def display_map(self, Map, path8, SearchStart, SearchGoal, explored8):
        # Create a color map
        cmap = plt.cm.get_cmap('Greys', 3)  # 0: free, 1: obstacle, 2: explored

        # Create a display map
        display = np.array(Map)
        for cell in explored8:
            display[cell] = 2  # Mark explored cells

        for cell in path8:
            display[cell] = 3  # Mark path cells

        display[SearchStart] = 4  # Start cell
        display[SearchGoal] = 5   # Goal cell

        # Define colors for each type of cell
        colors = {
            0: (1, 1, 1),      # Free space - white
            1: (0, 0, 0),      # Obstacle - black
            2: (0.7, 0.7, 0.7),# Explored - light gray
            3: (0, 1, 0),      # Path - green
            4: (0, 0, 1),      # Start - blue
            5: (1, 0, 0)       # Goal - red
        }

        # Create a color array for the display map
        color_array = np.zeros((display.shape[0], display.shape[1], 3))
        for key, color in colors.items():
            color_array[display == key] = color

        plt.figure(figsize=(12, 12))
        plt.imshow(color_array, interpolation='nearest')
        plt.title('Map with Path and Explored Cells')
        plt.axis('off')
        plt.show()

    def look_ahead(self, velocity):
        ''' Simple look-ahead distance based on velocity '''
        # Parameters (get param from config)
        alpha = self.config['navigation']['look_ahead_alpha']
        constant = self.config['navigation']['look_ahead_constant']
        return int(velocity * alpha) + constant  # e.g., 2 cells per unit velocity
    
    def direction_attraction(self, robot_pose, target_pose):
        dir_vector = np.array(target_pose) - np.array(robot_pose)
        norm = np.linalg.norm(dir_vector)
        if norm == 0:
            return np.array([0.0, 0.0])
        return dir_vector / norm