# imports
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt

class GlobalNavigation:
    
    def __init__(self):
        self.Map100 = np.zeros((100, 100))
        self.Map100[40:70, 40:70] = 0.
        self.Map100[60:80, 60:80] = -1
        self.Map100[85:95, 20:50] = -1
        self.Map100[20:50, 80:90] = -1
        self.Map100[90:95, 80:100] = -1

        self.SearchStart = (2, 2)
        self.SearchGoal = (700, 700)

    def set_map(self, map_data):
        ''' Set the environment map for navigation.'''
        self.Map100 = map_data

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

    def grid_path_smoothing(self, path, Map):
        ''' Smooth the given path by removing unnecessary waypoints.'''
        if not path or len(path) < 3:
            return path

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.is_line_free(path[i], path[j], Map):
                    break
                j -= 1
            smoothed_path.append(path[j])
            i = j
        return smoothed_path
    
    def is_line_free(self, start, end, Map):
        ''' Check if the line between start and end is free of obstacles.'''
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if Map[x1, y1] != 0:  # obstacle
                return False
            if (x1, y1) == (x2, y2):
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x1 += sx
            if err2 < dx:
                err += dx
                y1 += sy
        return True

    def find_closest_point_on_path(self, robot_pose, path):
        ''' Find the closest point on the path to the robot's current position.'''
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(path):
            dist = np.sqrt((point[0] - robot_pose[0])**2 + (point[1] - robot_pose[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx
    
    def find_closest_point_smoothed_path(self, robot_pose, smoothed_path):
        '''Find closest perpendicular point on smoothed path to robot_pose.'''
        min_dist = float('inf')
        closest_point = None
        for i in range(len(smoothed_path) - 1):
            segment_start = smoothed_path[i]
            segment_end = smoothed_path[i + 1]
            proj = self.project_point_onto_line(robot_pose, segment_start, segment_end)
            if self.is_point_on_segment(proj, segment_start, segment_end):
                dist = np.linalg.norm(proj - robot_pose)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = proj
        return closest_point
    
    def project_point_onto_line(self, point, line_start, line_end):
        ''' Project a point onto a line defined by two points.'''
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return np.array(line_start)
        t = np.dot(point_vec, line_vec) / line_len_sq
        projection = np.array(line_start) + t * line_vec
        return projection
    
    def is_point_on_segment(self, point, seg_start, seg_end):
        ''' Check if a point is on the line segment defined by seg_start and seg_end.'''
        crossproduct = (point[1] - seg_start[1]) * (seg_end[0] - seg_start[0]) - (point[0] - seg_start[0]) * (seg_end[1] - seg_start[1])
        if abs(crossproduct) > 1e-6:
            return False  # not collinear
        dotproduct = (point[0] - seg_start[0]) * (seg_end[0] - seg_start[0]) + (point[1] - seg_start[1]) * (seg_end[1] - seg_start[1])
        if dotproduct < 0:
            return False
        squaredlengthba = (seg_end[0] - seg_start[0])**2 + (seg_end[1] - seg_start[1])**2
        if dotproduct > squaredlengthba:
            return False
        return True
    
    def target_point(self, closest_point_on_path, look_ahead_distance, path=None):
        '''Travel a certain distance along a path starting from a point on the path.

        Parameters
        - closest_point_on_path: either an (x,y) coordinate (numpy array or tuple) lying on
          a smoothed path segment, or an integer index into a discrete grid path list.
        - look_ahead_distance: distance (in the same units as path coordinates, e.g. cells)
          to travel forward along the path from the closest point to produce the
          target point.
        - path: required when closest_point_on_path is a coordinate — the ordered list
          of waypoints (list/array of (r,c) or (x,y) pairs) that defines the path.

        Returns
        - target_point: numpy array [x, y] of the point located look_ahead_distance ahead
          along the path. If the lookahead extends beyond the path end, returns the final
          path waypoint (goal).

        Notes
        - Coordinates are treated as (row, col) / (y, x) consistently with the rest of
          this module (i.e. index 0 is the vertical axis).
        '''
        # If the caller passed an integer index, we require a discrete path list
        if isinstance(closest_point_on_path, int):
            if path is None:
                raise ValueError("path must be provided when closest_point_on_path is an index")
            idx = int(closest_point_on_path)
            # start from the path node at idx, walk forward accumulating Euclidean distance
            remaining = float(look_ahead_distance)
            # convert path entries to numpy arrays
            pts = [np.array(p, dtype=float) for p in path]
            if idx >= len(pts):
                return pts[-1]
            # if lookahead is zero or negative, return the current node
            if remaining <= 0:
                return pts[idx]
            for i in range(idx, len(pts) - 1):
                seg_start = pts[i]
                seg_end = pts[i + 1]
                seg_len = np.linalg.norm(seg_end - seg_start)
                if seg_len >= remaining:
                    # interpolate along this segment
                    t = remaining / seg_len if seg_len > 0 else 0.0
                    return seg_start + t * (seg_end - seg_start)
                remaining -= seg_len
            # if we reach here, lookahead extends beyond goal
            return pts[-1]

        # otherwise assume closest_point_on_path is a coordinate on a smoothed path
        cp = np.array(closest_point_on_path, dtype=float)
        if path is None:
            raise ValueError("path must be provided when closest_point_on_path is a coordinate")

        # walk along the path starting from the projection point
        pts = [np.array(p, dtype=float) for p in path]
        # find which segment contains the closest point (or nearest segment)
        seg_idx = None
        proj_point = None
        for i in range(len(pts) - 1):
            s = pts[i]
            e = pts[i + 1]
            proj = self.project_point_onto_line(cp, s, e)
            if self.is_point_on_segment(proj, s, e):
                seg_idx = i
                proj_point = proj
                break
        # if none, find nearest segment by distance
        if seg_idx is None:
            min_dist = float('inf')
            for i in range(len(pts) - 1):
                s = pts[i]
                e = pts[i + 1]
                proj = self.project_point_onto_line(cp, s, e)
                dist = np.linalg.norm(proj - cp)
                if dist < min_dist:
                    min_dist = dist
                    seg_idx = i
                    proj_point = proj

        # now advance from proj_point along the path by look_ahead_distance
        remaining = float(look_ahead_distance)
        # First consume the remainder of current segment
        start_seg_end = pts[seg_idx + 1]
        to_end = np.linalg.norm(start_seg_end - proj_point)
        if remaining <= to_end:
            t = remaining / to_end if to_end > 0 else 0.0
            return proj_point + t * (start_seg_end - proj_point)
        remaining -= to_end
        # advance through subsequent full segments
        for j in range(seg_idx + 1, len(pts) - 1):
            s = pts[j]
            e = pts[j + 1]
            seg_len = np.linalg.norm(e - s)
            if seg_len >= remaining:
                t = remaining / seg_len if seg_len > 0 else 0.0
                return s + t * (e - s)
            remaining -= seg_len
        # if we get here, return the last waypoint
        return pts[-1]

    def display_map(self, Map, path8, SearchStart, SearchGoal, explored8, smoothed_path=None, robot_pose=None, closest_point=None, target_point=None, attractive_vector=None):
        """Display the grid map with semi-transparent overlays for sand and explored cells.

        - free cells (Map == 0) shown as white
        - obstacles (Map == -1) shown as black
        - sand cells (Map > 0) blended with a sandy tint
        - explored cells blended with a light gray overlay
        - path, start, goal drawn solid on top
        """
        h, w = Map.shape

        # Base image: free = white, obstacles = black
        base = np.ones((h, w, 3), dtype=float)
        obs_mask = (Map == -1)
        base[obs_mask] = (0.0, 0.0, 0.0)

        # Sand overlay: Map > 0
        sand_mask = (Map > 0)
        sand_color = np.array([0.92, 0.84, 0.60])
        sand_alpha = 0.5
        if sand_mask.any():
            base[sand_mask] = base[sand_mask] * (1.0 - sand_alpha) + sand_color * sand_alpha

        # Explored overlay (semi-transparent)
        explored_mask = np.zeros((h, w), dtype=bool)
        for cell in explored8:
            explored_mask[cell] = True
        explored_color = np.array([0.6, 0.6, 0.6])
        explored_alpha = 0.35
        if explored_mask.any():
            base[explored_mask] = base[explored_mask] * (1.0 - explored_alpha) + explored_color * explored_alpha

        # Draw path, start, goal on top as solid colors
        for cell in path8:
            base[cell] = (0.0, 1.0, 0.0)  # Path - green

        base[SearchStart] = (0.0, 0.0, 1.0)  # Start - blue
        base[SearchGoal] = (1.0, 0.0, 0.0)   # Goal - red

        plt.figure(figsize=(12, 12))
        plt.imshow(base, interpolation='nearest')
        plt.title('Map with Path, Sand, and Explored Cells')
        plt.axis('off')

        if smoothed_path:
            # Overlay the smoothed path
            for i in range(len(smoothed_path) - 1):
                plt.plot([smoothed_path[i][1], smoothed_path[i + 1][1]],
                         [smoothed_path[i][0], smoothed_path[i + 1][0]], color='cyan', linewidth=2)
                
        if robot_pose is not None:
            theta = robot_pose[2]
            plt.plot(robot_pose[1], robot_pose[0], 'mo', markersize=10, label='Robot Position')
            plt.arrow(robot_pose[1], robot_pose[0],np.cos(theta)*3, np.sin(theta)*3,
                      head_width=1.0, head_length=1.0, fc='m', ec='m', label='Robot direction')
        if closest_point is not None:
            plt.plot(closest_point[1], closest_point[0], 'yx', markersize=10, label='Closest Point on Path')
        if target_point is not None:
            plt.plot(target_point[1], target_point[0], 'ks', markersize=10, label='Target Point')

        if attractive_vector is not None:
                plt.arrow(robot_pose[1], robot_pose[0],
                          attractive_vector[1]*5, attractive_vector[0]*5,
                          head_width=1.0, head_length=1.0, fc='k', ec='k', label='Attractive Vector')
        plt.legend()
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