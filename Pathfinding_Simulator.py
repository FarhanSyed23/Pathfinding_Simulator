import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
import tkinter.font as tkFont
import numpy as np
import heapq
import tensorflow as tf
import time
import random
import threading
import queue
import math

# Define constants for grid cell values
EMPTY = 0
REGULAR_OBSTACLE = 1
AGENT1_START = 2
AGENT1_END = 3
AGENT2_START = 4
AGENT2_END = 5
WEIGHTED_OBSTACLE = 6
DYNAMIC_OBSTACLE = 7

# Node class for pathfinding algorithms
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position  # (row, column)
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost estimate to end
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

# Heuristic functions
def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def euclidean_distance(a, b):
    """Euclidean distance heuristic."""
    return ((b[0] - a[0])**2 + (b[1] - a[1])**2) ** 0.5

# Load scaler parameters
def load_scaler(scaler_mean_path='scaler_mean.npy', scaler_scale_path='scaler_scale.npy'):
    try:
        # Load scaler parameters
        scaler_mean = np.load(scaler_mean_path)
        scaler_scale = np.load(scaler_scale_path)
        print("Scaler parameters loaded successfully.")

        return scaler_mean, scaler_scale
    except Exception as e:
        messagebox.showerror("Scaler Load Error", f"Failed to load scaler: {e}")
        return None, None

# A* Pathfinding Algorithm
def astar(grid, start, end, heuristic_func):
    """
    Performs the A* algorithm to find the shortest path from start to end on the grid.
    Considers weighted obstacles and uses an ML-based heuristic if available.
    """
    start_time = time.perf_counter()

    open_list = []
    closed_set = set()
    start_node = Node(None, start)
    start_node.g = 0
    start_node.h = heuristic_func(start, end)
    start_node.f = start_node.g + start_node.h
    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        closed_set.add(current_node.position)

        if current_node.position == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6  # Convert to microseconds
            return path[::-1], len(closed_set), runtime

        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            node_position = (
                current_node.position[0] + move[0],
                current_node.position[1] + move[1]
            )

            if 0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0]):
                cell_value = grid[node_position[0]][node_position[1]]
                if cell_value in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]:
                    continue  # Untraversable obstacle

                # Determine traversal cost
                traversal_cost = 1
                if cell_value == WEIGHTED_OBSTACLE:
                    traversal_cost = 5  # Weighted obstacle

                neighbor = Node(current_node, node_position)
                neighbor.g = current_node.g + traversal_cost
                neighbor.h = heuristic_func(neighbor.position, end)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor.position in closed_set:
                    continue

                # Check if neighbor is in open_list with a higher g cost
                in_open = False
                for open_node in open_list:
                    if neighbor == open_node[1] and neighbor.g >= open_node[1].g:
                        in_open = True
                        break
                if not in_open:
                    heapq.heappush(open_list, (neighbor.f, neighbor))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # Convert to microseconds
    return [], len(closed_set), runtime

# Rapidly-exploring Random Trees (RRT) Algorithm
def rrt(grid, start, end, heuristic_func, max_iterations=1000, step_size=1):
    """
    Performs the RRT algorithm to find a path from start to end on the grid.
    Incorporates an ML-based heuristic by dynamically adjusting the sampling probability based on the heuristic.
    """
    start_time = time.perf_counter()

    tree = {start: None}
    nodes = [start]

    # Precompute heuristic values for all walkable cells
    heuristic_map = {}
    walkable_points = []
    heuristic_values = []

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]:
                continue  # Skip obstacles
            point = (i, j)
            walkable_points.append(point)
            heuristic_map[point] = heuristic_func(point, end)
            # To avoid division by zero, add a small epsilon
            heuristic_values.append(1 / (heuristic_map[point] + 1e-5))

    heuristic_values = np.array(heuristic_values)
    # Normalize to create a probability distribution
    sampling_prob = heuristic_values / heuristic_values.sum()

    # Convert walkable points to a NumPy array for efficient indexing
    walkable_points = np.array(walkable_points)

    for iteration in range(max_iterations):
        # Sample a random point based on the probability distribution
        rand_index = np.random.choice(len(walkable_points), p=sampling_prob)
        rand_point = tuple(walkable_points[rand_index])

        # Find the nearest node in the tree using Euclidean distance
        nearest_point = min(tree.keys(), key=lambda node: euclidean_distance(node, rand_point))

        # Steer towards the random point by step_size
        theta = math.atan2(rand_point[0] - nearest_point[0], rand_point[1] - nearest_point[1])
        new_pos = (
            nearest_point[0] + step_size * round(math.sin(theta)),
            nearest_point[1] + step_size * round(math.cos(theta))
        )

        # Ensure new_pos is within grid bounds
        new_pos = (
            max(0, min(new_pos[0], len(grid) - 1)),
            max(0, min(new_pos[1], len(grid[0]) - 1))
        )

        # Check if the new position is walkable and not already in the tree
        if grid[new_pos[0]][new_pos[1]] in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]:
            continue
        if new_pos in tree:
            continue

        # Check line of sight between nearest_point and new_pos to prevent crossing dynamic obstacles
        if not has_line_of_sight(grid, nearest_point, new_pos):
            continue

        # Add the new node to the tree
        tree[new_pos] = nearest_point
        nodes.append(new_pos)

        # Check if the new node is close enough to the end
        if euclidean_distance(new_pos, end) <= step_size:
            # Reconstruct the path
            path = [end]
            current = new_pos
            while current != start:
                path.append(current)
                current = tree[current]
            path.append(start)
            path.reverse()

            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6  # Convert to microseconds
            return path, len(tree), runtime

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # Convert to microseconds
    return [], len(tree), runtime  # Return empty path if no path found

# Dijkstra's Algorithm
def dijkstra(grid, start, end):
    """
    Performs Dijkstra's algorithm to find the shortest path from start to end on the grid.
    Considers weighted obstacles.
    """
    start_time = time.perf_counter()

    open_list = []
    closed_set = set()
    start_node = Node(None, start)
    start_node.g = 0
    start_node.f = 0  # In Dijkstra's, f = g
    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        closed_set.add(current_node.position)

        if current_node.position == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6  # Convert to microseconds
            return path[::-1], len(closed_set), runtime

        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            node_position = (
                current_node.position[0] + move[0],
                current_node.position[1] + move[1]
            )

            if 0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0]):
                cell_value = grid[node_position[0]][node_position[1]]
                if cell_value in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]:
                    continue  # Untraversable obstacle

                # Determine traversal cost
                traversal_cost = 1
                if cell_value == WEIGHTED_OBSTACLE:
                    traversal_cost = 5  # Weighted obstacle

                neighbor = Node(current_node, node_position)
                neighbor.g = current_node.g + traversal_cost
                neighbor.f = neighbor.g  # In Dijkstra's, f = g

                if neighbor.position in closed_set:
                    continue

                # Check if neighbor is in open_list with a higher g cost
                in_open = False
                for open_node in open_list:
                    if neighbor == open_node[1] and neighbor.g >= open_node[1].g:
                        in_open = True
                        break
                if not in_open:
                    heapq.heappush(open_list, (neighbor.f, neighbor))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # Convert to microseconds
    return [], len(closed_set), runtime

# Breadth-First Search Algorithm
def bfs(grid, start, end):
    """
    Performs the Breadth-First Search algorithm to find the shortest path from start to end on the grid.
    Ignores traversal costs (for comparison purposes).
    """
    start_time = time.perf_counter()

    if not start or not end:
        end_time = time.perf_counter()
        runtime = (end_time - start_time) * 1e6  # Convert to microseconds
        return [], 0, runtime

    queue_nodes = [Node(None, start)]
    visited = set([start])
    closed_set = set()
    expansions = 0

    while queue_nodes:
        current_node = queue_nodes.pop(0)
        current = current_node.position
        closed_set.add(current)
        expansions += 1

        if current == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6  # Convert to microseconds
            return path[::-1], expansions, runtime

        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            node_position = (
                current[0] + move[0],
                current[1] + move[1]
            )
            if (0 <= node_position[0] < len(grid) and
                0 <= node_position[1] < len(grid[0]) and
                grid[node_position[0]][node_position[1]] not in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE] and
                node_position not in visited):
                visited.add(node_position)
                queue_nodes.append(Node(current_node, node_position))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # Convert to microseconds
    return [], expansions, runtime

# Depth-First Search Algorithm
def dfs(grid, start, end):
    """
    Performs the Depth-First Search algorithm to find a path from start to end on the grid.
    Ignores traversal costs (for comparison purposes).
    """
    start_time = time.perf_counter()

    if not start or not end:
        end_time = time.perf_counter()
        runtime = (end_time - start_time) * 1e6  # Convert to microseconds
        return [], 0, runtime

    stack_nodes = [Node(None, start)]
    visited = set([start])
    closed_set = set()
    expansions = 0

    while stack_nodes:
        current_node = stack_nodes.pop()
        current = current_node.position
        closed_set.add(current)
        expansions += 1

        if current == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6  # Convert to microseconds
            return path[::-1], expansions, runtime

        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            node_position = (
                current[0] + move[0],
                current[1] + move[1]
            )
            if (0 <= node_position[0] < len(grid) and
                0 <= node_position[1] < len(grid[0]) and
                grid[node_position[0]][node_position[1]] not in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE] and
                node_position not in visited):
                visited.add(node_position)
                stack_nodes.append(Node(current_node, node_position))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # Convert to microseconds
    return [], expansions, runtime

# Greedy Best-First Search Algorithm
def greedy_best_first_search(grid, start, end, heuristic_func):
    """
    Performs the Greedy Best-First Search algorithm using the provided heuristic function.
    """
    start_time = time.perf_counter()

    open_list = []
    closed_set = set()
    start_node = Node(None, start)
    start_node.h = heuristic_func(start, end)
    start_node.f = start_node.h
    heapq.heappush(open_list, (start_node.f, start_node))
    expansions = 0

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        closed_set.add(current_node.position)
        expansions += 1

        if current_node.position == end:
            # Reconstruct path
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6
            return path[::-1], expansions, runtime

        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            node_position = (
                current_node.position[0] + move[0],
                current_node.position[1] + move[1]
            )

            if 0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0]):
                cell_value = grid[node_position[0]][node_position[1]]
                if cell_value in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]:
                    continue

                traversal_cost = 1
                if cell_value == WEIGHTED_OBSTACLE:
                    traversal_cost = 5  # Weighted obstacle

                neighbor = Node(current_node, node_position)
                neighbor.g = current_node.g + traversal_cost
                neighbor.h = heuristic_func(neighbor.position, end)
                neighbor.f = neighbor.h  # Only heuristic for Greedy Best-First

                if neighbor.position in closed_set:
                    continue

                # Check if neighbor is in open_list with a higher h cost
                in_open = False
                for open_node in open_list:
                    if neighbor == open_node[1] and neighbor.h >= open_node[1].h:
                        in_open = True
                        break
                if not in_open:
                    heapq.heappush(open_list, (neighbor.f, neighbor))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6
    return [], expansions, runtime

# Modified Theta* Search Algorithm with Enhanced Collision Avoidance
def theta_star(grid, start, end, heuristic_func):
    """
    Implements the Theta* algorithm using the provided heuristic function.
    Enhanced to better handle collision avoidance.
    """
    start_time = time.perf_counter()

    open_list = []
    closed_set = set()
    start_node = Node(None, start)
    start_node.g = 0
    start_node.h = heuristic_func(start, end)
    start_node.f = start_node.g + start_node.h
    start_node.parent = start_node  # Parent of start node is itself
    heapq.heappush(open_list, (start_node.f, start_node))
    expansions = 0

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        if current_node.position in closed_set:
            continue
        closed_set.add(current_node.position)
        expansions += 1

        if current_node.position == end:
            # Reconstruct path
            path = []
            while current_node != current_node.parent:
                path.append(current_node.position)
                current_node = current_node.parent
            path.append(start)
            path.reverse()
            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6
            return path, expansions, runtime

        for move in [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]:
            node_position = (
                current_node.position[0] + move[0],
                current_node.position[1] + move[1]
            )

            if (0 <= node_position[0] < len(grid) and
                0 <= node_position[1] < len(grid[0]) and
                grid[node_position[0]][node_position[1]] not in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]):

                traversal_cost = euclidean_distance(current_node.position, node_position)
                if grid[node_position[0]][node_position[1]] == WEIGHTED_OBSTACLE:
                    traversal_cost *= 5

                neighbor = Node(current_node, node_position)

                # Check if line of sight exists between current_node's parent and neighbor
                if has_line_of_sight(grid, current_node.parent.position, neighbor.position):
                    # Path 2: From parent of current node to neighbor
                    tentative_g = current_node.parent.g + traversal_cost
                    tentative_parent = current_node.parent
                else:
                    # Path 1: From current node to neighbor
                    tentative_g = current_node.g + traversal_cost
                    tentative_parent = current_node

                if neighbor.position in closed_set:
                    continue

                if tentative_g < neighbor.g or neighbor.position not in [n.position for f, n in open_list]:
                    neighbor.g = tentative_g
                    neighbor.h = heuristic_func(neighbor.position, end)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = tentative_parent
                    heapq.heappush(open_list, (neighbor.f, neighbor))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6
    return [], expansions, runtime

def has_line_of_sight(grid, start, end):
    """
    Determines if there is a direct line of sight between start and end.
    Uses Bresenham's Line Algorithm.
    Corrected to use grid[y][x] indexing and includes boundary checks.
    """
    x0, y0 = start[1], start[0]
    x1, y1 = end[1], end[0]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2

    for _ in range(n):
        # Boundary check to prevent IndexError
        if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):
            return False
        if grid[y][x] in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]:
            return False
        if (x, y) == (x1, y1):
            return True
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return False

# Modified Jump Point Search (JPS) Algorithm with Enhanced Collision Avoidance
def jump_point_search(grid, start, end, heuristic_func):
    """
    Performs the Jump Point Search (JPS) algorithm to find a path from start to end on the grid.
    Enhanced to better handle collision avoidance by integrating dynamic obstacles into forced neighbor checks.
    Uses 8-directional movement and supports collision avoidance.
    """
    start_time = time.perf_counter()

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    expansions = 0

    while open_list:
        current_f, current = heapq.heappop(open_list)

        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6  # microseconds
            return path, expansions, runtime

        expansions += 1

        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = jump(grid, current, direction, end)
            if neighbor:
                tentative_g = g_score[current] + euclidean_distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic_func(neighbor, end)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # microseconds
    return [], expansions, runtime

def jump(grid, current, direction, end):
    """
    Jump function for JPS with enhanced collision avoidance.
    """
    x, y = current
    dx, dy = direction

    while True:
        x += dx
        y += dy

        if not (0 <= x < len(grid) and 0 <= y < len(grid[0])):
            return None
        if grid[x][y] in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]:
            return None
        if (x, y) == end:
            return (x, y)
        if has_forced_neighbors(grid, (x, y), direction):
            return (x, y)
        # Diagonal movements need to check for horizontal and vertical jump points
        if dx != 0 and dy != 0:
            if (jump(grid, (x - dx, y), (dx, 0), end) or
                jump(grid, (x, y - dy), (0, dy), end)):
                return (x, y)

def has_forced_neighbors(grid, position, direction):
    """
    Checks if the current position has any forced neighbors.
    Enhanced boundary checks to prevent IndexError.
    """
    x, y = position
    dx, dy = direction

    # Diagonal movement
    if dx != 0 and dy != 0:
        # Check for forced neighbors on the horizontal and vertical axes
        if ((is_blocked(grid, x - dx, y + dy) and not is_blocked(grid, x - dx, y)) or
            (is_blocked(grid, x + dx, y - dy) and not is_blocked(grid, x, y - dy))):
            return True
    else:
        # Horizontal movement
        if dx != 0:
            if ((is_blocked(grid, x + dx, y + 1) and not is_blocked(grid, x, y + 1)) or
                (is_blocked(grid, x + dx, y - 1) and not is_blocked(grid, x, y - 1))):
                return True
        else:
            # Vertical movement
            if ((is_blocked(grid, x + 1, y + dy) and not is_blocked(grid, x + 1, y)) or
                (is_blocked(grid, x - 1, y + dy) and not is_blocked(grid, x - 1, y))):
                return True
    return False

def is_blocked(grid, x, y):
    """
    Helper function to check if a cell is blocked.
    """
    if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
        return True
    return grid[x][y] in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]

# D* Lite Algorithm
class DStarLite:
    def __init__(self, grid, start, goal, heuristic_func=heuristic):
        self.grid = grid
        self.s_start = start
        self.s_goal = goal
        self.k_m = 0
        self.rhs = {}
        self.g = {}
        self.U = []
        self.OPEN = set()
        self.s_last = self.s_start
        self.heuristic_func = heuristic_func  # Added heuristic function
        self.init()

    def init(self):
        self.rhs[self.s_goal] = 0
        self.g[self.s_goal] = float('inf')
        heapq.heappush(self.U, (self.calculate_key(self.s_goal), self.s_goal))
        self.OPEN.add(self.s_goal)

    def calculate_key(self, s):
        g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_rhs + self.heuristic_func(self.s_start, s) + self.k_m, g_rhs)

    def update_vertex(self, u):
        if u != self.s_goal:
            min_rhs = float('inf')
            for s in self.get_successors(u):
                cost = self.cost(u, s)
                min_rhs = min(min_rhs, self.g.get(s, float('inf')) + cost)
            self.rhs[u] = min_rhs
        if u in self.OPEN:
            self.OPEN.discard(u)
            self.U = [item for item in self.U if item[1] != u]
            heapq.heapify(self.U)
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heapq.heappush(self.U, (self.calculate_key(u), u))
            self.OPEN.add(u)

    def compute_shortest_path(self):
        expansions = 0
        while self.U and (self.U[0][0] < self.calculate_key(self.s_start) or
                          self.rhs.get(self.s_start, float('inf')) != self.g.get(self.s_start, float('inf'))):
            k_old, u = heapq.heappop(self.U)
            self.OPEN.discard(u)
            k_new = self.calculate_key(u)
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
                self.OPEN.add(u)
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                expansions += 1
                for s in self.get_predecessors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                expansions += 1
                for s in self.get_predecessors(u) + [u]:
                    self.update_vertex(s)
        return expansions

    def get_successors(self, s):
        successors = []
        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            node_position = (s[0] + move[0], s[1] + move[1])
            if (0 <= node_position[0] < len(self.grid) and
                0 <= node_position[1] < len(self.grid[0]) and
                self.grid[node_position[0]][node_position[1]] not in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]):
                successors.append(node_position)
        return successors

    def get_predecessors(self, s):
        return self.get_successors(s)

    def cost(self, a, b):
        if self.grid[b[0]][b[1]] == WEIGHTED_OBSTACLE:
            return 5
        else:
            return 1

    def reconstruct_path(self):
        path = []
        s = self.s_start
        if self.g.get(s, float('inf')) == float('inf'):
            return path
        while s != self.s_goal:
            path.append(s)
            min_rhs = float('inf')
            next_s = None
            for s_prime in self.get_successors(s):
                cost = self.cost(s, s_prime)
                g_rhs = self.g.get(s_prime, float('inf')) + cost
                if g_rhs < min_rhs:
                    min_rhs = g_rhs
                    next_s = s_prime
            if next_s is None:
                return []  # No path found
            s = next_s
        path.append(self.s_goal)
        return path

def run_d_star_lite(grid, start, goal, heuristic_func=heuristic):
    start_time = time.perf_counter()  # Start time
    dstar = DStarLite(grid, start, goal, heuristic_func)
    expansions = dstar.compute_shortest_path()
    path = dstar.reconstruct_path()
    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # Convert to microseconds
    return path, expansions, runtime

# Beam Search Algorithm
def beam_search(grid, start, end, heuristic_func, beam_width=3):
    """
    Performs Beam Search to find a path from start to end on the grid.
    Beam Search explores the most promising nodes based on a heuristic, limited by beam_width.
    """
    start_time = time.perf_counter()

    # Initialize the beam with the start node
    beam = [Node(None, start)]
    visited = set()
    visited.add(start)
    expansions = 0

    while beam:
        next_beam = []
        for current_node in beam:
            expansions += 1
            if current_node.position == end:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                end_time = time.perf_counter()
                runtime = (end_time - start_time) * 1e6  # microseconds
                return path[::-1], expansions, runtime

            for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                node_position = (
                    current_node.position[0] + move[0],
                    current_node.position[1] + move[1]
                )

                if (0 <= node_position[0] < len(grid) and
                    0 <= node_position[1] < len(grid[0]) and
                    grid[node_position[0]][node_position[1]] not in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE] and
                    node_position not in visited):

                    traversal_cost = 1
                    if grid[node_position[0]][node_position[1]] == WEIGHTED_OBSTACLE:
                        traversal_cost = 5  # Weighted obstacle

                    neighbor = Node(current_node, node_position)
                    neighbor.g = current_node.g + traversal_cost
                    neighbor.h = heuristic_func(neighbor.position, end)
                    neighbor.f = neighbor.g + neighbor.h

                    next_beam.append(neighbor)
                    visited.add(node_position)

        if not next_beam:
            break

        # Sort next_beam based on heuristic (h) and select top beam_width nodes
        next_beam.sort(key=lambda node: node.h)
        beam = next_beam[:beam_width]

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # microseconds
    return [], expansions, runtime

def run_rrt(grid, start, end, heuristic_func, max_iterations=1000, step_size=1):
    """
    Wrapper function to run RRT algorithm and return path, visited nodes, and runtime.
    """
    path, visited, runtime = rrt(grid, start, end, heuristic_func, max_iterations, step_size)
    return path, visited, runtime

# PathfindingGUI Class
class PathfindingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Multi-Agent Pathfinding Simulator")
        master.resizable(True, True)  # Allow window resizing

        # Set window size
        master.geometry("1145x790")  # Increased width to accommodate more algorithms

        # Grid configuration
        self.grid_size = 15  # 15x15 grid
        self.cell_size = 40  # Each cell is 40x40 pixels

        # Initialize grid
        self.grid = [[EMPTY for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Agents configuration
        self.agents = []
        self.max_agents = 2  # Support up to 2 agents

        # Colors
        self.colors = {
            "regular_obstacle": "black",
            "weighted_obstacle": "orange",
            "dynamic_obstacle": "red",
            "agent1_start": "green",
            "agent1_end": "light green",
            "agent2_start": "blue",
            "agent2_end": "light blue",
            "path_agent1": "purple",
            "path_agent2": "cyan",
            "agent1": "purple",
            "agent2": "cyan"
        }

        # Initialize dynamic obstacles
        self.dynamic_obstacles = []
        self.max_dynamic_obstacles = 3  # Maximum number of dynamic obstacles
        self.max_dynamic_obstacle_resets = 3  # Maximum resets per obstacle

        # Load scaler parameters
        self.scaler_mean, self.scaler_scale = load_scaler()
        if self.scaler_mean is None:
            master.destroy()
            return

        # Model path
        self.model_path = 'heuristic_model.tflite'

        # Initialize time variables for agents BEFORE creating widgets
        self.agent1_time_var = tk.StringVar()
        self.agent2_time_var = tk.StringVar()
        self.agent1_time_var.set("Agent 1 Time: 0.000000 s")
        self.agent2_time_var.set("Agent 2 Time: 0.000000 s")

        # Create GUI components
        self.create_widgets()

        # Initialize dynamic obstacles AFTER widgets are created
        self.initialize_dynamic_obstacles()

        # Pathfinding queue
        self.queue = queue.Queue()

        # Start processing the queue
        self.master.after(100, self.process_queue)

        # Simulation variables
        self.simulation_running = False

        # Obstacle management
        self.max_obstacles = int(self.grid_size * self.grid_size * 0.3)  # Max 30% of grid can be obstacles

    def create_widgets(self):
        # Top frame for controls (First Row)
        control_frame = tk.Frame(self.master)
        control_frame.pack(side=tk.TOP, pady=10, fill=tk.X)

        # Algorithm Selection Dropdown
        algo_label = tk.Label(control_frame, text="Select Algorithm:", font=('Helvetica', 12))
        algo_label.pack(side=tk.LEFT, padx=(5, 0))  # Adjust padding for better alignment

        self.selected_algorithm = tk.StringVar()
        self.selected_algorithm.set("A*")  # Default value
        self.informed_algorithms = ["A*", "Greedy Best-First", "Theta*", "Beam Search", "Jump Point Search", "D* Lite", "RRT"]
        self.uninformed_algorithms = ["BFS", "DFS", "Dijkstra"]
        self.algo_menu = tk.OptionMenu(control_frame, self.selected_algorithm, *self.informed_algorithms, command=self.on_algorithm_change)
        self.algo_menu.config(width=25)
        self.algo_menu.pack(side=tk.LEFT, padx=5)

        # Collision avoidance option
        self.collision_var = tk.BooleanVar()
        self.collision_var.set(True)  # Default to collision avoidance enabled
        self.collision_checkbox = tk.Checkbutton(control_frame, text="Avoid Path Collisions", variable=self.collision_var, font=('Helvetica', 12))
        self.collision_checkbox.pack(side=tk.LEFT, padx=5)

        # Run All button
        self.run_button = tk.Button(control_frame, text="Run All", command=self.run_pathfinding_thread, bg='light green', width=12)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Start Simulation button
        self.simulate_button = tk.Button(control_frame, text="Start Simulation", command=self.start_simulation, bg='light blue', width=18)
        self.simulate_button.pack(side=tk.LEFT, padx=5)

        # Stop Simulation button
        self.stop_button = tk.Button(control_frame, text="Stop Simulation", command=self.stop_simulation, bg='light coral', width=18, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Export Log button
        export_button = tk.Button(control_frame, text="Export Log", command=self.export_log, bg='light yellow', width=12)
        export_button.pack(side=tk.LEFT, padx=5)

        # Reset button
        reset_button = tk.Button(control_frame, text="Reset", command=self.reset_grid, bg='light gray', width=12)
        reset_button.pack(side=tk.LEFT, padx=5)

        # Algorithm Category Selection Frame (Second Row)
        category_frame = tk.Frame(self.master)
        category_frame.pack(side=tk.TOP, pady=5, fill=tk.X)

        category_label = tk.Label(category_frame, text="Select Algorithm Category:", font=('Helvetica', 12))
        category_label.pack(side=tk.LEFT, padx=5)

        self.algorithm_category = tk.StringVar()
        self.algorithm_category.set("Informed Search")
        informed_rb = tk.Radiobutton(category_frame, text="Informed Search", variable=self.algorithm_category,
                                     value="Informed Search", command=self.update_algorithm_options, font=('Helvetica', 12))
        informed_rb.pack(side=tk.LEFT, padx=5)

        uninformed_rb = tk.Radiobutton(category_frame, text="Uninformed Search", variable=self.algorithm_category,
                                       value="Uninformed Search", command=self.update_algorithm_options, font=('Helvetica', 12))
        uninformed_rb.pack(side=tk.LEFT, padx=5)

        help_button = tk.Button(category_frame, text="Help", command=self.open_help_window, bg='light pink', width=10)
        help_button.pack(side=tk.LEFT, padx=10)


        # Legend
        legend_frame = tk.Frame(self.master)
        legend_frame.pack(side=tk.TOP, padx=10, fill=tk.X)

        legends = [
            ("Agent1 Start", self.colors["agent1_start"]),
            ("Agent1 End", self.colors["agent1_end"]),
            ("Agent2 Start", self.colors["agent2_start"]),
            ("Agent2 End", self.colors["agent2_end"]),
            ("Regular Obstacle", self.colors["regular_obstacle"]),
            ("Weighted Obstacle", self.colors["weighted_obstacle"]),
            ("Dynamic Obstacle", self.colors["dynamic_obstacle"]),
            ("Agent1 Path", self.colors["path_agent1"]),
            ("Agent2 Path", self.colors["path_agent2"])
        ]

        # Arrange legends in multiple rows
        cols = 3
        for idx, (name, color) in enumerate(legends):
            # Set text color based on background color
            if color == 'black' or name in ["Agent2 Start", "Dynamic Obstacle", "Agent1 Path"]:
                text_color = 'white'
            else:
                text_color = 'black'

            lbl = tk.Label(legend_frame, text=name, bg=color, fg=text_color, width=20, anchor='w', relief='solid')
            lbl.grid(row=idx // cols, column=idx % cols, padx=2, pady=2)

        # Main frame for canvas and status box
        main_frame = tk.Frame(self.master)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left frame for status box and time labels
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)

        # Right frame for canvas
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for grid
        self.canvas = tk.Canvas(right_frame, width=self.grid_size * self.cell_size, height=self.grid_size * self.cell_size, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.left_click)   # Left-click for obstacles and agent positions
        self.canvas.bind("<Button-3>", self.right_click)  # Right-click for weighted obstacles

        # Initialize grid visualization
        self.draw_grid()

        # Status area
        status_label = tk.Label(left_frame, text="Status:", font=('Helvetica', 12))
        status_label.pack(anchor='nw')

        # Time labels for agents
        time_frame = tk.Frame(left_frame)
        time_frame.pack(anchor='nw', pady=5)

        agent1_time_label = tk.Label(time_frame, textvariable=self.agent1_time_var, font=('Helvetica', 10), fg='blue')
        agent1_time_label.pack(anchor='w')

        agent2_time_label = tk.Label(time_frame, textvariable=self.agent2_time_var, font=('Helvetica', 10), fg='blue')
        agent2_time_label.pack(anchor='w')

        # Status text box
        self.status_text = scrolledtext.ScrolledText(left_frame, width=60, height=35, state='disabled')
        self.status_text.pack(fill=tk.BOTH, expand=True)


    def open_help_window(self):
        """Open a new window displaying help and instructions with formatted text."""
        help_window = tk.Toplevel(self.master)
        help_window.title("Help & Instructions")
        help_window.geometry("700x700")  # Adjust size as needed
        help_window.resizable(True, True)

        # Add a scrollbar
        scrollbar = tk.Scrollbar(help_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a Text widget
        help_text = tk.Text(help_window, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=('Helvetica', 10))
        help_text.pack(expand=True, fill=tk.BOTH)
        scrollbar.config(command=help_text.yview)

        # Define custom fonts
        bold_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
        italic_font = tkFont.Font(family="Helvetica", size=10, slant="italic")
        heading_font = tkFont.Font(family="Helvetica", size=14, weight="bold")
        subheading_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
        
        # Configure tags for styling
        help_text.tag_configure("heading", font=heading_font, foreground="dark blue")
        help_text.tag_configure("subheading", font=subheading_font, foreground="blue")
        help_text.tag_configure("bold", font=bold_font)
        help_text.tag_configure("italic", font=italic_font, foreground="dark red")
        help_text.tag_configure("important", font=italic_font, foreground="red")
        help_text.tag_configure("bullet", font=('Helvetica', 10), lmargin1=20, lmargin2=40)
        help_text.tag_configure("number", font=('Helvetica', 10), lmargin1=40, lmargin2=60)

        # Define the help content with tags
        instructions = [
            ("Advanced Multi-Agent Pathfinding Simulator - Help & Instructions\n\n", "heading"),
            ("---\n\n", ""),
            
            ("Running the Simulation\n\n", "subheading"),
            
            ("1. Assigning Agents:\n", "number"),
            ("   - Left-Click on the grid to set the Start and End points for up to two agents.\n", "bullet"),
            ("   - The first left-click assigns the Start position for Agent 1 (green circle).\n", "bullet"),
            ("   - The second left-click assigns the End position for Agent 1 (light green circle).\n", "bullet"),
            ("   - Similarly, the third and fourth left-clicks set the Start and End for Agent 2 (blue and light blue circles).\n\n", "bullet"),
            
            ("2. Adding Obstacles:\n", "number"),
            ("   - Left-Click on any grid cell to add or remove a Regular Obstacle (black).\n", "bullet"),
            ("   - Right-Click on any grid cell to add or remove a Weighted Obstacle (orange).\n", "bullet"),
            ("     - Weighted Obstacles have a higher traversal cost, making the path longer and more resource-consuming.\n\n", "bullet"),
            
            ("3. Dynamic Obstacles:\n", "number"),
            ("   - Red cells represent Dynamic Obstacles that move randomly across the grid.\n", "bullet"),
            ("   - These obstacles can reset or be removed if they become stuck after several attempts.\n\n", "bullet"),
            
            ("4. Selecting Algorithms:\n", "number"),
            ("   - Choose between Informed Search and Uninformed Search using the radio buttons.\n", "bullet"),
            ("   - From the dropdown menu, select the desired pathfinding algorithm (e.g., A*, BFS, DFS).\n\n", "bullet"),
            
            ("5. Collision Avoidance:\n", "number"),
            ("   - Enable or disable collision avoidance using the \"Avoid Path Collisions\" checkbox.\n", "bullet"),
            ("   - Note: Collision avoidance is disabled for Theta* and Jump Point Search (JPS) algorithms as they inherently handle path optimization differently.\n\n", "bullet"),
            
            ("6. Running the Pathfinding:\n", "number"),
            ("   - Click the \"Run All\" button to compute paths for all agents based on the selected algorithms and current grid setup.\n\n", "bullet"),
            
            ("7. Simulation Controls:\n", "number"),
            ("   - Start Simulation: Begins dynamic obstacle movement and periodically updates paths.\n", "bullet"),
            ("   - Stop Simulation: Halts dynamic obstacle movement and path recalculations.\n", "bullet"),
            ("   - Reset: Clears the entire grid, removing all obstacles and agent assignments.\n", "bullet"),
            ("   - Export Log: Saves the status log detailing pathfinding outcomes and performance metrics to a `.txt` file.\n\n", "bullet"),
            
            ("---\n\n", ""),
            
            ("Understanding Obstacles and Their Colors\n\n", "subheading"),
            
            ("- White: Empty cells; traversable with no additional cost.\n", "bullet"),
            ("- Black: Regular Obstacles; impassable barriers blocking the path.\n", "bullet"),
            ("- Orange: Weighted Obstacles; traversable but with increased cost, influencing path choice.\n", "bullet"),
            ("- Red: Dynamic Obstacles; moving barriers that change position over time, adding unpredictability.\n", "bullet"),
            ("- Green Circle: Agent 1's Start position.\n", "bullet"),
            ("- Light Green Circle: Agent 1's End position.\n", "bullet"),
            ("- Blue Circle: Agent 2's Start position.\n", "bullet"),
            ("- Light Blue Circle: Agent 2's End position.\n", "bullet"),
            ("- Purple Lines: Agent 1's computed path.\n", "bullet"),
            ("- Cyan Lines: Agent 2's computed path.\n\n", "bullet"),
            
            ("---\n\n", ""),
            
            ("Using Mouse Clicks for Obstacles\n\n", "subheading"),
            
            ("- Left-Click:\n", "bullet"),
            ("  - Add/Remove Regular Obstacle: Toggles a cell between empty and a regular obstacle.\n", "bullet"),
            ("  - Set Agent Positions: Assigns start and end points for agents if they haven't been set already.\n\n", "bullet"),
            
            ("- Right-Click:\n", "bullet"),
            ("  - Add/Remove Weighted Obstacle: Toggles a cell between empty and a weighted obstacle.\n", "bullet"),
            ("  - Convert Regular to Weighted: Right-clicking a regular obstacle converts it into a weighted obstacle.\n\n", "bullet"),
            
            ("---\n\n", ""),
            
            ("Algorithm Selection and Collision Avoidance\n\n", "subheading"),
            
            ("- Algorithm Categories:\n", "bullet"),
            ("  - Informed Search: Utilizes heuristic functions to optimize pathfinding (e.g., A*, Greedy Best-First).\n", "bullet"),
            ("  - Uninformed Search: Explores the grid without heuristic guidance (e.g., BFS, DFS).\n\n", "bullet"),
            
            ("- Collision Avoidance:\n", "bullet"),
            ("  - When enabled, the simulator avoids overlapping paths between agents, ensuring they do not collide.\n", "bullet"),
            ("  - Limitations: Algorithms like Theta* and Jump Point Search (JPS) do not support collision avoidance in this simulator due to their inherent path optimization mechanisms.\n\n", "bullet"),
            
            ("---\n\n", ""),
            
            ("Log Description\n\n", "subheading"),
            
            ("- Path Found:\n", "bullet"),
            ("  - Path Length: Number of steps from start to end.\n", "bullet"),
            ("  - Visited Nodes: Total nodes explored during the search.\n", "bullet"),
            ("  - Runtime: Time taken to compute the path (in microseconds).\n\n", "bullet"),
            
            ("- No Path Found:\n", "bullet"),
            ("  - Indicates that no viable path exists between the agent's start and end points given the current grid configuration.\n\n", "bullet"),
            
            ("- Errors:\n", "bullet"),
            ("  - Any issues encountered during pathfinding (e.g., missing agent positions, algorithm failures) are logged for user awareness.\n\n", "bullet"),
            
            ("---\n\n", ""),
            
            ("Additional Tips\n\n", "subheading"),
            
            ("- Resetting the Grid: Use the \"Reset\" button to clear all configurations and start fresh.\n", "bullet"),
            ("- Exporting Logs: Regularly export logs to monitor performance metrics and pathfinding outcomes.\n", "bullet"),
            ("- Dynamic Obstacles: Observe how moving obstacles influence agent paths and adjust strategies accordingly.\n\n", "bullet"),
            
            ("---\n\n", ""),
            
            ("Enjoy navigating through your pathfinding simulations! If you encounter any issues or have suggestions, feel free to reach out.", "important")
        ]

        # Insert the help content with appropriate tags
        for text, tag in instructions:
            if tag:
                help_text.insert(tk.END, text, tag)
            else:
                help_text.insert(tk.END, text)

        help_text.config(state='disabled')  # Make the text read-only



    def update_algorithm_options(self):
        """Update the algorithm options based on the selected category."""
        category = self.algorithm_category.get()
        menu = self.algo_menu['menu']
        menu.delete(0, 'end')
        if category == "Informed Search":
            for algo in self.informed_algorithms:
                menu.add_command(label=algo, command=lambda value=algo: self.selected_algorithm.set(value))
            # Reset algorithm selection to first informed algorithm
            self.selected_algorithm.set(self.informed_algorithms[0])
            self.update_status("Switched to Informed Search algorithms.\n")
        else:
            for algo in self.uninformed_algorithms:
                menu.add_command(label=algo, command=lambda value=algo: self.selected_algorithm.set(value))
            # Reset algorithm selection to first uninformed algorithm
            self.selected_algorithm.set(self.uninformed_algorithms[0])
            self.update_status("Switched to Uninformed Search algorithms.\n")

        # After updating algorithm options, handle collision avoidance checkbox
        self.handle_collision_checkbox()

    def on_algorithm_change(self, value):
        """Handle changes in the algorithm selection dropdown."""
        self.handle_collision_checkbox()

    def handle_collision_checkbox(self):
        """Enable or disable the collision avoidance checkbox based on selected algorithm."""
        algo = self.selected_algorithm.get()
        if algo in ["Theta*", "Jump Point Search"]:
            # Disable collision avoidance and uncheck it
            if self.collision_var.get():
                self.collision_var.set(False)
            self.collision_checkbox.config(state=tk.DISABLED)
            self.update_status(f"Collision avoidance is not supported with the {algo} algorithm.\n")
        else:
            # Enable collision avoidance
            self.collision_checkbox.config(state=tk.NORMAL)
            self.update_status(f"Collision avoidance is enabled for the {algo} algorithm.\n")


    def draw_grid(self):
        self.canvas.delete("all")  # Clear any existing drawings
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                fill_color = 'white'
                if self.grid[i][j] == REGULAR_OBSTACLE:
                    fill_color = self.colors["regular_obstacle"]
                elif self.grid[i][j] == WEIGHTED_OBSTACLE:
                    fill_color = self.colors["weighted_obstacle"]
                elif self.grid[i][j] == DYNAMIC_OBSTACLE:
                    fill_color = self.colors["dynamic_obstacle"]
                elif self.grid[i][j] in [AGENT1_START, AGENT1_END, AGENT2_START, AGENT2_END]:
                    fill_color = 'white'
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline='gray')

        # Draw dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            for idx, pos in enumerate(obstacle['positions']):
                i, j = pos
                # Differentiate head and body
                if idx == 0:
                    color = "dark red"
                else:
                    color = self.colors["dynamic_obstacle"]
                self.update_cell(i, j, color)

        # Draw agents' start and end points
        for agent_id, agent in enumerate(self.agents):
            if agent.get('start'):
                self.draw_circle(agent['start'], self.colors["agent1_start"] if agent_id == 0 else self.colors["agent2_start"], tag=f"agent_start_{agent_id}")
            if agent.get('end'):
                self.draw_circle(agent['end'], self.colors["agent1_end"] if agent_id == 0 else self.colors["agent2_end"], tag=f"agent_end_{agent_id}")

    def update_cell(self, i, j, color):
        x1 = j * self.cell_size
        y1 = i * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        # Redraw the cell with the new color
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray')

    def draw_circle(self, position, color, tag=None):
        """Draw a circle at the given grid position."""
        i, j = position
        x = j * self.cell_size + self.cell_size / 2
        y = i * self.cell_size + self.cell_size / 2
        radius = 10 if tag and ("start" in tag or "end" in tag) else 5
        if tag:
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=color, outline='', tags=tag
            )
        else:
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=color, outline=''
            )

    def left_click(self, event):
        """Handle left-click to add/remove regular obstacles or set agent start/end."""
        j = event.x // self.cell_size
        i = event.y // self.cell_size

        if i >= self.grid_size or j >= self.grid_size:
            return

        clicked_pos = (i, j)

        # Check if clicked on any agent's start or end
        for agent_id, agent in enumerate(self.agents):
            if agent.get('start') == clicked_pos:
                self.remove_agent_start(agent_id)
                return
            if agent.get('end') == clicked_pos:
                self.remove_agent_end(agent_id)
                return

        # Assign start/end points for agents
        for agent_id in range(self.max_agents):
            if len(self.agents) <= agent_id:
                self.agents.append({'start': None, 'end': None, 'path': []})
            agent = self.agents[agent_id]
            if not agent['start']:
                self.set_agent_start(agent_id, clicked_pos)
                return
            elif not agent['end'] and clicked_pos != agent['start']:
                self.set_agent_end(agent_id, clicked_pos)
                return

        # After all agents have start and end, toggle regular obstacles
        if self.grid[i][j] == EMPTY:
            if self.grid[i][j] != DYNAMIC_OBSTACLE and not self.is_agent_position(clicked_pos):
                self.grid[i][j] = REGULAR_OBSTACLE
                self.update_cell(i, j, self.colors["regular_obstacle"])
                self.update_status(f"Added Regular Obstacle at {clicked_pos}\n")
        elif self.grid[i][j] == REGULAR_OBSTACLE:
            self.grid[i][j] = EMPTY
            self.update_cell(i, j, 'white')
            self.update_status(f"Removed Regular Obstacle from {clicked_pos}\n")
        elif self.grid[i][j] == WEIGHTED_OBSTACLE:
            # Allow removal of weighted obstacles during simulation
            self.grid[i][j] = EMPTY
            self.update_cell(i, j, 'white')
            self.update_status(f"Removed Weighted Obstacle from {clicked_pos}\n")

    def right_click(self, event):
        """Handle right-click to add/remove weighted obstacles."""
        j = event.x // self.cell_size
        i = event.y // self.cell_size

        if i >= self.grid_size or j >= self.grid_size:
            return

        clicked_pos = (i, j)

        # Prevent placing weighted obstacles on agent start/end or dynamic obstacles
        if self.is_agent_position(clicked_pos) or self.grid[i][j] == DYNAMIC_OBSTACLE:
            return

        # Toggle weighted obstacle
        if self.grid[i][j] == WEIGHTED_OBSTACLE:
            self.grid[i][j] = EMPTY
            self.update_cell(i, j, 'white')
            self.update_status(f"Removed Weighted Obstacle from {clicked_pos}\n")
        elif self.grid[i][j] == EMPTY:
            self.grid[i][j] = WEIGHTED_OBSTACLE
            self.update_cell(i, j, self.colors["weighted_obstacle"])
            self.update_status(f"Added Weighted Obstacle at {clicked_pos}\n")
        elif self.grid[i][j] == REGULAR_OBSTACLE:
            # Convert regular obstacle to weighted
            self.grid[i][j] = WEIGHTED_OBSTACLE
            self.update_cell(i, j, self.colors["weighted_obstacle"])
            self.update_status(f"Converted Regular Obstacle to Weighted at {clicked_pos}\n")

    def is_agent_position(self, pos):
        """Check if the position is assigned to any agent's start or end."""
        for agent in self.agents:
            if agent.get('start') == pos or agent.get('end') == pos:
                return True
        return False

    def set_agent_start(self, agent_id, position):
        """Set the start position for an agent."""
        # Check if position is already occupied
        if self.is_agent_position(position):
            self.update_status(f"Position {position} is already occupied by another agent's start/end.\n")
            return

        self.agents[agent_id]['start'] = position
        i, j = position
        self.grid[i][j] = AGENT1_START if agent_id == 0 else AGENT2_START
        color = self.colors["agent1_start"] if agent_id == 0 else self.colors["agent2_start"]
        self.update_cell(i, j, color)
        self.draw_circle(position, color, tag=f"agent_start_{agent_id}")
        self.update_status(f"Agent {agent_id + 1} Start set at {position}\n")
        self.handle_collision_checkbox()  # Update collision checkbox based on new algorithm selection

    def set_agent_end(self, agent_id, position):
        """Set the end position for an agent."""
        # Check if position is already occupied
        if self.is_agent_position(position):
            self.update_status(f"Position {position} is already occupied by another agent's start/end.\n")
            return

        self.agents[agent_id]['end'] = position
        i, j = position
        self.grid[i][j] = AGENT1_END if agent_id == 0 else AGENT2_END
        color = self.colors["agent1_end"] if agent_id == 0 else self.colors["agent2_end"]
        self.update_cell(i, j, color)
        self.draw_circle(position, color, tag=f"agent_end_{agent_id}")
        self.update_status(f"Agent {agent_id + 1} End set at {position}\n")
        self.handle_collision_checkbox()  # Update collision checkbox based on new algorithm selection

    def remove_agent_start(self, agent_id):
        """Remove the start position for an agent."""
        pos = self.agents[agent_id]['start']
        if not pos:
            return
        i, j = pos
        self.grid[i][j] = EMPTY
        self.update_cell(i, j, 'white')
        self.canvas.delete(f"agent_start_{agent_id}")
        self.agents[agent_id]['start'] = None
        self.agents[agent_id]['path'] = []
        self.clear_path(agent_id)  # Clear the path from the GUI
        self.update_status(f"Agent {agent_id + 1} Start removed from {pos}\n")
        if self.simulation_running:
            self.update_status(f"Need starting point for Agent {agent_id + 1}\n")

    def remove_agent_end(self, agent_id):
        """Remove the end position for an agent."""
        pos = self.agents[agent_id]['end']
        if not pos:
            return
        i, j = pos
        self.grid[i][j] = EMPTY
        self.update_cell(i, j, 'white')
        self.canvas.delete(f"agent_end_{agent_id}")
        self.agents[agent_id]['end'] = None
        self.agents[agent_id]['path'] = []
        self.clear_path(agent_id)  # Clear the path from the GUI
        self.update_status(f"Agent {agent_id + 1} End removed from {pos}\n")
        if self.simulation_running:
            self.update_status(f"Need ending point for Agent {agent_id + 1}\n")

    def run_pathfinding_thread(self):
        """Start the pathfinding process in a separate thread."""
        threading.Thread(target=self.run_pathfinding, daemon=True).start()

    def run_pathfinding(self):
        """Run pathfinding for all agents with or without collision avoidance."""
        category = self.algorithm_category.get()
        algorithm = self.selected_algorithm.get()
        avoid_collisions = self.collision_var.get()

        agent_paths = []  # To store paths of all agents

        for agent_id, agent in enumerate(self.agents):
            start = agent.get('start')
            end = agent.get('end')

            if not start or not end:
                # Clear the agent's path if it exists
                if agent['path']:
                    self.agents[agent_id]['path'] = []
                    self.clear_path(agent_id)
                # Log the missing start/end point
                if not start and not end:
                    self.queue.put(("log", f"Need starting and ending points for Agent {agent_id + 1}\n"))
                elif not start:
                    self.queue.put(("log", f"Need starting point for Agent {agent_id + 1}\n"))
                elif not end:
                    self.queue.put(("log", f"Need ending point for Agent {agent_id + 1}\n"))
                continue

            start = agent['start']
            end = agent['end']

            # Prepare grid for pathfinding
            grid_copy = [row[:] for row in self.grid]  # Deep copy

            # Treat other agents' start and end points as obstacles
            for other_id, other_agent in enumerate(self.agents):
                if other_id != agent_id:
                    other_start = other_agent.get('start')
                    other_end = other_agent.get('end')
                    if other_start:
                        grid_copy[other_start[0]][other_start[1]] = REGULAR_OBSTACLE
                    if other_end:
                        grid_copy[other_end[0]][other_end[1]] = REGULAR_OBSTACLE

            # If collision avoidance is enabled and there are previous agents, treat their paths as dynamic obstacles
            if avoid_collisions and agent_paths:
                for prev_path in agent_paths:
                    for pos in prev_path:
                        # Avoid marking start and end points again
                        if self.grid[pos[0]][pos[1]] not in [AGENT1_START, AGENT1_END, AGENT2_START, AGENT2_END]:
                            grid_copy[pos[0]][pos[1]] = DYNAMIC_OBSTACLE

            # Set agent's own start and end to empty in the copy to avoid blocking
            si, sj = start
            ei, ej = end
            grid_copy[si][sj] = EMPTY
            grid_copy[ei][ej] = EMPTY

            # Select algorithm and prepare heuristic function
            heuristic_func = heuristic  # Default heuristic

            # Initialize ML-based heuristic if applicable
            if category == "Informed Search":
                try:
                    # Load interpreter for this agent
                    interpreter = tf.lite.Interpreter(model_path=self.model_path)
                    interpreter.allocate_tensors()
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                except Exception as e:
                    self.queue.put(("error", f"Failed to load ML model for Agent {agent_id + 1}: {e}\n"))
                    # Fallback to default heuristic
                    heuristic_func = heuristic
                    interpreter = None

                # Define ML-based heuristic lambda if interpreter is loaded
                if interpreter:
                    heuristic_func = lambda a, b: self.ml_heuristic(
                        interpreter, input_details, output_details, a, b, grid_copy
                    )

            # Run the selected algorithm
            try:
                if algorithm == "A*":
                    path, visited, runtime = astar(grid_copy, start, end, heuristic_func)
                elif algorithm == "Dijkstra":
                    path, visited, runtime = dijkstra(grid_copy, start, end)
                elif algorithm == "BFS":
                    path, visited, runtime = bfs(grid_copy, start, end)
                elif algorithm == "DFS":
                    path, visited, runtime = dfs(grid_copy, start, end)
                elif algorithm == "Greedy Best-First":
                    path, visited, runtime = greedy_best_first_search(grid_copy, start, end, heuristic_func)
                elif algorithm == "Theta*":
                    path, visited, runtime = theta_star(grid_copy, start, end, heuristic_func)
                elif algorithm == "D* Lite":
                    path, visited, runtime = run_d_star_lite(grid_copy, start, end, heuristic_func)
                elif algorithm == "Beam Search":
                    path, visited, runtime = beam_search(grid_copy, start, end, heuristic_func, beam_width=3)
                elif algorithm == "Jump Point Search":
                    path, visited, runtime = jump_point_search(grid_copy, start, end, heuristic_func)
                elif algorithm == "RRT":
                    # Incorporate ML heuristic by adjusting goal bias based on heuristic
                    path, visited, runtime = run_rrt(grid_copy, start, end, heuristic_func, max_iterations=1000, step_size=1)
                else:
                    self.queue.put(("error", f"The algorithm '{algorithm}' is not implemented.\n"))
                    continue
            except Exception as e:
                self.queue.put(("error", f"{algorithm} algorithm failed for Agent {agent_id + 1}: {e}\n"))
                continue

            if path:
                agent_paths.append(path)  # Store the path to avoid collision with subsequent agents
                self.queue.put(("path_found", (agent_id, path, visited, runtime)))
            else:
                self.queue.put(("path_not_found", (agent_id, visited, runtime)))

    def ml_heuristic(self, interpreter, input_details, output_details, start, end, grid):
        """Heuristic function using the ML model."""
        try:
            # Prepare input features
            grid_features = np.array(grid).flatten()
            start_features = np.array(start)
            end_features = np.array(end)
            input_features = np.concatenate((grid_features, start_features, end_features))
            input_features = (input_features - self.scaler_mean) / self.scaler_scale
            input_features = input_features.astype(np.float32).reshape(1, -1)

            # Set tensor
            interpreter.set_tensor(input_details[0]['index'], input_features)

            # Invoke interpreter
            interpreter.invoke()

            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index']).copy()
            predicted = float(output_data[0][0])

            return max(predicted, 0)  # Ensure non-negative
        except Exception as e:
            # Fallback to Manhattan distance if prediction fails
            print(f"Heuristic prediction failed: {e}")
            return heuristic(start, end)

    def process_queue(self):
        """Process messages from the pathfinding thread."""
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()
                if message[0] == "path_found":
                    agent_id, path, visited, runtime = message[1]
                    self.agents[agent_id]['path'] = path
                    self.draw_path(agent_id, path)
                    # Update time labels
                    seconds = runtime / 1e6
                    if agent_id == 0:
                        self.agent1_time_var.set(f"Agent 1 Time: {seconds:.6f} s")
                    elif agent_id == 1:
                        self.agent2_time_var.set(f"Agent 2 Time: {seconds:.6f} s")
                    self.update_status(f"Agent {agent_id + 1}: Path found. Path Length: {len(path)} | Visited Nodes: {visited} | Runtime: {runtime:.2f} μs\n")
                elif message[0] == "path_not_found":
                    agent_id, visited, runtime = message[1]
                    self.agents[agent_id]['path'] = []
                    self.clear_path(agent_id)
                    # Update time labels
                    seconds = runtime / 1e6
                    if agent_id == 0:
                        self.agent1_time_var.set(f"Agent 1 Time: {seconds:.6f} s")
                    elif agent_id == 1:
                        self.agent2_time_var.set(f"Agent 2 Time: {seconds:.6f} s")
                    self.update_status(f"Agent {agent_id + 1}: No path found. Visited Nodes: {visited} | Runtime: {runtime:.2f} μs\n")
                elif message[0] == "error":
                    error_msg = message[1]
                    self.update_status(f"Error: {error_msg}")
                elif message[0] == "log":
                    log_msg = message[1]
                    self.update_status(log_msg)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)

    def draw_path(self, agent_id, path):
        """Draw the path on the grid."""
        # Remove existing path
        tags = f"path_agent_{agent_id}"
        self.canvas.delete(tags)

        if not path:
            return

        path_color = self.colors["path_agent1"] if agent_id == 0 else self.colors["path_agent2"]

        for idx in range(len(path)-1):
            pos1 = path[idx]
            pos2 = path[idx+1]
            x1 = pos1[1] * self.cell_size + self.cell_size / 2
            y1 = pos1[0] * self.cell_size + self.cell_size / 2
            x2 = pos2[1] * self.cell_size + self.cell_size / 2
            y2 = pos2[0] * self.cell_size + self.cell_size / 2
            self.canvas.create_line(x1, y1, x2, y2, fill=path_color, width=2, tags=tags)

    def clear_path(self, agent_id):
        """Clear the path of an agent from the GUI."""
        tags = f"path_agent_{agent_id}"
        self.canvas.delete(tags)

    def update_status(self, message):
        """Update the status text area."""
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')

    def initialize_dynamic_obstacles(self):
        """Initialize dynamic moving obstacles."""
        # Ensure we don't exceed the maximum number of dynamic obstacles
        while len(self.dynamic_obstacles) < self.max_dynamic_obstacles:
            self.create_dynamic_obstacle(size=6)  # Initialize with size greater than 5

    def move_dynamic_obstacles(self):
        """Move dynamic obstacles."""
        obstacles_to_remove = []
        for obstacle in self.dynamic_obstacles:
            if not obstacle['positions']:
                continue

            # Calculate new head position
            current_head = obstacle['positions'][0]
            possible_directions = [(-1,0), (1,0), (0,-1), (0,1)]
            random.shuffle(possible_directions)
            moved = False

            for direction in possible_directions:
                di, dj = direction
                new_head = (current_head[0] + di, current_head[1] + dj)
                if (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size and
                    self.grid[new_head[0]][new_head[1]] == EMPTY and not self.is_agent_position(new_head)):
                    obstacle['direction'] = direction
                    # Move forward
                    obstacle['positions'].insert(0, new_head)
                    self.grid[new_head[0]][new_head[1]] = DYNAMIC_OBSTACLE  # Dynamic Obstacle
                    self.update_cell(new_head[0], new_head[1], self.colors["dynamic_obstacle"])

                    # Remove tail
                    tail = obstacle['positions'].pop()
                    self.grid[tail[0]][tail[1]] = EMPTY
                    self.update_cell(tail[0], tail[1], 'white')

                    moved = True
                    break

            if not moved:
                # No available moves
                if obstacle.get('resets', 0) < self.max_dynamic_obstacle_resets:
                    obstacle['resets'] = obstacle.get('resets', 0) + 1
                    self.reset_dynamic_obstacle(obstacle)
                else:
                    # Mark obstacle for removal
                    obstacles_to_remove.append(obstacle)

        # Remove obstacles that have reached max resets
        for obstacle in obstacles_to_remove:
            self.remove_dynamic_obstacle(obstacle)

        # Ensure we don't have more than max_dynamic_obstacles
        while len(self.dynamic_obstacles) > self.max_dynamic_obstacles:
            obstacle = self.dynamic_obstacles.pop()
            self.remove_dynamic_obstacle(obstacle)

    def reset_dynamic_obstacle(self, obstacle):
        """Reset the dynamic obstacle to a new position if stuck."""
        # Clear current positions
        for pos in obstacle['positions']:
            self.grid[pos[0]][pos[1]] = EMPTY
            self.update_cell(pos[0], pos[1], 'white')

        # Remove obstacle from the list
        if obstacle in self.dynamic_obstacles:
            self.dynamic_obstacles.remove(obstacle)

        # Create new obstacle if we have fewer than the maximum allowed
        if len(self.dynamic_obstacles) < self.max_dynamic_obstacles:
            # Create a smaller obstacle of size 3 or 4
            size = random.choice([3, 4])
            self.create_dynamic_obstacle(size)

    def create_dynamic_obstacle(self, size):
        """Create a dynamic obstacle of a given size."""
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            start_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if self.grid[start_pos[0]][start_pos[1]] == EMPTY and not self.is_agent_position(start_pos):
                break
            attempts += 1
        else:
            return  # Failed to find a valid position

        direction = random.choice([(-1,0), (1,0), (0,-1), (0,1)])
        obstacle = {
            'positions': [start_pos],
            'direction': direction,
            'color': self.colors["dynamic_obstacle"],
            'resets': 0
        }

        for _ in range(1, size):
            last = obstacle['positions'][-1]
            new_pos = (last[0] - direction[0], last[1] - direction[1])

            if (0 <= new_pos[0] < len(self.grid) and 0 <= new_pos[1] < len(self.grid[0]) and
                self.grid[new_pos[0]][new_pos[1]] == EMPTY and not self.is_agent_position(new_pos)):
                obstacle['positions'].append(new_pos)
                self.grid[new_pos[0]][new_pos[1]] = DYNAMIC_OBSTACLE
                self.update_cell(new_pos[0], new_pos[1], self.colors["dynamic_obstacle"])
            else:
                break

        # Mark positions in the grid
        for pos in obstacle['positions']:
            self.grid[pos[0]][pos[1]] = DYNAMIC_OBSTACLE

        self.dynamic_obstacles.append(obstacle)

    def remove_dynamic_obstacle(self, obstacle):
        """Remove a dynamic obstacle from the grid."""
        for pos in obstacle['positions']:
            self.grid[pos[0]][pos[1]] = EMPTY
            self.update_cell(pos[0], pos[1], 'white')
        if obstacle in self.dynamic_obstacles:
            self.dynamic_obstacles.remove(obstacle)

    def start_simulation(self):
        """Start the simulation with dynamic obstacle movement."""
        if not self.agents:
            messagebox.showwarning("No Agents", "Please assign at least one agent before starting the simulation.")
            return

        # Check if all agents have start and end points
        for agent_id, agent in enumerate(self.agents):
            if not agent['start'] or not agent['end']:
                messagebox.showwarning("Incomplete Agents", f"Please assign both start and end points for Agent {agent_id + 1}.")
                return

        self.run_button.config(state=tk.DISABLED)
        self.simulate_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("Simulation started.\n")
        self.simulation_running = True
        self.simulation_step()

    def stop_simulation(self):
        """Stop the ongoing simulation."""
        self.simulation_running = False
        self.run_button.config(state=tk.NORMAL)
        self.simulate_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("Simulation stopped.\n")

    def simulation_step(self):
        """Perform one step of the simulation."""
        if not self.simulation_running:
            return

        self.move_dynamic_obstacles()
        self.add_random_obstacles()
        self.remove_random_obstacles()
        self.run_pathfinding_thread()  # Recalculate paths

        # Schedule the next simulation step
        self.master.after(1000, self.simulation_step)

    def add_random_obstacles(self):
        """Add regular and weighted obstacles at random positions."""
        total_obstacles = sum(row.count(REGULAR_OBSTACLE) + row.count(WEIGHTED_OBSTACLE) for row in self.grid)
        if total_obstacles >= self.max_obstacles:
            return  # Do not add more obstacles if max limit reached

        num_new_obstacles = random.randint(1, 3)  # Add 1 to 3 new obstacles
        for _ in range(num_new_obstacles):
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if self.grid[i][j] == EMPTY and not self.is_agent_position((i, j)):
                obstacle_type = random.choice([REGULAR_OBSTACLE, WEIGHTED_OBSTACLE])
                self.grid[i][j] = obstacle_type
                color = self.colors["regular_obstacle"] if obstacle_type == REGULAR_OBSTACLE else self.colors["weighted_obstacle"]
                self.update_cell(i, j, color)
                obstacle_name = "Regular Obstacle" if obstacle_type == REGULAR_OBSTACLE else "Weighted Obstacle"
                self.update_status(f"Added {obstacle_name} at { (i, j) }\n")

    def remove_random_obstacles(self):
        """Remove obstacles randomly to prevent overcrowding."""
        total_obstacles = sum(row.count(REGULAR_OBSTACLE) + row.count(WEIGHTED_OBSTACLE) for row in self.grid)
        if total_obstacles <= self.max_obstacles * 0.7:
            return  # Do not remove obstacles if not overcrowded

        num_remove = random.randint(1, 3)
        removed = 0
        while removed < num_remove:
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if self.grid[i][j] in [REGULAR_OBSTACLE, WEIGHTED_OBSTACLE]:
                obstacle_type = "Regular Obstacle" if self.grid[i][j] == REGULAR_OBSTACLE else "Weighted Obstacle"
                self.grid[i][j] = EMPTY
                self.update_cell(i, j, 'white')
                removed += 1
                self.update_status(f"Removed {obstacle_type} at { (i, j) }\n")

    def reset_grid(self):
        """Reset the entire grid and simulation."""
        self.simulation_running = False
        self.run_button.config(state=tk.NORMAL)
        self.simulate_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.grid = [[EMPTY for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.agents = []
        self.dynamic_obstacles = []
        self.initialize_dynamic_obstacles()
        self.canvas.delete("all")
        self.draw_grid()
        self.update_status("Grid reset.\n")
        # Reset time labels
        self.agent1_time_var.set("Agent 1 Time: 0.000000 s")
        self.agent2_time_var.set("Agent 2 Time: 0.000000 s")

    def export_log(self):
        """Export the status log to a .txt file."""
        log_content = self.status_text.get("1.0", tk.END).strip()
        if not log_content:
            messagebox.showinfo("Export Log", "The log is empty. Nothing to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                                                 title="Save Log As")
        if file_path:
            try:
                # Open the file with 'utf-8' encoding to support special characters
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(log_content)
                messagebox.showinfo("Export Log", f"Log successfully saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Log Error", f"Failed to save log: {e}")


def main():

    root = tk.Tk()
    root.title('Pathfinding Simulator')
    root.iconbitmap('icon.ico')
    gui = PathfindingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
