import numpy as np
import random
from multiprocessing import Pool, cpu_count
import heapq
import time

# Define constants for grid cell values
EMPTY = 0
REGULAR_OBSTACLE = 1
WEIGHTED_OBSTACLE = 4
DYNAMIC_OBSTACLE = 7  # Not used in dataset generation but kept for consistency

# Node class for A* algorithm
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

# Heuristic function: Manhattan distance
def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

# A* Pathfinding Algorithm
def astar(grid, start, end, heuristic_func=heuristic):
    """
    Performs the A* algorithm to find the shortest path from start to end on the grid.
    Considers weighted obstacles.
    Returns the path as a list of positions and the total traversal cost.
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
            # Reconstruct path
            path = []
            total_cost = current_node.g
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            end_time = time.perf_counter()
            runtime = (end_time - start_time) * 1e6  # Convert to microseconds
            return path[::-1], total_cost, runtime

        # Generate children
        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            node_position = (
                current_node.position[0] + move[0],
                current_node.position[1] + move[1]
            )

            # Check within range
            if (0 <= node_position[0] < len(grid)) and (0 <= node_position[1] < len(grid[0])):
                # Check if walkable
                cell_value = grid[node_position[0]][node_position[1]]
                if cell_value in [REGULAR_OBSTACLE, DYNAMIC_OBSTACLE]:
                    continue  # Untraversable obstacle

                # Determine traversal cost
                traversal_cost = 1
                if cell_value == WEIGHTED_OBSTACLE:
                    traversal_cost = 5  # Weighted obstacle

                # Create new node
                neighbor = Node(current_node, node_position)
                neighbor.g = current_node.g + traversal_cost
                neighbor.h = heuristic_func(neighbor.position, end)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor.position in closed_set:
                    continue

                # Check if neighbor is already in open_list with a lower g
                in_open = False
                for open_node in open_list:
                    if neighbor == open_node[1] and neighbor.g >= open_node[1].g:
                        in_open = True
                        break
                if not in_open:
                    heapq.heappush(open_list, (neighbor.f, neighbor))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 1e6  # Convert to microseconds
    return [], grid_size * grid_size, runtime  # Assign high cost if no path found

def generate_single_sample(grid_size=15, obstacle_prob=0.2, weighted_prob=0.1):
    """
    Generates a single grid sample with start and end positions and computes the actual path cost.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Randomly select start and end positions
    start = tuple(np.random.randint(0, grid_size, size=2))
    end = tuple(np.random.randint(0, grid_size, size=2))
    while end == start:
        end = tuple(np.random.randint(0, grid_size, size=2))

    # Assign obstacles
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == start or (i, j) == end:
                continue
            rand = random.random()
            if rand < obstacle_prob:
                grid[i][j] = REGULAR_OBSTACLE  # Regular obstacle
                if rand < obstacle_prob + weighted_prob:
                    grid[i][j] = WEIGHTED_OBSTACLE  # Weighted obstacle

    # Compute the actual shortest path cost using A*
    path, actual_cost, runtime = astar(grid, start, end, heuristic)
    if not path:
        actual_cost = grid_size * grid_size  # Assign a high cost if no path found

    return {
        'input': grid.flatten().tolist() + list(start) + list(end),
        'output': actual_cost  # Actual path cost
    }

def generate_dataset(num_samples=5000, grid_size=15, obstacle_prob=0.2, weighted_prob=0.1):
    """
    Generates a dataset of grid configurations and their corresponding actual path costs.
    Utilizes multiprocessing for faster generation.
    """
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            generate_single_sample,
            [(grid_size, obstacle_prob, weighted_prob) for _ in range(num_samples)]
        )
    return results

def main():
    num_samples = 5000
    grid_size = 15
    obstacle_prob = 0.2
    weighted_prob = 0.1

    print(f"Generating dataset with {num_samples} samples...")
    dataset = generate_dataset(num_samples, grid_size, obstacle_prob, weighted_prob)
    print("Dataset generation completed.")

    # Save to binary NumPy file for faster loading
    np.save('heuristic_dataset.npy', dataset)
    print("Dataset saved to 'heuristic_dataset.npy'.")

if __name__ == "__main__":
    main()
