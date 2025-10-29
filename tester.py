import os
import sys

# Ensure Python can find src/
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graphparts import Graph

if __name__ == "__main__":
    # Build the graph using the dummy CSV
    g = Graph()  # uses DATA_PATH inside crime.py

    # Pick any start and end node
    start_node = 1
    end_node = 18

    # Run Dijkstra's algorithm
    total_cost, path = g.dijkstra(start_node, end_node)

    # Print results
    print(f"Shortest path from B{start_node} → B{end_node}: {path}")
    print(f"Total cost: {total_cost:.4f}")
