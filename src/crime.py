# LUMENROUTE BACKEND
# CRIME CLASS AND CALCULATE WEIGHTS

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
from time import sleep

# dummy_data = read csv
# CSV we're assuming (source, destination, weight)

class Graph:
    # IMPLEMENTED BY ARY AND ETHAN
    # assume all methods implemented
    def __init__(self, csv_path) -> None:
        self.adj_list = defaultdict(list)

        with open(csv_path, 'r') as f:
            next(f)  # skip header if any
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue

                node1, node2, weight = int(parts[0]), int(parts[1]), float(parts[2])

                # Undirected edge
                self.adj_list[node1].append((node2, weight))
                self.adj_list[node2].append((node1, weight))


    def get_neighbors(self, node_id: int):
        """Return list of (neighbor_id, weight) pairs."""
        return self.adj_list.get(node_id, [])
    
    def dijkstra(self, start: int, target: int):
        """
        Find the cheapest (minimum-weight) path from start → target using Dijkstra’s algorithm.
        Returns: (total_cost, path_list)
        """
        # Min-heap priority queue: (distance_so_far, current_node)
        pq = [(0, start)]
        distances = {start: 0}
        previous = {start: None}

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            # Early exit: reached target
            if current_node == target:
                break

            # If this distance is outdated, skip
            if current_dist > distances.get(current_node, float('inf')):
                continue

            for neighbor, weight in self.get_neighbors(current_node):
                new_dist = current_dist + weight

                # Found a shorter path to neighbor
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct the path
        if target not in previous and target != start:
            return float('inf'), []  # no path found

        path = []
        node = target
        while node is not None:
            path.append(node)
            node = previous.get(node)
        path.reverse()

        return distances.get(target, float('inf')), path

class Node:
    # IMPLEMENTED BY JOJO
    # assume all methods implemented
    def __init__(self, lat: float, lon: float, id: int, g: Graph) -> None:
        self.lat = lat
        self.lon = lon
        self.id = id
        self.g = g

class Crime(Node):
    def __init__(self, lat: float, lon: float, g: Graph, dt: datetime) -> None:
        self.lat = lat
        self.lon = lon
        self.wt = 0.0
        self.crds = np.array(lat, lon)
        self.dt = dt
        self.g = g
        # add more data from DP's table

    # calculates the weight of a crime — change this later lol
    def calc_weight(self) -> int:
        priority = dummy_data['Priority'].str.extract(r'\w+_(\d{1})').astype('int')
        magnitude = 1 # get user pref later
        recency = now - timedelta(days=7) # AIM: recency = now - datetime of crime
        return np.sum(priority, magnitude, recency) / 3

    def update_weight(self) -> None:
        # while the mapping algorithm is running
        while True:
            weight = calc_weight(self)
            sleep(30)
    
    # calculates the distance from a crime to nodes 1 and 2
    # helper method for calc_weight
    def calc_distance(self, n1: Node, n2: Node) -> int:
        d1 = np.sqrt(((n1.lat - self.lat) ** 2) + (n1.lon - self.lon) ** 2)
        d2 = np.sqrt(((n2.lat - self.lat) ** 2) + (n2.lon - self.lon) ** 2)
        if (d1 == d2):
            return d1
        else:
            return min(d1, d2)
        




