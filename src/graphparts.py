# LUMENROUTE BACKEND
# CRIME CLASS AND CALCULATE WEIGHTS

import pandas as pd
import numpy as np
import heapq
from datetime import datetime
from collections import defaultdict
import os
import math
from typing import Dict, List, Tuple, Optional


# for dummy, get the csv
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dummy.csv')
DATA_PATH = os.path.abspath(DATA_PATH)

# true nodes/edges for after mid-sem
if not os.path.exists("data/berkeley_adj_list.csv"):
    from backend.csv_builder import build_csv
    build_csv()

# CSV we're assuming (source, destination, weight)

class Graph:
    # IMPLEMENTED BY ARY AND ETHAN
    # assume all methods implemented
    def __init__(self, csv_path=DATA_PATH) -> None:
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
        Find the cheapest path from start → target using Dijkstra’s algorithm.
        Returns: (total_cost, [(node, edge_weight), ..., (target, 0)])
        """
        pq = [(0, start)]
        distances = {start: 0}
        previous = {start: None}

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node == target:
                break

            if current_dist > distances.get(current_node, float('inf')):
                continue

            for neighbor, weight in self.get_neighbors(current_node):
                new_dist = current_dist + weight
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    previous[neighbor] = (current_node, weight)
                    heapq.heappush(pq, (new_dist, neighbor))

        if target not in previous and target != start:
            return float('inf'), []

        # Reconstruct path with weights
        path_with_weights = []
        node = target
        while node is not None:
            prev = previous.get(node)
            if prev is None:
                path_with_weights.append((node, 0))  # start node
                break
            prev_node, edge_weight = prev
            path_with_weights.append((node, edge_weight))
            node = prev_node

        path_with_weights.reverse()
        total_cost = distances.get(target, float('inf'))
        return total_cost, path_with_weights

class Node:
    def __init__(self, lat: float, lon: float, id: int, g: Graph) -> None:
        self.lat = lat
        self.lon = lon
        self.id = id
        self.g = g

    # get lat
    def get_lat(self) -> float:
        return self.lat
    
    # set lat
    def set_lat(self, new_lat: float) -> None:
        self.lat = new_lat

     # get lon
    def get_lon(self) -> float:
        return self.lon
    
    # set lat
    def set_lon(self, new_lon: float) -> None:
        self.lon = new_lon

    # get id
    def get_id(self) -> int:
        return self.id
    
    def get_neighbours(self) -> list:
        return self.g.get_neighbors(self.id)
    
    def __str__(self) -> str:
        return f"Node(id={self.id}, lat={self.lat:.6f}, lon={self.lon:.6f})"


class Crime(Node):
    """
    Crime is a Node + attributes that produce a weight:
      weight = average(priority_norm, magnitude_norm, recency_norm)
    Where:
      priority_norm  in [0,1] (e.g., priority 1..5 -> / 5)
      magnitude_norm in [0,1] (user pref 1..5 -> / 5, or already 0..1)
      recency_norm   in [0,1] via exponential decay or linear window
    """
    def __init__(
        self,
        lat: float,
        lon: float,
        g: Graph,
        dt: datetime,
        priority: int,
        magnitude: float,
        *,
        priority_max: int = 5,
        magnitude_max: float = 5.0,
        recency_half_life_days: float = 7.0,
        node_id: Optional[int] = None
    ) -> None:
        super().__init__(lat, lon, node_id if node_id is not None else -1, g)
        self.dt = dt
        self.priority = int(priority)               # e.g., 1..5 (use of force)
        self.magnitude = float(magnitude)           # user preference (1..5 or 0..1)
        self.priority_max = int(priority_max)
        self.magnitude_max = float(magnitude_max)
        self.recency_half_life_days = float(recency_half_life_days)
        self.coords = np.array([self.lat, self.lon], dtype=float)

    def calc_weight(self, now: Optional[datetime] = None) -> float:
        # normalize each factor to [0,1]
        priority_norm = min(max(self.priority / self.priority_max, 0.0), 1.0)
        magnitude_norm = min(max(self.magnitude / self.magnitude_max, 0.0), 1.0)
        recency_norm = recency_score(self.dt, now, self.recency_half_life_days)
        return (priority_norm + magnitude_norm + recency_norm) / 3.0

    def calc_distance_km(self, n1: Node, n2: Node) -> float:
        """Distance from this crime to the closer of n1 or n2 (in km)."""
        d1 = haversine_km(self.lat, self.lon, n1.lat, n1.lon)
        d2 = haversine_km(self.lat, self.lon, n2.lat, n2.lon)
        return min(d1, d2)

    # If you really want periodic refreshing, prefer an external scheduler.
    # Kept here as an explicit one-shot update method.
    def update_weight_once(self, now: Optional[datetime] = None) -> float:
        return self.calc_weight(now)
     

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * (2 * math.asin(math.sqrt(a)))

def recency_score(dt: datetime, now: Optional[datetime] = None, half_life_days: float = 7.0) -> float:
    """
    Exponential decay in [0,1]. Now = 1.0, decays with half-life.
    score = 0.5 ** (age_days / half_life_days)
    """
    now = now or datetime.now(datetime.timezone.utc)
    age_days = max((now - dt).total_seconds(), 0) / 86400.0
    return float(0.5 ** (age_days / half_life_days))