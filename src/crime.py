# LUMENROUTE BACKEND
# CRIME CLASS AND CALCULATE WEIGHTS

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
from time import sleep

# dummy_data = read csv

class Graph:
    # IMPLEMENTED BY ARY AND ETHAN
    # assume all methods implemented
    def __init__(self) -> None:
        self.adj = {}
    
    def dijkstra(self, start_node: Node) -> dict:
        # assume method implemented
        pass

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

        