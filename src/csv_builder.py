import osmnx as ox
import pandas as pd
import os

# unfinished
def build_csv(center_coords=(37.8719, -122.2585), radius_m=4828, outfile="data/berkeley_adj_list.csv"): # 4828 ~= 3mi
    # get road network
    G = ox.graph_from_point(center_coords, dist=radius_m, network_type='drive')

    # extract edges
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges.reset_index()  # ensures u, v become columns
    adj_df = edges[['u', 'v', 'length']].rename(columns={'u': 'source', 'v': 'destination', 'length': 'weight'})


    # simplify node IDs
    # node_map = {node_id: i for i, node_id in enumerate(G.nodes())}
    # adj_df['source'] = adj_df['source'].map(node_map)
    # adj_df['destination'] = adj_df['destination'].map(node_map)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    adj_df.to_csv(outfile, index=False)

if __name__ == "__main__":
    build_csv()
