"""Dataclass représentant le réseau routier unifié."""

from dataclasses import dataclass

import geopandas as gpd
import igraph as ig


@dataclass
class UrbanNetwork:
    graph: ig.Graph
    nodes_xy: dict          # {node_idx: (x_lambert93, y_lambert93)}
    edges_gdf: gpd.GeoDataFrame
    crs: int

    def summary(self) -> None:
        print(f"Graphe : {self.graph.vcount():,} nœuds, {self.graph.ecount():,} arcs")
        print(f"CRS    : EPSG:{self.crs}")
        print(f"Sources: {self.edges_gdf['source'].value_counts().to_dict()}")
