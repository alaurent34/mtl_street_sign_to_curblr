"""Module providing graph operation for Geobase GeoDataFrame"""

import numpy as np
import networkx as nx
import momepy
import geopandas as gpd

from signe.tools.geom import DEFAULT_CRS


ROAD_CLASS_MAPPING = {
    0: 'residential',
    1: 'pedestrian',
    2: 'service',
    3: 'service',
    4: 'service',
    5: 'tertiary',
    6: 'secondary',
    7: 'primary',
    8: 'motorway',
    9: 'motorway'
}


def construct_graph(lines: gpd.GeoDataFrame,
                    limits: gpd.GeoDataFrame = None,
                    crs: str = DEFAULT_CRS) -> nx.Graph:
    """
    Converts a serie of geometric lines into a graph with nodes and edges.

    Parameters
    ----------
    lines : gpd.GeoDataFrame
        The set of lines to transform.
    limits : gpd.GeoDataFrame, optional
        Optional boundaries to select a subset of the lines with (lines that
        intersected the boundaries and kept whole). The default is None.
    crs : str, optional
        The crs in which the objects in `lines` are projected. The default is
        DEFAULT_CRS (see tools.geom).

    Returns
    -------
    graph : nx.Graph
        The graph object.

    """

    lines = lines.copy()
    lines = lines.to_crs(crs)

    if isinstance(limits, gpd.GeoDataFrame):
        columns = lines.columns
        limits = limits.copy()

        # 1 - project to same crs
        limits = limits.to_crs(crs)

        # 2 - create graph
        lines = gpd.sjoin(
            left_df=lines,
            right_df=limits,
            predicate='intersects',
            how='inner'
        ).reset_index()

        # 3 - remove duplicates created by join
        lines = lines[columns].drop_duplicates().reset_index(drop=True)

    # prepare data
    index_names = lines.index.names
    lines = lines.reset_index()

    # needs pygeos
    lines.geometry = momepy.close_gaps(lines, 1)

    # re-index
    index_names = index_names if len(index_names) > 1 else index_names[0]
    if index_names:
        lines = lines.set_index(index_names)
    else:
        lines = lines.set_index('index')

    graph = momepy.gdf_to_nx(lines, approach='primal')

    return graph


def get_segment_intersection_name(graph: nx.Graph, seg_id: int,
                                  seg_col: str = 'segment',
                                  col_name: str = 'Rue') -> tuple[str, str]:
    """
    Query the name of the intersections (as street1, street2) from a road
    segment taken in a graph built using the construct_graph function

    Parameters
    ----------
    graph : nx.Graph
        The graph to search.
    seg_id : int
        The id from the segment to document.
    seg_col : str, optional
        The name of the attribute in which the edges names are stored.
        The default is 'segment'.
    col_name : str, optional
        The name of the attribute in which the node names are stored.
        The default is 'Rue'.

    Returns
    -------
    tuple[str, str]
        The name of the two intersecting streets, in the edge's vector
        direction.

    """
    _, edges = momepy.nx_to_gdf(graph)

    try:
        nodes_start = edges.set_index(seg_col).loc[seg_id, 'node_start']
        nodes_end = edges.set_index(seg_col).loc[seg_id, 'node_end']
        street_name = edges.set_index(seg_col).loc[seg_id, col_name]
    except KeyError:
        return np.repeat('Segment not found', 2)
    try:
        rfrom = edges[
            ((edges.node_start == nodes_start) |
             (edges.node_end == nodes_start)) &
            (edges[col_name] != street_name)
        ][col_name].unique()[0]
    except IndexError:
        rfrom = "N/A"

    try:
        rto = edges[
            ((edges.node_start == nodes_end) |
             (edges.node_end == nodes_end)) &
            (edges[col_name] != street_name)
        ][col_name].unique()[0]
    except IndexError:
        rto = "N/A"

    return rfrom, rto
