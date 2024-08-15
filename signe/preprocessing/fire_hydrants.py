""" Process point fire hydrants to lines on road network
"""
import geopandas as gpd

from signe.tranform.points_to_line import (
    buffered_point_to_ligne,
    v_project,
    infer_side_of_street,
    match_points_on_roads
)
from signe.tools.geom import MONTREAL_CRS


def process_fire_hydrants(
    data: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    limits: gpd.GeoDataFrame = gpd.GeoDataFrame()
) -> gpd.GeoDataFrame:
    """_summary_

    Parameters
    ----------
    data : gpd.GeoDataFrame
        _description_
    roads : gpd.GeoDataFrame
        _description_

    Returns
    -------
    gpd.GeoDataFrame
        _description_
    """

    data = data.copy()
    roads = roads.copy()
    roads = roads.rename_geometry('road_geom')

    if not limits.empty:
        limit_cols = list(limits.columns)
        limit_cols.remove('geometry')
        data = gpd.sjoin(
            data,
            limits,
            how='inner',
            predicate='intersects'
        ).drop(
            columns=['index_right'] + limit_cols
        )

    # Compute the closest road to each sig_sta
    roads = roads.to_crs(MONTREAL_CRS)
    data = data.to_crs(MONTREAL_CRS)
    data = gpd.sjoin_nearest(
        data,
        roads,
        distance_col='dist_match',
        max_distance=10
    )
    # columns = ['ID_TRC', 'road_geom']
    # data[columns] = match_points_on_roads(
    #     data,
    #     roads,
    #     columns
    # )

    # Compute side of street of a sign on a street
    data['side_of_street'] = infer_side_of_street(
        points=data,
        roads_network=roads,
        roads_id_col='ID_TRC'
    )

    # linear referencing
    data['dist_on_roads'] = v_project(
        data.road_geom,
        data.geometry
    )

    data = buffered_point_to_ligne(
        data=data,
        roads=roads,
        meters=6,
        linear_ref_field='dist_on_roads',
        roads_id_col='ID_TRC'
    )

    return data
