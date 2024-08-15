""" Module for testing
"""
from itertools import repeat

import pandas as pd
import geopandas as gpd

from signe.tools.geom import MONTREAL_CRS
from signe.tranform.points_to_line import project


# FIXME: This only compare 2 by 2 pannel. It does not do a real job at
# checking if there is a problematic disposition of start's and end
def _detect_unspecified_end(
    sig_sta: pd.DataFrame,
    side_of_street: str = 'side_of_street',
    chain: str = 'chainage',
    class_panel: str = 'CODE_RPA',
    dist: str = 'dist_on_road',
    geom_col: str = 'geometry'
) -> pd.DataFrame:
    """ Detect if there is a regulation that have no ending while a new
    regulation of the same type is starting.

    Parameters
    ----------
    sig_sta : pd.DataFrame
        Dataframe containing all the signalisation pannels with the
        positions on a street, linear referencement and side of street already
        computed.
    side_of_street : str, optional
       Column containing the side of street information, 1 is right, -1 is
       left. Right and left are determined by the digitalization direction. By
       default 'side_of_street'.
    chain : str, optional
        Column containing the chainage ifnormation, by default 'chainage'.
        Chainage should be formatted as 1: start, 2: repeat, 3: end.
    class_panel : str, optional
        Column containing the type of the regulation, by default 'CODE_RPA'.
        Only the same regulations should no start when one other has already
        started.
    dist : str, optional
        Linear referencing columns, by default 'dist_on_road'. Computed
        alongside the digitalization direction of the road network.
    geom_col : str, optional
        Column containing geometry info, by default 'geometry'.

    Returns
    -------
    pd.DataFrame
        Modified dataframe.

    Example
    -------

    If we have this signalisation on the road, and that B and A cannot apply on
    the same street then we end A on first B presence and start B on first B
    prensence as shown above:

    Before:

            | A |   | B |   | B |
        _>____|_______|_______|_____>___
              2       2       2

    After:
                    | A |
            | A |   | B |   | B |
        _>____|_______|_______|_____>___
              2       3       2
                      1
    """
    sig_sta = sig_sta.copy()
    sig_sta = sig_sta.sort_values([side_of_street, dist])
    sig_sta = sig_sta.reset_index(drop=False)

    insert = []
    update = []

    for i in range(sig_sta.shape[0] - 1):
        if (
            sig_sta.loc[i, side_of_street] !=
            sig_sta.loc[i + 1, side_of_street]
        ):
            continue
        if (
            sig_sta.loc[i, class_panel] !=
            sig_sta.loc[i + 1, class_panel]
        ):
            # check if last sign is a end (chain=3)
            if sig_sta.loc[i, chain] != 3:
                # insert new end
                row_to_insert = sig_sta.loc[i].copy()
                row_to_insert[geom_col] = sig_sta.loc[i + 1, geom_col]
                row_to_insert[chain] = 3
                row_to_insert[dist] = sig_sta.loc[i + 1, dist]
                row_to_insert[
                    ['Latitude', 'Longitude']
                ] = sig_sta.loc[i + 1, ['Latitude', 'Longitude']]
                row_to_insert[['X', 'Y']] = sig_sta.loc[i + 1, ['X', 'Y']]
                # save the row
                insert.append(row_to_insert)

                # state the other pannel as begining
                update.append(sig_sta.loc[i + 1, 'index'])

    return insert, update


def clean_street_cleaning_signs(
    points_gdf,
    road_df,
    side_of_street='side_of_street',
    road_id='ID_TRC',
    class_panel='CODE_RPA',
    circulation_dir='SENS_CIR',
    chain='chainage',
    point_geom='geometry',
    road_geom='road_geom',
    crs_points='epsg:4326',
):
    """This function handle street cleaning sign error on the street curb. As
    different street cleaning period cannot apply to the same street side. If
    there is two different sign, one should end before the other begin.
    """

    road_df = road_df.copy().to_crs(MONTREAL_CRS)
    points_gdf = points_gdf.copy().to_crs(MONTREAL_CRS)
    points = []

    for seg_id, subpoints in points_gdf.groupby(road_id):

        roads_ls = road_df.loc[
            road_df[road_id] == seg_id, road_geom
        ].iloc[0]
        road_circ = road_df.loc[
            road_df[road_id] == seg_id, circulation_dir
        ].iloc[0]

        subpoints['dist_on_road'] = list(map(
            project,
            list(repeat(roads_ls, subpoints.shape[0])),
            subpoints[point_geom].values
        ))

        if road_circ == 1:
            # nothing to do
            pass
        elif road_circ == 0:
            # inverser les distances du sens de circ opposé
            dist_road = roads_ls.length
            subpoints.loc[
                subpoints[side_of_street] == -1,
                'dist_on_road'
            ] = dist_road - subpoints.loc[
                subpoints[side_of_street] == -1,
                'dist_on_road'
            ]
        else:
            # inverser toutes les distances
            dist_road = roads_ls.length
            subpoints['dist_on_road'] = dist_road - subpoints['dist_on_road']

        # check if pannel has a start, no end and another pannel start after
        # (replacement)
        points_street_c = subpoints[
            subpoints['CODE_RPA'].str.match(
                r'(.*)? [0-9]{1,2} (AVRIL|MARS) AU [0-9]{1,2} (NOV|DEC)'
            )]

        insertions, update = _detect_unspecified_end(
            points_street_c,
            class_panel=class_panel
        )
        subpoints.loc[update, chain] = 1
        subpoints.loc[update, 'manualy_added'] = 1

        subpoints = pd.concat([subpoints, pd.DataFrame(insertions)], axis=0)

        # Ensure that CRS is set for subpoints before adding to points list
        if subpoints.crs is None:
            subpoints = gpd.GeoDataFrame(
                subpoints,
                geometry=point_geom,
                crs=MONTREAL_CRS
            )
        points.append(subpoints)

    points_gdf = pd.concat(points, axis=0)
    points_gdf = gpd.GeoDataFrame(
        points_gdf,
        geometry=point_geom,
        crs=MONTREAL_CRS
    )
    points_gdf = points_gdf.to_crs(crs_points)

    return points_gdf
