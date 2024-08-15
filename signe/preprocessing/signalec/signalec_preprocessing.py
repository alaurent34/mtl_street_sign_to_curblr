#!/usr/bin/env python
# coding: utf-8
import argparse
import itertools

import geopandas as gpd
import pandas as pd

from signe.tranform.points_to_line import (
    infer_side_of_street,
    v_project,
    compute_chainage,
    point_to_line
)
from signe.tools.geom import MONTREAL_CRS
from signe.tools.graph import (
    construct_graph,
    get_segment_intersection_name
)


# Transform fleche to chaining enum sharedstreet standard
# {1: start, 2: middle, 3: end}
MTL_FLECHE_MAP = {
    0: {
        -1:
            {0: '2',   # no arrow -> middle
             2: '1',   # left arrow -> start
             3: '3',   # right arrow -> end
             8: '2'},  # both direction ?
        1:
            {0: '2',  # no arrow -> middle
             2: '1',  # left arrow -> start
             3: '3',  # right arrow -> end
             8: '2'}  # both direction ?
    },
    1: {
        -1:
            {0: '2',  # no arrow -> middle
             2: '3',  # left arrow -> end
             3: '1',  # right arrow -> start
             8: '2'},  # both direction ?
        1:
            {0: '2',  # no arrow -> middle
             2: '1',  # left arrow -> start
             3: '3',  # right arrow -> end
             8: '2'}  # both direction ?
    },
    -1: {
        1:
            {0: '2',  # no arrow -> middle
             2: '3',  # left arrow -> end
             3: '1',  # right arrow -> start
             8: '2'},  # both direction ?
        -1:
            {0: '2',  # no arrow -> middle
             2: '1',  # left arrow -> start
             3: '3',  # right arrow -> end
             8: '2'}  # both direction ?
    },
}


def transform_signalec(
    sig_sta: pd.DataFrame,
    geobase: gpd.GeoDataFrame,
    limit: gpd.GeoDataFrame = gpd.GeoDataFrame()
):
    """_summary_

    Parameters
    ----------
    sig_sta : pd.DataFrame
        _description_
    geobase : gpd.GeoDataFrame
        _description_
    limit : gpd.GeoDataFrame, optional
        _description_, by default gpd.GeoDataFrame()

    Returns
    -------
    _type_
        _description_
    """
    sig_sta = sig_sta.copy()
    geobase = geobase.copy()

    sig_sta = gpd.GeoDataFrame(
        sig_sta, geometry=gpd.points_from_xy(sig_sta.Longitude,
                                             sig_sta.Latitude),
        crs='epsg:4326'
    )
    geobase = geobase.rename_geometry('road_geom')

    if not limit.empty:
        limit_columns = list(limit.columns)
        limit_columns.remove('geometry')
        sig_sta = gpd.sjoin(
            sig_sta,
            limit,
            predicate='within',
            how='inner'
        ).drop(
            columns=['index_right'] + list(limit_columns)
        )
        geobase = gpd.sjoin(
            geobase,
            limit,
            predicate='intersects',
            how='inner'
        ).drop(
            columns=['index_right'] + list(limit_columns)
        )

    # keep only up to date sign without banners
    filter_ = (
        ~sig_sta.DESCRIPTION_RPA.str.startswith('PANONCEAU') &
        ((sig_sta.DESCRIPTION_REP == 'Réel') |
         (sig_sta.DESCRIPTION_REP == 'En conception')) &
        (sig_sta.CODE_RPA == 'R-TA')
    )

    geobase = geobase.reset_index(drop=True)
    filtered_signs = sig_sta[filter_].copy().reset_index(drop=True)

    # Compute the closest road to each sig_sta
    geobase = geobase.to_crs(MONTREAL_CRS)
    filtered_signs = filtered_signs.to_crs(MONTREAL_CRS)
    filtered_signs = gpd.sjoin_nearest(
        filtered_signs,
        geobase,
        distance_col='dist_match',
        max_distance=10
    )

    # Compute the closest road to each sig_sta
    # columns = ['ID_TRC', 'SENS_CIR', 'road_geom']
    # filtered_signs[columns] = match_points_on_roads(
    #     filtered_signs,
    #     geobase,
    #     columns
    # )

    # Compute side of street of a sign on a street
    filtered_signs['side_of_street'] = infer_side_of_street(
        points=filtered_signs,
        roads_network=geobase,
        roads_id_col='ID_TRC'
    )

    roads_lines = geobase.set_index("ID_TRC").copy()
    roads_lines = roads_lines.loc[
        filtered_signs['ID_TRC'],
        roads_lines.geometry.name
    ].values

    # linear referencing
    filtered_signs['dist_on_roads'] = v_project(
        roads_lines,
        filtered_signs.geometry
    )

    # Compute chaining of each sign on a roads
    filtered_signs['chainage'] = list(map(
        compute_chainage,
        itertools.repeat(MTL_FLECHE_MAP),
        filtered_signs.SENS_CIR,
        filtered_signs.side_of_street,
        filtered_signs.FLECHE_PAN
    ))

    # filtered_signs.to_file('debug_rpa.geojson')

    # filtered_signs.crs = 'epsg:4326'
    # cleans unwanted chaining behavior on street cleaning sign
    # this does not work
    # filtered_signs = clean_street_cleaning_signs(filtered_signs, geobase)

    # create geometry
    filtered_signs = point_to_line(
        data=filtered_signs,
        roads=geobase,
        sequence_field='chainage',
        match_field=['CODE_RPA', 'DESCRIPTION_RPA'],
        linear_ref_field='dist_on_roads',
        side_of_street_field='side_of_street',
        roads_id_col='ID_TRC',
        traffic_dir_field='SENS_CIR',
    )

    geobase_graph = geobase.rename_geometry('geometry').copy()
    graph = construct_graph(
        lines=geobase_graph,
    )
    geobase_x_filt = geobase[geobase.ID_TRC.isin(filtered_signs.ID_TRC)].copy()
    geobase_x_filt['De'] = geobase_x_filt.apply(
        lambda x: get_segment_intersection_name(
            graph=graph,
            seg_id=x['ID_TRC'],
            seg_col='ID_TRC',
            col_name='NOM_VOIE'
        )[0],
        axis=1
    )
    geobase_x_filt['À'] = geobase_x_filt.apply(
        lambda x: get_segment_intersection_name(
            graph=graph,
            seg_id=x['ID_TRC'],
            seg_col='ID_TRC',
            col_name='NOM_VOIE'
        )[1],
        axis=1
    )
    filtered_signs['Rue'] = geobase.set_index("ID_TRC")\
                                   .loc[filtered_signs.ID_TRC, 'NOM_VOIE']\
                                   .values
    filtered_signs['De'] = geobase_x_filt.set_index("ID_TRC")\
                                         .loc[filtered_signs.ID_TRC, 'De']\
                                         .values
    filtered_signs['À'] = geobase_x_filt.set_index("ID_TRC")\
                                        .loc[filtered_signs.ID_TRC, 'À']\
                                        .values

    return filtered_signs


def parse_arguments():
    """ Argument parser

    """
    parser = argparse.ArgumentParser(
        description='Signalec Preprocessing Script'
    )
    parser.add_argument(
        '-s',
        '--sig_sta',
        required=True,
        help='Path to the signs file'
    )
    parser.add_argument(
        '-g',
        '--geobase',
        required=True,
        help='Path to the geobase file'
    )
    return parser.parse_args()


def main(args):
    """_summary_

    Parameters
    ----------
    args : _type_
        _description_
    """
    sig_sta_path = args.sig_sta
    geobase_path = args.geobase

    print(f'Loading data from {sig_sta_path} and {geobase_path} ...')
    sig_sta = gpd.read_file(sig_sta_path)
    geobase = gpd.read_file(geobase_path)
    print('Data loaded successfully.\n')

    print('Running preprocessing ...')
    result = transform_signalec(
        sig_sta,
        geobase
    )
    output_path = './data/signalec_signs_preprocessed.geojson'
    result.to_file(output_path, driver='GeoJSON')
    print(f'Preprocessing completed. Output saved to {output_path}')


if __name__ == '__main__':

    args_ = parse_arguments()
    main(args_)
