#!/usr/bin/env python
# coding: utf-8
import argparse
from itertools import repeat

import geopandas as gpd
import pandas as pd

from tools.geohelper import MONTREAL_CRS
from tools.geohelper import right_or_left
from tools.geohelper import rtreenearest


def _detect_unspecified_end(sig_sta, side_of_street='side_of_street', chain='chainage',
                            code_rpa='CODE_RPA', dist='dist_on_road', geom_col='geometry'):
    """ Detect if there is a regulation that have no ending while a new regulation appear

    Example :

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
        if sig_sta.loc[i, side_of_street] != sig_sta.loc[i + 1, side_of_street]:
            continue
        if sig_sta.loc[i, code_rpa] != sig_sta.loc[i + 1, code_rpa]:
            # check if last sign is a end (chain=3)
            if sig_sta.loc[i, chain] != 3:
                # insert new end
                row_to_insert = sig_sta.loc[i].copy()
                row_to_insert[geom_col] = sig_sta.loc[i + 1, geom_col]
                row_to_insert[chain] = 3
                row_to_insert[dist] = sig_sta.loc[i + 1, dist]
                row_to_insert[['Latitude', 'Longitude']] = sig_sta.loc[i + 1, ['Latitude', 'Longitude']]
                row_to_insert[['X', 'Y']] = sig_sta.loc[i + 1, ['X', 'Y']]
                # save the row
                insert.append(row_to_insert)

                # state the other pannel as begining
                update.append(sig_sta.loc[i + 1, 'index'])

    return insert, update


def clean_street_cleaning_signs(points_gdf, road_df, side_of_street='side_of_street',
                                desc_reg='DESCRIPTION_RPA', code_rpa='CODE_RPA',
                                circulation_dir='SENS_CIR', chain='chainage',
                                point_geom='geometry', road_geom='road_geom',
                                road_id='ID_TRC',
                                crs_points='epsg:4326', crs_roads='epsg:4326'):
    """This function handle street cleaning sign error on the street curb. As different street
    cleaning period cannot apply to the same street side. If there is two different sign,
    one should end before the other begin. 
    """

    road_df = road_df.copy().to_crs(MONTREAL_CRS)
    points_gdf = points_gdf.copy().to_crs(MONTREAL_CRS)
    points = []

    for seg_id, subpoints in points_gdf.groupby(road_id):

        roads_ls = road_df.loc[road_df[road_id] == seg_id, road_geom].iloc[0]
        road_circ = road_df.loc[road_df[road_id] == seg_id, circulation_dir].iloc[0]

        def dist_on_road(road, point):
            return road.project(point)

        subpoints['dist_on_road'] = list(
            map(dist_on_road, list(repeat(roads_ls, subpoints.shape[0])), subpoints[point_geom].values))

        if road_circ == 1:
            # nothing to do
            pass
        elif road_circ == 0:
            # inverser les distances du sens de circ opposé
            dist_road = roads_ls.length
            subpoints.loc[subpoints[side_of_street] == -1, 'dist_on_road'] = dist_road - subpoints.loc[
                subpoints[side_of_street] == -1, 'dist_on_road']
        else:
            # inverser toutes les distances
            dist_road = roads_ls.length
            subpoints['dist_on_road'] = dist_road - subpoints['dist_on_road']

        # check if pannel has a start, no end and another pannel start after (replacement)
        points_street_c = subpoints[
            subpoints[desc_reg].str.match(r'(.*)? [0-9]{1,2} (AVRIL|MARS) AU [0-9]{1,2} (NOV|DEC)')]

        insertions, update = _detect_unspecified_end(points_street_c)
        subpoints.loc[update, chain] = 1
        subpoints.loc[update, 'manualy_added'] = 1

        subpoints = pd.concat([subpoints, pd.DataFrame(insertions)], axis=0)

        # Ensure that CRS is set for subpoints before adding to points list
        if subpoints.crs is None:
            subpoints = gpd.GeoDataFrame(subpoints, geometry=point_geom, crs=MONTREAL_CRS)
        points.append(subpoints)

    points_gdf = pd.concat(points, axis=0)
    points_gdf = gpd.GeoDataFrame(points_gdf, geometry=point_geom, crs=MONTREAL_CRS)
    points_gdf = points_gdf.to_crs(crs_points)

    return points_gdf


def main(sig_sta, geobase, limit):
    sig_sta = sig_sta.copy()
    geobase = geobase.copy()

    sig_sta = gpd.GeoDataFrame(sig_sta, geometry=gpd.points_from_xy(sig_sta.Longitude, sig_sta.Latitude),
                               crs='epsg:4326')
    geobase = geobase.copy().rename(columns={'geometry': 'road_geom'}).set_geometry('road_geom')

    if not limit.empty:
        sig_sta = gpd.sjoin(sig_sta, limit, predicate='within', how='inner').drop(columns='index_right')
        geobase = gpd.sjoin(geobase, limit, predicate='intersects', how='inner')

    # keep only up to date sign without banners
    filter_ = ~sig_sta.DESCRIPTION_RPA.str.startswith('PANONCEAU') & \
              ((sig_sta.DESCRIPTION_REP == 'Réel') | (sig_sta.DESCRIPTION_REP == 'En conception'))

    geobase = geobase.reset_index(drop=True)
    filtered_signs = sig_sta[filter_].copy().reset_index(drop=True)

    # Compute the closest road to each sig_sta
    filtered_signs = rtreenearest(filtered_signs, geobase, ['road_geom', 'SENS_CIR', 'ID_TRC'])

    # Compute side of street of a sign on a street
    filtered_signs['side_of_street'] = list(map(right_or_left, filtered_signs.road_geom, filtered_signs.geometry))

    # Transform fleche to chaining enum sharedstreet standard {1:start, 2:middle, 3:end}
    fleche_map = {
        0: {
            -1:
                {0: '2',  # no arrow -> middle
                 2: '1',  # left arrow -> start
                 3: '3'},  # right arrow -> end
            1:
                {0: '2',  # no arrow -> middle
                 2: '1',  # left arrow -> start
                 3: '3'}  # right arrow -> end
        },
        1: {
            -1:
                {0: '2',  # no arrow -> middle
                 2: '3',  # left arrow -> end
                 3: '1'},  # right arrow -> start
            1:
                {0: '2',  # no arrow -> middle
                 2: '1',  # left arrow -> start
                 3: '3'}  # right arrow -> end
        },
        -1: {
            1:
                {0: '2',  # no arrow -> middle
                 2: '3',  # left arrow -> end
                 3: '1'},  # right arrow -> start
            -1:
                {0: '2',  # no arrow -> middle
                 2: '1',  # left arrow -> start
                 3: '3'}  # right arrow -> end
        },
    }

    def compute_chainage(col):  # FIXME : do not ignore entries that do not have corresponding keys in fleche_map.
        try:
            return fleche_map[col.SENS_CIR][col.side_of_street][col.FLECHE_PAN]
        except KeyError as e:
            # Print the problematic row and continue
            print(f"KeyError for col: {col.to_dict()}, missing key: {e}")
            return None

    # Compute chaining of each sign on a roads
    filtered_signs['chainage'] = (filtered_signs[['SENS_CIR', 'side_of_street', 'FLECHE_PAN']]
                                  .apply(compute_chainage, axis=1))

    filtered_signs.crs = 'epsg:4326'
    # cleans unwanted chaining behavior on street cleaning sign
    filtered_signs = clean_street_cleaning_signs(filtered_signs, geobase)

    return filtered_signs.drop(columns='road_geom')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Signalec Preprocessing Script')
    parser.add_argument('-s', '--sig_sta', required=True, help='Path to the signs file')
    parser.add_argument('-g', '--geobase', required=True, help='Path to the geobase file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    sig_sta_path = args.sig_sta
    geobase_path = args.geobase

    print(f'Loading data from {sig_sta_path} and {geobase_path} ...')
    sig_sta = gpd.read_file(sig_sta_path)
    geobase = gpd.read_file(geobase_path)
    print('Data loaded successfully.\n')

    print('Running preprocessing ...')
    result = main(sig_sta, geobase)
    output_path = './data/signalec_signs_preprocessed.geojson'
    result.to_file(output_path, driver='GeoJSON')
    print(f'Preprocessing completed. Output saved to {output_path}')
