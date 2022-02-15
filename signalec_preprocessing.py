#!/usr/bin/env python
# coding: utf-8

import sys
import getopt

import pandas as pd
import geopandas as gpd
from shapely import wkt
from itertools import repeat
from geopy.distance import geodesic

from tools.geohelper import rtreenearest
from tools.geohelper import right_or_left
from tools.geohelper import distance, vectorized_dist
from tools.geohelper import MONTREAL_CRS


def load_wkt(wkt_str):
    """ Return shapely geometry from string.
    """
    try: 
        return wkt.loads(wkt_str)
    except ValueError:
        return None

def _detect_unspecified_end(signs, side_of_street='side_of_street', chain='chainage',
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
    signs = signs.copy()
    signs = signs.sort_values([side_of_street, dist])
    signs = signs.reset_index(drop=False)
    
    insert = []
    update = []
    
    for i in range(signs.shape[0]-1):
        if signs.loc[i, side_of_street] != signs.loc[i+1, side_of_street]:
            continue
        if signs.loc[i, code_rpa] != signs.loc[i+1, code_rpa]:
            # check if last sign is a end (chain=3)
            if signs.loc[i, chain] != 3:
                # insert new end
                row_to_insert = signs.loc[i].copy()
                row_to_insert[geom_col] = signs.loc[i+1, geom_col]
                row_to_insert[chain] = 3
                row_to_insert[dist] = signs.loc[i+1, dist]
                row_to_insert[['Latitude', 'Longitude']] = signs.loc[i+1,['Latitude', 'Longitude']]
                row_to_insert[['X', 'Y']] = signs.loc[i+1,['X', 'Y']]
                # save the row
                insert.append(row_to_insert)
                
                # state the other pannel as begining
                update.append(signs.loc[i+1, 'index'])
            
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

        subpoints['dist_on_road'] = list(map(dist_on_road, list(repeat(roads_ls, subpoints.shape[0])), subpoints[point_geom].values))

        if road_circ == 1:
            # nothing to do
            pass
        elif road_circ == 0:
            # inverser les distances du sens de circ opposé
            dist_road = roads_ls.length
            subpoints.loc[subpoints[side_of_street] == -1, 'dist_on_road'] = dist_road - subpoints.loc[subpoints[side_of_street] == -1, 'dist_on_road']
        else:
            # inverser toutes les distances
            dist_road = roads_ls.length
            subpoints['dist_on_road'] = dist_road - subpoints['dist_on_road']

        # check if pannel has a start, no end and another pannel start after (replacement)
        points_street_c = subpoints[subpoints[desc_reg].str.match(r'(.*)? [0-9]{1,2} (AVRIL|MARS) AU [0-9]{1,2} (NOV|DEC)')]

        insertions, update = _detect_unspecified_end(points_street_c)
        subpoints.loc[update, chain] = 1
        subpoints.loc[update, 'manualy_added'] = 1

        subpoints = pd.concat([subpoints, pd.DataFrame(insertions)], axis=0)
        points.append(subpoints)
    
    points_gdf = pd.concat(points, axis=0)
    points_gdf = gpd.GeoDataFrame(points_gdf, geometry=point_geom, crs=MONTREAL_CRS)
    points_gdf = points_gdf.to_crs(crs_points)
    
    return points_gdf



def main(signs, geobase, limit):

    signs = signs.copy()
    signs = gpd.GeoDataFrame(signs, geometry=gpd.points_from_xy(signs.Longitude, signs.Latitude), crs='epsg:4326')
    if not limit.empty:
        signs = gpd.sjoin(signs, limit, op='within', how='inner').drop(columns='index_right')
        geobase = geobase.copy().rename(columns={'geometry':'road_geom'}).set_geometry('road_geom')
    geobase = gpd.sjoin(geobase, limit, op='intersects', how='inner').reset_index(drop=True)

    # keep only up to date sign without banners
    filter_ = ~signs.DESCRIPTION_RPA.str.startswith('PANONCEAU') &\
            ( (signs.DESCRIPTION_REP == 'Réel') |\
              (signs.DESCRIPTION_REP == 'En conception') )

    filtered_signs = signs[filter_].copy().reset_index(drop=True)

    # Transform fleche to chaining enum sharedstreet standard 
    #                               {1:start, 2:middle, 3:end}
    fleche_map={ 
        0: {
            -1:
                {0:2,  # no arrow -> middle
                2:1,   # left arrow -> start
                3:3},   # right arrow -> end
            1:
                {0:2,  # no arrow -> middle
                2:1,   # left arrow -> start
                3:3}   # right arrow -> end
        },
        1: {
            -1:
                {0:2,  # no arrow -> middle
                2:3,   # left arrow -> end
                3:1},  # right arrow -> start
            1:
                {0:2,  # no arrow -> middle
                2:1,   # left arrow -> start
                3:3}   # right arrow -> end
        },
        -1: {
            1:
                {0:2,  # no arrow -> middle
                2:3,   # left arrow -> end
                3:1},  # right arrow -> start
            -1:
                {0:2,  # no arrow -> middle
                2:1,   # left arrow -> start
                3:3}   # right arrow -> end
        },
        
    }

    # Compute closest road to each signs
    filtered_signs = rtreenearest(filtered_signs, geobase, ['road_geom', 'SENS_CIR', 'ID_TRC'])
    # Compute side of street of a sign on a street
    filtered_signs['side_of_street'] = list(map(right_or_left, filtered_signs.road_geom, filtered_signs.geometry))
    # Compute chaining of each sign on a roads
    filtered_signs['chainage'] = filtered_signs[['SENS_CIR', 'side_of_street', 'FLECHE_PAN']].apply(lambda x: fleche_map[x.SENS_CIR][x.side_of_street][x.FLECHE_PAN], axis=1)
    filtered_signs.crs = 'epsg:4326'

    # cleans unwanted chaining behavior on street cleaning sign
    filtered_signs = clean_street_cleaning_signs(filtered_signs, geobase)

    # save
    return filtered_signs.drop(columns='road_geom')


if __name__ == '__main__':

    signsfile = ''
    geobasefile = ''
    try:
      opts, args = getopt.getopt(sys.argv,"hs:g:",["signs=","geobase="])
    except getopt.GetoptError:
      print('signalec_preprocessing.py -s <signsfile> -g <geobasefile>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('signalec_preprocessing.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-s", "--signs"):
         signsfile = arg
      elif opt in ("-g", "--geobase"):
         geobasefile = arg

    signs = gpd.read_file(signsfile)
    geobase = gpd.read_file(geobasefile)

    main(signs, geobase).to_file('./data/signalec_signs_preprocessed.geojson', driver='GeoJSON')
