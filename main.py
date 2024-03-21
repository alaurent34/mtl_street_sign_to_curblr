import os
import sys
import getopt
import argparse
import subprocess

import geopandas as gpd

from data import load_data
from connections import SOURCE
from signalec_preprocessing import main as signalec
from paid_parking_preprocessing import main as gsm_paid_parking
from paid_parking_preprocessing import post_processing
from catalogue_preprocessing import main as catalogue
from curblr_creation import main as populate_curblr

def parse_args() -> argparse.Namespace:
    """
    Argument parser for the script

    Return
    ------
    argparse.Namespace
        Parsed command line arguments
    """
    # Initialize parser
    parser = argparse.ArgumentParser(
            description="This script provide an automation to transform point to curblr"
            )

    # Adding optional argument
    parser.add_argument('-p', '--preprocessing', action="store_true",
                        dest='prepro', help='Compute preprocessing')
    parser.add_argument('-c', '--curblr', action="store_true", default=True,
                        dest='curblr', help='Compute transformation to curblr')

    # Read arguments from command line
    args = parser.parse_args()

    return args

def assert_same_crs(gdf1, gdf2):
    if gdf1.crs != gdf2.crs:
        return gdf1.to_crs(gdf2.crs)

    else:
        return gdf1

if __name__ == '__main__':

    config = parse_args()

    if config.prepro:
        print('Preprocessing...')
        # preprocessing of data
        datas = load_data(SOURCE)
        limits = assert_same_crs(datas['geo_limit'], datas['hydrants'])

        signs = signalec(datas['signs'], datas['geobase'], limits)
        paid_parking = gsm_paid_parking(datas['parking_slot'], limits)
        rpa = catalogue(datas['catalogue'])
        hydrants = datas['hydrants'].copy()
        if not limits.empty:
            hydrants = gpd.sjoin(hydrants, limits, how='inner', predicate='intersects')

        # saving information
        os.makedirs('./output/shst', exist_ok=True)
        signs.to_file('./output/signs_preprocessed.geojson', driver='GeoJSON')
        paid_parking.to_file('./output/paid_parking_preprocessed.geojson', driver='GeoJSON')
        rpa.to_csv('./output/catalogue_rpa_preprocessed.csv', index=False)
        hydrants.to_file('./output/hydrants_preprocessed.geojson', driver='GeoJSON')
        print('Done !')

        paid_parking_buf_length=7
        hydrants_buf_length=6

        try:
            subprocess.call(["shst", "match", "output/signs_preprocessed.geojson",
                             "--join-points",
                             "--join-point-sequence-field=chainage",
                             "--join-points-match-fields=CODE_RPA,PANNEAU_ID_RPA",
                             "--snap-intersections",
                             "--trim-intersections-radius=10",
                             "--search-radius=15",
                             "--out=output/shst/signs_preprocessed.geojson"])
            subprocess.call(["shst", "match", "output/paid_parking_preprocessed.geojson",
                             "--buffer-points",
                             f"--buffer-points-length={paid_parking_buf_length}",
                             "--snap-intersections",
                             "--trim-intersections-radius=10",
                             "--out=output/shst/paid_parking_preprocessed.geojson"])
            subprocess.call(["shst", "match", "output/hydrants_preprocessed.geojson",
                             "--buffer-points",
                             f"--buffer-points-length={hydrants_buf_length}",
                             "--snap-intersections",
                             "--trim-intersections-radius=10",
                             "--out=output/shst/hydrants_preprocessed.geojson"])
        except FileNotFoundError:
            print("shst not found. You have to run it manually before continuing.")
            sys.exit(3)

    if config.curblr:
        print('Transformation CurbLR')
        # paid parking post processing
        post_processing()
        populate_curblr()
        print('Done.')
        sys.exit()
