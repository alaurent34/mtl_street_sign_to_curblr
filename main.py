import argparse
import os
import subprocess
import sys

import geopandas as gpd

from catalogue_preprocessing import main as catalogue
from curblr_creation import main as populate_curblr
from data import load_data, SIG_RPA_NAME, SIG_STA_NAME, GEOBASE_NAME, HYDRANTS_NAME, GEO_LIMIT_NAME, PAID_PARKING_NAME
from paid_parking_preprocessing import main as gsm_paid_parking
from signalec_preprocessing import main as signalec

OUTPUT_DIR = './output/'
SHST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'shst/')


def run_shst_process(input_path, output_path, shst_args):
    print(f'\t Running SHST process ...')
    process = subprocess.run([
        "shst", "match", input_path,
        *shst_args,
        f"--out={output_path}"
    ], capture_output=True, text=True)
    print(f'\t {process.stderr}')
    print(f'\t {process.stdout}')


def preprocess_signalec(datas):
    print(f'Preprocessing {SIG_STA_NAME} ...')
    sig_sta = signalec(datas[SIG_STA_NAME], datas[GEOBASE_NAME], datas[GEO_LIMIT_NAME])
    sig_sta_path = os.path.join(OUTPUT_DIR, 'sig_sta_preprocessed.geojson')
    sig_sta_shst_path = os.path.join(SHST_OUTPUT_DIR, 'sig_sta_preprocessed_shst.geojson')
    sig_sta.to_file(sig_sta_path, driver='GeoJSON')
    print(f'\t Generated file {sig_sta_path}')
    run_shst_process(sig_sta_path, sig_sta_shst_path, [
        "--join-points",
        "--join-point-sequence-field=chainage",
        "--join-points-match-fields=CODE_RPA,PANNEAU_ID_RPA",
        "--snap-intersections",
        "--trim-intersections-radius=10",
        "--search-radius=15"
    ])
    print(f'{SIG_STA_NAME} preprocessing done.\n')


def preprocess_paid_parking(datas):
    print(f'Preprocessing {PAID_PARKING_NAME} ...')
    paid_parking = gsm_paid_parking(datas[PAID_PARKING_NAME], datas[GEO_LIMIT_NAME])
    paid_parking_path = os.path.join(OUTPUT_DIR, 'paid_parking_preprocessed.geojson')
    paid_parking_shst_path = os.path.join(SHST_OUTPUT_DIR, 'paid_parking_preprocessed_shst.geojson')
    paid_parking.to_file(paid_parking_path, driver='GeoJSON')
    print(f'\t Generated file {paid_parking_path}')
    run_shst_process(paid_parking_path, paid_parking_shst_path, [
        "--buffer-points",
        "--buffer-points-length=7",
        "--snap-intersections",
        "--trim-intersections-radius=10"
    ])
    print(f'{PAID_PARKING_NAME} preprocessing done.\n')


def preprocess_catalogue(datas):
    print(f'Preprocessing {SIG_RPA_NAME} ...')
    sig_rpa = catalogue(datas[SIG_RPA_NAME])
    sig_rpa_path = os.path.join(OUTPUT_DIR, 'sig_rpa_preprocessed.csv')
    sig_rpa.to_csv(sig_rpa_path, index=False)
    print(f'\t Generated file {sig_rpa_path}')
    print(f'{SIG_RPA_NAME} preprocessing done.\n')


def preprocess_hydrants(datas):
    print(f'Preprocessing {HYDRANTS_NAME} ...')
    hydrants = datas[HYDRANTS_NAME].copy()
    if not datas[GEO_LIMIT_NAME].empty:
        hydrants = gpd.sjoin(hydrants, datas[GEO_LIMIT_NAME], how='inner', op='intersects')
    hydrants_path = os.path.join(OUTPUT_DIR, 'hydrants_preprocessed.geojson')
    hydrants_shst_path = os.path.join(SHST_OUTPUT_DIR, 'hydrants_preprocessed_shst.geojson')
    hydrants.to_file(hydrants_path, driver='GeoJSON')
    print(f'\t Generated file {hydrants_path}')
    run_shst_process(hydrants_path, hydrants_shst_path, [
        "--buffer-points",
        "--buffer-points-length=6",
        "--snap-intersections",
        "--trim-intersections-radius=10"
    ])
    print(f'{HYDRANTS_NAME} preprocessing done.\n')


def preprocessing():
    result = True
    try:
        datas = load_data()
        os.makedirs(SHST_OUTPUT_DIR, exist_ok=True)

        preprocess_signalec(datas)
        # preprocess_paid_parking(datas) FIXME
        preprocess_catalogue(datas)
        preprocess_hydrants(datas)
    except Exception as e:
        print(f"An error occurred: {e}")
        result = False
    return result


def convert():
    result = True
    try:
        print('Transformation CurbLR')
        # post_processing()  # paid parking post processing # FIXME
        populate_curblr()
        print('Done.')
    except Exception as e:
        print(f"An error occurred: {e}")
        result = False
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description='Main python script')
    parser.add_argument('-p', '--preprocessing', action='store_true', help='Enable preprocessing')
    parser.add_argument('-c', '--curblr', action='store_true', help='Start processing')
    args = parser.parse_args()
    return args.preprocessing, args.curblr


if __name__ == '__main__':
    requires_preprocessing, should_convert = parse_arguments()
    if not (should_convert or requires_preprocessing):
        print('Nothing to do.')
        sys.exit(2)
    if requires_preprocessing:
        should_convert = preprocessing()
    if should_convert:
        convert()
    sys.exit(0)
