""" Main module
"""
import argparse
import os
import sys


from signe.tranform.curblr_creation import main as populate_curblr
from signe.data import (
    load_data,
    SIG_RPA_NAME,
    SIG_STA_NAME,
    GEOBASE_NAME,
    HYDRANTS_NAME,
    GEO_LIMIT_NAME,
    PAID_PARKING_NAME
)
from signe.preprocessing import (
    process_catalog,
    process_mtl_paid_parking,
    process_signalec,
    process_fire_hydrants
)

OUTPUT_DIR = './output/'
SHST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'shst/')


def preprocess_signalec(datas):
    print(f'Preprocessing {SIG_STA_NAME} ...')
    sig_sta = process_signalec(
        datas[SIG_STA_NAME],
        datas[GEOBASE_NAME],
        datas[GEO_LIMIT_NAME]
    )
    sig_sta_path = os.path.join(OUTPUT_DIR, 'sig_sta_preprocessed.geojson')
    sig_sta.to_file(
        sig_sta_path,
        driver='GeoJSON'
    )
    print(f'\t Generated file {sig_sta_path}')
    print(f'{SIG_STA_NAME} preprocessing done.\n')


def preprocess_paid_parking(datas):
    print(f'Preprocessing {PAID_PARKING_NAME} ...')
    paid_parking = process_mtl_paid_parking(
        datas[PAID_PARKING_NAME],
        datas[GEO_LIMIT_NAME])
    paid_parking_path = os.path.join(
        OUTPUT_DIR,
        'paid_parking_preprocessed.geojson'
    )
    paid_parking.to_file(paid_parking_path, driver='GeoJSON')
    print(f'\t Generated file {paid_parking_path}')
    print(f'{PAID_PARKING_NAME} preprocessing done.\n')


def preprocess_catalogue(datas):
    print(f'Preprocessing {SIG_RPA_NAME} ...')
    sig_rpa = process_catalog(datas[SIG_RPA_NAME])
    sig_rpa_path = os.path.join(OUTPUT_DIR, 'sig_rpa_preprocessed.csv')
    sig_rpa.to_csv(sig_rpa_path, index=False)
    print(f'\t Generated file {sig_rpa_path}')
    print(f'{SIG_RPA_NAME} preprocessing done.\n')


def preprocess_hydrants(datas):
    print(f'Preprocessing {HYDRANTS_NAME} ...')
    hydrants = datas[HYDRANTS_NAME].copy()
    hydrants = process_fire_hydrants(
        datas[HYDRANTS_NAME],
        datas[GEOBASE_NAME],
        datas[GEO_LIMIT_NAME]
    )
    hydrants_path = os.path.join(OUTPUT_DIR, 'hydrants_preprocessed.geojson')
    hydrants.to_file(hydrants_path, driver='GeoJSON')
    print(f'\t Generated file {hydrants_path}')
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
        raise e
        print(f"An error occurred: {e.with_traceback()}")
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
    parser.add_argument(
        '-p',
        '--preprocessing',
        action='store_true',
        help='Enable preprocessing'
    )
    parser.add_argument(
        '-c',
        '--curblr',
        action='store_true',
        help='Start processing'
    )
    args = parser.parse_args()
    return args.preprocessing, args.curblr


if __name__ == '__main__':
    requires_preprocessing, should_convert = parse_arguments()
    if not (should_convert or requires_preprocessing):
        print('Nothing to do.')
        sys.exit(2)
    if requires_preprocessing:
        should_convert = preprocessing() and should_convert
    if should_convert:
        convert()
    sys.exit(0)
