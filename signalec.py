""" Main module
"""
import argparse
import os
import sys
import traceback

from cygne.tranform.curblr_creation import main as populate_curblr
from cygne.data import (
    load_data,
    SIG_RPA_NAME,
    SIG_STA_NAME,
    GEOBASE_NAME,
    HYDRANTS_NAME,
    GEO_LIMIT_NAME,
    PAID_PARKING_NAME
)
from cygne.preprocessing import (
    process_mtl_paid_parking,
    process_signalec,
    process_fire_hydrants,
    process_catalog
)
from cygne.preprocessing.paid_parking_preprocessing import post_processing

OUTPUT_DIR = './output/'
SHST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'shst/')


def preprocess_signalec(datas):
    print(f'Preprocessing {SIG_STA_NAME} ...')
    sig_sta = process_signalec(
        datas[SIG_STA_NAME],
        datas[GEOBASE_NAME],
        datas[GEO_LIMIT_NAME]
    )

    if 'CODE_RPA' in sig_sta.columns:
        sig_sta['CODE_RPA'] = sig_sta['CODE_RPA'].astype(str).str.strip().str.upper()

    # Try to normalize the ID column name right now
    # to make the next script's job easier.
    poteau_candidates = ['POTEAU_ID', 'ID_POTEAU', 'NO_POTEAU', 'No_Support', 'ID']
    if 'POTEAU_ID_POT' not in sig_sta.columns:
        found = False
        for col in poteau_candidates:
            # Case-insensitive search
            match = next((c for c in sig_sta.columns if c.lower() == col.lower()), None)
            if match:
                print(f"\tStandardization: Renaming '{match}' to 'POTEAU_ID_POT'")
                sig_sta['POTEAU_ID_POT'] = sig_sta[match]
                found = True
                break
        if not found:
            print("\tWarning: Post ID not found at this stage.")

    sig_sta_path = os.path.join(OUTPUT_DIR, 'signs_preprocessed.geojson')
    sig_sta.to_file(
        sig_sta_path,
        driver='GeoJSON'
    )
    print(f'\t Generated file {sig_sta_path}')
    print(f'{SIG_STA_NAME} preprocessing done.\n')


def preprocess_paid_parking(datas):
    print(f'Preprocessing {PAID_PARKING_NAME} ...')
    # Check if raw data exists
    if datas.get(PAID_PARKING_NAME) is None:
        print(f"\tSkipping {PAID_PARKING_NAME} (Data not found)")
        return

    paid_parking = process_mtl_paid_parking(
        datas[PAID_PARKING_NAME],
        datas[GEO_LIMIT_NAME]
    )

    # Extra safety: If processing failed or returned None
    if paid_parking is None or paid_parking.empty:
        print(f"\tWarning: Processing of {PAID_PARKING_NAME} returned empty result.")
        return

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
    
    sig_rpa.columns = sig_rpa.columns.str.strip()

    if 'CODE_RPA' in sig_rpa.columns:
        sig_rpa['CODE_RPA'] = sig_rpa['CODE_RPA'].astype(str).str.strip().str.upper()

    # CREATION OF MISSING COLUMN
    if 'DESCRIPTION_RPA' in sig_rpa.columns:
        sig_rpa['DESCRIPTION_REP'] = sig_rpa['DESCRIPTION_RPA']
    elif 'Description' in sig_rpa.columns:
        sig_rpa['DESCRIPTION_REP'] = sig_rpa['Description']
    elif 'DESCRIPTION' in sig_rpa.columns:
        sig_rpa['DESCRIPTION_REP'] = sig_rpa['DESCRIPTION']
    
    # Final check
    if 'DESCRIPTION_REP' not in sig_rpa.columns:
        print(f"\tWARNING: Unable to create DESCRIPTION_REP. Available columns: {sig_rpa.columns.tolist()}")
        sig_rpa['DESCRIPTION_REP'] = "Inconnu"

    sig_rpa_path = os.path.join(OUTPUT_DIR, 'sig_rpa_preprocessed.csv')
    sig_rpa.to_csv(sig_rpa_path, index=False)
    print(f'\t Generated file {sig_rpa_path}')
    print(f'{SIG_RPA_NAME} preprocessing done.\n')


def preprocess_hydrants(datas):
    print(f'Preprocessing {HYDRANTS_NAME} ...')
    if datas.get(HYDRANTS_NAME) is None:
        print(f"\tSkipping {HYDRANTS_NAME} (Data not found)")
        return

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
        preprocess_paid_parking(datas)
        preprocess_catalogue(datas)
        preprocess_hydrants(datas)
    except Exception:
        print("An error occurred during preprocessing:")
        traceback.print_exc()
        result = False
    return result


def convert():
    result = True
    try:
        print('CurbLR Transformation')
        # Smart check for SHST files
        # If "matched" file exists, we can run parking meter post-processing
        if os.path.exists('./output/shst/paid_parking_preprocessed.matched.geojson'):
             post_processing()
        else:
             print("\tSkipping paid parking post-processing (Matched file not found)")

        populate_curblr()
        print('Done.')
    except Exception:
        print("An error occurred during conversion:")
        traceback.print_exc()
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
        print('Nothing to do. Use -p for preprocessing or -c for conversion.')
        sys.exit(2)
    if requires_preprocessing:
        success = preprocessing()
        # If preprocessing fails, stop everything
        if not success:
            sys.exit(1)
            
    if should_convert:
        convert()
    sys.exit(0)