import os
import sys
import getopt
import subprocess

import geopandas as gpd

from data import load_data
from connections import SOURCE
from signalec_preprocessing import main as signalec
from paid_parking_preprocessing import main as gsm_paid_parking
from paid_parking_preprocessing import post_processing
from catalogue_preprocessing import main as catalogue
from curblr_creation import main as populate_curblr

if __name__ == '__main__':
   
    prepro = False # for debug
    curblr = True
    try:
      opts, args = getopt.getopt(sys.argv,"hpc",["preprocessing","curblr"])
    except getopt.GetoptError:
      print('main.py -p --preprocessing -c --curblr')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('main.py -p --preprocessing -c --curblr')
         sys.exit()
      elif opt in ("-p", "--preprocessing"):
         prepro = True
      elif opt in ("-c", "--curblr"):
         curblr = True

    if not (curblr or prepro):
        print('Nothing to do.')
        sys.exit(2)

    if prepro:
        print('Preprocessing...')
        # preprocessing of data
        datas = load_data(SOURCE)
        signs = signalec(datas['signs'], datas['geobase'], datas['geo_limit'])
        paid_parking = gsm_paid_parking(datas['parking_slot'], datas['geo_limit'])
        rpa = catalogue(datas['catalogue'])
        hydrants = datas['hydrants'].copy()
        if not datas['geo_limit'].empty:
            hydrants = gpd.sjoin(hydrants, datas['geo_limit'], how='inner', op='intersects')

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

    if curblr:
        print('Transformation CurbLR')
        # paid parking post processing
        post_processing()
        populate_curblr()
        print('Done.')
        sys.exit()
