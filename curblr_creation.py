#!/usr/bin/env python
# coding: utf-8

import re
import os
import json
import datetime
import pandas as pd
import geopandas as gpd
from shapely import wkt

from helpers.curblr import CurbLRHelper
from signalec.pannonceaux_rpa import PANONCEAUX

def load_wkt(wkt_str):
    try: 
        return wkt.loads(wkt_str)
    except ValueError:
        return None

def update_pannonceau(rpa, pannonceau):
    for key, value in pannonceau.items():
        # ajout d'une règle qui n'existe pas
        if not key in rpa.keys():
            print(key)
            rpa.update(panonceau)
        if key in ['timeSpans', 'userClasses']:
            rpa[key].extend(value)
        if key == 'rule':
            for rule_k, rule_v in value.items():
                rpa[key][rule_k] = rule_v 


def main():

    # Get signs
    signs = gpd.read_file('./output/signs_preprocessed.geojson')

    # Filter only real pans
    filter_ = ((signs.DESCRIPTION_REP == 'Réel') |
               (signs.DESCRIPTION_REP == 'En conception'))

    signs = signs[filter_].copy().reset_index(drop=True)

    # Get SharedStreets processed signs
    with open("./output/shst/signs_preprocessed.joined.geojson") as f: 
        signs_shst = json.load(f)

    # Get RPA to Curb data
    rpa_to_curb_data = pd.read_csv('./output/catalogue_rpa_preprocessed.csv')
    rpa_to_curb_data['CurbLR'] = rpa_to_curb_data.CurbLR.apply(lambda x: eval(x)[0])

    # JSON CurbLR data
    rpa_to_curb = rpa_to_curb_data.set_index('CODE_RPA')['CurbLR'].to_dict()
    panonceaux = PANONCEAUX

    # Make an error regulations
    error_curb = [{"rule": {
                    "activity": "no standing",
                    "priorityCategory": "error",
                }}]

    # Sort signs by position on a pole for banner processing
    panonceaux_boyer = signs[
        signs.POTEAU_ID_POT.isin(
            signs[signs.DESCRIPTION_RPA.str.startswith('PANONCEAU')].POTEAU_ID_POT
        )
    ].sort_values(['POTEAU_ID_POT', 'POSITION_POP'], ascending=True).copy()


    # Update CurbLR sign code with extra banner information
    for pot, pan_pot in panonceaux_boyer.groupby('POTEAU_ID_POT'):
        this_pan=0
        panon_rpa = ''
        for idx, row in pan_pot.iterrows():
            gate = 1
            if re.match(r'.*PANONCEAU.*', row['DESCRIPTION_RPA']):
                this_pan = 1
                gate = 0 
                panon_rpa = row['CODE_RPA']
                
            if this_pan == 1 and gate == 1:
                this_pan = 0
                if panon_rpa in panonceaux.keys():
                    #rpa_to_curb[row['CODE_RPA']].update(panonceaux[panon_rpa])
                    update_pannonceau(rpa_to_curb[row['CODE_RPA']], panonceaux[panon_rpa])

    # CurbLR Creation
    geojson = {};
    geojson["manifest"] = {
        "createdDate": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "lastUpdatedDate": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "curblrVersion": "1.1.0",
        "priorityHierarchy": ["no standing", "construction", "temporary restriction", "restricted standing", "standing", "restricted loading", "loading", "restricted parking", "paid parking", "free parking"],
        "timeZone": "America/Montréal",
        "currency": "CAD",
        "authority": {
         "name": "Agence de mobilité durable",
         "url": "https://www.agencemobilitedurable.ca/"
        }
    }
    geojson['type'] = 'FeatureCollection';
    geojson['features'] = [];

    for feature in signs_shst['features']:
        newTargetFeature = {
            'type':feature['type'],
            'geometry':feature['geometry'],
            'properties':{
                'location':{
                    'shstRefId':feature['properties']['referenceId'],
                    'sideOfStreet':feature['properties']['sideOfStreet'],
                    'shstLocationStart':feature['properties']['section'][0],
                    'shstLocationEnd':feature['properties']['section'][1],
                    'assetType':'sign',
                    'assetSubType':feature['properties']['pp_code_rpa'],
                    'objectId':feature['properties']['pp_panneau_id_rpa'],
                },
                'regulations':[rpa_to_curb[feature['properties']['pp_code_rpa']]] if feature['properties']['pp_code_rpa'] in rpa_to_curb.keys() else error_curb
              }
            }
        geojson['features'].append(newTargetFeature)

    print('Signs to CurbLR done.')
    # Add paid parking

    with open('./output/shst/paid_parking_postprocessed.buffered.geojson') as f:
        paid_parking = json.load(f)

    for feature in paid_parking['features']:
        newTargetFeature = {
            'type':feature['type'],
            'geometry':feature['geometry'],
            'properties':{
                'location':{
                    'shstRefId':feature['properties']['referenceId'],
                    'sideOfStreet':feature['properties']['sideOfStreet'],
                    'shstLocationStart':feature['properties']['loc_start'],
                    'shstLocationEnd':feature['properties']['loc_end'],
                    'assetType':'PlaceTarifées',
                    'assetSubType':feature['properties']['pp_sk_d_troncon'],
                    'objectId':feature['properties']['pp_no_place'],
                },
                'regulations':[{
                    "rule": {
                        "activity": "parking",
                        "priorityCategory": "paid parking",
                        "payment": 'true'
                    },
                    "timeSpans": [{
                        "timesOfDay": [
                            {"from": "09:00", "to": "21:00"}
                        ]
                    }],               
                    "payment":{
                        'rate':[{
                            'fees':[feature['properties']['pp_tarif_hr']],
                            'durations':[60]
                        }]
                    }
                }]
              }
            }
        geojson['features'].append(newTargetFeature)
    print('Paid parking to CurbLR done.')

    with open('./output/shst/hydrants_preprocessed.buffered.geojson') as f:
        paid_parking = json.load(f)

    for feature in paid_parking['features']:
        newTargetFeature = {
            'type':feature['type'],
            'geometry':feature['geometry'],
            'properties':{
                'location':{
                    'shstRefId':feature['properties']['referenceId'],
                    'sideOfStreet':feature['properties']['sideOfStreet'],
                    'shstLocationStart':feature['properties']['section'][0],
                    'shstLocationEnd':feature['properties']['section'][1],
                    'assetType':'Fire hydrant',
                    'objectId':feature['properties']['pp_id_aq_bi'],
                },
                'regulations':[{
                    "rule": {
                        "activity": "no parking",
                        "priorityCategory": "no parking",
                    },
                }]
              }
            }
        geojson['features'].append(newTargetFeature)
    print('Fire hydrants to CurbLR done.')

    with open('./output/curblr_limit.curblr.json', 'w') as f:
        json.dump(geojson, f, indent=True)
