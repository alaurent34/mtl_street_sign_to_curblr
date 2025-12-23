#!/usr/bin/env python
# coding: utf-8

import re
import json
import os
import datetime
import pandas as pd
import geopandas as gpd
from shapely import wkt

from cygne.preprocessing.signalec.pannonceaux_rpa import PANONCEAUX

def load_wkt(wkt_str):
    try:
        return wkt.loads(wkt_str)
    except ValueError:
        return None

def update_pannonceau(rpa, pannonceau):
    for key, value in pannonceau.items():
        if key not in rpa.keys():
            rpa.update(pannonceau)
        if key in ['timeSpans', 'userClasses']:
            rpa[key].extend(value)
        if key == 'rule':
            for rule_k, rule_v in value.items():
                rpa[key][rule_k] = rule_v


def main():

    # Get signs
    signs_path = './output/signs_preprocessed.geojson'
    if not os.path.exists(signs_path):
        print(f"Error: File {signs_path} not found.")
        return

    signs = gpd.read_file(signs_path)
    
    # GeoJSON/CurbLR format requires EPSG:4326 (Latitude/Longitude)
    # Montreal data is often in EPSG:32188 (MTM Zone 8, in meters)
    if signs.crs is not None and signs.crs.to_string() != "EPSG:4326":
        print(f"Info: Converting coordinates from {signs.crs} to EPSG:4326 (Lat/Lon)...")
        signs = signs.to_crs("EPSG:4326")
    elif signs.crs is None:
        # If no CRS is defined, assume MTM Zone 8 (standard Mtl) and convert
        print("Warning: No coordinate system detected. Assuming MTM Zone 8 (EPSG:32188).")
        signs.set_crs("EPSG:32188", inplace=True)
        signs = signs.to_crs("EPSG:4326")

    if 'DESCRIPTION_REP' not in signs.columns:
        if 'DESCRIPTION_RPA' in signs.columns:
            signs['DESCRIPTION_REP'] = signs['DESCRIPTION_RPA']
        elif 'Description' in signs.columns:
            signs['DESCRIPTION_REP'] = signs['Description']
        else:
            print("Warning: Description column not found, using default values.")
            signs['DESCRIPTION_REP'] = 'Réel'

    poteau_candidates = ['POTEAU_ID_POT', 'POTEAU_ID', 'ID_POTEAU', 'NO_POTEAU', 'No_Support', 'ID']
    
    found_col = None
    for col in poteau_candidates:
        match = next((c for c in signs.columns if c.lower() == col.lower()), None)
        if match:
            found_col = match
            break
    
    if found_col:
        signs['POTEAU_ID_POT'] = signs[found_col]
    else:
        print("Info: No Sign Post ID found. Generating ID based on GPS position.")
        # Transform geometry to text to serve as unique identifier
        signs['POTEAU_ID_POT'] = signs['geometry'].apply(lambda x: str(x))

    if 'POSITION_POP' not in signs.columns:
         if 'POSITION' in signs.columns:
             signs['POSITION_POP'] = signs['POSITION']
         else:
             signs['POSITION_POP'] = 0 
    
    if 'DESCRIPTION_REP' in signs.columns:
        print(f"DEBUG - Sign statuses found: {signs['DESCRIPTION_REP'].unique()}")

    # We comment out the strict filter to allow data through (even 'Enlevé' - Removed)
    # filter_ = ((signs.DESCRIPTION_REP == 'Réel') | (signs.DESCRIPTION_REP == 'En conception'))
    # signs = signs[filter_].copy().reset_index(drop=True)
    
    print(f"DEBUG - Number of signs to process: {len(signs)}")

    shst_signs_path = "./output/shst/signs_preprocessed.joined.geojson"
    if not os.path.exists(shst_signs_path):
        print(f"WARNING: SharedStreets file not found ({shst_signs_path}).")
        print("Linear matching step skipped. Using raw data.")
        
        # Convert GeoDataFrame -> JSON Dict
        # Using the GeoDataFrame converted to Lat/Lon here
        signs_json = json.loads(signs.to_json())
        signs_shst = {'features': []}
        
        # Reconstruct compatible structure
        for feat in signs_json['features']:
            props = feat['properties']
            
            props['referenceId'] = str(props.get('ID_TRC', 'unknown'))
            props['sideOfStreet'] = props.get('side_of_street', 'unknown')
            props['section'] = [props.get('start', 0), props.get('end', 0)]
            
            props['pp_code_rpa'] = props.get('CODE_RPA', 'inconnu')
            props['pp_panneau_id_rpa'] = str(props.get('POTEAU_ID_POT', 'unknown'))
            
            signs_shst['features'].append(feat)

    else:
        with open(shst_signs_path) as f:
            signs_shst = json.load(f)

    rpa_path = './output/sig_rpa_preprocessed.csv' 
    if not os.path.exists(rpa_path):
        print(f"Error: Catalog file {rpa_path} not found.")
        return
        
    rpa_to_curb_data = pd.read_csv(rpa_path)
    rpa_to_curb_data['CurbLR'] = rpa_to_curb_data.CurbLR.apply(lambda x: eval(x)[0])

    rpa_to_curb = rpa_to_curb_data.set_index('CODE_RPA')['CurbLR'].to_dict()
    panonceaux = PANONCEAUX
    error_curb = [{"rule": {"activity": "no standing", "priorityCategory": "error"}}]

    # Sort signs by position on a pole for banner processing
    desc_col = 'DESCRIPTION_RPA' if 'DESCRIPTION_RPA' in signs.columns else 'DESCRIPTION_REP'
    signs[desc_col] = signs[desc_col].astype(str)

    panonceaux_boyer = signs[
        signs.POTEAU_ID_POT.isin(
            signs[signs[desc_col].str.startswith('PANONCEAU')].POTEAU_ID_POT
        )
    ].sort_values(['POTEAU_ID_POT', 'POSITION_POP'], ascending=True).copy()


    for pot, pan_pot in panonceaux_boyer.groupby('POTEAU_ID_POT'):
        this_pan=0
        panon_rpa = ''
        for idx, row in pan_pot.iterrows():
            gate = 1
            val_desc = row[desc_col]
            
            if re.match(r'.*PANONCEAU.*', str(val_desc)):
                this_pan = 1
                gate = 0
                panon_rpa = row['CODE_RPA']

            if this_pan == 1 and gate == 1:
                this_pan = 0
                if row['CODE_RPA'] in rpa_to_curb and panon_rpa in panonceaux:
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
    geojson['type'] = 'FeatureCollection'
    geojson['features'] = []

    for feature in signs_shst['features']:
        props = feature.get('properties', {})
        
        # Intelligent fallback
        code_rpa = props.get('pp_code_rpa', props.get('CODE_RPA'))
        obj_id = props.get('pp_panneau_id_rpa', props.get('POTEAU_ID_POT'))
        
        section = props.get('section', [0,0])
        if section is None: section = [0,0]

        newTargetFeature = {
            'type':feature['type'],
            'geometry':feature['geometry'],
            'properties':{
                'location':{
                    'shstRefId':props.get('referenceId', str(props.get('ID_TRC', 'unknown'))),
                    'sideOfStreet':props.get('sideOfStreet', props.get('side_of_street')),
                    'shstLocationStart':section[0],
                    'shstLocationEnd':section[1],
                    'assetType':'sign',
                    'assetSubType':code_rpa,
                    'objectId':str(obj_id),
                },
                'regulations':[rpa_to_curb[code_rpa]] if code_rpa in rpa_to_curb.keys() else error_curb
              }
            }
        geojson['features'].append(newTargetFeature)

    print(f'Signs to CurbLR done. ({len(geojson["features"])} features processed)')
    
    # Add paid parking
    pp_path = './output/shst/paid_parking_postprocessed.buffered.geojson'
    if os.path.exists(pp_path):
        with open(pp_path) as f:
            paid_parking = json.load(f)

        for feature in paid_parking['features']:
            newTargetFeature = {
                'type':feature['type'],
                'geometry':feature['geometry'],
                'properties':{
                    'location':{
                        'shstRefId':feature['properties'].get('referenceId'),
                        'sideOfStreet':feature['properties'].get('sideOfStreet'),
                        'shstLocationStart':feature['properties'].get('loc_start'),
                        'shstLocationEnd':feature['properties'].get('loc_end'),
                        'assetType':'PlaceTarifées',
                        'assetSubType':feature['properties'].get('pp_sk_d_troncon'),
                        'objectId':feature['properties'].get('pp_no_place'),
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
                                'fees':[feature['properties'].get('pp_tarif_hr')],
                                'durations':[60]
                            }]
                        }
                    }]
                }
                }
            geojson['features'].append(newTargetFeature)
        print('Paid parking to CurbLR done.')
    else:
        print(f"Skipping Paid Parking conversion (File {pp_path} missing)")

    hydrants_path = './output/shst/hydrants_preprocessed.buffered.geojson'
    if os.path.exists(hydrants_path):
        with open(hydrants_path) as f:
            hydrants_data = json.load(f)

        for feature in hydrants_data['features']:
            newTargetFeature = {
                'type':feature['type'],
                'geometry':feature['geometry'],
                'properties':{
                    'location':{
                        'shstRefId':feature['properties'].get('referenceId'),
                        'sideOfStreet':feature['properties'].get('sideOfStreet'),
                        'shstLocationStart':feature['properties']['section'][0],
                        'shstLocationEnd':feature['properties']['section'][1],
                        'assetType':'Fire hydrant',
                        'objectId':feature['properties'].get('pp_id_aq_bi'),
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
    else:
        print(f"Skipping Hydrants conversion (File {hydrants_path} missing)")

    output_path = './output/curblr_limit.curblr.json'
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=True)
    print(f"SUCCESS: CurbLR file generated: {output_path}")

if __name__ == '__main__':
    main()