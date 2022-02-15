import pandas as pd
import geopandas as gpd

CON_DATABASE = {
    'Axes': {
        'server': 'prisqlbiprod01',
        'database': 'Axes',
        'autocommit': False,
    },
}

SOURCE = [
    {
        'name':'catalogue',
        'type':'flat_file',
        'connection':{
            'path':'https://data.montreal.ca/dataset/c5bf5e9c-528b-4c28-b52f-218215992e35/resource/0795f422-b53b-41ca-89be-abc1069a88c9/download/signalisation-codification-rpa.json', 
            'reader':pd.read_json,
            'args':{}
        }
    },
    {
        'name':'signs',
        'type':'flat_file',
        'connection':{
            'path':'https://data.montreal.ca/dataset/8ac6dd33-b0d3-4eab-a334-5a6283eb7940/resource/7f1d4ae9-1a12-46d7-953e-6b9c18c78680/download/signalisation_stationnement.csv', 
            'reader':pd.read_csv,
            'args':{}
        }
    },
    {
        'name':'geobase',
        'type':'flat_file',
        'connection':{
            'path':'https://data.montreal.ca/dataset/984f7a68-ab34-4092-9204-4bdfcca767c5/resource/9d3d60d8-4e7f-493e-8d6a-dcd040319d8d/download/geobase.json',
            'reader':gpd.read_file,
            'args':{}
        }
    },
    {
        'name':'hydrants',
        'type':'flat_file',
        'connection':{
            'path':'https://data.montreal.ca/dataset/cb4de65e-138b-4936-9d5c-2d9a0bc9b4ce/resource/434e80ca-4fec-4b38-a6b5-070a06757233/download/ati_geomatique.aqu_borneincendie_p_j.json',
            'reader':gpd.read_file,
            'args':{}
        }
    },
    {
        'name':'geo_limit',
        'type':'flat_file',
        'connection':{
            'path':'data/limit.geojson',
            'reader':gpd.read_file,
            'args':{}
        }
    },
    {
        'name':'parking_slot',
        'type':'sql_server',
        'connection':{
            'path':'''SELECT DISTINCT SK_D_Place, No_Place, SK_D_Troncon, P.Tarif_Hr, Latitude, Longitude
                          FROM D_Place P
                          WHERE 1=1 
                              AND P.Ind_Actif = 'Actif'
                              AND Ind_SurRue_HorsRue = 'Sur rue'
                              AND Latitude IS NOT NULL            -- Don't want null point object
                              AND MD_Vers_Courant = 'Oui'
                       ''',
            'reader':pd.read_sql,
            'args':{
                'con': CON_DATABASE['Axes'] 
            }
        }
    },

]
