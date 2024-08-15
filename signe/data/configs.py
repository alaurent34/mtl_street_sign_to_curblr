""" Default config for Montreal open data
"""
SIG_RPA_NAME = 'sig_rpa'
SIG_RPA_URL = 'https://data.montreal.ca/dataset/c5bf5e9c-528b-4c28-b52f-218215992e35/resource/0795f422-b53b-41ca-89be-abc1069a88c9/download/signalisation-codification-rpa.json'

SIG_STA_NAME = 'sig_sta'
SIG_STA_URL = 'https://data.montreal.ca/dataset/8ac6dd33-b0d3-4eab-a334-5a6283eb7940/resource/7f1d4ae9-1a12-46d7-953e-6b9c18c78680/download/signalisation_stationnement.csv'

GEOBASE_NAME = 'geobase'
GEOBASE_URL = 'https://data.montreal.ca/dataset/984f7a68-ab34-4092-9204-4bdfcca767c5/resource/9d3d60d8-4e7f-493e-8d6a-dcd040319d8d/download/geobase.json'

HYDRANTS_NAME = 'hydrants'
HYDRANTS_URL = 'https://data.montreal.ca/dataset/cb4de65e-138b-4936-9d5c-2d9a0bc9b4ce/resource/434e80ca-4fec-4b38-a6b5-070a06757233/download/ati_geomatique.aqu_borneincendie_p_j.json'

GEO_LIMIT_NAME = 'geo_limit'
GEO_LIMIT_PATH = 'data/limit.geojson'

PAID_PARKING_NAME = 'paid_parking'
PAID_PARKING_QUERY = '''
    SELECT DISTINCT
      SK_D_Place,
      No_Place,
      SK_D_Troncon,
      P.Tarif_Hr,
      Latitude,
      Longitude
    FROM D_Place P
    WHERE 1=1
      AND P.Ind_Actif = 'Actif'
      AND Ind_SurRue_HorsRue = 'Sur rue'
      AND Latitude IS NOT NULL
      AND MD_Vers_Courant = 'Oui'
'''

SOURCE_FILES = [
    {
        'name': SIG_RPA_NAME,
        'type': 'json',
        'source': {
            'type': 'remote',
            'path': SIG_RPA_URL,
            'args': {}
        }
    },
    {
        'name': SIG_STA_NAME,
        'type': 'csv',
        'source': {
            'type': 'remote',
            'path': SIG_STA_URL,
            'args': {}
        }
    },
    {
        'name': GEOBASE_NAME,
        'type': 'geofile',
        'source': {
            'type': 'remote',
            'path': GEOBASE_URL,
            'args': {}
        }
    },
    {
        'name': HYDRANTS_NAME,
        'type': 'geofile',
        'source': {
            'type': 'remote',
            'path': HYDRANTS_URL,
            'args': {}
        }
    },
    {
        'name': GEO_LIMIT_NAME,
        'type': 'geofile',
        'source': {
            'type': 'local',
            'path': GEO_LIMIT_PATH,
            'args': {}
        }
    },
]

SOURCE_SQL = [
    {
        'name': PAID_PARKING_NAME,
        'query': PAID_PARKING_QUERY,
        'args': {
            'host': 'prisqlbiprod01',
            'database': 'Axes',
            'autocommit': False,
            'trusted_connection': True
        }
    }
]

DFLT_HEADER = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' +
        'AppleWebKit/537.36 (KHTML, like Gecko)' +
        'Chrome/91.0.4472.124 Safari/537.36'
    )
}
