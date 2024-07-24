import io
import geopandas as gpd
import pandas as pd
import requests
import pymssql

# Shorter variable names for URLs and names
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
PAID_PARKING_QUERY = '''SELECT DISTINCT SK_D_Place, No_Place, SK_D_Troncon, P.Tarif_Hr, Latitude, Longitude
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
            'server': 'prisqlbiprod01',
            'database': 'Axes',
            'autocommit': False,
        }
    }
]


def fetch_data(source, file_type):
    """
    Fetches data from a specified source.

    Parameters:
    - source (dict): A dictionary object containing the source type and path.
        - source['type']: The type of the data source. Must be either 'remote' or 'local'.
        - source['path']: The path or URL of the data source.

    - file_type (str): The type of the data file. Must be one of the following: 'json', 'csv', 'geofile'.

    Returns:
    - data (pandas.DataFrame or geopandas.GeoDataFrame): The fetched data, represented as either a pandas DataFrame or a geopandas GeoDataFrame.

    Note:
    - For 'json' and 'csv' file types, the data is read using pandas.
    - For 'geofile' file type, the data is read using geopandas.
    - If the source type or file type is not supported, a ValueError will be raised.
    - If there is an error while fetching or reading the data, an error message will be printed and None will be returned.

    Examples:
    # Fetch data from a remote JSON file
    source = {'type': 'remote', 'path': 'https://example.com/data.json'}
    file_type = 'json'
    data = fetch_data(source, file_type)
    """
    try:
        source_type = source['type']
        path = source['path']

        if source_type == 'remote':
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(path, headers=headers)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch data from {path}: {response.status_code}")
            content = response.content
        elif source_type == 'local':
            with open(path, 'rb') as file:
                content = file.read()
        else:
            raise ValueError("Unsupported source type. Please use 'remote' or 'local'.")

        if file_type == 'json':
            data = pd.read_json(io.BytesIO(content))
        elif file_type == 'csv':
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_type == 'geofile':
            data = gpd.read_file(io.BytesIO(content))
        else:
            raise ValueError("Unsupported file type. Please use 'json', 'csv', or 'geofile'.")

        return data

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None


def load_file_data(data_sources):
    """
    Loads data from multiple file sources.

    :param data_sources: A list of dictionaries describing the file sources.
           Each dictionary should contain the following keys:
           - 'name': The name of the data source.
           - 'source': The source path of the file.
           - 'type': The type of the file.

    :return: A dictionary containing the loaded data from each file source.
    """
    datas = {}
    for ds in data_sources:
        ds_name = ds['name']
        print(f'Loading {ds_name} data ... ', end='')
        datas[ds_name] = fetch_data(source=ds['source'], file_type=ds['type'])
        print('OK')
    return datas


def load_sql_data(data_sources):
    """
    Load SQL data from multiple data sources.

    Parameters:
    - data_sources (list): A list of dictionaries representing the data sources.
                           Each dictionary should contain the following keys:
                           - 'name' (str): The name of the data source.
                           - 'args' (dict): A dictionary containing the arguments
                                           to connect to the SQL database.
                           - 'query' (str): The SQL query to execute on the database.

    Returns:
    - datas (dict): A dictionary mapping the data source names to the corresponding
                    pandas DataFrame objects containing the SQL query results.

    Example Usage:
    data_sources = [{'name': 'Source A',
                     'args': {'host': 'example.com', 'user': 'user', 'password': 'pass', 'database': 'db'},
                     'query': 'SELECT * FROM table_a'},
                    {'name': 'Source B',
                     'args': {'host': 'example.com', 'user': 'user', 'password': 'pass', 'database': 'db'},
                     'query': 'SELECT * FROM table_b'}]
    result = load_sql_data(data_sources)

    Note:
    - This method uses the `pymssql` library to connect to the SQL database.
    - Each connection is closed after retrieving the data."""
    datas = {}
    for ds in data_sources:
        ds_name = ds['name']
        print(f'Loading {ds_name} data ... ', end='')
        con = pymssql.connect(**ds['args'])
        datas[ds_name] = pd.read_sql(ds['query'], con)
        con.close()
        print('OK')
    return datas


def load_data():
    file_data = load_file_data(SOURCE_FILES)
    # sql_data = load_sql_data(SOURCE_SQL) # FIXME : An error occurred: (20009, b'DB-Lib error message 20009, severity 9:\nUnable to connect: Adaptive Server is unavailable or does not exist (prisqlbiprod01)\n')
    # return {**file_data, **sql_data}
    return file_data

if __name__ == "__main__":
    data = load_data()
    for key, value in data.items():
        print(f"{key}: {type(value)}")
