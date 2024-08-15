""" Helper module to load needed datasets

Provide function fetch_data, load_data, load_file_data, load_sql_data
"""
import io
import geopandas as gpd
import pandas as pd
import requests

from signe.data.configs import DFLT_HEADER, SOURCE_FILES
from signe.io.sqlalchemy_utils import get_engine


def fetch_data(
    source: dict[str, str],
    file_type: str
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Fetches data from a specified source.

    Parameters
    ----------
    source: dict[str, str]
        A dictionary object containing the source type and path.
        - source['type']: The type of the data source. Must be either
        'remote' or 'local'.
        - source['path']: The path or URL of the data source.

    file_type: str
        The type of the data file. Must be one of the following:
        'json', 'csv', 'geofile'.

    Returns
    -------
    pandas.DataFrame | geopandas.GeoDataFrame
        The fetched data, represented as either a pandas DataFrame
        or a geopandas GeoDataFrame.

    Raise
    -----
    ValueError
        source file type not supported

    Note
    ----
    - For 'json' and 'csv' file types, the data is read using pandas.
    - For 'geofile' file type, the data is read using geopandas.
    - If the source type or file type is not supported, a ValueError will be
    raised.
    - If there is an error while fetching or reading the data, an error
    message will be printed and None will be returned.

    Examples
    --------
    # Fetch data from a remote JSON file
    source = {'type': 'remote', 'path': 'https://example.com/data.json'}
    file_type = 'json'
    data = fetch_data(source, file_type)
    print(data)
    >>> Failed to fetch data: Failed to fetch data from
    >>> https://example.com/data.json: 404
    """
    try:
        source_type = source['type']
        path = source['path']

        if source_type == 'remote':
            response = requests.get(
                path,
                headers=DFLT_HEADER,
                timeout=600
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to fetch data from {path}: {response.status_code}"
                )
            content = response.content
        elif source_type == 'local':
            with open(path, 'rb') as file:
                content = file.read()
        else:
            raise ValueError(
                "Unsupported source type. Please use 'remote' or 'local'."
            )

        if file_type == 'json':
            data = pd.read_json(io.BytesIO(content))
        elif file_type == 'csv':
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_type == 'geofile':
            data = gpd.read_file(io.BytesIO(content))
        else:
            raise ValueError(
                "Unsupported file type. Please use 'json', " +
                "'csv', or 'geofile'."
            )

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
                     'args': {
                         'host': 'example.com',
                         'user': 'user',
                         'password': 'pass',
                         'database': 'db'
                     },
                     'query': 'SELECT * FROM table_a'},
                    {'name': 'Source B',
                     'args': {
                         'host': 'example.com',
                         'database': 'db',
                         'trusted_connection': True
                      },
                     'query': 'SELECT * FROM table_b'}]
    result = load_sql_data(data_sources)

    Note:
    - This method uses the `pymssql` library to connect to the SQL database.
    - Each connection is closed after retrieving the data."""
    datas = {}
    for ds in data_sources:
        ds_name = ds['name']
        print(f'Loading {ds_name} data ... ', end='')
        con = get_engine(driver='SQLServer', **ds['args'])
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
