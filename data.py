import urllib.parse
from sqlalchemy import create_engine

def get_engine(server, database, driver, Trusted_Connection='yes', autocommit=True,
              fast_executemany=True, user='', pwd=''):
    """ Create a connection to a sql server via sqlalchemy
    Arguments:
    server -- The server name (str). e.g.: 'SQL2012PROD03'
    database -- The specific database within the server (str). e.g.: 'LowFlows'
    driver -- The driver to use for the connection (str). e.g.: SQL Server
    trusted_conn -- Is the connection to be trusted. Values are 'yes' or 'No' (str).
    """

    if driver == 'SQL Server':
        engine = create_engine(
            f"mssql+pyodbc://{server}/{database}"
            f"?driver={driver}"
            f"&Trusted_Connection={Trusted_Connection}"
            f"&autocommit={autocommit}",
            fast_executemany=fast_executemany
        )
    elif driver == 'postgresql':
        user = urllib.parse.quote_plus(user)
        pwd  = urllib.parse.quote_plus(pwd)
        engine = create_engine(
            f"postgresql+psycopg2://{user}:{pwd}@{server}/{database}"
        )
    else:
        raise NotImplementedError('No other connections supported')
    return engine

def load_data(data_sources):
    """ Take a connection list of data to load and return all the loaded values
    """
    datas = {}
    for data_source in data_sources:
        print(f'Loading data : {data_source["name"]}')

        if data_source['type'] == 'sql_server':
            #con = pymssql.connect(**data_source['connection']['args']['con'])
            con = get_engine(**data_source['connection']['args']['con'])
            data_source['connection']['args']['con'] = con

        datas[data_source['name']] = data_source['connection']['reader'](
            data_source['connection']['path'],
            **data_source['connection']['args']
        )

    return datas
