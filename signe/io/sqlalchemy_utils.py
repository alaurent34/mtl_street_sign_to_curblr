""" Module providing function to read data from database
"""
import urllib.parse
from sqlalchemy import create_engine


def get_engine(
    host: str,
    database: str,
    driver: str,
    trusted_connection: str = 'yes',
    autocommit: bool = True,
    fast_executemany: bool = True,
    user: str = '',
    pwd: str = '',
    **kwargs
):
    """ Create a connection to a sql server via sqlalchemy

    Arguments
    ---------
    server: str
        The server name. e.g.: 'SQL2012PROD03'
    database: str
        The specific database within the server. e.g.: 'LowFlows'
    driver: str
        The driver to use for the connection. e.g.: SQL Server
    trusted_conn: str, optional
        Is the connection to be trusted. Values are 'yes' or 'No', by default
        'yes'.
    autocommit : bool, optional
        Automatically commit after transaction, by default True
    fast_executemany : bool, optional
        MSSQL parameter, by default True
    user : str, optional
        Login, by default ''
    pwd : str, optional
        Password, by default ''
    """

    if driver == 'SQL Server':
        engine = create_engine(
            f"mssql+pyodbc://{host}/{database}"
            f"?driver={driver}"
            f"&Trusted_Connection={trusted_connection}"
            f"&autocommit={autocommit}",
            fast_executemany=fast_executemany,
            **kwargs
        )
    elif driver == 'postgresql':
        user = urllib.parse.quote_plus(user)
        pwd = urllib.parse.quote_plus(pwd)
        engine = create_engine(
            f"postgresql+psycopg2://{user}:{pwd}@{host}/{database}",
            **kwargs
        )
    else:
        raise NotImplementedError('No other connections supported')
    return engine
