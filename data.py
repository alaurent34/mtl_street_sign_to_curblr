import pymssql

def load_data(data_sources):
    """ Take a connection list of data to load and return all the loaded values
    """
    datas = {}
    for data_source in data_sources:
        print(f'Loading data : {data_source["name"]}')

        if data_source['type'] == 'sql_server':
            con = pymssql.connect(**data_source['connection']['args']['con'])
            data_source['connection']['args']['con'] = con

        datas[data_source['name']] = data_source['connection']['reader'](
            data_source['connection']['path'],
            **data_source['connection']['args']
        )

    return datas
