#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd

from shapely.ops import linemerge
from shapely.geometry import (
    LineString,
    MultiLineString
)


def main(paid_parking, limit):
    """_summary_

    Parameters
    ----------
    paid_parking : _type_
        _description_
    limit : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    paid_parking = gpd.GeoDataFrame(paid_parking, geometry=gpd.points_from_xy(paid_parking.Longitude, paid_parking.Latitude), crs='epsg:4326')
    if not limit.empty:
        paid_parking = gpd.sjoin(paid_parking, limit, op='within', how='inner').drop(columns='index_right')

    return  paid_parking


def post_processing(buff_length=7):
    """_summary_

    Parameters
    ----------
    buff_length : int, optional
        _description_, by default 7
    """
    place_loc = gpd.read_file('./output/shst/paid_parking_preprocessed.matched.geojson')

    place_loc['start'] = place_loc.location - (buff_length/2)
    place_loc['end'] = place_loc.location + (buff_length/2)

    paid_parking_buff = gpd.read_file('./output/shst/paid_parking_preprocessed.buffered.geojson')
    paid_parking_buff[['loc_start', 'loc_end']] = place_loc[['start', 'end']]
    paid_parking_buff.sort_values(['pp_sk_d_troncon', 'pp_no_place'], inplace=True)

    # compute cluster of overlaping segments
    g = paid_parking_buff.groupby('pp_sk_d_troncon')
    res = []
    for _, d in g:
        dd = d[:-1].reset_index().intersects(d[1:].reset_index())
        cluster_lines = [0]
        cluster_id = 0
        for i in range(dd.shape[0]):
            if dd[i] == False:
                cluster_id += 1
            cluster_lines.append(cluster_id)

        d['cluster_id'] = cluster_lines
        res.append(d)

    paid_parking_buff = pd.concat(res)

    # process overlapping segments
    columns=['referenceId', 'sideOfStreet', 'pp_sk_d_place', 'pp_no_place', 'pp_sk_d_troncon', 'pp_tarif_hr', 'geometry', 'loc_start', 'loc_end']
    res = []
    g = paid_parking_buff.sort_values(['pp_sk_d_troncon', 'pp_no_place']).groupby(['referenceId', 'pp_sk_d_troncon', 'cluster_id'])

    for _, data in g:
        if data.shape[0] == 1:
            res.append(list(data[columns].values[0]))
            continue

        assert data['pp_tarif_hr'].unique().shape[0] == 1, f'Error in clustering (Tarif) : {data}'
        assert data['referenceId'].unique().shape[0] == 1, f'Error in clustering (ReferenceId) : {data}'
        assert data['referenceId'].unique().shape[0] == 1, f'Error in clustering (SideOfStreet) : {data}'
        assert data['pp_sk_d_troncon'].unique().shape[0] == 1, f'Error in clustering (pp_sk_d_troncon) : {data}'

        place_to_line = linemerge(data['geometry'].to_list())
        to_add = [
            data['referenceId'].iloc[0],
            data['sideOfStreet'].iloc[0],
            list(data['pp_sk_d_place'].values),
            list(data['pp_no_place'].values),
            data['pp_sk_d_troncon'].iloc[0],
            data['pp_tarif_hr'].iloc[0],
            LineString([place_to_line.geoms[0].coords[0], place_to_line.geoms[-1].coords[-1]]) if isinstance(place_to_line, MultiLineString) else place_to_line,
            data['loc_start'].min(),
            data['loc_end'].max()
        ]
        res.append(to_add)


    paid_parking_buff_merge = gpd.GeoDataFrame(data=res, columns=columns, geometry='geometry', crs='epsg:4326')


    paid_parking_buff_merge['pp_sk_d_place'] = paid_parking_buff_merge['pp_sk_d_place'].apply(lambda x: x.__str__())
    paid_parking_buff_merge['pp_no_place'] = paid_parking_buff_merge['pp_no_place'].apply(lambda x: x.__str__())

    paid_parking_buff_merge.to_file('./output/shst/paid_parking_postprocessed.buffered.geojson', driver='GeoJSON')

if __name__ == '__main__':
    # TODO args passing
    pass
