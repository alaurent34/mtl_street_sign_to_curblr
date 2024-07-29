import sys

import pandas as pd
import geopandas as gpd
import numpy as np
import osrm
import pyproj
from pyproj import CRS, transform
from shapely.geometry import Point, LineString

import tools.geom as geom

# Matcher
OSRM_MATCH_COLUMNS = ['u', 'v', 'lat_match', 'lng_match']
OSRM_2_MATCHER = {
    'u': 'u',
    'v': 'v',
    'lat_match': 'lat_match',
    'lng_match': 'lng_match'
}
UNIVERSAL_CRS = CRS.from_epsg(3857)
DEFAULT_CRS = CRS.from_epsg(4326)
MONTREAL_CRS = CRS.from_epsg(32188)

EARTH_RADIUS = 6371000


def create_segments(points, order, sequence):
    """ Tranform a series of point into a ligne in function of their order and
    sequence.

    points: List[Point]
        Collection of points
    order: List[float]
        Linear referencing of the point onto a segment

    Returns
    -------
    List[Tuple[float, float]]

    """
    # transform to array
    elements = list(zip(points, order, sequence))
    dtype = [('points', Point), ('order', float), ('sequence', int)]
    elements = np.array(elements, dtype=dtype)

    # sort elements by order and sequence
    elements = np.sort(elements, order=['order', 'sequence'])

    #start = elements[0][1]
    #end = elements[-1][1]

    # special case of 1 point
    if elements.shape[0] == 1:
        if elements[0][2] == 2:  # 2
            return [(0, np.inf)]
        elif elements[0][2] == 1:  # 1
            return [(elements[0][1], np.inf)]
        else:  # 3
            return [(0, elements[0][1])]

    lines = []
    curr_line = None
    for i in range(elements.shape[0]):
        # start a new segment
        if not curr_line:
            curr_line = []

            if elements[i][2] == 1:
                curr_line.append(elements[i][1])
            elif elements[i][2] > 1:
                curr_line.append(0)
                if elements[i][2] == 3:
                    curr_line.append(elements[i][1])
                    lines.append(curr_line)
                    curr_line = None
        elif elements[i][2] == 3:
            curr_line.append(elements[i][1])
            lines.append(curr_line)
            curr_line = None

    if curr_line:
        curr_line.append(np.inf)
        lines.append(curr_line)
        curr_line = None

    return lines

def point_to_line(data, roads, sequence_field, match_field,
                  linear_ref_field, roads_id_col='ID_TRC'):
    """ Allows a series of points to be snapped to the street and transformed
    into a street segment.

    Parameters
    ----------
    data: gpd.GeoDataFrame
        Point Geometry collection
    sequence_field: List[str]
        Specifies the name of the field containing point sequence info (1=start,
        2=middle, 3=terminus)
    match_field: List[str]
        When turning points into line segments, there may be multiple sequences
        of points on one street. Columns of the fields that must match in order
        for points to considered part of the same segment and merged into a
        line
    linear_reference_field: str
        Specifies the name of the field containing linear reference to the road
        network

    Return
    ------
    gpd.GeoDataFrame
        Collection of LineString
    """
    data = data.copy()
    roads = roads.to_crs(MONTREAL_CRS)

    groups = data.groupby(match_field)

    lines_df = []
    for match_field_g, g_data in groups:
        lines = create_segments(
                    points=g_data.geometry,
                    order=g_data[linear_ref_field],
                    sequence=g_data[sequence_field]
                )
        lines = pd.DataFrame(lines, columns=['start', 'end'])
        lines[match_field] = match_field_g
        lines_df.append(lines)

    lines_df = pd.concat(lines_df)

    # Extract lines from roads
    lines_df = lines_df.join(
            roads.set_index(roads_id_col)[['geometry', 'SENS_CIR']],
            on=roads_id_col,
            how='left'
    )
    lines_df.rename(columns={'geometry': 'road_geom'}, inplace=True)

    # reverse start_end when direction is reversed
    mask = ((lines_df.SENS_CIR == 0) & (lines_df.side_of_street == -1)) | \
            (lines_df.SENS_CIR == -1)
    lines_df = gpd.GeoDataFrame(lines_df, geometry='road_geom', crs=roads.crs)
    length = lines_df.loc[mask].geometry.length
    lines_df.loc[mask, 'start'] = length - lines_df.loc[mask, 'start']
    lines_df.loc[mask, 'end'] = length - lines_df.loc[mask, 'end']

    # find geometry
    lines_df['geometry'] = lines_df.apply(
            lambda x: LineString([
                x.road_geom.interpolate(x.start),
                x.road_geom.interpolate(x.end)
            ]) if x.start < x.end else \
                      LineString([
                x.road_geom.interpolate(x.end),
                x.road_geom.interpolate(x.start)
            ]), axis=1)

    lines_df = gpd.GeoDataFrame(lines_df, geometry='geometry', crs=roads.crs)
    lines_df.drop(columns='road_geom', inplace=True)

    return lines_df


def osrm_parse_matchs(response):
    """Retrieve match points
    """

    # create an empty response
    empty_response = {'location':[0,0], 'name':None, 'hint':None, 'matchings_index':None, 'waypoint_index':None, 'alternatives_count':None}
    # create list of matchs
    tracepoints = [x if x is not None else empty_response for x in response['tracepoints']]
    # create a DataFrame of matchs
    res = pd.DataFrame.from_records(tracepoints)

    return res


def osrm_parse_legs(response):
    """ Parse OSRM response
    """

    # retrieve each edge on match_points
    legs_df = pd.DataFrame(
        columns=['matchings_index', 'waypoint_index', 'u', 'v']
    )
    for i, matching in enumerate(response['matchings']):
        for j, leg in enumerate(matching['legs']):
            u = leg['annotation']['nodes'][0]
            v = leg['annotation']['nodes'][1]
            leg_i = j
            # append new row
            legs_df.loc[-1] = [i, leg_i, u, v]  # adding a row
            legs_df.index = legs_df.index + 1  # shifting index
            legs_df = legs_df.sort_index()  # sorting by index

            # on est au dernier point, il faut l'ajouter aussi
            if j == len(matching['legs']) - 1:
                u = leg['annotation']['nodes'][-2]
                v = leg['annotation']['nodes'][-1]
                leg_i = j+1
                # append last point
                legs_df.loc[-1] = [i, leg_i, u, v]  # adding a row
                legs_df.index = legs_df.index + 1  # shifting index
                legs_df = legs_df.sort_index()  # sorting by index

    # legs_df = legs_df.merge(gdf_edges, on=['u','v'], how='inner')
    legs_df = legs_df[['matchings_index', 'waypoint_index', 'u', 'v']]

    return legs_df


def osrm_parse_response(response):
    """
    doc
    """

    matchs = osrm_parse_matchs(response)
    legs = osrm_parse_legs(response)

    matchs = pd.merge(matchs, legs,
                      on=['matchings_index', 'waypoint_index'],
                      how='left')
    # format location
    locations = np.stack(matchs['location'], axis=0)
    matchs['lng_match'] = locations[:, 0]
    matchs['lat_match'] = locations[:, 1]

    return matchs[OSRM_MATCH_COLUMNS]


class MapMatcher():
    """
    doc
    """

    def __init__(self, host, return_edge=True):

        self.type = ''
        self.return_edge = return_edge
        self.host = host
        self.client = None

    def _start_client(self):
        client = osrm.Client(self.host, timeout=60)
        self._check_client(client)

    def _check_client(self, client):
        if isinstance(client, osrm.Client):
            self.type = 'osrm'
            self.client = client
            self.columns = OSRM_MATCH_COLUMNS
        else:
            raise TypeError('MapMatcher constructor called with incompatible client and dtype: {e}'.format(e=type(client)))

    def match(self, coords, **kwargs):
        if self.client is None:
            self._start_client()

        if self.type == 'osrm':
            # assert kwargs['edges_osm_df']
            # TODO: Checker pour ce param si osm
            return self._osrm_match(coords, **kwargs)
        else:
            raise NotImplementedError('There is no implementation for this matcher')

    def _osrm_match(self, coords, **kwargs):
        """ Coords is in format lat, lng np.array
        """

        coords = np.asarray(coords)
        assert coords.ndim == 2, 'Coordinates should be 2 dimensions'

        try:
            if kwargs:
                response = self.client.match(coordinates=coords[:, ::-1], **kwargs)
            else:
                response = self.client.match(
                    coordinates=coords[:, ::-1],
                    overview=osrm.overview.full,
                    annotations=True
                )
            if response['code'] != 'Ok':
                return pd.DataFrame()
        except osrm.OSRMClientException as e:
            print(f'Error happened when mapmatching : {e}')
            return pd.DataFrame()

        # retrieve paresed matchs
        matchs = osrm_parse_response(response)
        matchs = matchs.rename(columns=OSRM_2_MATCHER)

        # concat with ori points
        assert matchs.shape[0] == coords.shape[0], f'Something went wrong during map matching: match shape {matchs.shape[0]} != coords shape {coords.shape[0]}'

        # return edges info ?
        if self.return_edge:
            return matchs
        col = self.columns.copy()
        col.remove('u')
        col.remove('v')

        return matchs[col]
