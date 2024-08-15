# -*- coding: utf-8 -*-
"""
A module that hosts basic geometric calculations

Created on Wed Jun 9 2021 10:12:56

@author: lgauthier
@author: alaurent
"""
import math
from operator import itemgetter
import itertools
from typing import List, Tuple

import shapely
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import geopandas
from geopy.distance import geodesic, great_circle
from scipy.spatial import cKDTree
from pyproj import CRS

EARTH_RADIUS = 6371000
UNIVERSAL_CRS = CRS.from_epsg(3857)
DEFAULT_CRS = CRS.from_epsg(4326)
MONTREAL_CRS = CRS.from_epsg(32188)


Point = Tuple[float, float]


def localisation_distance(loc1: Point, loc2: Point):
    """Default distance measure between two localisations

    Parameters
    ----------
    loc1 : Tuple[float, float]
        first localisation. In latitude, longitude decimal format.
    loc2 : Tuple[float, float]
        second localisation. In latitude, longitude decimal format.

    Returns
    -------
    float
        distance in meter
    """
    return geodesic_distance(loc1, loc2)


def great_circle_distance(loc1: Point, loc2: Point) -> float:
    """Great circle formula between two localisations

    Parameters
    ----------
    loc1 : Tuple[float, float]
        first localisation. In latitude, longitude decimal format.
    loc2 : Tuple[float, float]
        second localisation. In latitude, longitude decimal format.

    Returns
    -------
    float
        great circle distance in meter
    """

    return great_circle(loc1[::-1], loc2[::-1]).m


def geodesic_distance(loc1: Point, loc2: Point) -> float:
    """Geodesic formula between two localisations

    Parameters
    ----------
    loc1 : Tuple[float, float]
        first localisation. In latitude, longitude decimal format.
    loc2 : Tuple[float, float]
        second localisation. In latitude, longitude decimal format.

    Returns
    -------
    float
        geodesic distance in meter
    """

    return geodesic(loc1[::-1], loc2[::-1]).m


def haversine_distance(loc1: Point, loc2: Point) -> float:
    """Haversine formula between two localisations

    Parameters
    ----------
    loc1 : Tuple[float, float]
        first localisation. In latitude, longitude decimal format.
    loc2 : Tuple[float, float]
        second localisation. In latitude, longitude decimal format.

    Returns
    -------
    float
        haversine distance in meters
    """
    ""

    lat1, lon1 = loc1
    lat2, lon2 = loc2

    # convert to radians
    lon1 = lon1 * math.pi / 180.0
    lon2 = lon2 * math.pi / 180.0
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2.0))**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0-a))
    km = EARTH_RADIUS * c

    return km * 1000


def vector_left_or_right(
    vector: shapely.LineString,
    point: shapely.LineString
) -> int:
    """
    Determines if a point is to the right or left of a vector.

    Parameters
    ----------
    vector : shapely.LineString
        The vector to compare. The object must be made of a maximum of two
        points.
    point : shapely.LineString
        The point to compare.

    Raises
    ------
    TypeError
        vector is not a shapely.LineString object or point shapely.Point
        object.
    ValueError
        The number of coordinates in vector is greater than 2.

    Returns
    -------
    result: int
        The result of the comparison is a -1/0/1 indication with the following
        meaning:
            -1 for "left"
            0 for "on the line" (neither right nor left)
            1 for "right"
    """
    # RIGHT HAND RULE: Right of forwards+++++++; Left of forwards-------

    # Vectorial multiplication of both vectors gives us information about the
    # orientation of the snapped vector
    # To follow the right hand rule, we must use:
    #       if z is (-), P is to the right of the spline and direction_y
    #       is thus +++++
    #       if z is (+), P is to the left of the spline and direction_y
    #       is thus ----

    if not isinstance(vector, shapely.LineString):
        raise TypeError(f'Expecting a shapely LineString object, received {vector.__class__}')
    if not isinstance(point, shapely.Point):
        raise TypeError(f'Expecting a shapely Point object, received {point.__class__}')

    if len(vector.coords.xy[0]) != 2:
        raise ValueError(f'Expecting 2 points, received {len(vector.coords.xy[0])}')

    v_x1, v_x2, v_y1, v_y2 = *vector.coords.xy[0], *vector.coords.xy[1]
    p_x, p_y = point.x, point.y

    result = (p_x - v_x1) * (v_y2 - v_y1) - (p_y - v_y1) * (v_x2 - v_x1)

    if result > 0 :
        return 1
    if result < 0 :
        return -1
    return 0


def multipointobject_to_points(
    multipointobject: shapely.geometry.base.BaseMultipartGeometry
) -> List[shapely.Point]:
    """
    Convert the coordinates of a shapely multi point geometry object into a
    list of shapely.Point objects

    Parameters
    ----------
    multipointobject : shapely.geometry.base.BaseMultipartGeometry
        The object to extract the points from.

    Returns
    -------
    points: List[shapely.Point, ...]
        The points, with all shapely.Point methods availaible.

    """
    x_list,y_list = multipointobject.coords.xy
    return [shapely.Point(x_list[i], y_list[i]) for i in range(len(x_list))]


def norm_2_distance(p_1:shapely.Point, p_2:shapely.Point)-> float:
    """Calculate the 2-norm distance (also called the Euclidean distance) between
    two points.

    Parameters
    ----------
    p_1: shapely.Point
        The first point.
    p_2: shapely.Point
        The second point.

    Returns
    -------
    result: float
        The distance between both points. This distance is always positive.
    """
    _d = p_1-p_2
    return math.sqrt(_d.x**2+_d.y**2)


def find_closest_to_point(candidates:shapely.geometry.base.BaseMultipartGeometry,
                          point:shapely.Point, great_circle:bool=True)-> shapely.LineString:
    """
    The the closest point on a multi point geometry (polyline, polygon, etc.)
    closest to a given point.

    Parameters
    ----------
    candidates : shapely.geometry.base.BaseMultipartGeometry
        Themulti part geometry object the calculate a distance from. This object
        must have a `geoms` attribute.
    point : shapely.Point
        The point whose distance with the multi part geometry must be calculated.
    great_circle : bool, optional
        When True, the geodesic distance is used. When False, the euclidien
        distance is used. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if float('.'.join(shapely.__version__.split('.')[:2])) >= 1.8:
        candidates = candidates.geoms

    closest = (None, np.inf)
    for candidate in candidates:
        if isinstance(candidate, shapely.Point):
            if great_circle:
                dist = geodesic((candidate.x, candidate.y), (point.x, point.y)).m
            else:
                dist = norm_2_distance(candidate, point)
        else:
            dist = candidate.distance(point)
        if dist < closest[-1]:
            closest = (candidate, dist)
        else:
            pass
    return closest[0]


def vectorize(points:List[shapely.Point])->shapely.LineString:
    """
    Transforms a list of shapely.Point into a shapely.LineString object

    Parameters
    ----------
    points : List[shapely.Point, ...]
        A list of points to form a polyline.

    Returns
    -------
    line: shapely.LineString

    """
    return shapely.LineString(coordinates=points)


def uv_coords(vector:shapely.LineString)->ArrayLike:
    """
    Returns the u and v components of a vector, corresponding to dx and dy.

    Parameters
    ----------
    vector : shapely.LineString
        The object must be made of a maximum of two points.

    Raises
    ------
    ValueError
        The number of coordinates in vector is greater than 2.

    Returns
    -------
    ArrayLike[float, float]
        The u and v components.

    """
    if len(vector.coords.xy[0]) != 2:
        raise ValueError(f'Expecting 2 points, received {len(vector.coords.xy[0])}')

    d_x = vector.coords.xy[0][-1] - vector.coords.xy[0][0]
    d_y = vector.coords.xy[1][-1] - vector.coords.xy[1][0]
    return np.array([d_x, d_y])


def dot_product(vector1:shapely.LineString, vector2:shapely.LineString)->float:
    """
    Calculate the dot product (also called scalar product) of two vectors.

    Parameters
    ----------
    vector1 : shapely.LineString
       The object must be made of a maximum of two points.
    vector2 : shapely.LineString
       The object must be made of a maximum of two points.

    Raises
    ------
    ValueError
        The number of coordinates in vector1 or vector2 is greater than 2.

    Returns
    -------
    float

    """
    if len(vector1.coords.xy[0]) != 2:
        raise ValueError(f'Expecting 2 points, received {len(vector1.coords.xy[0])}')
    if len(vector2.coords.xy[0]) != 2:
        raise ValueError(f'Expecting 2 points, received {len(vector2.coords.xy[0])}')

    a_1, a_2 = uv_coords(vector1)
    b_1, b_2 = uv_coords(vector2)
    return a_1*b_1 + a_2*b_2


def normalize_vector(vector:shapely.LineString)->shapely.LineString:

    '''from first_point, computes the new last_point such as the lenght equals 1'''
    _u, _v = uv_coords(vector) / vector.length
    p_1, _ = multipointobject_to_points(vector)
    return shapely.LineString(coordinates=[[p_1.x, p_1.y], [p_1.x + _u, p_1.y + _v]])


def azimuth(point1:shapely.Point, point2:shapely.Point)->float:
    '''azimuth between 2 shapely points (interval 0 - 360)'''
    angle = np.arctan2(point2[0] - point1[0], point2[-1] - point1[1])
    return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360


def line_azimuth(line: shapely.LineString | shapely.MultiLineString,
                 reversed_line: bool = False
                 ) -> float:
    """


    Parameters
    ----------
    line : shapely.LineString | shapely.MultiLineString
        DESCRIPTION.
    reversed_line : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    float
        DESCRIPTION.

    """
    if isinstance(line, shapely.MultiLineString):
        return None
    line = list(line.coords)
    return azimuth(line[0], line[-1]) if not reversed_line else azimuth(line[-1], line[0])


def angle_between_vectors(vector1:shapely.LineString, vector2:shapely.LineString,
                          as_degree:bool=False)-> float:
    """
    Calculates the absolute minimum angle (in radians) between two vectors.

    Parameters
    ----------
    vector1 : shapely.LineString
        First vector.
    vector2 : shapely.LineString
        Second vector.
    as_degree : bool, optional
        If True, returns the result in degrees. The default is False.

    Returns
    -------
    angle: float
        The minimum angle between vector1 and vector2.

    """
    #to avoid the machine error, lets first check if they are equal and force the zero
    if (uv_coords(vector1) == uv_coords(vector2)).all():
        angle = 0
    else:
        scal_prod = dot_product(normalize_vector(vector1), normalize_vector(vector2))
        scal_prod = np.clip(scal_prod, -1, 1)
        angle = math.acos(scal_prod)
    if as_degree:
        return math.degrees(angle)
    return angle


def multipointobject_left_or_right(
    multipointobject: shapely.MultiLineString | shapely.LineString,
    point: shapely.Point
) -> int:
    """Determines if the point is to the right or left of a multipointobject.

    Parameters
    ----------
    multipointobject : shapely.MultiLineString or shapely.LineString
        The object to compare

    point : shapely.Point
        The point to compare

    Returns
    -------
    result: int
        The result of the comparison is a -1/0/1 indication with the following
        meaning:
            -1 for "left"
            0 for "on the line" (neither right nor left)
            1 for "right"

    Note
    ----
    For Polygons, this is not an indication that the point is inside or outside
    of the polygons, since traversing the boundary in the opposite direction
    would give the opposite result regardless of wether the point is inside or
    outside.
    """
    # RIGHT HAND RULE: Right of forwards+++++++; Left of forwards-------

    # Vectorial multiplication of both vectors gives us information about the
    # orientation of the snapped vector
    # To follow the right hand rule, we must use:
    #       if z is (-), P is to the right of the spline and direction_y is
    #       thus +++++
    #       if z is (+), P is to the left of the spline and direction_y is
    #       thus ----
    if isinstance(multipointobject, shapely.MultiLineString):
        # find the closest part
        multipointobject = find_closest_to_point(multipointobject, point)

    # first, we try to parse points directly on the line
    if multipointobject.distance(point) == 0:
        return 0

    points = multipointobject_to_points(multipointobject)

    closest = find_closest_to_point(shapely.MultiPoint(points), point)
    index = points.index(closest)

    if index == 0:
        return vector_left_or_right(vectorize([points[0], points[1]]), point)
    if index == len(points)-1:
        return vector_left_or_right(vectorize([points[-2], points[-1]]), point)

    # at this point we found the closest point, which can be either the
    # starting point or the end point of a vector. For this reason we
    # need to test two vectors: the one linking i-1 and i and the one
    # linking i and i+1
    cands = [vectorize([points[index-1], points[index]]),
             vectorize([points[index], points[index+1]])]

    # two things can happen here:
    #  1 - The vector is placed such that a projection falls on one of
    #      the two vectors without needing the project those vectors
    #      out of their bounds.
    #  2 - The vector needs one of the vectors to be projected.
    #
    # Convexe junctions prensent an edge case that need to be avoided:
    # if the point is close to vector1, it could be left of it while
    # being right of the projection of vector 2 (or the opposite). To
    # avoid this, we check both sides and decide the outcome based on
    # the direction of the second vector compared to the first.

    # if the angle is 360°, there's not much we can do. Either everything
    # is left, or everything is right, or a combinaison of both and
    # it entirely depends on how you see the problem at hand. Here we
    # chose LEFT.
    if angle_between_vectors(cands[0], cands[1], as_degree=True) == 180:
        angle = -1
    else:
        side_1 = vector_left_or_right(cands[0], point)
        side_2 = vector_left_or_right(cands[1], point)
        #when both checks agree, we can't go wrong by choosing this answer
        if side_1 == side_2:
            angle = side_1
        #Otherwise, the point is actually considered to be on the opposite
        #side of where the second vector is pointing
        else:
            angle = -1 * vector_left_or_right(cands[0], points[index+1])
    return angle


def distance_line_from_point(
    line: shapely.LineString,
    point: shapely.Point
) -> float:
    """ Compute the geodesic distance (in meters) between a LineString and a
    Point. The distance is computed as the projected distance between the
    LineString and the point.

    Parameters
    ----------
    line : shapely.geometry.LineString
        LineString to compute the distance on
    point : shapely.geometry.Point
        The point to compute the distance on

    Returns
    -------
    dist_tmp : float
        The distance between 'point' and the projected point on the line.
        In meters.
    """
    point_projected = line.interpolate(line.project(point))
    dist_tmp = geodesic(
        (point.x, point.y),
        (point_projected.x, point_projected.y)
    ).m

    return dist_tmp


def polyline_to_vectors(line:shapely.LineString)->List[shapely.LineString]:
    """Divides a polyline in a list of 2-points segments.

    Parameters
    ----------
    line : shapely.LineString
        The polyline to divide.

    Raises
    ------
    ValueError
        A MultiLinestring was passed.

    Returns
    -------
    vectors : list[shapely.LineString]
        The list of segments.
    """

    if not line.is_simple:
        raise ValueError('Line must be simple.')

    line_size = len(line.coords)

    vectors = np.zeros((line_size-1, 2, 2))
    for i in range(line_size):
        if i == line_size -1:
            break
        vectors[i][0] = line.coords[i]
        vectors[i][1] = line.coords[i+1]

    return vectors


def get_closest_sub_line(line:shapely.LineString, point:shapely.Point)->shapely.LineString:
    """Find the closest segment (vector) of a polyline from a given point.

    Parameters
    ----------
    line : shapely.LineString
        Polyline to analyse.
    point : shapely.Point
        Point to compare.

    Raises
    ------
    ValueError
        A MultiLinestring was passed.

    Returns
    -------
    subline : shapely.LineString
        The closest segment.
    """

    if not line.is_simple:
        raise ValueError('Line must be simple.')

    line_size = len(line.coords)

    subsline = map(shapely.LineString, polyline_to_vectors(line))
    points = itertools.repeat(point, line_size-1)

    id_line = np.argmin(list(map(distance_line_from_point, subsline, points)))
    id_line = int(id_line)
    subline = shapely.LineString([line.coords[id_line], line.coords[id_line+1]])

    return subline


def is_left(line:shapely.LineString, point:shapely.Point) -> bool:
    """Determine if a point is on the left of a given line

    Parameters
    ----------
    line : shapely.LineString
        Line.
    point : shapely.Point
        Point to compare.

    Returns
    -------
    bool
    """
    x_list, y_list = line.xy
    _a = shapely.Point(x_list[0], y_list[0])
    _b = shapely.Point(x_list[-1], y_list[-1])

    return ((_b.x - _a.x)*(point.y - _a.y) - (_b.y - _a.y)*(point.x - _a.x)) > 0


def ckdnearest(gdf_a: geopandas.GeoDataFrame,
               gdf_b: geopandas.GeoDataFrame,
               gdf_b_cols: List[str] = None
               ) -> geopandas.GeoDataFrame:
    """For each geometry in gdf_a, find the closest geometry in gdf_b using an KDTree.

    Parameters
    ----------
    gdf_a : geopandas.GeoDataFrame
        Right dataframe.
    gdf_b : geopandas.GeoDataFrame
        Left dataframe.
    gdf_b_cols : List[str, ...], optional
        The columns to join from B to A. The default is ['Place'].

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Returns gdfa enriched with columns from gdfb according to the geometry
        proximity.

    """
    if gdf_b_cols is None:
        raise ValueError("Must provide at least one column name")
    if not isinstance(gdf_b_cols, (list, tuple, np.ndarray)):
        gdf_b_cols = [gdf_b_cols]
    _a = np.concatenate(
        [np.array(geom.coords) for geom in gdf_a.geometry.to_list()])
    _b = [np.array(geom.coords) for geom in gdf_b.geometry.to_list()]
    b_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, _b)))]))
    _b = np.concatenate(_b)
    ckd_tree = cKDTree(_b)
    dist, idx = ckd_tree.query(_a, k=1)
    idx = itemgetter(*idx)(b_ix)
    gdf = pd.concat(
        [gdf_a, gdf_b.loc[idx, gdf_b_cols].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf


def rtreenearest(gdf_a: geopandas.GeoDataFrame,
                 gdf_b: geopandas.GeoDataFrame,
                 gdf_b_cols: List[str] = None
                 ) -> geopandas.GeoDataFrame:
    """For each geometry in gdf_a, find the closest geometry in gdf_b using an Rtree.

    Parameters
    ----------
    gdf_a : geopandas.GeoDataFrame
        Right dataframe.
    gdf_b : geopandas.GeoDataFrame
        Left dataframe.
    gdf_b_cols : List[str, ...], optional
        The columns to join from B to A. The default is ['ID'].

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Returns gdfa enriched with columns from gdf_b according to the geometry
        proximity and 'dist' columns.

    """
    if gdf_b_cols is None:
        raise ValueError("Must provide at least one column name")
    if not isinstance(gdf_b_cols, (list, tuple, np.ndarray)):
        gdf_b_cols = [gdf_b_cols]

    gdf_b = gdf_b.copy()
    gdf_b.loc[-1, gdf_b_cols] = np.nan

    idx = []
    dist_l = []
    for point in gdf_a.geometry.to_list():
        road_i = gdf_b.sindex.nearest(point, return_all=True)[1]
        roads = gdf_b.loc[road_i, gdf_b.geometry.name].to_list()

        if not roads:
            idx.append(-1)
            dist_l.append(-1)
            continue

        dist = np.inf
        id_nearest = 0

        for j, road in enumerate(roads):
            place_projected = road.interpolate(road.project(point))
            vector = shapely.LineString([point, place_projected])
            # plus coure distance de la place à la route

            if vector.length < dist:
                dist = vector.length
                id_nearest = j

        idx.append(road_i[id_nearest])
        dist_l.append(dist)

    gdf = pd.concat([gdf_a,
                     gdf_b.loc[idx, gdf_b_cols].reset_index(drop=True),
                     pd.Series(dist_l, name='dist')],
                    axis=1)
    return gdf


def cut_line_at_dist(line:shapely.LineString,
                     dist: float) -> Tuple[shapely.LineString, shapely.LineString]:
    """ Cuts a line in two at a distance from its starting point.
    This is taken from shapely manual and modified to always return
    a tuple of two Linestring.

    Parameters
    ----------
    line : shapely.LineString
        Line to cut
    distance : float
        Cuting distance

    Returns
    -------
    Tuple[shapely.LineString, shapely.LineString]
        First and last half of the Line cut at distance `dist`
    """
    if np.round(dist,0) <= 0.0:
        return [[], shapely.LineString(line)]
    if np.ceil(dist) >= line.length:
        return [shapely.LineString(line), []]

    coords = list(line.coords)
    for i, p in enumerate(coords):
        point_dist = line.project(shapely.Point(p))
        if point_dist < dist:
            continue
        if point_dist == dist:
            return [
                shapely.LineString(coords[:i+1]),
                shapely.LineString(coords[i:])
            ]

        cp = line.interpolate(dist)
        return [
            shapely.LineString(coords[:i] + [(cp.x, cp.y)]),
            shapely.LineString([(cp.x, cp.y)] + coords[i:])
        ]

def split_line_with_points(line:shapely.LineString,
                           points:List[shapely.Point]) -> List[shapely.LineString]:
    """Splits a line string in several segments considering a list of points.

    The points used to cut the line are assumed to be in the line string
    and given in the order of appearance they have in the line string.

    Taken from https://stackoverflow.com/a/39574007 with minor modifications.

    Parameters
    ----------
    line: shapely.LineString
        Line to split
    points: List[shapely.Point]
        Collection of point where to split the linestring. Should be in
        sequential order.

    Returns
    -------
    List[shapely.LineString]
        Lines segments cut at each point

    Example
    -------
    >>> line = LineString( [(1,2), (8,7), (4,5), (2,4), (4,7), (8,5), (9,18),
    ...        (1,2),(12,7),(4,5),(6,5),(4,9)] )
    >>> points = [Point(2,4), Point(9,18), Point(6,5)]
    >>> [str(s) for s in split_line_with_points(line, points)]
    ['LINESTRING (1 2, 8 7, 4 5, 2 4)', 'LINESTRING (2 4, 4 7, 8 5, 9 18)',
    'LINESTRING (9 18, 1 2, 12 7, 4 5, 6 5)', 'LINESTRING (6 5, 4 9)']

    """
    segments = []
    current_line = line
    for p in points:
        d = current_line.project(p)
        seg, current_line = cut_line_at_dist(current_line, d)
        segments.append(seg)
    segments.append(current_line)
    return segments


vectorized_r_o_l = np.vectorize(multipointobject_left_or_right)
vectorized_dist = np.vectorize(distance_line_from_point)
