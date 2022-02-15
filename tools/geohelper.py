import itertools
from operator import itemgetter
import warnings

import pandas as pd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString
from shapely.ops import transform
from scipy.spatial import cKDTree
from geopy.distance import geodesic

from .cgeom import Polyline2D
from .cgeom import Point as cPoint


UNIVERSAL_CRS = CRS.from_epsg(3857)
DEFAULT_CRS = CRS.from_epsg(4326)
MONTREAL_CRS = CRS.from_epsg(32188)

EARTH_RADIUS = 6371000

def distance(line, point):
    """ Compute the distance bewtwen a LineString and a Point.
    The distance is computed as the projected distance between the LineString
    and the point.

    Parameters
    ----------
    line : shapely.geometry.LineString
        LineString to compute the distance on
    point : shapely.geometry.Point
        The point to compute the distance on 

    Returns
    -------
    dist_tmp : int
        The distance between 'point' and the projected point on the line
        
    """
    point_projected = line.interpolate(line.project(point))
    dist_tmp = geodesic((point.x, point.y), (point_projected.x, point_projected.y)).m

    return dist_tmp

def line_to_vectors(line):

    if not line.is_simple:
        raise ValueError('Line must be simple.')

    line_size = len(line.coords)

    vectors = np.zeros((line_size-1, 2, 2))
    for i in range(len(line.coords)):
        if i == line_size -1:
            break
        vectors[i][0] = line.coords[i]
        vectors[i][1] = line.coords[i+1]

    return vectors

def get_closest_sub_line(line, point):
    
    if not line.is_simple:
        raise ValueError('Line must be simple.')
    
    line_size = len(line.coords)

    subsline = map(LineString, line_to_vectors(line))
    points = itertools.repeat(point, line_size-1)
    
    id_line = np.argmin(list(map(distance, subsline, points)))
    id_line = int(id_line)
    subline = LineString([line.coords[id_line], line.coords[id_line+1]])
    
    return subline

def right_or_left(line, point, crs=DEFAULT_CRS, proj_crs=MONTREAL_CRS):
        ''' Determines if the point is to the right or left of self
        
        RIGHT HAND RULE: Right of forwards+++++++; Left of forwards-------)    
        
        vectorial multiplication of both vectors gives us information about the orientation of the snapped vector
        To follow the right hand rule, we must use: 
                       if z is (-), P is to the right of the spline and direction_y is thus +++++
                       if z is (+), P is to the left of the spline and direction_y is thus ----         
        
        Parameters
        ----------
        line: shapely.geometry.LineString
            The road
        point: shapely.geometry.Point
            The plaque

        Returns
        -------
        -1 for "left"
         0 for "on the line" (neither right nor left)
         1 for "right"
         2 for None
        
        '''
        if line is None:
            return 2
        #if line.xy[0].shape[0] == 0 or line.xy[1].shape[0] == 0:
       #     return 2
        if not line.is_simple:
            raise ValueError('Line must be simple.')

        #local = crs
        #proj = proj_crs
        #project = pyproj.Transformer.from_crs(wgs84, local, always_xy=True).transform

        line = Polyline2D.from_shapely(line)#, crs=crs)
        point = cPoint.from_shapely(point)#, crs=crs)

        # transform
        # line = line.to_crs(proj_crs)
        # point = point.to_crs(proj_crs)

        return line.left_or_right(point) 

        #if line is None:
        #    return 2

        ## get subline
        #if len(line.coords) > 2:
        #    line = get_closest_sub_line(line, point)
        
        #ax = line.xy[0][0]
        #ay = line.xy[1][0]
        
        #bx = line.xy[0][-1]
        #by = line.xy[1][-1]
    
        #d = (point.x - ax) * (by - ay) - (point.y - ay)*(bx - ax)
        #if d > 0 :
        #    return 1
        #elif d < 0 :
        #    return -1
        #else:
        #    return 0

vectorized_r_o_l = np.vectorize(right_or_left)
vectorized_dist = np.vectorize(distance)

def ckdnearest(gdfA, gdfB, gdfB_cols=['Place']):
    A = np.concatenate(
        [np.array(geom.coords) for geom in gdfA.geometry.to_list()])
    B = [np.array(geom.coords) for geom in gdfB.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)
    gdf = pd.concat(
        [gdfA, gdfB.loc[idx, gdfB_cols].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf

def rtreenearest(gdfA, gdfB, gdfB_cols=['ID']):
    """ There is a implementation change in geopandas.sindex.nearest that will take change in next upgrade.

    FutureWarning: sindex.nearest using the rtree backend was not previously documented and this behavior
    is deprecated in favor of matching the function signature provided by the pygeos backend (see
    PyGEOSSTRTreeIndex.nearest for details)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return warning_rtreenearest(gdfA, gdfB, gdfB_cols)

def warning_rtreenearest(gdfA, gdfB, gdfB_cols=['ID']):
    """ Compute the spatial jointure on two GeoDataFrame based on the operation "nearest".
    """
    A = gdfA.geometry.to_list()
    rtree = gdfB.sindex
    
    idx = []
    dist_l = []
    for point in A:
        road_i = list(rtree.nearest((point.x, point.y), 4))
        roads = gdfB.loc[road_i, gdfB.geometry.name].to_list()

        dist = np.inf
        id_nearest = 0

        for j, road in enumerate(roads):
            place_projected = road.interpolate(road.project(point))
            vector = LineString([point, place_projected])
            # plus coure distance de la place à la route
            dist_tmp = vector.length
        
            if dist_tmp < dist:
                dist = dist_tmp
                id_nearest = j
                
        idx.append(road_i[id_nearest])
        dist_l.append(dist)
        
    gdf = pd.concat(
        [gdfA, gdfB.loc[idx, gdfB_cols].reset_index(drop=True), pd.Series(dist_l, name='dist')], axis=1)
    return gdf
