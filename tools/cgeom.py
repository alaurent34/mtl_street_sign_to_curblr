# -*- coding: utf-8 -*-
"""
Éditeur de Spyder





@author: laurent.gauthier
"""
######################
# Top level imports
######################
import re, warnings, math
import numpy as np
import operator

######################
# Specialized imports
######################
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely
import geopandas
import pandas
import shapely.wkt
import pyproj
import numbers

##############
### Utils
##############
def rounding(x, base=5, how='round'):
    """Version of math.round/math.ceil/math.floor that ceils to the nearest base multiple."""
    if how == 'round':
        func = round
    elif how == 'floor':
        func = math.floor
    elif how == 'ceil':
       func = math.ceil
    else:
        raise ValueError("`how` must be one of {'round', 'ceil', 'floor'}, "+
                         f"received {how}")

    if isinstance(base, int):
        return int(base * func((float(x)/base)))
    if isinstance(base, float):
        return base * func((float(x)/base))


def catch_machine_error(num, tolerance=10**-10):
    """Round the number just above the machine error threshold."""
    return rounding(num, base=tolerance)

def eval_truth(first, operator_text, second):
    """Evaluate a statement to see if it's True or False.

    Parameters
    ----------
    first : object
        The first object to compare.
    operator_text : string
        The operateur must be one of {'>', '<', '>=',
        '<=', '=', '==', '!='}.
    second : object
        The second object to compare.

    Returns
    -------
    Answer: bool

    Example
    -------
    >>> get_thruth(1, '>=', 2)
    False

    """
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq,
           '==':operator.eq,
           '!=':operator.ne
           }
    return ops[operator_text](first, second)

def sort2lists(list1,list2,ascending=True):
    """Sort list2 according to the sorting of the content of list1.

    Parameters
    ----------
    list1 : list
        must contain values that can be sorted.
    list2 : list
        may contain any kind of data.
    ascending : bool
        If True, sort list 1 from smallest to highest. If False, sort in the reverse order.

    Returns
    -------
    s1, s2: lists
        The sorted list1 and list2.

    Example
    -------
    >>> list1 = [2,6,8,1]
    >>> list2 = [1,2,3,4]
    >>> sort2lists(list1, list2, ascending=False)
    [8,6,2,1], [3,2,1,4]

    >>> sort2lists(list1, list2, ascending=True)
    [1,2,6,8], [4,1,2,3]

    """
    indexes = list(range(len(list1)))
    indexes.sort(key=list1.__getitem__)

    s1 = [list1.__getitem__(index) for index in indexes]
    s2 = [list2.__getitem__(index) for index in indexes]

    if ascending is True:
        return s1, s2
    return list(reversed(s1)), list(reversed(s2))

def getNestingLevel(item):
    count = 1
    if isinstance(item, (list, tuple, np.ndarray)):
        count += getNestingLevel(item[0])
    else:
        return count -1
    return count

def multireplace(string, replacements, ignore_case=False):
    """Given a string and a replacement map, it returns the replaced string.

    Parameters
    ----------
    string: str
        The string to execute replacements on.
    replacements: dict
        Replacement dictionary {value to find: value to replace}.

    Return
    ------
    Text: str
        The text containing the replacements.
    """
    if ignore_case:
        replacements = dict((pair[0].lower(), pair[1]) for pair in sorted(replacements.items()))
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)), re.I if ignore_case else 0)

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0).lower() if ignore_case else match.group(0)], string)

def get_key_from_numvalue(some_dict, val_pos=None, return_first=False, which='min'):
    """Get the key of some_dict containing the minimum/maximum value.

    If the value is a list or a tuple, val_pos has to be set to the index
    in of the value the list/tuple that is to be used for the match.

    Parameters
    ----------
    some_dict : dict
        The dictionnary to treat.
    val_pos : None or int
        If None, the function will try to match directly on the value of
        some_dict.get(key). If the some_dict.get(key) returns a list or a
        tuple, val_pos has to be set to the index of the value in the
        list/tuple that is to be used for the match.
        (Default value = None)
    return_first : bool
        If True, the function returns the first value in the list of matching
        keys.
        The default is False.

    which : {'min', 'max'}
        Determines if minimum or maximum is looked after
        (Default value = 'min')

    Returns
    -------
    keys: list
        A list of the minimum values

    Exemples
    >>> get_key_from_numvalue({320:1, 321:0, 322:3}, which='min')
    321.
    >>> get_key_from_numvalue({320:(1,2), 321:(0,1), 322:(3,0)}, which='min', val_pos=1)
    322

    In the last exemple, the 0 was matched vs 2 and 1.
    """
    positions = [] # output variable

    if which == 'min':
        current_value = float("inf")
        operator = '<'
    elif which == 'max':
        current_value = -float("inf")
        operator = '>'
    else:
        raise ValueError("which must be equal to 'min' or 'max', received "+str(which))

    for k, v in some_dict.items():

        if val_pos is None:
            to_test = v
        else:
            to_test = v[val_pos]


        if to_test == current_value:
            positions.append(k)

        if eval_truth(to_test, operator, current_value):
            current_value = to_test
            positions = [k] # reset the output previously build with the old best value

    if return_first:
        return positions[0]
    else:
        return positions

def latlong2mtm(_long, _lat, datum):
    """Transform a points coordinates from crs=epsg:4326 to another.

    Used to transfert to espg:2950.

    Note: This is a relic from a time long gone and should be eventually removed.
    """
    from pyproj import Proj

    proj = Proj(init=datum)
    return proj(_long,_lat)

def mtm2latlong(x, y, epsg):
    """Transform a points coordinates from some crs to crs=epsg:4326.

    Used to transfert from espg:2950.

    Note: This is a relic from a time long gone and should be eventually removed.
    """
    #To use on a pandas DataFrame:
    #df[['lon', 'lat']] = df.apply(lambda p: pandas.Series(mtm2latlong(p['X'], p['Y'], 'epsg:2950'),axis=1)
    from pyproj import Proj
    proj = Proj(init=epsg)
    return proj(x, y, inverse=True) #returns lng/lat

##############
### Shapely tools
##############
def clip_shp(shp, clip_polygons):
    """Clip a shape onto a shapely polygon."""
    # Create a single polygon object for clipping
    if isinstance(clip_polygons, geopandas.GeoDataFrame):
        poly = clip_polygons.geometry.unary_union

    elif isinstance(clip_polygons, (shapely.geometry.LineString,
                                    shapely.geometry.LinearRing,
                                    shapely.geometry.MultiLineString,
                                    shapely.geometry.MultiPoint,
                                    shapely.geometry.MultiPolygon,
                                    shapely.geometry.Point,
                                    shapely.geometry.Polygon)
                                    ):
        poly = clip_polygons

    else:
        poly = clip_polygons.geometry

    # Clip the data

    #parts of the shape wholy contained
    within = shp[shp.geometry.within(poly)]

    #parts of the shape that cross the bondaries and that need to be croped
    to_crop = shp[shp.geometry.crosses(poly)].geometry.intersection(poly)
    croped_df = pandas.DataFrame(columns=shp.columns)

    #separate the multilinestrings into linestrings
    for index, elem in to_crop.items():

        if re.match('MULTILINESTRING', elem.to_wkt()):
            parts = [obj.as_wkt() for obj in Polyline2D.from_wkt(elem.to_wkt())]
            for p in range(len(parts)):
                croped_df = croped_df.append({'FID':index*10000+p+1, 'geometry':parts[p]}, ignore_index=True)

        else:
            croped_df = croped_df.append({'FID':index*10000, 'geometry':elem.to_wkt()}, ignore_index=True)

    #transform the new df into a gdf
    croped_gdf = geopandas.GeoDataFrame(croped_df.drop('geometry', axis=1),
                                        geometry = [shapely.wkt.loads(elem) for elem in croped_df.geometry])

    #return the concatenation of both
    return geopandas.GeoDataFrame(pandas.concat([within, croped_gdf], ignore_index=True), crs=within.crs)

def near(p1, points):
    """Find the nearest point p contained in `points` from p1.

    Parameters
    ----------
    p1: shapely.geometry.Point
        The point to use as anchor.
    points: container
        A serie of points from which to extract the nearest. This can be a geopandas.GeoDataFrame, or any
        type of shapely object.
    """
    if isinstance(points, geopandas.GeoDataFrame):
        candidates = points.geometry.unary_union

    elif isinstance(points, (shapely.geometry.LineString,
                             shapely.geometry.LinearRing,
                             shapely.geometry.MultiLineString,
                             shapely.geometry.MultiPoint,
                             shapely.geometry.MultiPolygon,
                             shapely.geometry.Point,
                             shapely.geometry.Polygon)
                             ):
        candidates = points

    else:
        candidates = points.geometry

    # note: the function shapely.ops.nearest_points(p1, candidates) returns
    # a tuple containing (p1, the_closest_from_candidates), which is why we
    # return the second member
    return shapely.ops.nearest_points(p1, candidates)[1]

##############
### Arc
##############
def arc_from_two_points_and_radius(P, Q, radius, eps=1, nPoints=100):
    """Calculate a arc given two points (P, Q) and a radii.

    P: Point
        The starting point.
    Q: Point
        The end point.
    radius: float
        The radius of the circle.
    eps: int
        Indicates if the arc should go from P to Q counterclockwise (1)
        or counterclockwise (-1). The Default is 1.
    nPoints: int
        The number of points used to model the arc. The Default is 100.

    Returns
    -------
    arc: list
        A list of Points.
    """
    #distance from P to Q
    d = ((Q.x - P.x)**2 + (Q.y - P.y)**2 )**0.5

    #distance from C to midpoint(P, Q)
    h = (radius**2 - d**2/4)**0.5 #r > d

    #unit vector from P to Q
    u = (Q.x - P.x) / d
    v = (Q.y - P.y) / d

    #center point
    C = Point((P.x + Q.x) / 2 - eps * h * v, (P.y + Q.y) / 2 + eps * h * u)

    #angles between P and C and between Q and C
    s = math.atan2(P.y - C.y, P.x - C.x)
    t = math.atan2(Q.y - C.y, Q.x - C.x)

    delta_theta = (t - s) / nPoints

    return [Point(C.x + radius*math.cos(s + delta_theta * i), C.y + radius*math.sin(s + delta_theta * i)) for i in range(nPoints+1)]

def arc_from_center_theta_and_point(P, C, theta, eps=1, nPoints=100,
                                    as_degrees=False):
    """Calculate a arc given a point (P), the center (C) and an angle.

    P: Point
        The first point of the arc.
    C: Point
        The center of the circle.
    theta: float
        The angle used to calculate Q, the end point of the arc.
    eps: int
        Indicates if the arc should go from P to Q counterclockwise (1)
        or counterclockwise (-1). The Default is 1.
    nPoints: int
        The number of points used to model the arc. The Default is 100.
    as_degrees : bool
        If True, the angle is given as degrees, otherwise a radians.
        The Default is False.

    Returns
    -------
    arc: list
        A list of Points.
    """
    if as_degrees:
        theta = math.radians(theta)

    radius = Point.distanceNorm2(P, C)
    delta_theta = theta / nPoints * eps

    #make P be part of the output by forcing it's angle as a starting point in the calculaiton
    start_theta = math.atan2(P.y - C.y, P.x - C.x)

    return [Point(C.x + radius*math.cos(start_theta + delta_theta * i),
                  C.y + radius*math.sin(start_theta + delta_theta * i))
            for i in range(nPoints)]

def radius_and_center_of_circle_from_three_points(P1, P2, P3):
    """Calculate the center and radius of a circle given three points (P1, P2, P3).

    P1: Point
        The first point on the circumference.
    P2: Point
        The second point on the circumference.
    P3: Point
        The third point on the circumference.

    Returns
    -------
    C: Point
        The center of the circle.
    radius: float
        The radius of the circle.
    """
    #A = line from P1 to P2
    #B = line from P2 to P3
    m_a = (P2.y - P1.y) / (P2.x - P1.x)
    m_b = (P3.y - P2.y) / (P3.x - P2.x)

    #Center
    x = ( m_a * m_b * (P1.y - P3.y) + m_b * (P1.x + P2.x) - m_a * (P2.x + P3.x) ) / ( 2 * (m_b - m_a) )
    y = m_a * (x - P1.x) + P1.y

    C = Point(x, y)

    radius = Point.distanceNorm2(P1, C)

    return C, radius

##############
### Transformation
##############
def rotate_point_on_origin(point, theta):
    """Rotate a point [x, y] around (0,0).

    Parameters
    ----------
    point : list[x, y]
        The point to rotate araound (0,0)
    theta : float
        the angle (in degrees) of rotation

    Returns
    -------
    x', y'
    """
    r_matrix = [[math.cos(math.radians(theta)), -1*math.sin(math.radians(theta))],[math.sin(math.radians(theta)), math.cos(math.radians(theta))]]
    p_prime = np.matrix(r_matrix)*np.matrix(point).T
    return p_prime.A1[0], p_prime.A1[1]

def translate_point(point, vector):
    """Translate a point [x, y], along a vector [dx, dy].

    Parameters
    ----------
    point : list[x, y]
            The point to translate

    vector : list[dx, dy]
            The x distance and y distance to translate along

    Returns
    -------
    x', y'
    """
    return point[0]+vector[0], point[1]+vector[1]

def rotate_point(x1, y1, point, theta):
    """Rotate a point [x1, y1] around rotation_point[x2, y2] that can be
    different from (0,0).

    Parameters
    ----------
    point : list[x, y]
            The point to rotate

    rotation_point : list[x, y]
            The point to rotate around

    theta : float
            the angle (in degrees) of rotation

    Returns
    -------
    x', y'
    """
    P_1 = translate_point(point, [(0-x1), (0-y1)])
    P_2 = rotate_point_on_origin(P_1, theta)
    return translate_point(P_2, [(x1-0), (y1-0)])

def isbetween(x, x1, x2):
    """Find if x is between x1 and x2.

    Parameters
    ----------
    x : int or float
        The number to test.
    x1 : int or float
        The lowest bound.
    x2 : int or float
        The highest bound.

    Returns
    -------
    result: bool
    """
    if x >= x1 and x <= x2:
        return True
    else:
        return False

def find_middle_point(x0, y0, x1, y1):
    """Find the point(x, y) exactly between point 1 (x0, y0) and point 2 (x1, y1).

    Parameters
    ----------
    x0 : float
        The x coordinate of the first point
    y0 : float
        The y coordinate of the first point
    x1 : float
        The x coordinate of the second point
    y1 : float
        The y coordinate of the first point

    Returns
    -------
    x, y: float
        The coordinates of the middle point.
    """
    return (x0+x1)/2, (y0+y1)/2

def geopandas_crs_to_pyproj_crs(crs):
    """A utility function that forces geopandas' proj4 scheme to be translated
    to proj5+ scheme.

    Note: This should be removed as geopandas has now moved passed the problem.
    """
    if isinstance(crs, dict):
        if not 'init' in crs.keys():
            raise ValueError("Cannot interpret dict without an init keyword, "+
                             f"received {{crs.items()}}")
        crs = crs['init']
    return pyproj.CRS(crs)

def raw_crs_to_int(crs):
    #TODO: verify this. I'm not sur eit even works anymore.
    return geopandas_crs_to_pyproj_crs(crs).to_epsg()

def crs_transform(from_crs, to_crs, xes, yes):
    """A utility function to translate values from one crs to another.

    Parameters
    ----------
    from_crs : int or pyplot.CRS()
        The crs which the given values are set in.

    to_crs : int or pyplot.CRS()
        The crs into which the values are to be translated.

    xes :  float or list of floats
        The x coordinates in the original crs.

    xes :  float or list of floats
        The y coordinates in the original crs.

    Returns
    -------
    xes :  float or list of floats
        The x coordinates in the new crs. The type matches the input.

    yes :  float or list of floats
        The y coordinates in the new crs. The type matches the input.

    """
    transformer = pyproj.transformer.Transformer.from_crs(from_crs, to_crs,
                                                          always_xy=True)

    xes, yes = transformer.transform(xes, yes)

    return xes, yes
##############
### Maths
##############
class SortedOrderTree:
    """Sort some data in a list using a sigular sortable value entry for each
    data entry.

    This is a binairy tree where values are sorted left and right according to
    their relative value to the current leaf.

    Parameters
    ----------
    value : sortable, optional
        The values used to use during the sorting. These objects must be of a
        sortable nature. The class can be initialized with a None value.

        Default : None

    data : object, optional
        The data to sort. This can be any python object.

        Default : None

    """
    def __init__(self, value=None, data=None):
        self.left = None
        self.right = None
        self.data = data
        self.value = value

    def insert(self, value, data):
        """Insert a new ``data`` point, and sort it in the existing tree using
        it's ``value`` parameter.

        Parameters
        ----------
        value : sortable
            The object used as ``value`` must be compatible to be sorted with
            previously used objects when initializaing the class or calling this
            method.

        data : object
            The data to sort. This can be any python object.

        """
        #Compare the new value and class it compared to other leafs to class
        #them by x values.
        if self.value is not None:
            if value < self.value:
                if self.left is None:
                    self.left = SortedOrderTree(value=value, data=data)
                else:
                    self.left.insert(value, data)
            elif value > self.value:
                if self.right is None:
                    self.right = SortedOrderTree(value=value, data=data)
                else:
                    self.right.insert(value, data)

            elif value == self.value and not data == self.data:
                if self.right is None:
                    self.right = SortedOrderTree(value=value, data=data)
                else:
                    self.right.insert(value, data)

        else:
            self.value = value
            self.data = data

    def inorderTraversal(self, reverse=False):
        """Retreive the sorted list of ``data`` objects. The sorting uses the
        ascending order of their ``value`` parameters.

        Parameters
        ----------
        reverse : bool, optional
            If True, the list is instead returned in the descending order of
            their ``value`` parameters.

            Default : False
        Returns
        -------
        ordered : list
            The ordered objects
        """
        # Inorder traversal
        # Left -> Root -> Right
        ordered = []
        if self.data:
            if self.left:
                ordered = self.left.inorderTraversal()
            ordered.append(self.data)
            if self.right:
                ordered += self.right.inorderTraversal()
        if reverse:
            return list(reversed(ordered))
        return ordered

##############
### Maths
##############
def quad_formulae(a, b, c):
    '''0 = ax2 + bx + c'''
    if a == 0: return (None, None)
    if b**2 < 4*a*c:
        return (None, None)
    elif b**2 == 4*a*c:
        return ((-b + (b**2 - 4*a*c)**0.5)/(2*a), None)
    else:
        return ((-b + (b**2 - 4*a*c)**0.5)/(2*a), (-b - (b**2 - 4*a*c)**0.5)/(2*a))

##############
### Detect geometries
##############
def from_shapely(string, coerce=None):

    if coerce is not None:
        current = from_shapely(string)

        if coerce.lower()=='point':
            if isinstance(current, Point):
                return current
            elif isinstance(current, Vector2D):
                return [current.p1, current.p2]
            else:
                return current.points

        elif coerce.lower()=='vector' or coerce.lower()=='vector2d':
            if isinstance(current, Point):
                return Vector2D(current, current)
            elif isinstance(current, Vector2D):
                return current
            else:
                return current.vectors

        elif coerce.lower()=='polyline' or coerce.lower()=='polyline2d':
            if isinstance(current, Point):
                return Polyline2D.from_points([current])
            elif isinstance(current, Vector2D):
                return Polyline2D.from_vector_list([current])
            else:
                return Polyline2D.from_points(current.points)

        elif coerce.lower()=='polygon' or coerce.lower()=='polygon2d':
            if isinstance(current, Point):
                return Polygon2D.from_points([current])
            elif isinstance(current, Vector2D):
                return Polygon2D.from_vector_list([current])
            else:
                return Polygon2D.from_points(current.points)

    if isinstance(string, (list, tuple, np.ndarray)):
        return [from_shapely(elem, coerce=coerce) for elem in string]

    if isinstance(string, shapely.geometry.Point):
        return Point.from_shapely(string)

    elif isinstance(string, shapely.geometry.linestring.LineString):
        if len(string.coords) == 2: return Vector2D.from_shapely(string)
        else:                       return Polyline2D.from_shapely(string)

    elif isinstance(string, shapely.geometry.MultiPoint):
        if len(string.coords) == 2: return Vector2D.from_shapely(string)
        else:                       return Polyline2D.from_shapely(string)

    elif isinstance(string, shapely.geometry.MultiLineString):
        if len(string.coords) == 2: return Vector2D.from_shapely(string)
        else:                       return Polyline2D.from_shapely(string)

    elif isinstance(string, shapely.geometry.Polygon):
        return Polygon2D.from_shapely(string)

    elif isinstance(string, shapely.geometry.MultiPolygon):
        return Polygon2D.from_shapely(string)

##############
### Points
##############
class Point(object):
    __WKT_IDENTITY__ = 'POINT'
    __REPR_IDENTITY__ = 'Point'

    def __init__(self, x, y, dtype=float, crs=None):
        self.x = dtype(x)
        self.y = dtype(y)
        self.dtype = dtype

        if crs is None:
            self._crs = None
        else:
            self._crs = (crs)

    def __str__(self):
        return '({:f},{:f})'.format(self.x,self.y)

    def __add__(self, other):
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Point(other, other, dtype=self.dtype, crs=self.crs)
        elif isinstance(other, (tuple, list, np.ndarray)):
            other = Point(other[0], other[1], dtype=self.dtype, crs=self.crs)
        try:
            return Point(self.x+other.x, self.y+other.y, dtype=self.dtype, crs=self.crs)
        except:
            return NotImplemented

    def __radd__(self, other):
        return Point.__add__(self, other)

    def __sub__(self, other):
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Point(other, other, dtype=self.dtype, crs=self.crs)
        elif isinstance(other, (tuple, list, np.ndarray)):
            other = Point(other[0], other[1], dtype=self.dtype, crs=self.crs)
        try:
            return Point(self.x-other.x, self.y-other.y, dtype=self.dtype, crs=self.crs)
        except:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Point(other, other, dtype=self.dtype, crs=self.crs)
        elif isinstance(other, (tuple, list, np.ndarray)):
            other = Point(other[0], other[1], dtype=self.dtype, crs=self.crs)
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Point(other, other, dtype=self.dtype, crs=self.crs)
        elif isinstance(other, (tuple, list, np.ndarray)):
            other = Point(other[0], other[1], dtype=self.dtype, crs=self.crs)
        try:
            return Point.dot(self, other, dtype=self.dtype, crs=self.crs)
        except:
            return NotImplemented

    def __rmul__(self, other):
        return Point.__mul__(self, other)

    def __eq__(self, other):
        try:
            return (self.x == other.x) and (self.y == other.y)
        except:
            return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __neg__(self):
        return Point(-self.x, -self.y, dtype=self.dtype, crs=self.crs)

    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            raise IndexError("The x and y attributes can be accessed with " +
                             f"index 0 or 1 respectively, received: {i}")

    def __repr__(self):
        return "<{ident}(x={x}, y={y})>".format(ident=self.__REPR_IDENTITY__, x=self.x, y=self.y)

    @property
    def coords(self):
        return self.x, self.y

    @property
    def crs(self):
        if self._crs is None:
            return None
        return f"epsg:{self._crs}"

    @crs.setter
    def crs(self, crs):
        if crs is None:
            self._crs = None
        else:
            self._crs = (crs)

    def as_list(self):
        return [self.x, self.y]

    def as_repr(self):
        return self.__repr__

    def as_str(self):
        return '{} {}'.format(self.x,self.y)

    def as_tuple(self):
        return (self.x, self.y)

    def as_shapely(self):
        return shapely.wkt.loads(self.as_wkt())

    def as_wkt(self):
        """Return the point as well-known text"""
        return self.__WKT_IDENTITY__+'({} {})'.format(self.x,self.y)

    def as_latlong_tuple(self, datum, long_first=True):
        if long_first:
            return mtm2latlong(self.x,self.y,datum)
        else:
            long, lat = mtm2latlong(self.x,self.y,datum)
            return lat, long

    @staticmethod
    def cosine(p1, p2):
        return Point.dot(p1,p2)/(p1.norm2()*p2.norm2())

    @staticmethod
    def cross(p1, p2):
        """
        Calculates the cross product between two points

        Parameters
        ----------
        p1 : Point object

        p2 : Point object


        Returns
        -------
        Float

        """
        return p1.x*p2.y-p1.y*p2.x

    @staticmethod
    def distanceNorm2(p1, p2):
        return (p1-p2).norm2()

    def divide(self, alpha):
        """Warning, returns a new Point"""
        return Point(self.x/alpha, self.y/alpha)

    @staticmethod
    def distance_with_line(px,py,dx1,dy1,dx2,dy2):
        '''px, py the point, dx1, dy1 first point of the line, dx2, dy2 second point of the line'''
        px = float(px); py = float(py)
        dx1 = float(dx1); dy1 = float(dy1)
        dx2 = float(dx2); dy2 = float(dy2)
        return abs((dx2 - dx1)*(dy2 - py) - (dx1 - px)*(dy2 - dy1)) / ((dx2 - dx1)**2 + (dy2 - dy1)**2)**0.5

    @staticmethod
    def dot(p1, p2):
        """
        Calculates the scalar product between two points

        Parameters
        ----------
        p1 : Point object

        p2 : Point object


        Returns
        -------
        Float

        """
        return p1.x*p2.x+p1.y*p2.y

    @staticmethod
    def from_latlong(_long,_lat,datum, **kwargs):
        return Point.from_tuple(latlong2mtm(_long,_lat,datum), **kwargs)

    @staticmethod
    def from_tuple(point, **kwargs):
        """
        Builds a Point object from a tuple or list of coordinates

        Parameters
        ----------
        point : list of x and y coordinates
            ex: [1, 0]

        Returns
        -------
        Point object
        """
        return Point(point[0], point[1], **kwargs)

    @staticmethod
    def from_shapely(point, **kwargs):
        if isinstance(point, shapely.geometry.Point):
            return Point(point.x, point.y, **kwargs)
        else:
            raise TypeError('point must be of shapely.geometry.Point type')

    @staticmethod
    def from_wkt(text, **kwargs):
        if re.match('POINT', text):
            match = re.search('[+-]?((\d+\.?\d*)|(\.\d+)) [+-]?((\d+\.?\d*)|(\.\d+))', text)
            if match:
                x = float(match.group(0).split(' ')[0])
                y = float(match.group(0).split(' ')[1])
            return Point(x, y, **kwargs)
        return None

    def get_side_of_line(self, vector):
        '''returns 1 for Left, 0 for "on" and -1 for Right'''
        test= (vector.p2.x - vector.p1.x) * (self.y - vector.p1.y) - \
              (self.x - vector.p1.x) * (vector.p2.y - vector.p1.y)

        if test > 0: return 1
        if test < 0: return -1
        return test

    def is_bbox_enclosed(self, bbox):
        '''
        bbox as list 2D [[x1, y1],[x2, y2]]
        '''
        #TODO: add possibility to have BBox objects here
        xcheck = True if self.x >= min(bbox[0][0], bbox[1][0]) and self.x <= max(bbox[0][0], bbox[1][0]) else False
        ycheck = True if self.y >= min(bbox[0][1], bbox[1][1]) and self.y <= max(bbox[0][1], bbox[1][1]) else False
        if xcheck and ycheck:
            return True
        else:
            return False

    def is_inside_polygon(self, polygon):
        """Indicates if the Point is inside the polygon

        Use Polygon.contains if Shapely is installed

        Parameters
        ----------
        polygon : array
            (array of Nx2 coordinates of the polygon vertices)
            ex: np.asarray([[0,0],[0,3],[3,3],[3,0]])

        Returns
        -------
        Bool

        """
        if isinstance(polygon, Polygon2D):
            polygon = np.asarray([[point.x, point.y] for point in polygon.points])

        n = polygon.shape[0];
        counter = 0;

        p1 = polygon[0,:];
        for i in range(n+1):
            p2 = polygon[i % n,:];
            if self.y > min(p1[1],p2[1]):
                if self.y <= max(p1[1],p2[1]):
                    if self.x <= max(p1[0],p2[0]):
                        if p1[1] != p2[1]:
                            xinters = (self.y-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1])+p1[0];
                        if p1[0] == p2[0] or self.x <= xinters:
                            counter+=1;
            p1=p2
        return (counter%2 == 1);

    def inverse_xy(self):
        return Point(self.y, self.x, dtype=self.dtype, crs=self.crs)

    @staticmethod
    def midPoint(p1, p2, **kwargs):
        """Returns the middle of the segment [p1, p2]"""
        return Point(0.5*p1.x+0.5*p2.x, 0.5*p1.y+0.5*p2.y, **kwargs)

    def multiply(self, alpha):
        """Warning, returns a new Point"""
        return Point(self.x*alpha, self.y*alpha, dtype=self.dtype, crs=self.crs)

    def norm1(self):
        """1-norm distance (Manhattan distance)"""
        return abs(self.x)+abs(self.y)

    def norm2(self):
        """2-norm distance (Euclidean distance)"""
        return math.sqrt(self.norm2Squared())

    def norm2Squared(self):
        """2-norm distance (Euclidean distance)"""
        return self.x**2+self.y**2

    def normMax(self):
        """inf-norm distance (Tchebychev distance)"""
        return max(abs(self.x),abs(self.y))

    def orthogonal(self, clockwise = True):
        """Returns the orthogonal vector"""
        if clockwise:
            return Point(self.y, -self.x, dtype=self.dtype, crs=self.crs)
        else:
            return Point(-self.y, self.x, dtype=self.dtype, crs=self.crs)

    def plot(self, **kwargs):
        ax = kwargs.pop("ax", None)
        if ax is not None:
            ax.scatter([self.x],[self.y], **kwargs)
        else:
            plt.scatter([self.x],[self.y], **kwargs)

    @staticmethod
    def plot_point_list(points, **kwargs):
        return plt.scatter([p.x for p in points],[p.y for p in points], **kwargs)

    def rotate(self, base_point, theta):
        """base_point as Point class"""
        return Point.from_tuple(rotate_point(base_point.x, base_point.y, self.as_list(), float(theta)), dtype=self.dtype, crs=self.crs)

    def round_coords(self, ndigits=2, force_dtype=None):
        dtype = self.dtype if force_dtype is None else force_dtype
        return Point(round(self.x, ndigits), round(self.y, ndigits), dtype=dtype, crs=self.crs)

    def to_crs(self, crs):
        if self._crs is None:
            raise ValueError("Current crs not specified, cannot proceed")

        return Point(*crs_transform(self._crs, (crs),
                                    self.x, self.y),
                     dtype=self.dtype, crs=self.crs)

    def translate(self, x_distance=None, y_distance=None, vector=None):
        """x_distance, y_distance as float or vector as Vector class"""
        if vector is not None:
            if x_distance is not None:
                raise ValueError("Cannot specify vector and x_distance at the same time")
            if y_distance is not None:
                raise ValueError("Cannot specify vector and y_distance at the same time")
            d_x = vector.dx
            d_y = vector.dy
        else:
            if x_distance is None or y_distance is None:
                raise ValueError("If not specifying vector, both x_distance and y_distance are required")
            d_x = x_distance
            d_y = y_distance

        return Point.from_tuple(translate_point([self.x, self.y], [d_x, d_y]))

###############
###   Vectors
###############
class Vector2D(object):
    def __init__(self, first_point, last_point, dtype=float, crs=None):
        '''first_point, last_point as Point class'''
        self.p1 = first_point
        self.p2 = last_point

        self._dx = None
        self._dy = None
        self._lenght = None
        self._orientation = None
        self._perimeter = None
        self._min_x = None
        self._min_y = None
        self._max_x = None
        self._max_y = None
        self._crs=None
        self._dtype=float

        self.crs = crs
        self.dtype = dtype

    def __eq__(self, other):
        try:
            return (self.p1 == other.p1) and (self.p2 == other.p2)
        except:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __neg__(self):
        return Vector2D(self.p2, self.p1)

    def __lt__(self, other):
        try:
            return self.lenght < other.lenght
        except:
            return NotImplemented

    def __le__(self, other):
        try:
            return self.lenght <= other.lenght
        except:
            return NotImplemented

    def __gt__(self, other):
        try:
            return self.lenght > other.lenght
        except:
            return NotImplemented

    def __ge__(self, other):
        try:
            return self.lenght >= other.lenght
        except:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            return self.scale(self.lenght+other)
        try:
            return Vector2D(self.p1, self.p2.translate(other.dx, other.dy))
        except:
            return NotImplemented

    def __radd__(self, other):
        return Vector2D.__add__(self, other)

    def __sub__(self, other):
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            return self.scale(self.lenght-other)
        try:
            return Vector2D(self.p1, self.p2.translate(-other.dx, -other.dy))
        except:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            return self.scale(-1*self.lenght+other)
        try:
            return Vector2D(other.p1, other.p2.translate(-self.dx, -self.dy))
        except:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            return self.scale(self.lenght*other)
        try:
            return self.scalar_product(other)
        except:
            return NotImplemented

    def __rmul__(self, other):
        return Vector2D.__mul__(self, other)

    def __getitem__(self, i):
        if i == 0:
            return self.p1
        elif i == 1:
            return self.p2
        else:
            raise IndexError()

    def __contains__(self, value):

        if value.__class__ == self.p1.__class__:
            return value in [self.p1, self.p2]

        if isinstance(value, tuple):
            try:
                pvalue = Point.from_tuple(value)
                return pvalue in [self.p1, self.p2]
            except:
                pass

        if isinstance(value, list):
            try:
                pvalue = Point.from_list(value)
                return pvalue in [self.p1, self.p2]
            except:
                pass

        if isinstance(value, str):
            try:
                pvalue = Point.from_wkt(value)
                return pvalue in [self.p1, self.p2]
            except:
                pass

        return False

    def __repr__(self):
        return f"<Vector2D( Start(x={self.p1.x}, y={self.p1.y}), End(x={self.p2.x}, y={self.p2.y}) )>"

    def __hash__(self):
        return hash((self.p1, self.p2))

    @property
    def crs(self):
        if self._crs is None:
            return None
        return f"epsg:{self._crs}"

    @crs.setter
    def crs(self, crs):
        if crs is None:
            _crs = None
        else:
            _crs = (crs)

        if not _crs == self._crs:
            for point in self.points:
                point.crs = _crs

        self._crs = _crs

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype != self._dtype:
            for point in self.points:
                point.dtype = dtype
        self._dtype = dtype

    @property
    def dx(self):
        if self._dx is None:
            self._dx = self.p2.x - self.p1.x
        return self._dx

    @property
    def dy(self):
        if self._dy is None:
            self._dy = self.p2.y - self.p1.y
        return self._dy

    @property
    def is_singularity(self):
        return self.p1 == self.p2

    @property
    def lenght(self):
        if self._lenght is None:
            self._lenght = self.get_lenght()
        return self._lenght

    @property
    def orientation(self):
        if self._orientation is None:
            self._orientation = self.get_orientation()
        return self._orientation

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = self.get_s_lenght()
        return self._perimeter

    @property
    def max_x(self):
        if self._max_x is None:
            xes = [p.x for p in self.points]
            self._max_x = max(xes)
            self._min_x = min(xes)

        return self._max_x

    @property
    def max_y(self):
        if self._max_y is None:
            yes = [p.y for p in self.points]
            self._max_y = max(yes)
            self._min_y = min(yes)

        return self._max_y

    @property
    def min_x(self):
        if self._min_x is None:
            xes = [p.x for p in self.points]
            self._max_x = max(xes)
            self._min_x = min(xes)

        return self._min_x

    @property
    def min_y(self):
        if self._min_y is None:
            yes = [p.y for p in self.points]
            self._max_y = max(yes)
            self._min_y = min(yes)

        return self._min_y

    def _reset_attributes(self):
        self._dx = None
        self._dy = None
        self._lenght = None
        self._orientation = None
        self._perimeter = None
        self._min_x = None
        self._min_y = None
        self._max_x = None
        self._max_y = None

    @property
    def points(self):
        return [self.p1, self.p2]

    def as_list(self):
        return [self.p1.as_list(), self.p2.as_list()]

    def as_tuple(self):
        return [self.p1.as_tuple(), self.p2.as_tuple()]

    def as_UV(self):
        '''return the u and v components'''
        return (self.dx, self.dy)

    def as_shapely(self):
        return shapely.wkt.loads(self.as_wkt())

    def as_wkt(self):
        """Return the vector as well-known text"""
        return 'LINESTRING({}, {})'.format(self.p1.as_str(), self.p2.as_str())

    @staticmethod
    def dot_product(vector1, vector2):
        '''vector 1, vector2 as Vector2D objects'''
        a1, a2 = vector1.as_UV()
        b1, b2 = vector2.as_UV()
        return a1*b1 + a2*b2

    @staticmethod
    def from_dxdy(point, dx, dy):
        """Create a vector from an origin point and dx and dy"""
        return Vector2D(point, Point(point.x + dx, point.y + dy))

    @staticmethod
    def from_point_list(points, as_polyline=False):
        '''Point objects list'''
        if as_polyline:
            return Polyline2D.from_tuples(points)
        if len(points) > 2:
            return [Vector2D(points[i-1], points[i]) for i in range(1, len(points))]
        else:
            return Vector2D(points[0], points[1])

    @staticmethod
    def from_xylists(xlist, ylist, as_polyline=False):
        '''list objects'''
        if len(xlist) != len(ylist):
            raise ValueError("xlist and ylist the same dimension. Receive "+
                             f"{len(xlist)} and {len(ylist)}")
        if len(xlist) > 2:
            raise ValueError("A vector is only comprised of two points, "+
                             f"received {len(xlist)} coordinates for the "+
                             "x dimension.")
        if len(ylist) > 2:
            raise ValueError("A vector is only comprised of two points, "+
                             f"received {len(ylist)} coordinates for the "+
                             "y dimension.")

        return Vector2D.from_point_list([Point(xlist[p], ylist[p]) for p in range(len(xlist))], as_polyline=as_polyline)

    @staticmethod
    def from_2DList(list2D, as_polyline=False):
        return Vector2D.from_point_list([Point(tup[0], tup[1]) for tup in list2D], as_polyline=as_polyline)

    @staticmethod
    def from_shapely(shapely_object):
        if isinstance(shapely_object, shapely.geometry.LineString):
            return Vector2D.from_xylists(*shapely_object.xy)
        else:
            raise NotImplementedError

    @staticmethod
    def from_wkt(text):
        if re.match('LINESTRING', text):
            parts = text.split(',')
            if len(parts) == 2:
                points = []
                for part in parts:
                    points.append(Point.from_wkt('POINT'+part))
                return Vector2D(points[0], points[1])
            else:
                return Polyline2D.from_wkt(text)
        return None

    def get_angle_with_abscissa(self, as_degree=False):

        Vector2D(Point(0,0), Point(1,0)).plot(color='red')

        return self.get_angle_with_vector(Vector2D(Point(0,0), Point(1,0)), as_degree=as_degree)

    def get_angle_with_ordinate(self, as_degree=False):

        Vector2D(Point(0,0), Point(0,1)).plot(color='green')

        return self.get_angle_with_vector(Vector2D(Point(0,0), Point(0,1)), as_degree=as_degree)

    def get_angle_with_vector(self, other_vector, as_degree=False):
        '''absolute minimum angle between two vectors'''
        #to avoid the machine error, lets first check if they are equal and force the zero
        if self.as_UV() == other_vector.as_UV():
            angle = 0
        else:
            scal_prod = Vector2D.scalar_product(self.normalized(), other_vector.normalized())
            angle = math.acos(catch_machine_error(scal_prod, tolerance=10**-6))
        if as_degree:
            return math.degrees(angle)
        else:
            return angle

    def get_deltas(self):
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        return dx, dy

    def get_equation(self):
        '''return a, b from a*x+b'''
        return Vector2D.vector_equation(self.p1.x,self.p1.y,self.p2.x,self.p2.y,dim=1)

    def get_lenght(self):
        return Point.distanceNorm2(self.p1, self.p2)

    def get_mid_point(self):
        return Point.midPoint(self.p1, self.p2)

    def get_min_x(self):
        return min(self.p1.x, self.p2.x)

    def get_max_x(self):
        return max(self.p1.x, self.p2.x)

    def get_min_y(self):
        return min(self.p1.y, self.p2.y)

    def get_max_y(self):
        return max(self.p1.y, self.p2.y)

    def scale(self, lenght):
        '''returns a vector of given length with the same u and v components'''
        u, v = self.normalized_UV()
        return Vector2D(self.p1, Point(self.p1.x + u*lenght, self.p1.y + v*lenght))

    def normalized(self):
        '''from first_point, computes the new last_point such as the lenght equals 1'''
        u, v = self.normalized_UV()
        return Vector2D(self.p1, Point(self.p1.x + u, self.p1.y + v))

    def normalized_UV(self):
        '''return the u and v components of the normed vector'''
        #print (self)
        return ((self.p2.x - self.p1.x)/self.get_lenght(), (self.p2.y - self.p1.y)/self.get_lenght())

    def get_orientation(self):
        '''
        returns -1 for downward oriented vector
                 0 for strickly horizontaly oriented vector
                 1 for upward oriented vector
        '''
        if self.p1.y == self.p2.y:
            return 0
        if self.p1.y > self.p2.y:
            return -1
        if self.p1.y < self.p2.y:
            return 1

    def perpendicular(self):
        """This is the math definition, so the counter clockwise one"""
        return self.perpendicular_anticlockwise()

    def perpendicular_clockwise(self):
        return Vector2D.from_dxdy(Point(self.p1.x, self.p1.y), self.dy, -self.dx)

    def perpendicular_anticlockwise(self):
        return Vector2D.from_dxdy(Point(self.p1.x, self.p1.y), -self.dy, self.dx)

    def fraction_pt(self, fraction):
        return Point(fraction * self.dx + self.p1.x, fraction * self.dy + self.p1.y)

    def get_points(self):
        return [self.p1, self.p2]

    def get_signed_angle_with_vector(self, other_vector, as_degree=False):
        '''
        minimum angle between two vectors, with the sign as per the
        mathematical convention: ie: (-) is clockwise
                                     (+) is conterclockwise
        '''

        #to avoid the machine error, lets first check if they are equal and force the zero
        if self.as_UV() == other_vector.as_UV():
            angle = 0

        angle = self.get_angle_with_vector(other_vector, as_degree=as_degree)
        sign = Vector2D.perp_dot_product(self,other_vector)/(self.lenght*other_vector.lenght)
        return angle*sign

    def get_slope(self):
        if self.dx == 0:
            return float('inf')
        else:
            return self.dy / self.dx

    def get_x_intersect_at_y(self, y):
        if self.dy != 0:
            return (y - self.p1.y) * self.dx / self.dy + self.p1.x
        else:
            return self.p1.x

    def get_y_intersect_at_x(self, x):
        if self.dx != 0:
            return (x - self.p1.x) * self.dy / self.dx + self.p1.y
        else:
            return self.p1.y

    def inverse_xy(self):
        return Vector2D(self.p1.inverse_xy(), self.p2.inverse_xy())


    def on_self(self, point, tolerance=0.00001):
        if tolerance is None:
            return Vector2D.point_dist(point, self) == 0
        else:
            return Vector2D.point_dist(point, self) < tolerance

    @staticmethod
    def perp_dot_product(vector1, vector2):
        '''vector 1, vector2 as Vector2D objects

        The perp product of v1 and v2 returns the magnitude of
        the vector that would result from a regular 3D cross product of the
        input vectors, taking their Z values implicitly as 0.

        It can represent the area of the parallelogram between the two vectors

        In addition, this area is signed and can be used to determine whether
        rotating from V1 to V2 moves in an counter clockwise (anwser is +) or
        clockwise direction (anwser is 1).
        '''
        vector1_p = vector1.perpendicular()
        return vector1_p * vector2

    def plot(self, plot_points=False, show_arrows=False, arrow_kwars={}, **kwargs):

        ax = kwargs.pop("ax", plt.gca())

        if plot_points:
            self.p1.plot(**kwargs)
            self.p2.plot(**kwargs)

        if not show_arrows:
            ax.plot([self.p1.x, self.p2.x],[self.p1.y, self.p2.y], **kwargs)

        else:
            ax.arrow(self.p1.x, self.p1.y, self.dx, self.dy, **arrow_kwars, **kwargs)

            #since axes doesn't auto scale the plot like plt.plot or plt.scatter plot hidden points

            current_xlim = ax.get_xlim(); current_ylim = ax.get_ylim()
            if not (current_xlim[0] < self.p1.x and \
                    current_xlim[1] > self.p2.x and \
                    current_ylim[0] < self.p1.y and \
                    current_ylim[1] > self.p2.y):
                ax.scatter([self.p1.x, self.p2.x],[self.p1.y, self.p2.y], alpha=0)

        return ax

    def reverse(self):
        return Vector2D(self.p2, self.p1)

    def left_or_right(self, point):
        ''' Determines if the point is to the right or left of self

        RIGHT HAND RULE: Right of forwards+++++++; Left of forwards-------)

        vectorial multiplication of both vectors gives us information about the orientation of the snapped vector
        To follow the right hand rule, we must use:
                       if z is (-), P is to the right of the spline and direction_y is thus +++++
                       if z is (+), P is to the left of the spline and direction_y is thus ----

        =============
        Returns -1 for "left"
                 0 for "on the line" (neither right nor left)
                 1 for "right"

        '''
        d = (point.x - self.p1.x) * (self.p2.y - self.p1.y) - (point.y - self.p1.y)*(self.p2.x - self.p1.x)

        if d > 0 :
            return 1
        elif d < 0 :
            return -1
        else:
            return 0

    def rotate(self, base_point, theta):
        """base_point as Point class"""
        return Vector2D(self.p1.rotate(base_point, theta), self.p2.rotate(base_point, theta))

    def scalar_product(self, vector2):
        '''vector2 as Vector2D objects'''
        return Vector2D.dot_product(self, vector2)

    def snap_point(self, point, allow_expanding_vector=False, allow_out_of_bounds=True):
        '''point as Point class object'''
        if self.dx == self.dy == 0:  # the segment's just a point
            return (self.p1.x,self.p1.y)

        #Calculate the t that minimizes the distance.
        t = ((point.x - self.p1.x) * self.dx + (point.y - self.p1.y) * self.dy) / (self.dx**2 + self.dy**2)

        #See if this represents one of the segment's
        #end points or a point in the middle.
        if t < 0 and not allow_expanding_vector:
            if allow_out_of_bounds:
                return self.p1
            return None
        elif t > 1 and not allow_expanding_vector:
            if allow_out_of_bounds:
                return self.p2
            return None
        else:
            near_x = self.p1.x + t * self.dx
            near_y = self.p1.y + t * self.dy

            return Point(near_x, near_y)

    def shift(self, amount, side='right', inplace=False):

        if self.lenght == 0:
            return self

        if side == 'right':
            sidefactor = -1
        if side == 'left':
            sidefactor = 1

        move = self.perpendicular().normalized() * amount * sidefactor

        return Vector2D(move.p2, self.p2.translate(move.dx, move.dy))

    def split(self, point):
        """Split the vector using a Point as a cutting spot. The point is first
        snaped onto the vector object on the closest vertex using the snap_point
        method with ``allow_out_of_bounds=True``.

        Parameters
        ----------
        point : Point objects
            The point to use as cutters.

        Returns
        -------
        snaped : list of Vector2D objects
            The class of the return objects are the same as the caller object.
            Singularities are rejected from the output.
        """
        snaped = self.snap_point(point, allow_out_of_bounds=True,
                                 allow_expanding_vector=False)

        if any([snaped == self.p1, snaped == self.p2]):
            return [self]

        return [Vector2D(self.p1, snaped, dtype=self.dtype, crs=self.crs),
                Vector2D(snaped, self.p2, dtype=self.dtype, crs=self.crs)]

    def split_many(self, cut_lenght, ignore_rest=False, as_polyline=False):
        '''
        returns a list of subvectors of lenght equal to cut_lenght
        the first vector starts at (p1.x, p1.y)
        if the modulo of self.getLenght() and cut_lenght is not equal to zero,
        the last vector ending at (p2x, p2.y) will be shorter than cut_lenght

        if ignore_rest is True, the last vector will not be returned; if the
        modulo is not equal to zero, the sum of the lenghts of all returned
        subvectors will be shorter than self.getLenght()
        '''
        u, v = self.normalized_UV()
        xlist_of_complete_segments = [self.p1.x + n * u * cut_lenght for n in range(int(self.get_lenght() // cut_lenght)+1)]
        ylist_of_complete_segments = [self.p1.y + n * v * cut_lenght for n in range(int(self.get_lenght() // cut_lenght)+1)]

        rest = self.get_lenght() % cut_lenght

        if ignore_rest is False and rest != 0:
            xlist_of_complete_segments.append(self.p2.x)
            ylist_of_complete_segments.append(self.p2.y)

        return Vector2D.from_xylists(xlist_of_complete_segments, ylist_of_complete_segments, as_polyline=as_polyline)

    def to_crs(self, crs):
        if self._crs is None:
            raise ValueError("Current crs not specified, cannot proceed")

        new_x, new_y = crs_transform(self._crs, raw_crs_to_int(crs),
                                     [self.p1.x, self.p2.x],
                                     [self.p1.y, self.p2.y]
                                     )

        return Vector2D.from_xylists(new_x, new_y)

    def translate(self, x_dist, y_dist):
        return Vector2D(self.p1.translate(x_dist, y_dist), self.p2.translate(x_dist, y_dist))

    @staticmethod
    def point_dist(point, vector):
        '''Point class object, Vector2D class object'''
        closest = vector.snap_point(point)
        return Vector2D(point, closest).get_lenght()

    @staticmethod
    def vector_dist(vector1, vector2):
      """ distance between two segments in the plane:
          one segment is (x11, y11) to (x12, y12)
          the other is   (x21, y21) to (x22, y22)
      """
      if Vector2D.intersect(vector1, vector2): return 0.0
      # try each of the 4 vertices w/the other segment
      return min([Vector2D.point_dist(vector1.p1, vector2),
                  Vector2D.point_dist(vector1.p2, vector2),
                  Vector2D.point_dist(vector2.p1, vector1),
                  Vector2D.point_dist(vector2.p2, vector1)])

    @staticmethod
    def intersect(vector1, vector2):
      """ whether two segments in the plane intersect:
          Vector2D class objects
      """
      delta = vector2.dx * vector1.dy - vector2.dy * vector1.dx

      if delta == 0: return False  # parallel segments
      s = (vector1.dx * (vector2.p1.y - vector1.p1.y) + vector1.dy * (vector1.p1.x - vector2.p1.x)) / delta
      t = (vector2.dx * (vector1.p1.y - vector2.p1.y) + vector2.dy * (vector2.p1.x - vector1.p1.x)) / (-delta)

      return (0 <= s <= 1) and (0 <= t <= 1)

    @staticmethod
    def vector_equation(vector):
        a = float(vector.p2.x - vector.p1.x) / float(vector.p2.y - vector.p1.y)
        b = vector.p1.y - a*vector.p1.x
        return a,b

    @staticmethod
    def vector_intersection_point(vector1, vector2, allow_expand_vectors=False):
        '''
        Vector2D class objects
        '''

        delta = vector2.dx * vector1.dy - vector2.dy * vector1.dx

        if delta == 0: return None  # parallel segments
        s = (vector1.dx * (vector2.p1.y - vector1.p1.y) + vector1.dy * (vector1.p1.x - vector2.p1.x)) / delta
        t = (vector2.dx * (vector1.p1.y - vector2.p1.y) + vector2.dy * (vector2.p1.x - vector1.p1.x)) / (-delta)

        if not (0 <= s <= 1) and (0 <= t <= 1) and not allow_expand_vectors:
            return None #intersection point is outside both segements

        int_x = vector1.p1.x + t * vector1.dx
        int_y = vector1.p1.y + t * vector1.dy

        return Point(int_x, int_y)


##############
### Multipoints (n>2) Classes
##############
class BaseMultiPointClass(object):
    __WKT_IDENTITY__ = ''
    __WKT_N_BRACKET__ = 1
    __REPR_IDENTITY__ = ''

    def __init__(self, points=None, vectors=None, angles_as_degrees=True,
                 dtype=float, crs=None):
        self.angles_as_degrees = angles_as_degrees

        self._crs = None
        self._dtype = float

        self.points = []
        if isinstance(points, list):
            if np.asarray([isinstance(p, Point) for p in points]).all():
                self.points = points

        self.vectors = []
        if isinstance(vectors, list):
            if np.asarray([isinstance(v, Vector2D) for v in vectors]).all():
                self.vectors = vectors

        self.angles = []

        if self.points != [] and self.vectors == []:
            self._compute_vectors_from_points()

        if self.vectors != [] and self.points == []:
            self._compute_points_from_vectors()

        #if self.vectors != []:
        #    self._compute_angles_from_vectors(angles_as_degrees=self.angles_as_degrees)

        self.crs = crs     #this is to cascade it down to the points when changed
        self.dtype = dtype #this is to cascade it down to the points when changed

        self._max_x = None
        self._max_y = None
        self._min_x = None
        self._min_y = None

    def __repr__(self):
        _repr_ = "<"+self.__REPR_IDENTITY__+"("
        for p in range(len(self.points)):
            if p == 0:
                _repr_ += " "
            else:
                _repr_ += ", "
            _repr_ += "P{index}(x={px}, y={py})".format(index=p, px=self.points[p].x, py=self.points[p].y)
        _repr_ += " )>"
        return _repr_

    def __neg__(self):
        return self.from_points([p for p in reversed(self.points)])

    def __eq__(self, other):
        try:
            if sum([self.points[p] == other.points[p] for p in range(len(self.points))]) > 0:
                return True
            return False
        except:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        """Returns the current number of points in the Polyline"""
        return len(self.points)

    def __contains__(self, value):
        if value.__class__ == self.points[0].__class__:
            return value in self.points

        if isinstance(value, Vector2D):
            return value in self.vectors

        if isinstance(value, tuple):
            try:
                pvalue = Point.from_tuple(value)
                return pvalue in self.points
            except:
                pass

        if isinstance(value, list):
            try:
                pvalue = Point.from_list(value)
                return pvalue in self.points
            except:
                pass

        if isinstance(value, str):
            try:
                pvalue = Point.from_wkt(value)
                return pvalue in self.points
            except:
                pass

        return False

    def __hash__(self):
        return hash(((p.x, p.y) for p in self.points))

    def __getitem__(self, i):
        return self.points[i]

    def _compute_vectors_from_points(self, reset_vectors=False):
        if reset_vectors:
            self._reset_attributes()
            self.vectors = []

        for i in range(1, len(self.points)):
            self.vectors.append(Vector2D(self.points[i-1], self.points[i]))

        #self._compute_angles_from_vectors(angles_as_degrees=self.angles_as_degrees, reset_angles=reset_vectors)

    def _compute_angles_from_vectors(self, angles_as_degrees=True, reset_angles=False):
        if reset_angles:
            self.angles = []
        #TODO: make sure we are not trying to calculate on a singularity and that we are
        #      not feeding a division by 0 into get_signed_angle_with_vector
        for v in range(1,len(self.vectors)):
            self.angles.append(self.vectors[v-1].get_signed_angle_with_vector(self.vectors[v], as_degree=angles_as_degrees))

    def _compute_points_from_vectors(self, reset_points=False):
        if reset_points:
            self._reset_attributes()
            self.points = []

        self.points.append(self.vectors[0].p1)
        self.points.append(self.vectors[0].p2)
        for i in range(1, len(self.vectors)):
            if not self.vectors[i].p1 == self.points[-1]:
                self.points.append(self.vectors[i].p1)
            if not self.vectors[i].p2 == self.points[-1]:
                self.points.append(self.vectors[i].p2)

    def _push_down_crs(self):
        for point in self.points:
            point.crs = self._crs
        for vector in self.vectors:
            vector.crs = self._crs

    @property
    def crs(self):
        if self._crs is None:
            return None
        return f"epsg:{self._crs}"

    @crs.setter
    def crs(self, crs):
        if crs is None:
            _crs = None
        else:
            _crs = raw_crs_to_int(crs)

        push_down=False
        if not _crs == self._crs:
            push_down=True

        self._crs = _crs

        if push_down:
            self._push_down_crs()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype != self._dtype:
            for point in self.points:
                point.dtype = dtype
            for vector in self.vectors:
                vector.dtype = dtype
        self._dtype = dtype

    @property
    def is_singularity(self):
        return all([self.points[i-1] == self.points[i] for i in range(1, len(self.points))])

    @property
    def max_x(self):
        if self._max_x is None:
            xes = [p.x for p in self.points]
            self._max_x = max(xes)
            self._min_x = min(xes)

        return self._max_x

    @property
    def max_y(self):
        if self._max_y is None:
            yes = [p.y for p in self.points]
            self._max_y = max(yes)
            self._min_y = min(yes)

        return self._max_y

    @property
    def min_x(self):
        if self._min_x is None:
            xes = [p.x for p in self.points]
            self._max_x = max(xes)
            self._min_x = min(xes)

        return self._min_x

    @property
    def min_y(self):
        if self._min_y is None:
            yes = [p.y for p in self.points]
            self._max_y = max(yes)
            self._min_y = min(yes)

        return self._min_y

    def _reset_attributes(self):
        self._max_x = None
        self._max_y = None
        self._min_x = None
        self._min_y = None

    def as_lists(self):
        '''
        Returns
        -------
        List of [x, y] lists
        '''
        return [p.as_list() for p in self.points]

    def as_tuples(self):
        '''
        Returns
        -------
        List of (x, y) tuples
        '''
        return [p.as_tuple() for p in self.points]

    def as_shapely(self):
        return shapely.wkt.loads(self.as_wkt())

    def as_wkt(self):
        """Return the vector as well-known text"""
        if len(self.points) > 1:
            text = self.__WKT_IDENTITY__
            for i in range(self.__WKT_N_BRACKET__):
                text += '('
            text += '{}'.format(self.points[0].as_str())
            for p in range(1,len(self.points)):
                text += ',' + self.points[p].as_str()
            for i in range(self.__WKT_N_BRACKET__):
                text += ')'
            return text
        else:
            return self.points[0].as_wkt()

    def as_xylists(self):
        '''
        Returns
        -------
        List of x coords, List of y coords
        '''
        return [p.x for p in self.points], [p.y for p in self.points]

    @classmethod
    def circle_from_CR(cls, center, radius, nPoints):
        """
        Trace a circle circonference composed of n-1 strait vectors centered
        around the center point and of a given radius.

        Parameters
        ----------
        center : Point object
                The center point of the circle

        radius : float
                The radius of the circle

        nPoint : int
                The number of points used to trace the resulting circle

        Returns
        -------
        A Polyline2D or Polygon2D object (depending on the calling class)
        """
        delta_theta = 2 * math.pi / nPoints

        points = [Point(center.x + radius *math.sin(i*delta_theta), center.y + radius * math.cos(i*delta_theta)) for i in range(nPoints+1)]
        return cls.from_points(points)

    @classmethod
    def circle_from_3P(cls, P1, P2, P3, nPoints, base='P1', eps=1):
        """
        Trace a circle circonference composed of n-1 strait vectors centered
        around the center point and of a given radius.

        Parameters
        ----------
        P1, P2, P3 : Point object
             The points used to calculate the circle parameters

        nPoint : int
                The number of points used to trace the resulting circle

        Returns
        -------
        A Polyline2D or Polygon2D object (depending on the calling class)
        """
        center, radius = radius_and_center_of_circle_from_three_points(P1, P2, P3)
        delta_theta = 2 * math.pi / nPoints
        points = [Point(center.x + radius *math.sin(i*delta_theta), center.y + radius * math.cos(i*delta_theta)) for i in range(nPoints+1)]

        if base == 'P1':
            pass


        if eps == -1:
            return cls.from_points(points).rotate(center, ).reverse()
        elif eps == 1:
            return cls.from_points(points).rotate(center, )

    def find_closest_point(self, point):
        closest = (None, np.inf)
        for candidate in self.points:
            dist = Point.distanceNorm2(point, candidate)
            if closest[0] is None:
                closest = (candidate, dist)
            elif dist < closest[-1]:
                closest = (candidate, dist)
            else:
                pass
        return closest[0]

    @classmethod
    def from_points(cls, point_list, as_is=False, **kwargs):
        """Create the object from a list of Point objects.

        Parameters
        ----------
        point_list : list
            A list of Point objects used as continuous points in
            the multipoint object construction.
        as_is : bool
            If True, the algorithm does not check for back to back duplicates
            when constructing the point list and the vector list.
            The Default is False.
        kwargs : dict
            The dict is passed the class' init

        Returns
        -------
        An object from the calling class
        """
        bmpc = cls(**kwargs)

        if len(point_list) > 0:
            bmpc.points.append(point_list[0])

            for p in range(1, len(point_list)):
                if not bmpc.points[-1] == point_list[p] or as_is:
                    bmpc.points.append(point_list[p])

            bmpc._compute_vectors_from_points()

        if "crs" in kwargs.keys():
           bmpc._push_down_crs()

        return bmpc

    @classmethod
    def from_tuples(cls, *points, as_is=False, flip_xy_order=False, **kwargs):
        """Create the object from a list of (x, y) tuples."""
        if getNestingLevel(points) < 2:
            points=[points]
        else:
            while getNestingLevel(points) > 2:
                points = points[0]

        if flip_xy_order:
            points = [list(reversed(p)) for p in points]

        points = [Point.from_tuple(p) for p in points]
        return cls.from_points(points, as_is=as_is, **kwargs)

    @classmethod
    def from_wkt(cls, text, as_is=False, **kwargs):
        if re.match(cls.__WKT_IDENTITY__, text):
            text = multireplace(text, replacements={cls.__WKT_IDENTITY__:'', '(':'', ')':''}, ignore_case=True)
            parts = text.split(',')
            points = []
            for part in parts:
                match = re.search('[+-]?((\d+\.?\d*)|(\.\d+)) [+-]?((\d+\.?\d*)|(\.\d+))', part)
                if match:
                    x = float(match.group(0).split(' ')[0])
                    y = float(match.group(0).split(' ')[1])
                points.append(Point(x,y))
            return cls.from_points(points, as_is=as_is, **kwargs)
        return cls.from_points([], **kwargs)

    @classmethod
    def from_xylists(cls, x_list, y_list, as_is=False, **kwargs):
        """Create the object from a list of Point class objects."""
        return cls.from_points([Point(x_list[i], y_list[i]) for i in range(len(x_list))], as_is=as_is, **kwargs)

    def get_aabbox(self):
        """Get the axis aligned bounding box."""
        return AAMBoundingBox(self.points)

    def get_min_x(self):
        return min([p.x for p in self.points])

    def get_max_x(self):
        return max([p.x for p in self.points])

    def get_min_y(self):
        return min([p.y for p in self.points])

    def get_max_y(self):
        return max([p.y for p in self.points])

    def get_points(self, include_last_point=True):
        if include_last_point:
            return self.points
        return self.points[:-1]

    def get_s_coordinates(self):
        cumul=0
        coords=[(0,0)]
        for p in range(1, len(self.points)):
            cumul += Point.distanceNorm2(self.points[p], self.points[p-1])
            coords.append((cumul, 0))
        return coords

    def get_s_lenght(self):
        return sum([vector.lenght for vector in self.vectors])

    def get_vectors(self, include_last_vector=True):
        if include_last_vector:
            return self.vectors
        return self.vectors[:-1]

    def inverse_xy(self):
        return self.__class__.from_points([p.inverse_xy() for p in self.points])

    def left_or_right(self, point, tolerance=None):
        """Determines if the point is to the right or left of self.

        RIGHT HAND RULE: Right of forwards+++++++; Left of forwards-------)

        vectorial multiplication of both vectors gives us information about the orientation of the snapped vector
        To follow the right hand rule, we must use:
                       if z is (-), P is to the right of the spline and direction_y is thus +++++
                       if z is (+), P is to the left of the spline and direction_y is thus ----

        =============
        Returns -1 for "left"
                 0 for "on the line" (neither right nor left)
                 1 for "right"

        Note that for Polygons, this is not an indication that the point is
        inside or outside of the polygons, since traversing the boundary in the
        opposite direction would give the opposite result regardless of wether
        the point is inside or outside.
        """
        closest = self.find_closest_point(point)
        index = self.points.index(closest)

        if index == 0:
            return self.vectors[index].left_or_right(point)
        elif index == len(self.points)-1:
            return self.vectors[-1].left_or_right(point)
        else:
            #at this point we found the closest point, which can be either the
            #starting point or the end point of a vector. For this reason we
            #need to test two vectors: the one linking i-1 and i and the one
            #linking i and i+1
            cands = self.vectors[index-1: index+1]
            #two things can happen here:
            #  1 - The vector is placed such that a projection falls on one of
            #      the two vectors without needing the project those vectors
            #      out of their bounds.
            #  2 - The vector needs one of the vectors to be projected.
            #
            #Convexe junctions prensent an edge case that need to be avoided:
            #if the point is close to vector1, it could be left of it will
            #being right of the projection of vector 2 (or the opposite). To
            #avoid this, we check both sides and decide the outcome based on
            #the direction of the second vector compared to the first.

            #first, we try to parse points directly on the line
            if cands[0].on_self(point, tolerance=tolerance):
                return 0
            elif cands[1].on_self(point, tolerance=tolerance):
                return 0
            #if the angle is 360°, there's not much we can do. Either everything
            #is left, or everything is right, or a combinaison of both and
            #it entirely depends on how you see the problem at hand. Here we
            #chose LEFT.
            if cands[0].get_angle_with_vector(cands[1], as_degree=True) == 180:
                return -1

            side_1 = cands[0].left_or_right(point)
            side_2 = cands[1].left_or_right(point)
            #when both checks agree, we can't go wrong by choosing this answer
            if side_1 == side_2:
                return side_1
            #Otherwise, the point is actually considered to be on the opposite
            #side of where the second vector is pointing
            return -1 * cands[0].left_or_right(cands[1].p2)

    def on_self(self, point, tolerance=0.00001):

        if len(self.points) == 1:
            return self.points[0] == point

        closest = self.find_closest_point(point)
        index = self.points.index(closest)

        if index == 0:
            candidate_vectors = [self.vectors[index]]
        elif index == len(self.points)-1:
            candidate_vectors = [self.vectors[-1]]
        else:
            candidate_vectors = self.vectors[index-1: index+1]


        for vector in candidate_vectors:
            if vector.on_self(point, tolerance=tolerance):
                return True
        return False

    def plot(self, plot_points=False, show_arrows=False, arrow_kwars={'head_width':0.2, 'length_includes_head':True}, **kwargs):

        ax = kwargs.pop("ax", None)
        if ax is None:
            fig, ax = plt.subplots()

        if plot_points:
            ax.scatter([point.x for point in self.points],[point.y for point in self.points], **kwargs)

        if not show_arrows:
            for vector in self.vectors:
                vector.plot(**kwargs)
        else:
            for vector in self.vectors:
                ax.arrow(vector.p1.x, vector.p1.y, vector.dx, vector.dy, **arrow_kwars, **kwargs)

            #since axes doesn't auto scale the plot like plt.plot or plt.scatter plot hidden points
            for point in [Point(self.max_x, self.min_y), Point(self.max_x, self.max_y),
                          Point(self.min_x, self.min_y), Point(self.min_x, self.max_y)]:
                point.plot(alpha=0, ax=ax)

        return ax

    def remove(self, position):
        self.points.pop(position)
        self._compute_vectors_from_points(self, reset_vectors=True)

    def reverse(self):
        return self.__class__.from_points(list(reversed(self.points)))

    def rotate(self, theta, base_point=Point(0,0)):
        """base_point as Point class"""
        return self.__class__.from_points([point.rotate(base_point, theta) for point in self.points])

    def scale(self, factor=1.0):
        new_vectors=[]
        #fist vector keeps the same p1
        p1 = self.vectors[0].p1
        dx = self.vectors[0].dx * factor
        dy = self.vectors[0].dy * factor
        new_vectors.append(Vector2D.from_dxdy(p1, dx, dy))
        #new vectors snap to the previous ones
        for v in range(1, len(self.vectors)):
            pi = new_vectors[v-1].p2
            dx_i = self.vectors[v].dx * factor
            dy_i = self.vectors[v].dy * factor
            new_vectors.append(Vector2D.from_dxdy(pi, dx_i, dy_i))
        return self.__class__.from_vector_list(new_vectors)

    def shift(self, side='left', dist=1.0):
        if not side in ['left', 'right']:
            raise ValueError("`side` must be one of {'left', 'right'}, "+
                             f"received {side}")

        new_vectors = []
        #shift everything to it's left/right
        for v in range(len(self.vectors)):
            vec = self.vectors[v]
            if side == 'left':
                per = vec.perpendicular_clockwise()
            else:
                per = vec.perpendicular_anticlockwise()

            #scale it to dist
            per = per.scale(abs(dist))
            #the new shifted vector can be built at per.p2, using vec.dx and vec.dy
            vprime = Vector2D.from_dxdy(per.p2, vec.dx, vec.dy)
            #we can't correct the first one yet so let's just save it
            if v == 0:
                new_vectors.append(vprime)
                continue
            #now, if we shift everything, either they will overlap or they won't
            #connect, so we need to correct v-1.p2 and v.p1 at the intersection
            inter = Vector2D.vector_intersection_point(new_vectors[v-1], vprime,
                                                       allow_expand_vectors=True)
            #correct both points
            new_vectors[v-1].p2 = inter
            vprime.p1 = inter
            #save the new vector
            new_vectors.append(vprime)

        return self.__class__.from_vector_list(new_vectors)

    def snap_point(self, point, allow_out_of_bounds=True):
        """Snap a point onto the closest point that is part of the multipoint
        object's vector representation.

        Parameters
        ----------
        point : Point object
            The points to snap.

        allow_out_of_bounds : bool, optional
            If True, the point can be snaped to the extremities of the closest
            vector to the point. If False, the point needs to be able to be
            connected using a perpendicular line to the closest vector; use
            caution if setting it to False since it leaves massive holes on the
            outerside of an angle where a point won't be able to be snaped on
            the multipoint object.

            Default : False

        Returns
        -------
        snaped : Point object
            The snaped point
        """
        closest = self.find_closest_point(point)
        index = self.points.index(closest)

        if index == 0:
            candidate_vectors = [self.vectors[index]]
        elif index == len(self.points)-1:
            candidate_vectors = [self.vectors[-1]]
        else:
            candidate_vectors = self.vectors[index-1: index+1]

        snaped = (None, np.inf)
        for vector in candidate_vectors:
            candidate = vector.snap_point(point, allow_out_of_bounds=allow_out_of_bounds)
            if candidate is None:
                continue
            dist = Point.distanceNorm2(point, candidate)
            if snaped[0] is None:
                snaped = (candidate, dist)
            elif dist < snaped[-1]:
                snaped = (candidate, dist)
            else:
                pass

        return snaped[0]

    def split(self, points):
        """Split a multipoint object using a list if Points as cutting spots.
        The points are first snaped onto the multipoint object on the closest
        vertex using the snap_point method with ``allow_out_of_bounds=True``.

        Parameters
        ----------
        point : list of Point objects
            The point to use as cutters.

        Returns
        -------
        multis : list of multipoint objects
            The class of the return objects are the same as the caller object.
            Singularities are rejected from the output.
        """
        coords = self.get_s_coordinates()
        sto = SortedOrderTree()

        #snap the points on self and order them from fartest to closest to self.point[0]
        for point in points:
            snaped = self.snap_point(point, allow_out_of_bounds=True)

            closest = self.find_closest_point(snaped)
            index = self.points.index(closest)

            if index == 0:
                indexes = [index]
            elif index == len(self.points)-1:
                indexes = [index-1]
            else:
                indexes = [index-1, index]
            #print(point, len(self.points), index, indexes, len(self.vectors))
            for t in indexes:
                candidate = self.vectors[t].snap_point(snaped, allow_out_of_bounds=False)
                if candidate is None:
                    continue

                dist = Point.distanceNorm2(self.vectors[t].p1, candidate) + coords[t][0]
                sto.insert(dist, (candidate, t))
                break

        order = sto.inorderTraversal(reverse=True)

        #split the vectors
        vectors = self.vectors
        parts =[]
        for obj in order:
            i = obj[1]; candidate = obj[0]
            parts.append([Vector2D(candidate, vectors[i].p2)] +
                          vectors[i+1:])
            vectors = vectors[:i] + \
                      [Vector2D(vectors[i].p1, candidate)]
        parts.append(vectors)

        #create the resulting multipoints object
        multis=[]
        for part in parts:
            new = self.__class__.from_vector_list(part, dtype=self.dtype, crs=self.crs)
            if not new.is_singularity:
                multis.append(new)

        return list(reversed(multis))

    def to_crs(self, crs):
        if self._crs is None:
            raise ValueError("Current crs not specified, cannot proceed")

        new_x, new_y = crs_transform(self._crs, raw_crs_to_int(crs),
                                     [p.x for p in self.points],
                                     [p.y for p in self.points]
                                     )

        return self.__class__.from_xylists(new_x, new_y)

    def translate(self, x_dist, y_dist):
        return self.__class__().from_points([point.translate(x_dist, y_dist) for point in self.points])

class Polyline2D(BaseMultiPointClass):
    __WKT_IDENTITY__ = 'LINESTRING'
    __WKT_N_BRACKET__ = 1
    __REPR_IDENTITY__ = 'Polyline2D'

    def __init__(self, points=None, vectors=None, **kwargs):
        super().__init__(points=points, vectors=vectors, **kwargs)
        self._lenght = None

    @property
    def lenght(self):
        if self._lenght is None:
            self._lenght = self.get_s_lenght()
        return self._lenght

    def _reset_attributes(self):
        self._lenght = None
        super()._reset_attributes()

    def get_direct_lenght(self):
        return Vector2D(self.points[0], self.points[-1]).get_lenght()

    def get_direct_vector(self):
        return Vector2D(self.points[0], self.points[-1])

    def get_s_distance_from_origin(self, snaped_point):
        total = 0
        for vector in self.vectors:
            if not vector.on_self(snaped_point):
                total += vector.lenght
            else:
                total += Vector2D(vector.p1, snaped_point).lenght
                break
        return total

    @classmethod
    def from_points(cls, point_list, as_is=False, **kwargs):
        '''
        list of Point class objects

        if as_is is True, the algorithm does not check for back to back
        duplicates when constructing the point list and the vector list.
        '''
        return super(Polyline2D, cls).from_points(point_list, as_is=as_is, **kwargs)

    @staticmethod
    def from_shapely(polyline, **kwargs):
        if isinstance(polyline, shapely.geometry.LineString):
            x,y = polyline.coords.xy
            return Polyline2D.from_xylists(x, y, **kwargs)

        elif isinstance(polyline, shapely.geometry.MultiLineString):
            return [Polyline2D.from_shapely(poly, **kwargs) for poly in list(polyline)]

        else:
            raise TypeError('polyline must be of shapely.geometry.LineString or' +
                            'shapely.geometry.MultiLineString type, ' +
                            'received {}'.format(polyline.__class__))

    @staticmethod
    def from_vector_list(vector_list, as_is=False, **kwargs):
        '''
        Vector2D list

        if as_is is True, the algorithm does not check for gaps in the vector
        when constructing the point list and the vector list.

        **********************************************************************
        Exemple 1: "backtracking vectors":
                <Vector1( Start(x=p1.x, y=p1.y), End(x=p2.x, y=p2.y) )>
                                       and
                <Vector2( Start(x=p1.x, y=p1.y), End(x=p3.x, y=p3.y) )>

        will be added as the following point list for both case:

            <P1(p1.x, p1.y), P2(p2.x, p2.y), P3(p1.x, p1.y), P4(p3.x, p3.y)>

        the vector list will be treated differently:

        With as_is = True, they would be added as the following point list:
                <Vector1( Start(x=p1.x, y=p1.y), End(x=p2.x, y=p2.y) )>
                <Vector2( Start(x=p1.x, y=p1.y), End(x=p3.x, y=p3.y) )>

        With as_is = False, they would be added as the following point list:
                <Vector1( Start(x=p1.x, y=p1.y), End(x=p2.x, y=p2.y) )>
                <Vector2( Start(x=p2.x, y=p2.y), End(x=p1.x, y=p1.y) )>
                <Vector3( Start(x=p1.x, y=p1.y), End(x=p3.x, y=p3.y) )>

        **********************************************************************
        Exemple 2: "disconnected vectors":
                <Vector1( Start(x=p1.x, y=p1.y), End(x=p2.x, y=p2.y) )>
                                       and
                <Vector2( Start(x=p3.x, y=p3.y), End(x=p4.x, y=p4.y) )>

        will be added as the following point list for both case:

            <P1(p1.x, p1.y), P2(p2.x, p2.y), P3(p3.x, p3.y), P4(p4.x, p4.y)>

        the vector list will be treated differently:

        With as_is = True, they would be added as the following point list:
                <Vector1( Start(x=p1.x, y=p1.y), End(x=p2.x, y=p2.y) )>
                <Vector2( Start(x=p3.x, y=p3.y), End(x=p4.x, y=p4.y) )>

        With as_is = False, they would be added as the following point list:
                <Vector1( Start(x=p1.x, y=p1.y), End(x=p2.x, y=p2.y) )>
                <Vector2( Start(x=p2.x, y=p2.y), End(x=p3.x, y=p3.y) )>
                <Vector3( Start(x=p3.x, y=p3.y), End(x=p4.x, y=p4.y) )>

        '''
        pl2D = Polyline2D(**kwargs)

        if as_is:
            pl2D.vectors = vector_list
            #print (pl2D.vectors)
            pl2D._compute_points_from_vectors()
            return pl2D

        pl2D.points += [vector_list[0].p1, vector_list[0].p2]
        for v in range(1, len(vector_list)):

            if not vector_list[v-1]  == vector_list[v]:
                if vector_list[v].p1 in [vector_list[v-1].p1, vector_list[v-1].p2]:
                    if not pl2D.points[-1] == vector_list[v].p1:
                        #print ('case 1'+str(vector_list[v]))
                        pl2D.points.append(vector_list[v].p1)
                    if not pl2D.points[-1] == vector_list[v].p2:
                        #print ('case 2'+str(vector_list[v]))
                        pl2D.points.append(vector_list[v].p2)
                else:
                    #print ('case 3'+str(vector_list[v]))
                    pl2D.points += [vector_list[v].p1, vector_list[v].p2]
        pl2D._compute_vectors_from_points()

        if "crs" in kwargs.keys():
           pl2D._push_down_crs()

        return pl2D

    @classmethod
    def from_wkt(cls, text, as_is=False, **kwargs):
        if re.match('MULTILINESTRING', text):
            lines = []
            #clean it a bit
            text = multireplace(text, replacements={'MULTILINESTRING':''}, ignore_case=True).strip()
            #removing the outer parentheses
            text = re.sub('\(', '', re.sub('\)', '', text[::-1], count=1)[::-1], count=1)
            groups = re.split('[()]', text)
            for group in groups:
                group = multireplace(group, replacements={'(':'', ')':''}, ignore_case=True)
                if group.strip() != '' and group.strip() != ',':
                    lines.append(super(Polyline2D, cls).from_wkt(cls.__WKT_IDENTITY__+'('+group.strip()+')', as_is=as_is), **kwargs)
            return lines

        elif re.match(cls.__WKT_IDENTITY__, text):
                return super(Polyline2D, cls).from_wkt(text, as_is=as_is, **kwargs)

        else:
            raise ValueError('Pattern does not match a polyline')

    @staticmethod
    def polyline_to_polyline_distance(poly_a, poly_b):
        '''see https://stackoverflow.com/questions/45861488/distance-between-two-polylines'''
        import copy

        final_d = np.inf
        queue = {}
        counter = 0

        #initialize
        d_ab = BoundingBox.bbox_to_bbox_distance(poly_a.get_axis_aligned_bounding_box(), poly_b.get_axis_aligned_bounding_box())
        queue[counter] = (poly_a, poly_b, d_ab)

        while len(queue) > 0:
            pair_key = get_key_from_numvalue(queue, val_pos=-1, return_first=True, which='min')

            poly_a, poly_b, d_ab = queue[pair_key]
            queue.pop(pair_key)

            if final_d == 0.0:
                return final_d

            if d_ab > final_d:
                return final_d

            if len(poly_a.vectors) == 1 or len(poly_b.vectors) == 1:
                for vector_a in poly_a.vectors:
                    for vector_b in poly_b.vectors:
                        _dist = vector_a.vector_dist(vector_b)
                        if _dist < final_d:
                            final_d = copy.deepcopy(_dist)
            else:

                mid_point_a = int(len(poly_a.points) / 2)
                mid_point_b = int(len(poly_b.points) / 2)

                poly_a_1 = Polyline2D.from_points(poly_a.points[0:mid_point_a+1])
                poly_a_2 = Polyline2D.from_points(poly_a.points[mid_point_a:])

                poly_b_1 = Polyline2D.from_points(poly_b.points[0:mid_point_b+1])
                poly_b_2 = Polyline2D.from_points(poly_b.points[mid_point_b:])

                d_a1b1 = BoundingBox.bbox_to_bbox_distance(poly_a_1.get_axis_aligned_bounding_box(), poly_b_1.get_axis_aligned_bounding_box())
                d_a1b2 = BoundingBox.bbox_to_bbox_distance(poly_a_1.get_axis_aligned_bounding_box(), poly_b_2.get_axis_aligned_bounding_box())
                d_a2b1 = BoundingBox.bbox_to_bbox_distance(poly_a_2.get_axis_aligned_bounding_box(), poly_b_1.get_axis_aligned_bounding_box())
                d_a2b2 = BoundingBox.bbox_to_bbox_distance(poly_a_2.get_axis_aligned_bounding_box(), poly_b_2.get_axis_aligned_bounding_box())

                queue[counter+1] = (poly_a_1, poly_b_1, d_a1b1)
                queue[counter+2] = (poly_a_1, poly_b_2, d_a1b2)
                queue[counter+3] = (poly_a_2, poly_b_1, d_a2b1)
                queue[counter+4] = (poly_a_2, poly_b_2, d_a2b2)

                counter+=4

        return final_d

    @staticmethod
    def concat_polylines(poly_list):
        tmp = []
        for poly in poly_list:
            tmp += poly.as_tuples()
        return Polyline2D.from_tuples(tmp)

    @staticmethod
    def volume_between(poly_a, poly_b):
        raise NotImplementedError

    def shift(self, amount, side='right'):
        """
        Creates a new line on the specified side keeping the same directional
        vectors between the points
        """

        #first move all the vectors
        shifted_vectors = [vector.shift(amount, side=side) for vector in self.vectors]

        #we now need to connect (ie: shrink or expand) those vectors
        points = [shifted_vectors[0].points[0]]
        for v in range(1, len(shifted_vectors)):
            points.append(Vector2D.vector_intersection_point(shifted_vectors[v-1], shifted_vectors[v], allow_expand_vectors=True))
        points.append(shifted_vectors[v].points[-1])

        return self.__class__.from_points(points)

class Polygon2D(BaseMultiPointClass):
    __WKT_IDENTITY__ = 'POLYGON'
    __WKT_N_BRACKET__ = 2   #multi poly has 3: MULTIPOLYGON(((1 1,5 1,5 5,1 5,1 1),(2 2,2 3,3 3,3 2,2 2)),((6 3,9 2,9 4,6 3)))
    __REPR_IDENTITY__ = 'Polygon2D'

    def __init__(self, points=None, vectors=None, is_hole=False, **kwargs):
        super().__init__(points=points, vectors=vectors, **kwargs)
        if self.points != [] and self.vectors != []:
            self.vectors.append(Vector2D(self.points[-1], self.points[0]))

        self.is_hole = is_hole
        self.is_self_crossing = None

        self._area = None
        self._perimeter = None

    def _check_last_point(self):
        if self.points[-1] != self.points[0]:
            self.points.append(self.points[0])

    @property
    def area(self):
        if self._area is None:
            self._area = self.get_area()
        return self._area

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = self.get_s_lenght()
        return self._perimeter

    def _reset_attributes(self):
        self._area = None
        self._perimeter = None
        super()._reset_attributes()

    @classmethod
    def from_points(cls, point_list, is_hole=False, as_is=False, **kwargs):
        pg2D = super(Polygon2D, cls).from_points(point_list, as_is=as_is, **kwargs)
        pg2D.is_hole = is_hole
        pg2D._check_last_point()
        pg2D._compute_vectors_from_points(reset_vectors=True)

        if "crs" in kwargs.keys():
           pg2D._push_down_crs()
        return pg2D

    @staticmethod
    def from_vector_list(vector_list, is_hole=False, as_is=False, **kwargs):
        '''Vector2D list'''
        pg2D = Polygon2D(**kwargs)
        pg2D.points = Polyline2D.from_vector_list(vector_list, as_is=as_is).points
        pg2D.is_hole = is_hole
        pg2D._check_last_point()
        pg2D._compute_vectors_from_points(reset_vectors=True)

        if "crs" in kwargs.keys():
           pg2D._push_down_crs()
        return pg2D

    @staticmethod
    def from_polyline_list(polyline_list, is_hole=False, as_is=False, **kwargs):
        pg2D = Polygon2D(**kwargs)
        for p in range(len(polyline_list)):
            pg2D.points += Polyline2D.from_vector_list(polyline_list[p].vectors, as_is=as_is).points
        pg2D.is_hole = is_hole
        pg2D._check_last_point()
        pg2D._compute_vectors_from_points(reset_vectors=True)

        if "crs" in kwargs.keys():
           pg2D._push_down_crs()
        return pg2D

    @staticmethod
    def from_shapely(polygon, **kwargs):
        if isinstance(polygon, shapely.geometry.Polygon):
            x,y = polygon.exterior.coords.xy
            return Polygon2D.from_xylists(x, y, **kwargs)

        elif isinstance(polygon, shapely.geometry.MultiPolygon):
            return [Polygon2D.from_shapely(poly, **kwargs) for poly in list(polygon)]

        else:
            raise TypeError('polygon must be of shapely.geometry.Polygon or shapely.geometry.MultiPolygon type, received {}'.format(polygon.__class__))

    @classmethod
    def from_wkt(cls, text, as_is=False, **kwargs):
        if re.match('MULTIPOLYGON', text):
            raise NotImplementedError
            '''
            lines = []
            #clean it a bit
            text = multireplace(text, replacements={'MULTIPOLYGON':''}, ignore_case=True).strip()
            #removing the outer parentheses
            text = re.sub('\(', '', re.sub('\)', '', text[::-1], count=1)[::-1], count=1)

            and now... we might have this again!!!

            groups = re.split('[()]', text)
            for group in groups:
                group = multireplace(group, replacements={'(':'', ')':''}, ignore_case=True)
                if group.strip() != '' and group.strip() != ',':
                    lines.append(super(Polyline2D, cls).from_wkt(cls.__WKT_IDENTITY__+'('+group.strip()+')', as_is=as_is))
            return lines
            '''
        elif re.match(cls.__WKT_IDENTITY__, text):
                return super(Polygon2D, cls).from_wkt(text, as_is=as_is, **kwargs)

        else:
            raise ValueError('Pattern does not match a polyline')

    def get_area(self):
        '''
        https://www.mathopenref.com/coordpolygonarea2.html

        for testing purposes:

        point_list = [Point(-4, 6),
                      Point(-4,-8), Point( 8,-8),
                      Point( 8,-4), Point( 4,-4),
                      Point( 4, 6)
                      ]

        128.0 == Polygon2D.from_points(point_list).get_area()

        should be True
        '''
        area = 0.0

        if self.is_self_crossing is None:
            self._is_polygon_self_crossing()

        if not self.is_self_crossing:
            for i in range(len(self.points)-1): #need to remove the last point that is a duplicate from the first point
                if i == 0:
                    j = len(self.points) - 2 #The last vertex is the 'previous' one to the first
                else:
                    j = i-1 #j is previous vertex to i

                area += (self.points[j].x + self.points[i].x) * (self.points[j].y - self.points[i].y)

            return abs(area/2)

        else:
            #since the methodology fails on self crossing polygons, we must divide
            #it into smaller not crossing polygons and return the sum of their
            #respective areas

            subpoly_vectors = self.get_subpolygons()
            for subpoly in subpoly_vectors:
                #there are two possibilities here:
                #1. the main polygon is just folded on itself and every part must be added
                #2. the main polygon is superposed on itself and the superposed part must be added

                #the trick here is to check the wn number of the subpolygon
                #to avoid chosing a "bad point" to test (ie: a point that could be in another
                #subpolygon), we will move on the
                area += subpoly.get_area()

            return area

    @property
    def xsing_pts(self):
        if self.is_self_crossing is None:
            self._is_polygon_self_crossing()
        return self.xsing_pts

    def _is_polygon_self_crossing(self):
        return False
    '''
        #: debug this... dunno what the problem is

        self.is_self_crossing = False
        self._xsing_pts=[]
        for i in range(len(self.vectors)):
            for j in range(i+1, len(self.vectors)):
                print (i, j, '1')
                #checking if there is a common point to avoid costly calculations
                #if not self.vectors[i].p1 in self.vectors[j].get_points() and \
                #   not self.vectors[i].p2 in self.vectors[j].get_points() and \
                #   not self.vectors[j].p1 in self.vectors[i].get_points() and \
                #   not self.vectors[j].p2 in self.vectors[i].get_points():
                #    print (i, j, '2')
                #    #checking if there is an intersect witout calculating it
                #    if Vector2D.intersect(self.vectors[i], self.vectors[j]):
                #        xsing_point = Vector2D.vector_intersection_point(self.vectors[i], self.vectors[j])
                #        self.is_self_crossing = True
                #        self._xsing_pts.append({'point': xsing_point, 'vector1':self.vectors[i],'vector2':self.vectors[j]})

    def get_subpolygons(self):
        if self.is_self_crossing is not True:
            return None
        else:

            new_point_lookup = {}
            for point in self._xsing_pts:
                new_point_lookup[point['point']] = point
                new_point_lookup[point['point']]['life'] = 2

            all_points = self.get_points(include_last_point=False) + [point['point'] for point in self._xsing_pts]

            vectors = {}
            for vector in self.vectors:
                vectors[vector.p1] = vector

            #print( Vector2D.point_dist(self.xsing_pts[0]['point'], vectors[Point(3,5)]))
            subpoly_points=[]
            while len(all_points) > 0:

                #initialize
                sublist = []
                first = None
                direction = None

                while True:
                    if first is None:
                        if len(all_points) == 0: break
                        point = all_points[0]
                        first = point
                    else:
                        point = kept_candidate  #referenced below, cannot arrive here without a full circle once

                    sublist.append(point)
                    #get the vector
                    if point not in new_point_lookup:
                        direction = vectors[point]

                    else:
                        #detect the other vector and follow it
                        #if point == Point(7, 5): import pdb;pdb.set_trace()
                        if new_point_lookup[point]['vector1'] != direction:
                            direction = Vector2D(point, new_point_lookup[point]['vector1'].p2)
                        else:
                            direction = Vector2D(point, new_point_lookup[point]['vector2'].p2)

                    candidates = [p for p in all_points if direction.on_self(p) and p != point]

                    #order candidates by distance on the vector
                    distances = [Vector2D(direction.p1, candidate).lenght for candidate in candidates]
                    distances, candidates = sort2lists(distances, candidates)

                    if len(candidates) == 0:
                        kept_candidate = None
                    else:
                        kept_candidate = candidates[0]
                    #cleanup if a condition for termination of the current subpolygon arises
                    if kept_candidate == first or kept_candidate == None:
                        first = None
                        subpoly_points.append(sublist)
                        for point in reversed(all_points):
                            if point in sublist:
                                if point in new_point_lookup:
                                    if new_point_lookup[point]['life'] > 1:
                                        new_point_lookup[point]['life'] += -1
                                    else:
                                        all_points.pop(all_points.index(point))
                                else:
                                    all_points.pop(all_points.index(point))
                        break

            #build the polygons
            return [Polygon2D.from_points(poly) for poly in subpoly_points]
    '''

    def is_polygon_fully_contained(self, polygon):
        for point in polygon.points:
            if self.is_point_inside_crossing(point) == 0:
                return False
        return True

    def is_point_inside_bbox(self, point):
        if point.x >= self.min_x and point.x <= self.max_x:
            if point.y >= self.min_y and point.y <= self.max_y:
                return True
        return False

    def is_point_inside_crossing(self, point):
        '''
        http://geomalgorithms.com/a03-_inclusion.html

        returns 0 if point outside, 1 if point inside

        works for both clockwise and anti-clockwise polygons
        '''
        if not self.is_point_inside_bbox(point):
            return 0

        cn = 0
        for vector in self.vectors:
                    #crosses upward: includes its starting endpoint, and excludes its final endpoint
            if ( vector.orientation ==  1 and (point.y >= vector.p1.y and point.y < vector.p2.y) ) or \
               ( vector.orientation == -1 and (point.y >= vector.p2.y and point.y < vector.p1.y) ):
                    #crosses downward: excludes its starting endpoint, and includes its final endpoint
                if point.x <= vector.max_x:
                    if point.x < vector.get_x_intersect_at_y(point.y):
                        cn += 1

        return cn % 2

    def are_many_points_inside_crossing(self, pointList):

        #build the array: x,y,cn
        xes = np.asarray([p.x for p in pointList])
        yes = np.asarray([p.y for p in pointList])
        cn  = np.zeros(len(xes))

        for vector in self.vectors:

            #vector.get_x_intersect_at_y:
            if vector.dy != 0:
                x_intersect = (yes - vector.p1.y) * vector.dx / vector.dy + vector.p1.x
            else:
                x_intersect = np.ones(len(xes)) * vector.p1.x

            #test upward
            ori_up = (vector.orientation ==  1) * (yes >= vector.p1.y) * (yes < vector.p2.y)

            #test downward
            ori_dw = (vector.orientation == -1) * (yes >= vector.p2.y) * (yes < vector.p1.y)

            #apply tests
            cn = np.add(cn, np.ones_like(cn), out=cn, where=(np.logical_or(ori_up, ori_dw) * (xes < x_intersect) * (xes <= vector.max_x)))

        return list(cn % 2 > 0)

    def is_point_inside_winding(self, point):
        '''
        http://geomalgorithms.com/a03-_inclusion.html

        returns 0 if point outside

        if wn is greater than 1, it means the point is inside multiple times

        note: only works if the polygon is going clockwise....
        '''
        wn = 0
        for vector in self.vectors:
            if point.y > vector.p1.y and point.y < vector.p2.y: #the point actually can cross the vector horizontaly
                if vector.get_orientation() == 1: #crosses upward
                    if point.get_side_of_line(vector) == 1: #the point is strickly to the left
                        wn += 1

                elif vector.get_orientation() == -1: #crosses downward
                    if point.get_side_of_line(vector) == -1: #the point is strickly to the right
                        wn += -1
        return wn

    def shift(self, amount, side='right'):
        """
        Creates a new line on the specified side keeping the same directional
        vectors between the points
        """

        #first move all the vectors
        shifted_vectors = [vector.shift(amount, side=side) for vector in self.vectors]

        #we now need to connect (ie: shrink or expand) those vectors
        points = [Vector2D.vector_intersection_point(shifted_vectors[-1], shifted_vectors[0], allow_expand_vectors=True)]
        for v in range(1, len(shifted_vectors)):
            points.append(Vector2D.vector_intersection_point(shifted_vectors[v-1], shifted_vectors[v], allow_expand_vectors=True))

        #TODO: we want to avoid the overlaping when shrinking...

        return self.__class__.from_points(points)

##############
### Treat malformed multipolygons
##############
class NestedPolygons:

    def __init__(self, polygon, rank='surface'):
        self.childs = []
        self.rank = rank
        self.polygon = polygon

    @property
    def area(self):
        return self.polygon.area

    # Compare the new value with the parent node
    def insert(self, new_polygon):

        if isinstance(new_polygon, Polygon2D) and self.polygon.is_polygon_fully_contained(new_polygon) or \
           isinstance(new_polygon, NestedPolygons) and self.polygon.is_polygon_fully_contained(new_polygon.polygon):

            if self.childs == []:

                if isinstance(new_polygon, Polygon2D):
                    self.childs.append(NestedPolygons(new_polygon, rank='hole' if self.rank == 'surface' else 'surface'))
                else:
                    new_polygon.switch_rank()
                    new_polygon.switch_childs_rank()
                    self.childs.append(new_polygon)
                return 1

            else:
                #check if the new polygon is the child of a child
                get_out = False
                for child in self.childs:
                    if child.insert(new_polygon) > 0:
                        get_out = True
                        break

                #else insert it as a new child...
                if not get_out:
                    if isinstance(new_polygon, Polygon2D):
                        self.childs.append(NestedPolygons(new_polygon, rank='hole' if self.rank == 'surface' else 'surface'))
                    else:
                        new_polygon.switch_rank()
                        new_polygon.switch_childs_rank()
                        self.childs.append(new_polygon)

                return 1

        #if the old head in enclosed in the new propect, nest the old head and switch the rank of every childs
        elif isinstance(new_polygon, Polygon2D) and new_polygon.is_polygon_fully_contained(self.polygon) or \
             isinstance(new_polygon, NestedPolygons) and new_polygon.polygon.is_polygon_fully_contained(self.polygon):

            old = NestedPolygons(self.polygon, rank='hole' if self.rank == 'surface' else 'surface')
            old.childs = self.childs
            old.switch_childs_rank()

            if isinstance(new_polygon, Polygon2D):
                self.polygon = new_polygon
                self.childs = [old]
            else:
                self.polygon = new_polygon.polygon
                self.childs = new_polygon.childs
                self._calculate_AAMBB()
                self.insert(old)

            return 2

        else:
            return 0

    def switch_rank(self):
        self.rank = 'hole' if self.rank  == 'surface' else 'surface'

    def switch_childs_rank(self):
        for child in self.childs:
            child.switch_rank()

    #'''
    #    THIS VERSION IS DOWN TO TOP : 2.36MIN FOR TEST
    def _recursive_test_if_point_inside(self, point):
        if self.childs != []:
            for child in self.childs:
                inside, surface = child._recursive_test_if_point_inside(point)
                if inside and surface:
                    return True, True
                if inside and not surface:
                    return True, False

        test = self.polygon.is_point_inside_crossing(point)
        if test == 1:
            if self.rank == 'surface':
                return True, True
            else:
                return True, False
        return False, False
    #'''
    def is_point_inside_struct(self, point):
        inside, surface = self._recursive_test_if_point_inside(point)
        if inside and surface: return True
        return False

    def are_many_points_inside_struct(self, pointlist):
        result = np.zeros(len(pointlist))

        if self.childs != []:
            for child in self.childs:
                alpha = 1 if child.rank == 'surface' else -1

                result += alpha*np.asarray(child.polygon.are_many_points_inside_crossing(pointlist))

        alpha = 1 if self.rank == 'surface' else -1
        result += alpha*np.asarray(self.polygon.are_many_points_inside_crossing(pointlist))

        return list(result > 0)

    '''
    def _recursive_test_if_point_inside(self, point):

        #start with giant inequalities

        test = self.polygon.is_point_inside_crossing(point)
        if test == 0:
            return False, False

        if self.childs != []:
            for child in self.childs:
                inside, surface = child._recursive_test_if_point_inside(point)
                if inside and surface:
                    return True, True
                if inside and not surface:
                    return True, False

        if self.rank == 'surface':
            return True, True
        else:
            return True, False
    '''

    def print_tree(self, pad_before=0):
        add_str = ' '
        for i in range(pad_before):
            add_str += ' '

        print('{}{:7}: {}'.format(add_str, self.rank, self.polygon))

        if len(self.childs) > 0:
            print('{}  -> childs:'.format(add_str))

            for child in self.childs:
                child.print_tree(pad_before=7+pad_before)

    def plot(self, **kwargs):
        linestyle = '-' if self.rank == 'surface' else '-.'
        self.polygon.plot(linestyle=linestyle, **kwargs)
        for child in self.childs:
            child.plot(**kwargs)

def create_nested_polygon_list(polygons, print_steps=False, sort_with_area=False, sort_ascending=True):
    """
    polygons as a list of Polygon2D objects
    """

    #first check if theres a single polygon coming in, if yes return it
    if isinstance(polygons, Polygon2D):
        return NestedPolygons(polygons)


    heads = []
    count = 1
    for poly in polygons:

        get_out = False
        if heads == []:
            heads.append(NestedPolygons(poly))
            if print_steps: print ('init ', count, poly)
        else:
            for head in reversed(heads):
                old_head=head.polygon
                test = head.insert(poly)

                if test == 1:
                    if print_steps: print (count, ' got inserted inside', polygons.index(old_head)+1)#, poly)
                    get_out = True
                    break

                if test == 2:
                    if print_steps: print (count, 'replaced', polygons.index(old_head)+1)#, poly)

                    get_out = True
                    break

            if not get_out:
                heads.append(NestedPolygons(poly))
                if print_steps: print ('direct append ', count, poly)

            #since we are adding a new head, we need to make sure we checked all the
            #previous heads to make sure they do not belong as childs on the newest one
            if get_out and test == 2:

                for head in reversed(heads[:-1]): #the new head is the last one
                    if heads[-1].insert(head) == 1:
                        if print_steps: print ('head ', heads.index(head), ' got inserted into the newly created head')
                        heads.remove(head)
        count += 1

    if sort_with_area:
        areas = [p.area for p in heads]
        areas, heads = sort2lists(areas, heads, ascending=sort_ascending)

    return heads

def test_point_list_in_many_NestedPolygons(points,
                                           nestedPolygons,
                                           return_detailed=False,
                                           nestedPolygons_are_mutualy_exclusives=False):

    """
    to speed this ups, pass the nestedPolygons in descending order of their areas
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)

    tested = np.zeros((len(points), len(nestedPolygons)))

    for p in range(len(nestedPolygons)):

        if nestedPolygons_are_mutualy_exclusives:
            #since we know a points can only be in a single polygon at a time,
            #we can save time by only testing those points who were not
            #previously assigned to a particular polygon
            points_to_test_indices = np.nonzero(tested.sum(axis=1) == 0)[0]

            #if no more points are left to test, then we can skip all the
            #remaining tests since the matrix already contains False values (0)
            #for those
            if len(points_to_test_indices) == 0:
                break

        else:
            #since we dont know if a point can be in many polygons at a time,
            #we need to test them all
            points_to_test_indices = range(len(points))

        #gether the points that will be tested this run
        points_to_test = points[points_to_test_indices]

        #the actual test
        tested[points_to_test_indices,p] = nestedPolygons[p].are_many_points_inside_struct(points_to_test)

    if return_detailed:
        return tested > 0
    else:
        return tested.sum(axis=1) > 0

##############
### Arrows
##############
Arrow_shapes = {'strait':
                       {'start':{'side':'bot', 'frac':0.5},
                        'end':{'side':'top', 'frac':0.5},
                        'angle_range': '-22.5_<=_x_<=_22.5',
                        'default_A':90,
                        'default_B':0,
                        },

                'slight-left':
                       {'start':{'side':'bot', 'frac':0.5},
                        'end':{'side':'left', 'frac':0.75},
                        'angle_range': '22.5_<_x_<=_67.5',
                        'default_A':0,
                        'default_B':45
                        },

                'left':
                       {'start':{'side':'bot', 'frac':0.5},
                        'end':{'side':'left', 'frac':0.5},
                        'angle_range': '67.5_<_x_<=_112.5',
                        'default_A':0,
                        'default_B':90
                        },

                'sharp-left':
                       {'start':{'side':'bot', 'frac':0.5},
                        'end':{'side':'left', 'frac':0.25},
                        'angle_range': '112.5_<_x_<=_157.5',
                        'default_A':0,
                        'default_B':135
                        },

                'slight-right':
                       {'start':{'side':'bot', 'frac':0.5},
                        'end':{'side':'right', 'frac':0.75},
                        'angle_range': '-67.5_<=_x_<_-22.5',
                        'default_A':0,
                        'default_B':-60
                        },

                'right':
                       {'start':{'side':'bot', 'frac':0.5},
                        'end':{'side':'right', 'frac':0.5},
                        'angle_range': '-112.5_<=_x_<_-67.5',
                        'default_A':0,
                        'default_B':-90
                        },

                'sharp-right':
                       {'start':{'side':'bot', 'frac':0.5},
                        'end':{'side':'right', 'frac':0.25},
                        'angle_range': '-157.5_<=_x_<_-112.5',
                        'default_A':0,
                        'default_B':-135
                         },

                'u-turn':
                       {'start':{'side':'bot', 'frac':0.5},
                        'end':{'side':'left', 'frac':0},
                        'angle_range': '-157.5_<=_x_>_157.5',
                        'default_A':20,
                        'default_B':160
                        }
                }
#angle3,angleA=20,angleB=160

##############
### Bounding boxes
##############
class BoundingBox(Polygon2D):
    def __init__(self, center=None, height=None, width=None, rotate=None, rotation_angle=0, points=[]):
        '''center as Point class

        ATTENTION!
        witdh is on the x-axis and height is on the y-axis!
        '''
        if points != []:
            if not isinstance(points, (list, tuple, np.ndarray)):
                raise TypeError(f"`points` must be one of {{{', '.join(['list', 'tuple', 'np.ndarray'])}}}, receive {points.__class__}")
            for elem in points:
                if not isinstance(elem, Point):
                    raise TypeError(f"All elements of `points` must of type Point, receive {elem.__class__}")
            if len(points) != 5:
                raise ValueError(f"`points` must contain 5 points, received {len(points)}")

            self.center = Vector2D(points[0], points[2]).get_mid_point()
            self.height = Point.distanceNorm2(points[1], points[2])
            self.width = Point.distanceNorm2(points[0], points[1])

        else:

            if isinstance(center, (list, tuple, np.ndarray)):
                center = Point.from_tuple(center)

            self.center = center
            self.height = float(height)
            self.width  = float(width)

            points = BoundingBox._calculate_corners_from_params(center, height, width)

        self.rotation_angle = 0

        super().__init__(points=points, vectors=None)

        self._calculate_extents()

        if rotate:
            self.rotate(rotation_angle=rotation_angle, inplace=True)

    @staticmethod
    def from_corners(points, rotation_angle=0.0):
        """

        Parameters
        ----------


        rotation_angle : Float
             The angle of rotation in degrees. Positive is counterclockwise
             (Default value = 0.0)

        Returns
        -------
        BBox object
        """
        return BoundingBox(points=points, rotation_angle=rotation_angle)

    @classmethod
    def from_extent(cls, x0, y0, x1, y1, rotation_angle=0.0):
        """

        Parameters
        ----------
        xo: float
            The bottom left corner's x value

        y0: float
            The bottom left corner's y value

        x1: float
            The top right corner's x value

        y1: float
            The top right corner's y value

        rotation_angle : Float
             The angle of rotation in degrees. Positive is counterclockwise
             (Default value = 0.0)

        Returns
        -------
        BBox object
        """
        return cls(Point((x0+x1)/2, (y0+y1)/2),
                   y1-y0,
                   x1-x0,
                   rotation_angle=rotation_angle,
                   )

    @classmethod
    def from_limits(cls, xlim, ylim, rotation_angle=0.0):
        """

        Parameters
        ----------
        xlim : list
            xmin, xmax values as float

        ylim : list
            ymin, ymax values as float

        rotation_angle : Float
             The angle of rotation in degrees. Positive is counterclockwise
             (Default value = 0.0)

        Returns
        -------
        BBox object
        """
        return cls(Point(np.mean(xlim), np.mean(ylim)),
                   ylim[1] - ylim[0],
                   xlim[1] - xlim[0],
                   rotation_angle=rotation_angle,
                   form=form,
                   )

    def change_scale(self, factor=1.0, inplace=False):
        """
        Change the witdh and height of the bounding box.

        Parameters
        ----------
        factor : float
            changes the both the witdh and heigh value of the boundingbox

        Returns
        -------
        None
        """
        if inplace:
            self.width  = self.width  * factor
            self.height = self.height * factor
            if self.rotation_angle != 0:
                self._calculate_corners(calculate_rotation=True, calculate_points=True)
            else:
                self._calculate_corners(calculate_rotation=False, calculate_points=True)
        else:
            bcopy = self.__deepcopy__()
            bcopy.change_scale(factor=factor, inplace=True)
            return bcopy

    def change_hscale(self, h_factor=1.0, inplace=False):
        """
        Change the witdh and height of the bounding box.

        Parameters
        ----------
        h_factor : float
            changes the witdh value of the boundingbox

        Returns
        -------
        None
        """
        if inplace:
            self.height = self.height * h_factor
            if self.rotation_angle != 0:
                self._calculate_corners(calculate_rotation=True, calculate_points=True)
            else:
                self._calculate_corners(calculate_rotation=False, calculate_points=True)
        else:
            bcopy = self.__deepcopy__()
            bcopy.change_hscale(h_factor=h_factor, inplace=True)
            return bcopy

    def change_wscale(self, w_factor=1.0, inplace=False):
        """
        Change the heigh and height of the bounding box.

        Parameters
        ----------
        w_factor : float
            changes the witdh value of the boundingbox

        Returns
        -------
        None
        """
        if inplace:
            self.width  = self.width  * w_factor
            if self.rotation_angle != 0:
                self._calculate_corners(calculate_rotation=True, calculate_points=True)
            else:
                self._calculate_corners(calculate_rotation=False, calculate_points=True)
        else:
            bcopy = self.__deepcopy__()
            bcopy.change_wscale(w_factor=w_factor, inplace=True)
            return bcopy

    def __copy__(self):
        return self

    def __deepcopy__(self):
        #create a new instance
        new = self.__class__(self.center,
                             self.height,
                             self.width,
                             rotation_angle=self.rotation_angle,
                             form=self.form,
                             )

        #copy whatever was assigned in other fields of the class
        new.__dict__.update(self.__dict__)

        return new

    def copy(self, deep=False):
        if deep:
            return self.__deepcopy__()
        return self.__copy__()

    def translate(self, x_distance, y_distance, inplace=False):
        """
        Move the bounding box in x and y.

        Parameters
        ----------
        x_distance : float
                The x delta to apply

        y_distance : float
                The y delta to apply

        Returns
        -------
        None
        """
        if inplace:
            if x_distance != 0 and y_distance != 0:
                self.center = self.center.translate(x_distance, y_distance)
                if self.rotation_angle != 0:
                    self._calculate_corners(calculate_rotation=True, calculate_points=True)
                else:
                    self._calculate_corners(calculate_rotation=False, calculate_points=True)

        else:
            #create a new instance and transform it
            bcopy = self.__deepcopy__()
            bcopy.translate(x_distance, y_distance, inplace=True)
            return bcopy

    def _rotate(self, rotation_angle=0.0, base_point='center'):
        """
        Rotate the polyline forming the bbox around the center point.

        Parameters
        ----------
        rotation_angle : Float
             The angle of rotation in degrees. Positive is counterclockwise
             (Default value = 0.0)

        Returns
        -------
        poly: Polyline2D
            The rotated polyline
        """
        if base_point == 'center':
            base_point = self.center
        elif isinstance(base_point, (tuple, list, np.ndarray)):
            base_point = Point.from_tuple(base_point)

        if not isinstance(base_point, Point):
            raise TypeError(f"`base_point` must be a Point object or one of {{tuple, list, np.ndarray}} {base_point.__class__}")

        return Polygon2D.from_points(self.points).rotate(rotation_angle, base_point=base_point)

    def rotate(self, rotation_angle=0.0, base_point='center', inplace=False):
        """
        Rotate the bounding box around the center point.

        Parameters
        ----------
        rotation_angle : Float
             The angle of rotation in degrees. Positive is counterclockwise
             (Default value = 0.0)

        Returns
        -------
        None
        """
        poly = self._rotate(rotation_angle=rotation_angle)

        if inplace:
            self.__dict__.update(poly.__dict__)
            self.rotation_angle += rotation_angle

        else:
            #BoundingBox
            bbox = BoundingBox.from_corners(poly.points)
            bbox.rotation_angle = self.rotation_angle + rotation_angle

            return bbox

    def _calculate_extents(self):

        minx = min([p.x for p in self.points])
        miny = min([p.y for p in self.points])
        maxx = max([p.x for p in self.points])
        maxy = max([p.y for p in self.points])

        self.aabbLimits = [minx, miny, maxx, maxy]

    @staticmethod
    def _calculate_corners_from_params(center, height, width):
        """
        A submethod to calcule the corners from the attributes
        """
        #in order: lower_left, lower_right, upper_right, upper_left
        p1 = Point(center.x - height / 2,
                   center.y - width / 2)
        p2 = Point(center.x + height / 2,
                   center.y - width / 2)
        p3 = Point(center.x + height / 2,
                   center.y + width / 2)
        p4 = Point(center.x - height / 2,
                   center.y + width / 2)

        return [p1, p2, p3 , p4]

    @property
    def extent(self):
        return self.aabbLimits

    def is_point_in_BoundingBox(self, point, method='crossing'):
        if method == 'winding':
            return self.is_point_inside_winding(point) > 0
        if method == 'crossing':
            return self.is_point_inside_crossing(point) > 0

    @staticmethod
    def bbox_to_bbox_distance(bbox1, bbox2):

        outer_bbox = AAMBoundingBox(bbox1.get_points() + bbox2.get_points())
        return (max(outer_bbox.height - bbox1.height - bbox2.height, 0)**2 + max(outer_bbox.width - bbox1.width - bbox2.width, 0)**2)**0.5

class TurningArrow:
    _LOS_palette = {'Green':'#1ECB35',
                    'Yellow':'#ffff00',
                    'Orange':'#ffbf00',
                    'Red':'#D53134',
                    None:'none'}

    _LOS_colors = {'A':'Green',
                   'B':'Green',
                   'C':'Green',
                   'D':'Yellow',
                   'E':'Orange',
                   'F':'Red',
                   None:None}

    """
    Class used to create turning mouvement arrows for traffic simulation reports
    the arrow is built by detecting automatically the type of movement from a
    given angle and centered on a Point(x,y).

    Parameters
    ----------

    Returns
    -------
    TurningArrow object
    """
    def __init__(self, center, arrow_angle, rotate_bbox=False, bbox_rotation_angle=0,
                 LOS=None, arrow_scale=1, height=1.0, width=1.0, lock_on='display'):

        if not isinstance(arrow_angle, numbers.Number) or isinstance(arrow_angle, bool):
            raise ValueError(f'`arrow_angle` must be numeric, received {arrow_angle.__class__}')
        self.arrow_angle = arrow_angle
        self.center = center
        self.arrow_scale = arrow_scale
        self.height = height
        self.width = width

        self.bbox = BoundingBox(center, height, width)
        if rotate_bbox:
            self.bbox.rotate(rotation_angle=bbox_rotation_angle, inplace=True)

        if LOS is not None:
            color = TurningArrow._LOS_palette[TurningArrow._LOS_colors[LOS]]
        else:
            color = 'k'
        self.set_facecolor(color)
        self.set_edgecolor(color)

        self.set_transform(lock_on)
        self.textargs = None

    def __copy__(self):
        return self

    def __deepcopy__(self):
        #create a new instance
        new = self.__class__(self.bbox.center,
                             arrow_angle = self.arrow_angle,
                             rotation_angle=self._rotation_angle,
                             arrow_scale=self._arrow_scale,
                             height=self.height,
                             width=self.width
                             )

        #copy whatever was assigned in other fields of the class
        new.__dict__.update(self.__dict__)

        return new

    @property
    def rotation_angle(self):
        return self.bbox.rotation_angle

    def copy(self, deep=False):
        if deep:
            return self.__deepcopy__()
        return self.__copy__()

    def get_side(self, which='left'):
        #the vectors come in this order: lower_left, lower_right, upper_right, upper_left
        if which == 'bot':
            return self.bbox.get_vectors()[0]
        if which == 'right':
            return self.bbox.get_vectors()[1]
        if which == 'top':
            return self.bbox.get_vectors()[2]
        if which == 'left':
            return self.bbox.get_vectors()[3]

    def set_arrowStyle(self, arrow_scale=1, head_length=2, head_width=2.5, tail_width=1):
        """
        Modify the head and body parameters of the fancyarrow.
        See Matplotlib's FancyArrow doc for more details

        Parameters
        ----------
        arrow_scale : float
             (Default value = 1)
        head_length : float
             (Default value = 2)
        head_width : float
             (Default value = 2.5)
        tail_width : float
             (Default value = 1)

        Returns
        -------
        None
        """
        self.arrow_scale = arrow_scale
        self.head_length = head_length
        self.head_width = head_width
        self.tail_width = tail_width
        self.arrowStyle = 'simple,head_length={},head_width={},tail_width={}'.format(head_length*arrow_scale, head_width*arrow_scale, tail_width*arrow_scale)

    def set_connectionstyle(self, connectionstyle='angle3,angleA=90,angleB=0'):
        """
        Modify the curvature parameters of the fancyarrow

        Parameters
        ----------
        connectionstyle : any connection style string from matplotlib
             (Default value = 'angle3,angleA=90,angleB=0')

        Returns
        -------
        None
        """
        self.connectionstyle = connectionstyle

    def set_edgecolor(self, edgecolor='black'):
        """
        Changes the boundary color of the arrow

        Parameters
        ----------
        facecolor : any valid matplotlib color scheme
             (Default value = 'white')

        Returns
        -------
        None
        """
        self.edgecolor = edgecolor

    def set_facecolor(self, facecolor='white'):
        """
        Changes the filling color of the arrow

        Parameters
        ----------
        facecolor : any valid matplotlib color scheme
             (Default value = 'white')

        Returns
        -------
        None
        """
        self.facecolor = facecolor

    def set_scale(self, h_scale, w_scale, arrow_scale, inplace=False):
        """
        Changes the witdh and height of the bounding box and the size of the
        arrow enclosed

        Parameters
        ----------
        h_scale : float
            changes the heigh value of the boundingbox

        w_scale : float
            changes the witdh value of the boundingbox

        arrow_scale : float
            changes the size of the fancyarrow

        Returns
        -------
        None
        """
        if inplace:
            self.set_arrowStyle(arrow_scale=arrow_scale, head_length=self.head_length, head_width=self.head_width, tail_width=self.tail_width)
            self.bbox.change_hscale(h_scale)
            self.bbox.change_wscale(w_scale)
        else:
            tbcopy = self.__deepcopy__()
            tbcopy.set_scale(h_scale, w_scale, arrow_scale, inplace=True)
            return tbcopy

    def rotate(self, rotation_angle=0.0, inplace=False):
        """
        Rotates the arrow and it's bounding box around the center point

        Parameters
        ----------
        rotation_angle : Float
             The angle of rotation in degrees. Positive is counterclockwise
             (Default value = 0.0)

        Returns
        -------
        None
        """
        if inplace:
            self.bbox.rotate(rotation_angle=rotation_angle,
                             inplace=inplace
                             )

        else:
            __dict__ = self.__dict__.copy()
            __dict__.pop('bbox')
            ta = TurningArrow(__dict__.get('center'),
                              __dict__.get('arrow_angle'))

            ta.bbox = self.bbox.rotate(rotation_angle=rotation_angle)
            ta.__dict__.update(__dict__)

            return ta

    def translate(self, x_distance, y_distance, inplace=False):
        """
        Moves the arrow and it's bounding box in x and y

        Parameters
        ----------
        x_distance : float
                The x delta to apply

        y_distance : float
                The y delta to apply

        Returns
        -------
        None
        """
        if inplace:
            self.bbox.translate(x_distance, y_distance)
        else:
            tbcopy = self.__deepcopy__()
            tbcopy.set_scale(x_distance, y_distance, inplace=True)
            return tbcopy

    def detect_snapped_angle(self, arrow_angle):
        """
        Automatically chooses from Arrow_shapes dict the type of turn
        associated with the arrow to trace

        Parameters
        ----------
        arrow_angle : float
            The angle to use to detect the movement


        Returns
        -------
        A string from Arrow_shapes.keys()

        """
        for key in Arrow_shapes.keys():
            string_split = Arrow_shapes[key]['angle_range'].split('_')
            if  eval_truth(float(string_split[0]), string_split[1], arrow_angle) \
            and eval_truth(arrow_angle, string_split[-2], float(string_split[-1])):
                return key
        return 'u-turn'

    def set_transform(self, lock_on='display'):
        lockOnVal=['display', 'data']
        if not lock_on in lockOnVal:
            raise ValueError(f"`lock_on` must be one of {{{', '.join(lockOnVal)}}}, received {lock_on}")

        if lock_on == 'display':
            self.transform = transform=ax.transAxes
        if lock_on == 'data':
            self.transform = ax.transData

    def annotate(self, ax, showbbox=False, fixDisplay=False, bbox_plot_kwargs={}, arrow_kwargs={}):
        """
        Traces the arrow onto the

        Parameters
        ----------
        ax : ax object
            Corresponds to the figure where the arrow will be traced

        showbbox : Bool
             Wether to display the bounding box or not
             (Default value = False)

        fixDisplay

        Returns
        -------
        None
        """
        key = self.detect_snapped_angle(self.arrow_angle)

        connection_style = "angle3,angleA={A},angleB={B}".format(
            A=Arrow_shapes[key]['default_A'],# + self.rotation_angle,
            B=Arrow_shapes[key]['default_B'],# + self.rotation_angle
            )

        self.set_connectionstyle(connection_style)

        sVect = Arrow_shapes[key]['start']['side']
        sFrac = Arrow_shapes[key]['start']['frac']

        eVect = Arrow_shapes[key]['end']['side']
        eFrac = Arrow_shapes[key]['end']['frac']

        startPoint = self.get_side(sVect).fraction_pt(sFrac)
        endPoint =  self.get_side(eVect).fraction_pt(eFrac)

        self._annotate(ax, startPoint.x, startPoint.y, endPoint.x, endPoint.y, transform=self.transform, **arrow_kwargs)
        if showbbox:
            self.bbox.plot(ax=ax, transform=self.transform, **bbox_plot_kwargs)

        if self.textargs is not None:
            text = self._addText(
                     self.textargs['text'], self.textargs['dx'],
                     self.textargs['dy'], self.textargs['height'],
                     self.textargs['width'],
                     rotation=self.textargs['rotation'],
                     anchor=self.textargs['anchor'],
                     )

            text.plot(ax=ax, showbbox=showbbox, transform=self.transform, **self._textkwargs)
        return ax

    def _annotate(self, ax, x_start, y_start, x_end, y_end, **kwargs):
        """
        Internal function to trace the arrow on the figure

        Parameters
        ----------
        ax : ax object
            Corresponds to the figure where the arrow will be traced

        x_start : float
            x-coordinate of the first point of the arrow

        y_start : float
            y-coordinate of the first point of the arrow

        x_end : float
            x-coordinate of the pointy end of the arrow

        y_end : float
            y-coordinate of the pointy end of the arrow


        Returns
        -------
        None

        """
        #ax.annotate('', # only arrow (no text)
        #            xy= (x_end, y_end), xycoords='data',
        #            xytext = (x_start, y_start) ,textcoords='data',
        #            arrowprops=dict(arrowstyle=self.arrowStyle,
        #                            edgecolor=self.edgecolor,
        #                            facecolor=self.facecolor,
        #                            connectionstyle=self.connectionstyle)
        #                            )

        edgecolor = kwargs.pop("edgecolor", self.edgecolor)

        arrow = patches.FancyArrowPatch(posA=(x_start, y_start),
                                        posB=(x_end, y_end),
                                        #arrowstyle=kwargs.pop("arrowStyle", self.arrowStyle),
                                        edgecolor=kwargs.pop("edgecolor", self.edgecolor),
                                        facecolor=kwargs.pop("facecolor", self.facecolor),
                                        #connectionstyle=kwargs.pop("connectionstyle", self.connectionstyle),
                                        mutation_scale=self.arrow_scale,
                                        **kwargs)
        ax.add_patch(arrow)

    def addText(self, text, dx, dy, height, width, anchor='bot', rotation='auto', **kwargs):
        self._textkwargs = kwargs
        self.textargs = {'text':text, 'dx':dx, 'dy':dy, 'height':height,
                         'width':width, 'rotation':rotation, 'anchor':anchor}

    def _addText(self, text, dx, dy, height, width, anchor='bot', rotation='auto', **kwargs):
        if rotation == 'auto':
            rotation = -self.rotation_angle
        elif not isinstance(rotation, numbers.Numeric):
            raise TypeError(f"`rotation` must be either `auto` or a numeric object, received {auto.__class__}")

        #centerline
        cl = Vector2D(self.center, self.get_side('top').fraction_pt(0.5)).normalized()
        pr = cl.perpendicular_clockwise()

        if anchor == 'top':
            p1 = self.get_side('top').fraction_pt(0.5)
        elif anchor == 'bot':
            p1 = self.get_side('bot').fraction_pt(0.5)
        elif anchor == 'left':
            p1 = self.get_side('left').fraction_pt(0.5)
        elif anchor == 'right':
            p1 = self.get_side('right').fraction_pt(0.5)
        else:
            raise ValueError(f"`anchor` must be one of {{'bot', 'bot', 'bot', 'bot'}}, received {anchor}")

        text_center = Vector2D.from_dxdy(p1, cl.dx * dx + pr.dx * dx, cl.dy * dy + pr.dy * dy).p2


        return TextBBox(Point(text_center.x, text_center.y), height, width, text=text, rotation_angle=-rotation)


class TextBBox(BoundingBox):
    def __init__(self, center, height, width, text=None, rotation_angle=0.0):
        '''center as Point class'''
        self.text = text
        print(rotation_angle)
        super().__init__(center, height, width, rotate=True, rotation_angle=rotation_angle)

    @classmethod
    def from_extent(cls, x0, y0, x1, y1, text=None, rotation_angle=0.0):
        """

        Parameters
        ----------
        xo: float
            The bottom left corner's x value

        y0: float
            The bottom left corner's y value

        x1: float
            The top right corner's x value

        y1: float
            The top right corner's y value

        rotation_angle : Float
             The angle of rotation in degrees. Positive is counterclockwise
             (Default value = 0.0)

        Returns
        -------
        TextBBox object
        """
        tmp = super(TextBBox, cls).from_extent(x0, y0, x1, y1,
                                               rotation_angle=rotation_angle,
                                               )
        tmp.text = text
        return tmp

    @classmethod
    def from_limits(cls, xlim, ylim, text=None, rotation_angle=0.0):
        """

        Parameters
        ----------
        xlim : list
            xmin, xmax values as float

        ylim : list
            ymin, ymax values as float

        rotation_angle : Float
             The angle of rotation in degrees. Positive is counterclockwise
             (Default value = 0.0)

        Returns
        -------
        TextBBox object
        """
        tmp = super(TextBBox, cls).from_limits(xlim, ylim,
                                               rotation_angle=rotation_angle,
                                               )
        tmp.text = text
        return tmp

    def plot(self, showbbox=False, color='k', linewidth=2, position=None, **kwargs):

        xpoint = self.center.x
        ypoint = self.center.y

        for kwarg in kwargs:
            value = kwargs[kwarg]
            if kwarg == 'horizontalalignment' or kwargs == 'ha':
                if value == 'right':
                    xpoint = max([point.x for point in self.points])
                if value == 'center':
                    xpoint = self.center.x
                if value == 'left':
                    xpoint = min([point.x for point in self.points])

            if kwarg == 'verticalalignment' or kwargs == 'va':
                if value == 'top':
                    ypoint = max([point.y for point in self.points])
                if value == 'center':
                    ypoint = self.center.y
                if value == 'bottom':
                    ypoint = min([point.y for point in self.points])

        if position is not None:
            x, y = position
            if not (x >= -1 and x <= 1):
                raise ValueError('X component of the position must be between -1 and 1')
            if not (y >= -1 and y <= 1):
                raise ValueError('Y component of the position must be between -1 and 1')

            xpoint = Vector2D(self.center, Point(max([point.x for point in self.points]), self.center.y)).fraction_pt(x).x
            ypoint = Vector2D(self.center, Point(self.center.x, max([point.y for point in self.points]))).fraction_pt(y).y

        #this is for plotting the box on more complex figures
        ax=None
        if "ax" in kwargs:
            ax = kwargs.pop("ax")

        if ax is not None:
            print(self.rotation_angle)
            ax.text(xpoint, ypoint, self.text, rotation=self.rotation_angle, **kwargs)
        else:
            plt.text(xpoint, ypoint, self.text, rotation=self.rotation_angle, **kwargs)

        if showbbox:
            for vector in self.get_vectors():
                vector.plot(color=color, linewidth=linewidth, ax=ax)

class TableBBox(object):
    '''Create a table'''

    def __init__(self, cellText, colLabels=None):
        self.cellText = cellText
        self.colLabels = colLabels
        self.nbCols = len(cellText[0])
        self.nbRows = len(cellText)

    def plot(self, wCell, hCell, position=(0,0), fontsize=12, color='k', linewidth=1):
        '''
        wCell: the width of the cell
        hCell: the height of the cell
        position: the lower left corner of the table as a tuple, eg (1,2)
        '''

        # Reverse the list
        cellText = self.cellText[::-1]

        yPos = hCell/2 + position[1]
        for row in cellText:
            xPos = wCell/2 + position[0]
            for text in row:
                textBBox = TextBBox('{}'.format(text), Point(xPos,yPos), wCell, hCell)
                textBBox.plot(showbbox=True, ha='left',va='center', fontsize=fontsize, color=color, linewidth=linewidth)
                xPos += wCell
            yPos += hCell

        if self.colLabels is not None:
            xPos = wCell/2 + position[0]
            for label in self.colLabels:
                textBBox = TextBBox('{}'.format(label), Point(xPos,yPos), wCell, hCell)
                textBBox.plot(showbbox=True, ha='center',va='center', fontsize=fontsize, color=color, linewidth=linewidth)
                xPos += wCell

class AAMBoundingBox(BoundingBox):
    '''Axis-aligned minimum bounding box'''
    def __init__(self, point_list):
        self.point_list = point_list
        self.rotation_angle = 0
        center, height, width = self.calculate_min_max_coords()
        super().__init__(center, height, width, form='square')

    @classmethod
    def from_limits(cls, xlim, ylim):
        """

        Parameters
        ----------
        xlim : list
            xmin, xmax values as float

        ylim : list
            ymin, ymax values as float
        """
        return super(BoundingBox, cls).from_limits(xlim, ylim)

    def add_point(self, point):
        self.point_list.append(point)
        self.calculate_min_max_coords()

    def calculate_min_max_coords(self):
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

        for point in self.point_list:
            if self.min_x is None:
                self.min_x = point.x
            elif point.x < self.min_x:
                self.min_x = point.x

            if self.max_x is None:
                self.max_x = point.x
            elif point.x > self.max_x:
                self.max_x = point.x

            if self.min_y is None:
                self.min_y = point.y
            elif point.y < self.min_y:
                self.min_y = point.y

            if self.max_y is None:
                self.max_y = point.y
            elif point.y > self.max_y:
                self.max_y = point.y

        center = Point((self.max_x + self.min_x) / 2, (self.max_y + self.min_y) / 2)
        height = abs(self.max_x - self.min_x)
        width  = abs(self.max_y - self.min_y)

        return center, height, width

    def plot(self, **kwargs):
        for vector in self.get_vectors():
            vector.plot(**kwargs)

        for point in self.point_list:
            point.plot(**kwargs)

    @staticmethod
    def do_two_AAMBB_intersects(AAMBB1, AAMBB2):
       '''
       Calculates if AAMBB1 and AAMBB2 intersect each other
       '''

       #with X being AAMBB1:
       #                |       |
       #            H   |   A   |   B
       # max_y1 --------|-------|-------
       #            G   |   X   |   C
       # min_y1 --------|-------|-------
       #            F   |   E   |   D
       #                |       |
       #             max_x2  max_x2
       #
       #    A: AAMBB1.max_y < AAMBB2.min_y
       #    B: A & C
       #    C: AAMBB1.max_x < AAMBB2.min_x
       #    D: C & E
       #    E: AAMBB1.min_y > AAMBB2.max_y
       #    F: E & G
       #    G: AAMBB1.min_x < AAMBB2.max_x
       #    H: G & A

       #case 1: A, C or H
       if AAMBB1.max_y < AAMBB2.min_y:
           return True

       #case 2: B, C or D
       if AAMBB1.max_x < AAMBB2.min_x:
           return True

       #case 3: D, E or F
       if AAMBB1.min_y > AAMBB2.max_y:
           return True

       #case 4: F, G or H
       if AAMBB1.min_x > AAMBB2.max_x:
           return True

       return False

class AAMBoundingBox2(Polygon2D):
    __WKT_IDENTITY__ = Polygon2D.__WKT_IDENTITY__
    __WKT_N_BRACKET__ = Polygon2D.__WKT_N_BRACKET__
    __REPR_IDENTITY__ = Polygon2D.__REPR_IDENTITY__

    def __init__(self, to_enclose):

        if not isinstance(to_enclose, list):
            to_enclose = [to_enclose]

        xlist = [p.x for p in to_enclose]
        ylist = [p.y for p in to_enclose]

        corners = [min(xlist), min(ylist),
                   min(xlist), max(ylist),
                   max(xlist), max(ylist),
                   max(xlist), min(ylist)]

        super().__init__(points=points)
        self.is_hole = False
        self.is_self_crossing = None

    def from_limits(xlim, ylim):
        AAMBB = AAMBoundingBox2([[xlim[0], ylim[0]],
                                 [xlim[1], ylim[0]],
                                 [xlim[1], ylim[1]],
                                 [xlim[0], ylim[1]]
                                ],
                               )
        return AAMBB

    def add_elements(*to_enclose, inplace=False):
        if not isinstance(to_enclose, list):
            to_enclose = [to_enclose]

        xlist = [p.x for p in to_enclose] + [p.x for p in self.points]
        ylist = [p.y for p in to_enclose] + [p.y for p in self.points]

        corners = [min(xlist), min(ylist),
                   min(xlist), max(ylist),
                   max(xlist), max(ylist),
                   max(xlist), min(ylist)]

        AAMBB = AAMBoundingBox2(corners)

        if not inplace:
            return AAMBB

        self.__dict__ = AAMBB.__dict__.copy()


##############
### Shapely interactions
##############
def shapely_polygon_to_vectors(polygon, as_geometry_object=False):
    if isinstance(polygon, shapely.coords.CoordinateSequence):
        points = polygon[:]
    elif isinstance(polygon, shapely.geometry.Polygon):
        points = polygon.exterior.coords

    if points[0] != points[-1]:
        points.append(points[0])

    if as_geometry_object:
        return [Vector2D(points[i-1], points[i]) for i in range(1, len(points))]

    return [shapely.geometry.LineString(coordinates=(points[i-1], points[i])) for i in range(1, len(points))]

def add_shapely_vectors(*lists):
    out = lists[0]
    for sublist in lists[1:]:
        for vect in sublist:
            if vect not in out:
                out.append(vect)
    return out

def construct_triangles_from_vectors(interior_vectors, exterior_vectors):

    vectors = interior_vectors + exterior_vectors
    data = pandas.DataFrame.from_dict({'vectors':vectors,
                                       'p1':[v.p1 for v in vectors],
                                       'p2':[v.p2 for v in vectors],
                                       'n_triangles':[0 for v in vectors],
                                       'type': ['i' for e in interior_vectors] +
                                               ['e' for e in exterior_vectors],
                                       })

    triangles=[]
     #a variable storing problems for later fix (explained below)
    temp_pass=[]

    pass_again = True
    while pass_again:

        #remove from the search for the next candidate any vector that has
        #already reached its target or that was stored in the temp_pass
        #variable
        availaibles = data[(((data.type == 'i') & (data.n_triangles < 2))  |
                            ((data.type == 'e') & (data.n_triangles < 1))) &
                            ~(data.index.isin(temp_pass))
                            ]

        #sort the triangles by least found and pick the lowest one to work on
        #starting with the exterior vectors so we can fill missing inside mesh
        #items without causing too much trouble later on
        current = availaibles.sort_values(by=['n_triangles', 'type']).iloc[0]

        #get all vectors that are not our current vector,
        #                that have a point in comon with our current vector,
        #                and that didn't reach the max allocated triangles
        cand_p1 = data[(data.index != current.name) &
                       ((data.p1 == current.p1) | (data.p2 == current.p1)) &
                       ( ((data.type == 'i') & (data.n_triangles < 2)) |
                         ((data.type == 'e') & (data.n_triangles < 1)) )
                       ].copy()

        cand_p2 = data[(data.index != current.name) &
                       ((data.p1 == current.p2) | (data.p2 == current.p2)) &
                       ( ((data.type == 'i') & (data.n_triangles < 2)) |
                         ((data.type == 'e') & (data.n_triangles < 1)) )
                       ].copy()

        #create a column containing the far point. We'll join on that afterwards
        #print(current.name, len(cand_p1), len(cand_p2))
        #if len(cand_p1) == 0 or len(cand_p2) == 0:
        #    return data
        cand_p1['search_point'] = cand_p1.apply(lambda x: x.p1 if x.p1 != current.p1
                                                               else x.p2, axis=1)
        cand_p2['search_point'] = cand_p2.apply(lambda x: x.p1 if x.p1 != current.p2
                                                               else x.p2, axis=1)

        unioned = cand_p1.join(cand_p2.assign(index_cand_2=cand_p2.index)
                                      .set_index('search_point'),
                               on='search_point', how='inner',
                               lsuffix='_cand_p1', rsuffix='_cand_p2'
                               )

        if len(unioned) == 0:
            #if we get here, it means theres a missing link in the mesh:
            #    ie: len(interior_vectors) + len(exterior_vectors) is not
            #        a multiple of 3
            #    But we can't really use that check to detect it since it
            #    wouldn't register if we are missing, say, 3 links
            #
            #to fix this, we need to add a vector to the mix, and since we just
            #discovered which points are not linked by anything, we'll add it
            #on the fly.
            #
            #One this is important though: we would like to not cross other
            #vectors to avoid creating a blind spot
            #
            #The good news is that there's most likely another vector that will
            #break later on. We can save the number and wait for the other
            #problem to occur.
            if len(temp_pass) == 0:
                temp_pass.append(current.name)
                continue

            #if we got here, then we have *at least* two problematic vectors
            #to work with. Now, nothing tells us that they should be linked
            #together, bu we can try to see if it's the case.

            #Theres two possibilities:
            #  1) they share a common point but the third link of the triangle
            #     is missing. This one is rather easy: we can just force the
            #     new vector between the other two points
            #
            #  2) they float apart from each other and we can create a
            #     quadrilateral that we can then divide along one of it's
            #     diagonal to create two triangles.
            #
            #Let's try this, shall we?
            flag_success_match = False
            for i in range(len(temp_pass)):
                #create the candidates for the saved vector
                other_prob = data.loc[temp_pass[i]]



                #now let's see if we got a hit.
                #
                # 1) do our vectors share a point? And if yes, did they find each other?
                if other_prob.p1 in [current.p1, current.p2] or \
                   other_prob.p2 in [current.p1, current.p2]:
                   flag_success_match = True
                   break

                # 2 ) other, can we find two vectors in common?
                cand_p3 = data[(data.index != other_prob.name) &
                               ((data.p1 == other_prob.p1) | (data.p2 == other_prob.p1)) &
                               ( ((data.type == 'i') & (data.n_triangles < 2)) |
                                 ((data.type == 'e') & (data.n_triangles < 1)) )
                               ].copy()

                cand_p4 = data[(data.index != other_prob.name) &
                               ((data.p1 == other_prob.p2) | (data.p2 == other_prob.p2)) &
                               ( ((data.type == 'i') & (data.n_triangles < 2)) |
                                 ((data.type == 'e') & (data.n_triangles < 1)) )
                               ].copy()

                cand_p3['search_point'] = cand_p3.apply(lambda x: x.p1 if x.p1 != other_prob.p1
                                                               else x.p2, axis=1)
                cand_p4['search_point'] = cand_p4.apply(lambda x: x.p1 if x.p1 != other_prob.p2
                                                               else x.p2, axis=1)
                #now, merge the candidates
                cand_p1['origin'] = 'current.p1'
                cand_p2['origin'] = 'current.p2'
                cand_p3['origin'] = 'other_prob.p1'
                cand_p4['origin'] = 'other_prob.p2'

                current_cands = pandas.concat([cand_p1, cand_p2])
                other_cands = pandas.concat([cand_p3, cand_p4])

                in_both = other_cands[other_cands.index.isin(current_cands.index.tolist())].index.tolist()

                #oh... and make sure we actually found different vectors (set)
                if len(set(in_both)) >= 2:
                    flag_success_match = True
                    break

            #we can get here either with the break above or because the loop
            #was emptied. In the first case, we want to continue on with the
            #matching process. Otherwise we break from this part altogether
            if not flag_success_match:
                temp_pass.append(current.name)
                continue

            #if we get all the way here, we got a hit and can treat it
            # first case is easiest so we start with it
            if other_prob.p1 in [current.p1, current.p2] or \
               other_prob.p2 in [current.p1, current.p2]:
                #the triangle is easy to form, but we'll need the vector to
                #be pushed in the dataset because we are only using one pass
                #on it (and we are are going to make sure that it's located
                #inside the form)
                odd_one_o = other_prob.p1 if other_prob.p1 not in [current.p1, current.p2]\
                                        else other_prob.p2
                odd_one_c = current.p1 if current.p1 not in [other_prob.p1, other_prob.p2]\
                                        else current.p2
                #save the triangle
                triangles.append([current.p1,
                                  current.p2,
                                  odd_one_o
                                  ])
                #increment the use of our two vectors
                data.at[current.name,    'n_triangles'] += 1
                data.at[other_prob.name, 'n_triangles'] += 1

                #save this new vector
                new_vector = Vector2D(odd_one_o, odd_one_c)
                data = data.append(pandas.DataFrame([[new_vector, new_vector.p1, new_vector.p2, 1, 'i']],
                                                    columns=['vectors','p1','p2','n_triangles','type']
                                                    ),
                                   ignore_index=True
                                   )
                #print(f'   --> {current.name} {other_prob.name} {data.index.max()+1}     (creation of a new vector)')

            #In case two, we build a quadrilateral using the vectors we got
            else:
                #pick one to go with current and the other one to go with
                #other_prob

                #these should both be true:
                if not (current_cands.loc[in_both[0]].search_point in \
                                    other_prob[['p1', 'p2']].tolist()
                        and
                        other_cands.loc[in_both[1]].search_point in \
                                        current[['p1', 'p2']].tolist()
                        ):
                    temp_pass.append(current.name)
                    continue

                #the triangles are formed using:
                #    1)  current point:
                triangles.append([current.p1,
                                  current.p2,
                                  current_cands.loc[in_both[0]].search_point
                                  ])

                #    2)  other problematic point:
                triangles.append([other_prob.p1,
                                  other_prob.p2,
                                  other_cands.loc[in_both[1]].search_point
                                  ])

                #and the missing vector links combinaison of
                #   current.p1, other_prob.p1
                #   current.p1, other_prob.p2
                #   current.p2, other_prob.p1
                #   current.p2, other_prob.p2
                #
                #but since we automatically use both passes on it, it's kinda
                #useless to search which of the 2 combinaison are actually valid
                #(it only depends on the directions of both vectors) and to
                #arbitrairily pick one of the two.

                #print(f'   --> {current.name} {other_prob.name} FV     (new maxed out vector)')

                #update the 4 used vectors that we passed them at least once each
                data.at[current.name,    'n_triangles'] += 1
                data.at[other_prob.name, 'n_triangles'] += 1
                data.at[in_both[0],      'n_triangles'] += 1
                data.at[in_both[1],      'n_triangles'] += 1

            #finally, purge the problematic vectors from the list
            if other_prob.name in temp_pass:
                temp_pass.remove(other_prob.name)
            if current.name in temp_pass:
                temp_pass.remove(current.name)

        else:
            #now each line should be a match
            #   --> we are expecting 1 if current.type == 'i'
            #   --> we are expecting 2 if current.type == 'e'

            for i, row in unioned.iterrows():
                #assign them
                data.at[row.name, 'n_triangles'] += 1
                data.at[row.index_cand_2, 'n_triangles'] += 1
                data.at[current.name, 'n_triangles'] += 1
                #keep track of the triangle by it's 3 unique points
                #print(f'   --> {current.name} {row.name} {row.index_cand_2}')
                triangles.append(set([current.p1, current.p2,
                                      row.p1_cand_p1, row.p2_cand_p1,
                                      row.p1_cand_p2, row.p2_cand_p2
                                      ]))

                if row.name in temp_pass:
                    temp_pass.remove(row.name)
                if row.index_cand_2 in temp_pass:
                    temp_pass.remove(row.index_cand_2)

        #check if some vectors did not reach their target yet.
        #If none pop out, then we can break the cycle
        if len(data[(((data.type == 'i') & (data.n_triangles < 2))  |
                     ((data.type == 'e') & (data.n_triangles < 1))) &
                    ~(data.index.isin(temp_pass))
                    ]) == 0:

            #before ending the loop, make sure we treated every problematic
            #ones
            if len(temp_pass) == 0:
                pass_again = False

            else:
                print('got to the end with {}'.format(current.name))
                points = [p for p in set(data.p1.tolist() + data.p2.tolist())]
                triangles = [[points.index(p) for p in tri] for tri in triangles]
                x_list, y_list = np.array([p.as_tuple() for p in points]).transpose()
                return {'int_v':interior_vectors,
                        'ext_v':exterior_vectors,
                        'data':data,
                        'temp_pass':temp_pass,
                        'triangles':triangles,
                        'x_list':x_list,
                        'y_list':y_list
                        }

    #extract the unique list of points as x and y lists
    points = [p for p in set(data.p1.tolist() + data.p2.tolist())]
    #our triangle list is defined a list of raw points, but we actually need
    #pointers to the point list to be able to use mayavi.triangular_mesh
    triangles = [[points.index(p) for p in tri] for tri in triangles]
    #and the point has to be passed with two lists of raw coordinates: x and y
    x_list, y_list = np.array([p.as_tuple() for p in points]).transpose()

    return list(x_list), list(y_list), triangles

def build_triangles2D_from_point_lists(xlist, ylist, triangles):

    out=[]
    for vertex_1, vertex_2, vertex_3 in triangles:
        out.append(Polygon2D.from_tuples([(xlist[vertex_1], ylist[vertex_1]),
                                          (xlist[vertex_2], ylist[vertex_2]),
                                          (xlist[vertex_3], ylist[vertex_3])
                                          ])
                   )
    return out

def triangularize_shapely_polygon2D(polygon):
    #get all the legit edges
    edges = shapely.ops.triangulate(polygon, edges=True)
    #flush what is not contained in the polygon, since the triangulate function
    #also creates surfaces on top of holes and concave sections.
    #We'll convert them to geometry.Vector2D objects at the same time since
    #it's way easier to work with those elements
    int_mesh_parts = [Vector2D.from_shapely(edge) for edge in edges \
                      if polygon.contains(edge)]
    #The cleaning operation we just performed is getting rid of too much stuff
    #because of the definition of included/excluded rays plus some triangles
    #weirdly covering the holes in some circompstances. To convert for those
    #losses we need to query the contours
    ext_borders = shapely_polygon_to_vectors(polygon)
    int_borders = [shapely_polygon_to_vectors(interior.coords) for interior in polygon.interiors]

    #transform those vectors to geometry.Vector2D and add them in the same list
    #of external mesh objects
    ext_mesh_parts = [Vector2D.from_shapely(vect) for vect in \
                      add_shapely_vectors(ext_borders, *int_borders)]

    return construct_triangles_from_vectors(int_mesh_parts, ext_mesh_parts)

def furness(targetH, targetV, array=None, tol=0.01, maxIter=1000, silence_warning=False):

    dimH = len(targetH)
    dimV = len(targetV)

    targetH = np.asarray(targetH)
    targetV = np.asarray(targetV)

    if not silence_warning:
        if targetH.sum() != targetV.sum():
            warnings.warn("The sum of H and V don't match, the solution "+
                          "cannot reach an optimum")

    #generate a starting point if none is provided
    if array is None:
        #create a dummy array
        array = np.ones((dimV, dimH))
        #make sure we don't have any 0 to cover
        indH = np.nonzero(targetH==0)[0]
        for i in indH:
            array[:,i] = np.zeros(dimV) #we set the column in front of the line
        indV = np.nonzero(targetV==0)[0]
        for j in indV:
            array[j,:] = np.zeros(dimH) #we set the line in front of the column
        #make sure the total fits
        array = array * targetH.sum() / array.sum()

    targetH = targetH.reshape((dimH, 1))
    targetV = targetV.reshape((1, dimV))

    new = array
    newH = new.sum(axis=0).reshape((dimH, 1))
    newV = new.sum(axis=1).reshape((1, dimV))

    count=0
    while True:
        #we might have some a/0 or 0/0 divisions, that's why we do the
        #np.nan_to_num operation, we don,t need the warning so we'll silence it
        with np.errstate(divide='ignore', invalid='ignore'):
            xH = (targetH/newH).reshape((1,dimH))
        new = new*np.nan_to_num(xH, 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            xV = (targetV/newV).reshape((dimV,1))
        new = new*np.nan_to_num(xV, 0)

        newH = new.sum(axis=0).reshape((dimH, 1))
        newV = new.sum(axis=1).reshape((1, dimV))

        with np.errstate(divide='ignore', invalid='ignore'):
            convergence = max(np.nan_to_num(abs(targetV-newV)/newV,0).max(),
                              np.nan_to_num(abs(targetH-newH)/newH,0).max()
                              )

        count += 1
        if convergence < tol:
            break
        elif count > maxIter:
            break
        else:
            continue
    return new

if __name__ == '__main__':
    """
    fig, axes = plt.subplots(3,2)

    polys = [[Polyline2D.from_xylists([3, 5, 3], [2, 2, 0]),
              Polyline2D.from_xylists([3, 5, 3], [2, 2, 2]),
              Polyline2D.from_xylists([3, 5, 3], [2, 2, 4])
              ],
             [Polyline2D.from_xylists([3, 5, 7], [2, 2, 0]),
              Polyline2D.from_xylists([3, 5, 7], [2, 2, 2]),
              Polyline2D.from_xylists([3, 5, 7], [2, 2, 4])
             ]
            ]


    for j in range(len(polys)):
        for i in range(len(polys[j])):

            poly = polys[j][i]
            poly.plot(ax=axes[i][j])

            for x in range(30, 71):
                for y in range(15, 26):
                    point = Point(x/10, y/10)
                    side = poly.left_or_right(point)

                    if side == 0:
                        color = 'g'
                    elif side == 1:
                        color = 'r'
                    elif side == -1:
                        color = 'k'
                    elif side is None:
                        color = 'b'

                    point.plot(ax=axes[i][j], color=color)
    """
    fig, ax = plt.subplots(1)
    turn = TurningArrow(Point(0.1, 0.1), arrow_angle=0, height=0.1, width=0.1, rotate_bbox=True, bbox_rotation_angle=0, lock_on='data', arrow_scale=100)
    rot = turn.rotate(rotation_angle=45)
    rot.addText('N', 0, -0.2, 0.1, 0.1)
    rot.annotate(ax, showbbox=True)

    #p1, vx, vy = rot._addText('L', -0.2, 0, 0.1, 0.1)