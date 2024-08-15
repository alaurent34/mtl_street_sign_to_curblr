""" Module containing geometric transformation for the signes
"""
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
import numpy.typing as npt
from shapely import (
    Point,
    LineString
)
from signe.tools.geom import (
    MONTREAL_CRS,
    DEFAULT_CRS,
    rtreenearest,
    vectorized_r_o_l,
    cut_line_at_dist
)

logger = logging.getLogger(__name__)


def interpolate(line: LineString, dist: float) -> Point:
    """_summary_

    Parameters
    ----------
    line : LineString
        _description_
    dist : float
        _description_

    Returns
    -------
    Point
        _description_
    """
    return line.interpolate(dist)


def project(line: LineString, point: Point) -> float:
    """_summary_

    Parameters
    ----------
    line : LineString
        _description_
    point : Point
        _description_

    Returns
    -------
    float
        _description_
    """
    return line.project(point)


v_interpolate = np.vectorize(interpolate)
v_project = np.vectorize(project)


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


def sort_chainage_with_roads(
    data_sign: pd.DataFrame,
    roads: gpd.GeoDataFrame,
    linear_ref_field: str,
    roads_id_col: str = 'ID_TRC',
    side_of_street_field: str = 'side_of_street',
    traffic_dir_field: str = 'SENS_CIR',
) -> gpd.GeoDataFrame:
    """_summary_

    Parameters
    ----------
    data_sign : pd.DataFrame
        _description_
    roads : gpd.GeoDataFrame
        _description_
    linear_ref_field: str
        Field that store the linear referencing values
    roads_id_col : str, optional
        _description_, by default 'ID_TRC'
    side_of_street_field : str, optional
        _description_, by default 'side_of_street'
    traffic_dir_field : str, optional
        _description_, by default 'SENS_CIR'

    Returns
    -------
    gpd.GeoDataFrame
        _description_
    """
    data_sign = data_sign.copy()
    data_sign = data_sign.reset_index()
    roads = roads.copy()

    roads = roads.set_index(roads_id_col)

    traffic = roads.loc[data_sign[roads_id_col],
                        traffic_dir_field].reset_index(drop=True)
    side_of_street = data_sign[side_of_street_field]

    mask = ((traffic == 0) & (side_of_street == -1) | (traffic == -1))

    length = roads.loc[
        data_sign.loc[mask, roads_id_col],
        roads.geometry.name
    ].length.reset_index(drop=True)
    length.index = data_sign.loc[mask].index

    data_sign.loc[mask, 'dist_on_roads_sorted'] = (
        length -
        data_sign.loc[mask, linear_ref_field]
    )
    data_sign.loc[~mask, 'dist_on_roads_sorted'] = (
        data_sign.loc[~mask, linear_ref_field]
    )

    return data_sign


def cut_linestring(
    line: LineString,
    start: float,
    end: float
) -> LineString:
    """_summary_

    Parameters
    ----------
    line : LineString
        _description_
    start : float
        _description_
    end : float
        _description_

    Returns
    -------
    LineString
        _description_
    """
    if start > end:
        return cut_line_at_dist(
            line=cut_line_at_dist(
                line=line,
                dist=end
            )[1],
            dist=start - end
        )[0]
        # return LineString([
        #     line.interpolate(end),
        #     line.interpolate(start)
        # ])
    return cut_line_at_dist(
        line=cut_line_at_dist(
            line=line,
            dist=start
        )[1],
        dist=end - start
    )[0]
    # return LineString([
    #     line.interpolate(start),
    #     line.interpolate(end)
    # ])


v_cut_linestring = np.vectorize(cut_linestring)


def create_linestring_from_lr(
    data_sign: pd.DataFrame,
    roads: gpd.GeoDataFrame,
    roads_id_col: str = 'ID_TRC',
    start_field: str = 'start',
    end_field: str = 'end',
    traffic_dir_field: str = 'SENS_CIR',
    side_of_street_field: str = 'side_of_street'
) -> gpd.GeoDataFrame:
    """_summary_

    Parameters
    ----------
    data_sign : pd.DataFrame
        _description_
    roads : gpd.GeoDataFrame
        _description_
    roads_id_col : str, optional
        _description_, by default 'ID_TRC'

    Returns
    -------
    gpd.GeoDataFrame
        _description_
    """
    data_sign = data_sign.copy()
    data_sign = data_sign.reset_index()
    roads = roads.copy()

    roads = roads.set_index(roads_id_col)

    traffic = roads.loc[data_sign[roads_id_col],
                        traffic_dir_field].reset_index(drop=True)
    side_of_street = data_sign[side_of_street_field]

    mask = ((traffic == 0) & (side_of_street == -1) | (traffic == -1))

    lines = roads.loc[data_sign['ID_TRC'],
                      roads.geometry.name].values

    data_sign = gpd.GeoDataFrame(
        data_sign,
        geometry=lines,
        crs=lines.crs
    )

    # fill inf value
    data_sign.loc[
        np.isinf(data_sign[end_field]),
        end_field
    ] = data_sign.loc[
        np.isinf(data_sign[end_field]),
        'geometry'
    ].length

    data_sign.loc[mask, start_field] = (
        data_sign.loc[mask, 'geometry'].length -
        data_sign.loc[mask, start_field]
    )
    data_sign.loc[mask, end_field] = (
        data_sign.loc[mask, 'geometry'].length -
        data_sign.loc[mask, end_field]
    )

    start = data_sign[start_field].values
    end = data_sign[end_field].values
    lines = data_sign.geometry.values

    return v_cut_linestring(lines, start, end)


def point_to_line(
    data: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    sequence_field: list[str],
    match_field: list[str],
    linear_ref_field: str,
    roads_id_col: str = 'ID_TRC',
    traffic_dir_field: str = 'SENS_CIR',
    side_of_street_field: str = 'side_of_street'
) -> gpd.GeoDataFrame:
    """ Allows a series of points to be snapped to the street and transformed
    into a street segment.
)
    Parameters
    ----------
    data: gpd.GeoDataFrame
        Point Geometry collection
    sequence_field: List[str]
        Specifies the name of the field containing point sequence info
        (1=start, 2=middle, 3=terminus)
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

    # reverse start_end when direction is reversed
    data = sort_chainage_with_roads(
        data_sign=data,
        roads=roads,
        roads_id_col=roads_id_col,
        traffic_dir_field=traffic_dir_field,
        side_of_street_field=side_of_street_field,
        linear_ref_field=linear_ref_field
    )

    match_field.insert(0, roads_id_col)
    groups = data.groupby(match_field)

    lines_df = []
    for match_field_g, g_data in groups:
        # if there is two time the same panel, drop one of them
        uniq_idx = g_data[
            ['dist_on_roads_sorted',
             sequence_field]
        ].drop_duplicates().index
        g_data = g_data.loc[uniq_idx].copy()

        side_of_street = g_data[side_of_street_field].unique()[0]

        lines = create_segments(
            points=g_data.geometry,
            order=g_data['dist_on_roads_sorted'],
            sequence=g_data[sequence_field]
        )
        lines = pd.DataFrame(lines, columns=['start', 'end'])
        lines[match_field] = match_field_g
        lines[side_of_street_field] = side_of_street
        lines_df.append(lines)

    lines_df = pd.concat(lines_df)
    lines_df = lines_df.reset_index(drop=True)

    # find geometry
    geometry = create_linestring_from_lr(
        data_sign=lines_df,
        roads=roads,
        roads_id_col=roads_id_col,
        start_field='start',
        end_field='end'
    )

    lines_df = gpd.GeoDataFrame(
        lines_df,
        geometry=geometry,
        crs=MONTREAL_CRS
    )
    lines_df['length'] = lines_df.geometry.length

    return lines_df


def buffered_point_to_ligne(
    data: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    meters: float,
    linear_ref_field: str,
    roads_id_col: str = 'ID_TRC'
) -> gpd.GeoDataFrame:
    """_summary_

    Parameters
    ----------
    data : gpd.GeoDataFrame
        _description_
    roads : gpd.GeoDataFrame
        _description_
    meters : float
        _description_
    linear_ref_field : str
        _description_
    roads_id_col : str, optional
        _description_, by default 'ID_TRC'

    Returns
    -------
    gpd.GeoDataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
    data = data.copy()
    roads = roads.copy()

    if roads_id_col not in data.columns:
        raise ValueError('roads_id_col should be in data.')
    if linear_ref_field not in data.columns:
        raise ValueError('linear_ref_field should be in data.')
    if roads_id_col not in roads.columns:
        raise ValueError('roads_id_col shoulf be in roads.')

    roads.rename_geometry('line_geom', inplace=True)

    data = data.join(
        other=roads.set_index(roads_id_col)[['line_geom']],
        on=roads_id_col
    )

    half_meter = meters / 2
    first_lr = (data[linear_ref_field] - half_meter).values
    last_lr = (data[linear_ref_field] + half_meter).values
    lines = data['lines_geom'].values

    first_point = v_interpolate(lines, first_lr)
    last_point = v_interpolate(lines, last_lr)

    lines_geom = [LineString((first, last)) for first, last
                  in zip(first_point, last_point)]

    data['lines_geom'] = lines_geom

    return data


def match_points_on_roads(
    points: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    id_cols: list[str] = None
) -> npt.NDArray:
    """ Make mapmatching of point on side of street using shortest distance.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        Points to match to the road network
    roads : gpd.GeoDataFrame
        Road network geometry
    id_cols : list[str], optinal
        Columns to return back, by default None

    Returns
    -------
    np.NDArray
        The data contained in the id_cols
    """

    if not id_cols:
        id_cols = roads.columns

    points = points.copy()
    points = rtreenearest(
        points,
        roads,
        id_cols
    )

    return points[id_cols].values


def infer_side_of_street(
    points: gpd.GeoDataFrame,
    roads_network: gpd.GeoDataFrame,
    roads_id_col: str = 'ID_TRC'
) -> list[float]:
    """_summary_

    Parameters
    ----------
    points : gpd.GeoDataFrame
        _description_
    roads_network : gpd.GeoDataFrame
        _description_
    roads_id_col : str, optional
        _description_, by default 'ID_TRC'

    Returns
    -------
    list[float]
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    points = points.copy()
    roads_network.copy()

    points = points.to_crs(DEFAULT_CRS)
    roads_network = roads_network.to_crs(DEFAULT_CRS)

    if roads_id_col not in points.columns:
        raise ValueError('roads_id_col should be in points.')
    if roads_id_col not in roads_network.columns:
        raise ValueError('roads_id_col shoulf be in roads_network.')

    roads_network = roads_network.set_index(roads_id_col)

    lines = roads_network.loc[
        points[roads_id_col],
        roads_network.geometry.name
    ]
    return vectorized_r_o_l(
        lines,
        points.geometry
    )


def compute_chainage(
    arrow_mapping: dict[int, dict[int, dict[int, int]]],
    traffic_dir: int,
    side_of_street: int,
    arrow: int
) -> int:
    """_summary_

    Parameters
    ----------
    arrow_mapping : dict[int, dict[int, dict[int, int]]]
        _description_
    traffic_dir : int
        _description_
    side_of_street : int
        _description_
    arrow : int
        _description_

    Returns
    -------
    int
        _description_
    """
    try:
        return arrow_mapping[traffic_dir][side_of_street][arrow]
    except KeyError as e:
        # Print the problematic row and continue
        logger.warning(
            "KeyError for traffic_dir: %s, side_of_street: %s, arrow: %s.",
            traffic_dir, side_of_street, arrow
        )
        logger.warning("The missing key is : %e", e)
        return None


v_chainage = np.vectorize(compute_chainage)
