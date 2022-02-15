import copy
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiLineString

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import mlines
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as cx

day_french_to_eng = {
    'lun': 'mo',
    'mar': 'tu',
    'mer': 'we',
    'jeu': 'th',
    'ven': 'fr',
    'sam': 'sa',
    'dim': 'su'
}

def angle(linestring):
    if isinstance(linestring, MultiLineString):
        linestring = linestring[0]
    seg = np.array(linestring)
    seg = seg[1] - seg[0]
    return np.angle(complex(*(seg)), deg=True)

class CurbLRHelper(object):
    
    def __init__(self, geobase, columns_id='ID_TRC', crs='EPSG:3799'):
        self.geobase = geobase.to_crs(crs)
        self.geobase_id = columns_id
        self.crs = crs
    
    def curb_to_dataframe(self, filepath, offset=2, user_class_color_offest=0):
            
        curbLR = gpd.read_file(filepath)

        return self.fill_geodataframe(curbLR, offset=offset, user_class_color_offest=user_class_color_offest)
    
    def fill_geodataframe(self, gdf, offset=2, user_class_color_offest=0):
        def try_get_classes(row):
            try: 
                return eval(row)[0]['userClasses'][0]['classes'][0] 
            except (IndexError, KeyError): 
                return "everyone"

        def try_get_subclasses(row):
            try: 
                return eval(row)[0]['userClasses'][0]['subclasses'][0] 
            except (IndexError, KeyError): 
                return "everyone"

            
        def try_get_day_of_week(row):
            try:
                return eval(row)[0]['timeSpans'][0]['daysOfWeek']['days']
            except (KeyError, IndexError):
                return "always"
            
        def try_get_time_of_day(row):
            try:
                return eval(row)[0]['timeSpans'][0]['timesOfDay']
            except (KeyError, IndexError):
                return "always"
            
        def set_offset(row, offset):
            try:
                return row['geometry'].parallel_offset(offset+row['offset']*2, row['sideOfStreet'])
            except (ValueError):
                print(row['geometry'].__str__()+" failed to offset")
                return row['geometry']
            
        curbLR = gdf.copy()

        curbLR = curbLR.to_crs(self.crs)
        curbLR['shstRefId'] = curbLR.location.apply(lambda x: x['shstRefId'])
        #curbLR['id_trc'] = curbLR.location.apply(lambda x: int(x['idtrc']))
        curbLR['shstLocationStart'] = curbLR.location.apply(lambda x: x['shstLocationStart'])
        curbLR['shstLocationEnd'] = curbLR.location.apply(lambda x: x['shstLocationEnd'])
        curbLR['sideOfStreet'] = curbLR.location.apply(lambda x: x['sideOfStreet'])
        curbLR['assetType'] = curbLR.location.apply(lambda x: x['assetType'])
        curbLR['userClass'] = curbLR.regulations.apply(try_get_classes)
        curbLR['userSubClass'] = curbLR.regulations.apply(try_get_subclasses)        
        curbLR['activity'] = curbLR.regulations.apply(lambda x: eval(x)[0]['rule']['activity'])
        curbLR['priorityCategory'] = curbLR.regulations.apply(lambda x: eval(x)[0]['rule']['priorityCategory'])
        curbLR['day_of_week'] = curbLR.regulations.apply(try_get_day_of_week)
        curbLR['timeOfDay'] = curbLR.regulations.apply(try_get_time_of_day)

        userClassColor = pd.DataFrame(curbLR.userClass.unique(), columns=['userClass']).reset_index().rename(columns={'index':'userClassColor'})
        userClassColor['userClassColor'] += user_class_color_offest
        curbLR = curbLR.merge(userClassColor, on='userClass')
        
        curbLR['offset'] = 0
        grouped = curbLR.groupby(['shstRefId', 'sideOfStreet'])
        data = []
        for _, segment_regul in grouped:
            segment_regul['offset'] = np.arange(0, segment_regul.shape[0])
            data.append(segment_regul)
        curbLR = pd.concat(data).reset_index(drop=True)
        
        curbLR['offset_geom'] = curbLR.apply(lambda x: set_offset(x, offset), axis=1)
    
        return curbLR
    
    
    def plot(self, curbLR, geobase=False, foncier=False, nammed_buffer=50, **kwargs):
        
        fig, ax = plt.subplots(figsize=(15,15), dpi=150)
    
        if foncier:
            foncier = gpd.read_file('data/opendata/uniteevaluationfonciere.geojson')
            foncier_clip = gpd.clip(foncier.to_crs('epsg:3799'), get_enveloppe_gdf(gpd.GeoDataFrame(curbLR, geometry='offset_geom', crs=self.crs)))
            foncier = foncier.to_crs('epsg:3857')
        
        c = curbLR.userClassColor.unique()
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.tab10)
        cmap.set_array([])

        lines_kw = []
        for index, row in curbLR[['userClass', 'userClassColor']].drop_duplicates().iterrows():
            if row['userClass'] == '':
                lines_kw.append({'color':cmap.to_rgba(row["userClassColor"]), 'label': '(?)' , 'alpha':1})
                continue
            lines_kw.append({'color':cmap.to_rgba(row["userClassColor"]), 'label':row['userClass'], 'alpha':1})
            
        # plot curb data
        gpd.GeoDataFrame(curbLR, geometry='offset_geom', crs='EPSG:3799').to_crs('epsg:3857').reset_index().plot(column='userClassColor', ax=ax, zorder=3, cmap='tab10', linewidth=1.5);
        
        # Road segments or Buildings
        if foncier:
            foncier_clip.plot(color='lightgray', ax=ax, zorder=1, alpha=.5);
        elif geobase:
            # Road names
            geobase_clip = gpd.clip(self.geobase, get_enveloppe_gdf(gpd.GeoDataFrame(curbLR, geometry='offset_geom', crs=self.crs)))
            geobase_clip = geobase_clip.to_crs('epsg:3857')
            geobase_clip.loc[
                pd.concat([
                    #gpd.clip(geobase_clip, get_enveloppe_bbox(geobase_clip.total_bounds - nammed_buffer, crs=geobase_clip.crs))[['NOM_VOIE']].drop_duplicates(keep='first'),
                    gpd.clip(geobase_clip, get_enveloppe_bbox(geobase_clip.total_bounds - nammed_buffer, crs=geobase_clip.crs))[['NOM_VOIE']].drop_duplicates(keep='last')
                ]).index
            ].apply(lambda x: ax.annotate(text=x.NOM_VOIE, xy=x.geometry.centroid.coords[0], ha='center', rotation=angle(x.geometry), annotation_clip=True, alpha=0.7),axis=1);
            geobase_clip.plot(color='gray', ax=ax, zorder=2, alpha=.5);
    
        else:
            # add background
            cx.add_basemap(plt.gca(), source=cx.providers.OpenStreetMap.Mapnik)

        # Legend     
        _build_legend(ax=ax, lignes_kwords=lines_kw, **kwargs)

        bbox=gpd.GeoDataFrame(curbLR, geometry='offset_geom', crs='EPSG:3799').to_crs('epsg:3857').total_bounds
        
        
        ax.set_xlim(xmin=bbox[0], xmax=bbox[2])
        ax.set_ylim(ymin=bbox[1], ymax=bbox[3])

        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig, ax
    
    def correct_curblr_format(self, data_curb):
        '''Prend le fichier json CurbLR générer par Jakarto et le convertit aux normes CurbLR
        data_curb (dict): Dictionnaire contenant le curbLR généré par Jakarto 
        '''
        new_curb_lr = {}

        #manifest
        new_curb_lr['manifest'] = data_curb['manifest']['manifest']
        new_curb_lr['type'] = data_curb['CurbLR']['type']
        new_curb_lr['features'] = []

        for feature in data_curb['CurbLR']['features']:
            new_features = {}
            new_features['type'] = feature['type']

            # get the properties in correct format
            new_props = {}
            # location
            new_props['location'] = {}
            new_props['location']['shstRefId'] = feature['properties']['location']['shstRefId']
            new_props['location']['shstLocationStart'] = float(feature['properties']['location']['shstLocationStart'])
            new_props['location']['shstLocationEnd'] = float(feature['properties']['location']['shstLocationEnd'])
            new_props['location']['sideOfStreet'] = feature['properties']['location']['sideOfStreet'].lower()
            new_props['location']['assetType'] = feature['properties']['location']['assetType'].lower()
            # regulations
            new_props['regulations'] = []
            for old_reg in feature['properties']['regulations']:
                # rules
                rules = {}
                rules['activity'] = 'no parking' # default
                rules['priorityCategory'] = 'no parking' # default
                for key, value in old_reg['rule'].items():
                    if old_reg['rule'][key]:
                        try:
                            rules[key] = value.lower()
                        except AttributeError:
                            #it is int
                            rules[key] = value
                # userClasses (optional)
                user_classes = []
                for old_user_classes in old_reg['userClasses']:
                    classes = {}
                    for key, value in old_user_classes.items():
                        if value and value != ['']:
                            classes[key] = value
                    if classes:
                        user_classes.append(classes)
                if not user_classes:
                    user_classes = [{}]
                # timespan (optional)
                timespans = []
                for old_timespans in old_reg['timeSpans']:
                    # effectivesDates
                    timespan_iter = {}
                    try:
                        effectives_dates = []
                        for old_eff_dates in old_timespans['effectiveDates']:
                            eff_dates = {}
                            for key, value in old_eff_dates.items():
                                if old_eff_dates[key] and key in ['from', 'to']:
                                    eff_dates[key] = f"{value[3:]}-{value[:2]}"
                            if eff_dates:
                                effectives_dates.append(eff_dates)
                        if effectives_dates:
                            timespan_iter['effectiveDates'] = effectives_dates
                    except KeyError:
                        pass
                    # DayOfWeek
                    try:
                        dayofweek = {}
                        for key, value in old_timespans['daysOfWeek'].items():
                            if value and value != ['']:
                                val_eng = [day_french_to_eng[x] for x in value]
                                dayofweek[key] = value
                        if dayofweek:
                            timespan_iter['daysOfWeek'] = dayofweek
                    except KeyError:
                        pass
                    # timeOfDay
                    try:
                        timeofday = []
                        for old_tod in old_timespans['timesOfDay']:
                            tod_object = {}
                            for key, value in old_tod.items():
                                if value and key in ['from', 'to']:
                                    tod_object[key] = value
                            if tod_object:
                                timeofday.append(tod_object)
                        if timeofday:
                            timespan_iter['timesOfDay'] = timeofday
                    except KeyError:
                        pass
                    if timespan_iter:
                        timespans.append(timespan_iter)
                # update regulations
                new_props['regulations'].append({'rule': rules, 'userClasses': user_classes, 'timeSpans': timespans})
            # update features
            new_features['properties'] = new_props
            # update geometries
            new_features['geometry'] = feature['geometry']
            # append to NEW CURBLR DATA
            new_curb_lr['features'].append(new_features)
        return new_curb_lr
    
    

    def add_fake_srr(self, data_curb, fake_reg, user_class):
        ''' Ajoute une régulation SRR à la première zone de {user_class} rencontrée dans les données CurbLR {data_curb}
        '''
        new_data_curb = copy.deepcopy(data_curb)

        i = 0
        j = 0
        for feature in new_data_curb['features']:
            i+=1
            try: 
                if feature['properties']['regulations'][0]['userClasses'][0]['classes'][0] == user_class:
                    feature['properties']['regulations'].append(fake_reg)
                    break
            except (KeyError, AttributeError):
                j+=1
                pass
            
        return new_data_curb
    
    
    def change_class(self, data_curb, old_val, new_val):
        
        new_data_curb = copy.deepcopy(data_curb)
        i = 0
        for feature in new_data_curb['features']:
            j = 0
            for regulation in feature['properties']['regulations']:
                k = 0
                for user_class in regulation['userClasses']:
                    l = 0
                    for class_ in user_class['classes']:
                        if class_ == old_val:
                            new_data_curb['features'][i]['properties']['regulations'][j]['userClasses'][k]['classes'][l] = new_val
                            return new_data_curb
                        l+=1
                    k+=1
                j+=1
            i+=1
        
        print('Nothing changed')
        return new_data_curb
    
    def add_regulation(self, curb_data, road_id, start_m, end_m, regulation):
    
        new_curb_data = copy.deepcopy(curb_data)
        feature = {}
        
        # compute start and end points based on distance to road
        point_start = list(list(self.geobase.loc[self.geobase[self.geobase_id] == road_id].interpolate(start_m).to_crs('EPSG:4326').iloc[0].coords)[0])
        point_end = list(list(self.geobase.loc[self.geobase[self.geobase_id] == road_id].interpolate(end_m).to_crs('EPSG:4326').iloc[0].coords)[0])
        geometry = {'type': 'LineString', 'coordinates':[point_start, point_end]}
        
        # fill location
        location = {}
        location['shstRef'] = str(road_id)
        location['shstLocationStart'] = start_m
        location['shstLocationEnd'] = end_m
        location['sideOfStreet'] = 'left'
        location['assetType'] = 'sign'

        # fill geometry
        feature['geometry'] = geometry

        feature['type']="Feature"
        feature['properties'] = {'location': location}
        feature['properties'].update(regulation)
        
        new_curb_data['features'].append(feature)
        return new_curb_data

def get_enveloppe_gdf(gdf):
    bbox = gdf.total_bounds

    p1 = Point(bbox[0], bbox[3])
    p2 = Point(bbox[2], bbox[3])
    p3 = Point(bbox[2], bbox[1])
    p4 = Point(bbox[0], bbox[1])

    np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
    np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
    np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
    np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

    bb_polygon = Polygon([np1, np2, np3, np4])

    df2 = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=['geometry'], crs=gdf.crs)
    
    return df2

def get_enveloppe_bbox(bbox, crs):
    p1 = Point(bbox[0], bbox[3])
    p2 = Point(bbox[2], bbox[3])
    p3 = Point(bbox[2], bbox[1])
    p4 = Point(bbox[0], bbox[1])

    np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
    np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
    np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
    np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

    bb_polygon = Polygon([np1, np2, np3, np4])

    df2 = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=['geometry'], crs=crs)
    return df2

def _make_legend_line(ldict):
    #pop those so we don't conflict with the dict unpacking
    color=ldict.pop('color', None)
    label=ldict.pop('label', None)
    #create the patch
    return mlines.Line2D([], [], color=color, label=label, **ldict)
        
def _make_legend_point(pdict):
    #pop those so we don't conflict with the dict unpacking
    markerfacecolor=pdict.pop('markerfacecolor', None)
    markeredgecolor=pdict.pop('markeredgecolor', None)
    color=pdict.pop('color', None)
    label=pdict.pop('label', None)
    marker=pdict.pop('marker', '.')
    #decide which color to use. If both are None then screw the user
    mcolor = markerfacecolor if markerfacecolor is not None else color
    ecolor = markeredgecolor if markeredgecolor is not None else color
    #create the patch
    return mlines.Line2D([], [], color=None, markerfacecolor=mcolor,
                         markeredgecolor=ecolor, linewidth=0, marker=marker, 
                         label=label, **pdict)
def _make_legend_boxpatch(kdict):
    #pop those so we don't conflict with the dict unpacking
    color=kdict.pop('color', None)
    label=kdict.pop('label', None)
    boxstyle=kdict.pop('boxstyle', mpatches.BoxStyle("Round", pad=0.02))
    #create the patch
    return mpatches.FancyBboxPatch([0,0], 0.1, 0.1, color=color,
                                   label=label, boxstyle=boxstyle,
                                   **kdict)
def _build_legend(ax, title='Légende', bbox_to_anchor=(1.05, 1), loc='upper left', 
                  lignes_kwords=[], points_kwords=[], kdes_kwords=[],
                  label_order=[], align_title='left', **kwargs
                  ):
    """Build a legend to be attached to a map figure.
    
    Parameters
    ----------
    ax : matplotlib.axes
        The axe to attach the legend to.
        
    title : str, optional
        The tile of the legend box.
        
        Default : 'Légende'
    
    bbox_to_anchor : tuple of floats, optional
        Box that is used to position the legend in conjunction with loc. See
        matplotlib.pyplot.legend for more informations.
    
        Default : (1.05, 1)
        
    loc : str, optional
        Location of the legend. See matplotlib.pyplot.legend for more informations.
    
        Default : 'upper left'
        
    lignes_kwords : list of dicts, optional
        A list containing a dictionnary for every line entry to add to the legend.
        The dictionnary themselves must contain keywords compatible with 
        matplotlib.lines.Line2D. Be sure to provide at least values for 'color'
        and 'label' otherwise the legend will feel really empty.
        
        Default : []
    
    points_kwords : list of dicts, optional
        A list containing a dictionnary for every point entry to add to the legend.
        The dictionnary themselves must contain keywords compatible with 
        matplotlib.lines.Line2D. Be sure to provide at least values for 'color'
        and 'label' otherwise the legend will feel really empty.
        
        Default : []
    
    kdes_kwords : list of dicts, optional
        A list containing a dictionnary for every kde entry to add to the legend.
        The dictionnary themselves must contain keywords compatible with 
        matplotlib.lines.Line2D. Be sure to provide at least values for 'color'
        and 'label' otherwise the legend will feel really empty.
        
        Default : []
    
    label_order : list of str, optional
        List of labels that should be put on top of the legend. The oder of the
        list is respected. Labels not part of this list are added in the 
        dictionnaries's key order, starting with lines, then points and finally
        kdes.
    
        Default : []
    
    align_title : {'left', 'center', 'right'}, optional
        Force the alignement of the legend's title.
        
        Default : 'left'
    
    kwargs : dict, optional
        These parameters are passed to matplotlib.pyplot.legend
    
    Returns
    -------
    None
    
    Notes
    -----
    To merge multiple objects in the same legend, the keyword "multiple" can be
    used in either dictionnaries. The label is then used to group them as a
    single entity when building the legend patches. These objects don't need to
    be of the same type, tought combining them may require a certain order to
    give interesting results.
    
    """
    sorted_handles={}
    unsorted_handles=[]
    unsorted_handles_labels=[]
    memory={}
    #handle kde like patches
    for kdict in kdes_kwords:
        if kdict.pop("multiple", False):
            label=kdict.pop("label", None)
            if not label in memory.keys():
                memory[label] = []
            #create the patch with no label and save it to memory
            memory[label].append(_make_legend_boxpatch(kdict))
        
        else:
            #create the patch
            kde_patch = _make_legend_boxpatch(kdict)
            #add it to the correct list
            label = kde_patch.get_label()
            if label in label_order and label is not None:
                sorted_handles[label] = kde_patch
            else:
                unsorted_handles.append(kde_patch)
                unsorted_handles_labels.append(label)
    #handle lines
    for ldict in lignes_kwords:
        if ldict.pop("multiple", False):
            label=ldict.pop("label", None)
            if not label in memory.keys():
                memory[label] = []
            #create the patch with no label and save it to memory
            memory[label].append(_make_legend_line(ldict))
            
        else:
            #create the patch
            ligne_patch = _make_legend_line(ldict)
            #add it to the correct list
            label = ligne_patch.get_label()
            if label in label_order and label is not None:
                sorted_handles[label] = ligne_patch
            else:
                unsorted_handles.append(ligne_patch)
                unsorted_handles_labels.append(label)
    
    #handle points
    for pdict in points_kwords:
        if pdict.pop("multiple", False):
            label=pdict.pop("label", None)
            if not label in memory.keys():
                memory[label] = []
            #create the patch with no label and save it to memory
            memory[label].append(_make_legend_point(pdict))
        
        else:
            #create the patch
            point_patch = _make_legend_point(pdict)
            #add it to the correct list
            label = point_patch.get_label()
        
            if label in label_order and label is not None:
                sorted_handles[label] = point_patch
            else:
                unsorted_handles.append(point_patch)
                unsorted_handles_labels.append(label)
    #handle memorized patches
    if len(memory.keys()) > 0:
        for key in memory.keys():
            if key in label_order and key is not None:
                sorted_handles[key] = tuple(memory[key])
            else:
                unsorted_handles.append(memory[key])
                unsorted_handles_labels.append(key)
            
    #sort the handles' order
    handles = [sorted_handles[key] for key in label_order] + unsorted_handles
    labels = label_order + unsorted_handles_labels
    #generate the legend
    leg = ax.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor,
                    fancybox=True, title=title,
                    handler_map={tuple: HandlerTuple(ndivide=None)}, **kwargs)
    
    leg._legend_box.align = align_title
