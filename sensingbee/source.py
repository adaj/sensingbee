import time

import numpy as np
import pandas as pd

import geopandas as gpd
import fiona
import shapely

import sensingbee.utils


class Sensors(object):
    """
    Stores data and metadata about sensors. The methods implemented for
    this class is only to filter data on the spatial and temporal settings.

    Needs to be configured with a DATA_FOLDER, where all  information related
    to the object will be stored. There are some restrictions about data input
    format. First, it needs to be on csv. Also, must follow:

    * 1. sensors (path = DATA_FOLDER+'sensors.csv')
    columns = []
    index = []

    * 2. data (path = DATA_FOLDER+'data.csv')
    columns = []
    index = []
    """

    def __init__(self, configuration__):
        idx = pd.IndexSlice
        self.sensors = pd.read_csv(configuration__['DATA_FOLDER']+'sensors.csv',index_col='name')
        self.sensors = gpd.GeoDataFrame(self.sensors[['type','active','lon','lat']],
                                geometry=[shapely.geometry.Point(xy) for xy in zip(self.sensors['lon'], self.sensors['lat'])],
                                crs={'init': 'epsg:4326'}).to_crs(fiona.crs.from_epsg(4326))
        self.data = pd.read_csv(configuration__['DATA_FOLDER']+'data.csv')
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        self.data = self.data.set_index(['Variable','Sensor Name','Timestamp'])
        self.data = self.data.loc[idx[configuration__['variables_sensors']],:]
        self.sensors = self.sensors.loc[self.data.index.get_level_values(1).unique()]
        self.data, self.sensors = self.resample_by_frequency(configuration__['Sensors__frequency'])

    def delimit_by_geography(self, geography_city):
        idx = pd.IndexSlice
        self.sensors = gpd.sjoin(self.sensors, geography_city, how='inner' ,op='intersects')[self.sensors.columns]
        self.data = self.data.loc[idx[:,self.sensors.index,:],:]
        return self

    def delimit_by_osm_quantile(self, QUANTILES_PATH=None, osm_args=None):
        idx = pd.IndexSlice
        if QUANTILES_PATH is not None and osm_args is not None:
            osm_df = Features({},mode=None).make_osm_features(**osm_args)
            l = len(osm_df)
            quantiles = pd.read_csv(QUANTILES_PATH, index_col=0, header=-1, squeeze=True)
            osm_df = osm_df.loc[osm_df.apply(lambda x: True if np.any(x>quantiles) else False, axis=1)]
            self.sensors = self.sensors.loc[osm_df.index]
            self.data = self.data.loc[idx[:,osm_df.index,:],:]
            self.dropped_sensors = l-len(osm_df)
        return self

    def resample_by_frequency(self, frequency):
        idx = pd.IndexSlice
        level_values = self.data.index.get_level_values
        self.data = (self.data.groupby([level_values(i) for i in [0,1]]
                           +[pd.Grouper(freq=frequency, level=-1)]).median())
        self.data = self.data.loc[idx[:,self.sensors.index,:],:]
        self.sensors = self.sensors.loc[self.data.index.get_level_values(1).unique()]
        return self.data, self.sensors


class Geography(object):
    """
    Stores geospatial elements, such as city, osm_lines and osm_points. The
    methods implemented are to ...


    """
    def __init__(self, configuration__):
        self.city = gpd.read_file(configuration__['SHAPE_PATH']+'')
        self.city = self.filter_by_label(configuration__['Geography__filter_column'],configuration__['Geography__filter_label'])
        self.city = self.city.to_crs(fiona.crs.from_epsg(4326))
        self.city.crs = {'init': 'epsg:4326', 'no_defs': True}
        self.city = gpd.GeoDataFrame(geometry=gpd.GeoSeries(shapely.ops.cascaded_union(self.city['geometry'])))
        self.load_OSM_objects(configuration__['OSM_FOLDER'])

    def filter_by_label(self, column, label):
        return self.city[self.city[column].str.contains(label)]

    def load_OSM_objects(self, OSM_FOLDER):
        self.lines = gpd.read_file(OSM_FOLDER+'newcastle_streets.shp')
        self.lines = gpd.sjoin(self.lines, self.city, how='inner', op='intersects')[self.lines.columns]
        self.points = gpd.read_file(OSM_FOLDER+'newcastle_points.shp')
        self.points = gpd.sjoin(self.points, self.city, how='inner', op='intersects')[self.points.columns]
        return self

    def load_meshgrid_validregions(self, VALIDREGIONS_FILE):
        self.vr = pd.read_csv(VALIDREGIONS_FILE, index_col=0)
        self.vr['geometry'] = [shapely.geometry.Point(xy) for xy in zip(self.vr['lon'], self.vr['lat'])]
        return gpd.GeoDataFrame(self.vr, geometry=self.vr['geometry'])


class Features(object):
    """
    You can load or make meta-features matrixes (zx, zi) to after
    get your features for an specific variable with get_train_features(variable).
    Pay attention that if your meta-features matrixes aren't ready in
    DATA_FOLDER you will need to make it, and it can be very slow, taking even
    days of runnning in case of 1GB+ data_allsensors_8days.

    Examples:
        - to instantiate by making new features
            features = Features(configuration__, mode='make', make_args={
                'data':sensors.data,
                'sensors':sensors.sensors,
                'variables':configuration__['variables_sensors']
            }, osm_args={
                'Geography': geography,
                'input_pointdf': sensors.sensors,
                'line_objs': configuration__['osm_line_objs'],
                'point_objs': configuration__['osm_point_objs']
            })
        - to instantiate by loading zx and zi
            features = Features(configuration__)
        - having zx and zi, to get train features
            no2_x, no2_y = features.get_train_features('NO2')
            t_x, t_y = features.get_train_features('Temperature')
            co_x, co_y = features.get_train_features('CO')
    """
    def __init__(self, configuration__, mode='load', make_args=None, osm_args=None):
        if mode == 'load':
            self.zx, self.zi = self.load_csv__old(configuration__['DATA_FOLDER'])
        elif mode == 'make':
            t0 = time.time()
            make_args['osmf'] = self.make_osm_features(**osm_args)
            make_args['osmf'].quantile(0.5).to_csv(configuration__['DATA_FOLDER']+'median_quantiles_osmfeatures.csv')
            self.zx, self.zi = self.ingestion(make_args)
            self.zx.to_csv(configuration__['DATA_FOLDER']+'zx.csv')
            self.zi.to_csv(configuration__['DATA_FOLDER']+'zi.csv')
            print('Features ingested and saved in {} seconds',time.time()-t0)

    # should be updated when ingestion2 finished
    def load_csv__old(self, DATA_FOLDER):
        """
        This can only be done if ingestion have been executed a first time
        and zx.csv and zi.csv are available in DATA_FOLDER.
        """
        self.zx = pd.read_csv(DATA_FOLDER+'zx.csv')
        if 'Sensor Name' not in self.zx.columns:
            self.zx.rename({'Unnamed: 0':'Sensor Name'}, axis='columns', inplace=True)
        self.zx['Timestamp'] = pd.to_datetime(self.zx['Timestamp'])
        self.zx.set_index(['Sensor Name','Timestamp'], inplace=True)
        self.zx.rename(columns={key:key.split('.')[0] for key in self.zx.columns if '.' in key}, inplace=True)
        self.zi = pd.read_csv(DATA_FOLDER+'zi.csv')
        self.zi['Timestamp'] = pd.to_datetime(self.zi['Timestamp'])
        self.zi.set_index(['Sensor Name','Timestamp'], inplace=True)
        return self.zx, self.zi

    def resample_by_frequency(self, frequency):
        """
        You can resample the frequency of zx and zi matrixes, but
        pay attention on temporal features, if they are useless after
        the resampling, please drop it yourself. For instance,
        if you resampled from hour to day, drop 'hour' feature in zx.
        """
        idx = pd.IndexSlice
        level_values_zx = self.zx.index.get_level_values
        level_values_zi = self.zi.index.get_level_values
        self.zx = (self.zx.groupby([level_values_zx(i) for i in [0]]
                           +[pd.Grouper(freq=frequency, level=-1)]).median())
        self.zi = (self.zi.groupby([level_values_zi(i) for i in [0]]
                           +[pd.Grouper(freq=frequency, level=-1)]).median())
        return self

    def make_osm_features(self, Geography, input_pointdf, line_objs, point_objs):
        osmf = pd.DataFrame(columns=line_objs+point_objs)
        for m in range(input_pointdf.shape[0]):
            i = input_pointdf.iloc[m]
            d = {}
            for key in line_objs:
                d[key] = 1/Geography.lines.loc[(Geography.lines['highway']==key) | (Geography.lines['highway']=='{}_link'.format(key)),'geometry'].apply(lambda x: x.distance(i['geometry'])).min()
            for key in point_objs:
                d[key] = 1/Geography.points.loc[(Geography.points['highway']==key),'geometry'].apply(lambda x: x.distance(i['geometry'])).min()
            osmf.loc[i.name] = d
        return osmf

    def ingestion(self, make_args):
        """
        This is a very slow function, could run for days and should be
        used only for the first time that data is imported, as __init__
        saves the result in zx.csv and zi.csv
        """
        return sensingbee.utils.ingestion2(**make_args)

    def get_train_features(self, variable):
        var_y = self.zi.loc[self.zi['Variable']==variable]
        var_x = self.zx.loc[var_y.index]
        return var_x, var_y

    def mesh_ingestion(self, mesh_args, osm_args):
        # mesh_args['mesh'] =
        # mesh_args['osm_features'] = self.make_osm_features(osm_args)
        self.zmesh = sensingbee.utils.mesh_ingestion(**mesh_args)
        return self.zmesh

if __name__=='__main__':
    configuration__ = {
        'DATA_FOLDER':'/home/adelsondias/Repos/newcastle/air-quality/data_allsensors_8days/',
        'SHAPE_PATH':'/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp',
        'OSM_FOLDER':'/home/adelsondias/Downloads/newcastle_streets/',
        'VALIDREGIONS_FILE': '/home/adelsondias/Repos/newcastle/air-quality/data_30days/mesh_valid-regions.csv',
        'Sensors__frequency':'D',
        'Geography__filter_column':'msoa11nm',
        'Geography__filter_label':'Newcastle upon Tyne',
        'variables_sensors': ['NO2','Temperature']#,'O3','PM2.5','NO','Pressure','Wind Direction'],
        'osm_line_objs': ['primary','trunk','motorway','residential'],
        'osm_point_objs': ['traffic_signals','crossing']
    }

    geography = Geography(configuration__)
    sensors = Sensors(configuration__).delimit_by_geography(geography.city)
    # sensors.delimit_by_osm_quantile(configuration__['DATA_FOLDER']+'median_quantiles_osmfeatures.csv',osm_args={
    #     'Geography': geography,
    #     'input_pointdf': sensors.sensors,
    #     'line_objs': configuration__['osm_line_objs'],
    #     'point_objs': configuration__['osm_point_objs']
    # })

    # features = Features(configuration__)
    features = Features(configuration__, mode='make', make_args={
        'Sensors': sensors,
        'variables': configuration__['variables_sensors']
    }, osm_args={
        'Geography': geography,
        'input_pointdf': sensors.sensors,
        'line_objs': configuration__['osm_line_objs'],
        'point_objs': configuration__['osm_point_objs']
    })
