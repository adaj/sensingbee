import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
import fiona
import shapely

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, learning_curve
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

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

    def __init__(self, configuration__, mode='load', delimit_geography=None, delimit_quantiles=True):
        if mode=='make':
            idx = pd.IndexSlice
            self.sensors = pd.read_csv(configuration__['DATA_FOLDER']+'sensors.csv',index_col='name')
            self.sensors = gpd.GeoDataFrame(self.sensors[['type','active','lon','lat']],
                                    geometry=[shapely.geometry.Point(xy) for xy in zip(self.sensors['lon'], self.sensors['lat'])],
                                    crs={'init': 'epsg:4326'}).to_crs(fiona.crs.from_epsg(4326))
            self.data = pd.read_csv(configuration__['DATA_FOLDER']+'data.csv')
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            self.data = self.data.set_index(['Variable','Sensor Name','Timestamp'])
            self.data = self.data.loc[idx[configuration__['Sensors__variables']],:]
            self.sensors = self.sensors.loc[self.data.index.get_level_values(1).unique()]
            self.data, self.sensors = self.resample_by_frequency(configuration__['Sensors__frequency'])
            if delimit_geography is not None:
                self.delimit_sensors_by_geography(delimit_geography.city)
            if delimit_quantiles:
                self.delimit_sensors_by_osm_quantile(configuration__['DATA_FOLDER']+'median_quantiles_osmfeatures.csv', osm_args={
                    'Geography': delimit_geography,
                    'line_objs': configuration__['Features__osm_line_objs'],
                    'point_objs': configuration__['Features__osm_point_objs']
                })
            self.delimit_data_by_threshold(t_dict = {
                'Temperature':25,
                'NO2':80,
                'PM2.5':15
            })
            self.data.to_csv(configuration__['DATA_FOLDER']+'data__.csv')
            self.sensors.to_csv(configuration__['DATA_FOLDER']+'sensors__.csv')
        if mode=='load':
            self.sensors = pd.read_csv(configuration__['DATA_FOLDER']+'sensors__.csv', index_col=0)
            self.sensors = gpd.GeoDataFrame(self.sensors, geometry=[shapely.geometry.Point(xy) for xy in zip(self.sensors['lon'], self.sensors['lat'])])
            self.data = pd.read_csv(configuration__['DATA_FOLDER']+'data__.csv')
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            self.data = self.data.set_index(['Variable','Sensor Name','Timestamp'])

    def delimit_sensors_by_geography(self, geography_city):
        idx = pd.IndexSlice
        self.sensors.crs = geography_city.crs
        self.sensors = gpd.sjoin(self.sensors, geography_city, how='inner' ,op='intersects')[self.sensors.columns]
        self.data = self.data.loc[idx[:,self.sensors.index,:],:]
        return self

    def delimit_sensors_by_osm_quantile(self, QUANTILES_PATH, osm_args):
        idx = pd.IndexSlice
        osm_args['input_pointdf'] = self.sensors
        osm_df = Features({},mode=None).make_osm_features(**osm_args)
        l = len(osm_df)
        quantiles = pd.read_csv(QUANTILES_PATH, index_col=0, header=-1, squeeze=True)
        osm_df = osm_df.loc[osm_df.apply(lambda x: True if np.any(x>quantiles) else False, axis=1)]
        self.sensors = self.sensors.loc[osm_df.index]
        self.data = self.data.loc[idx[:,osm_df.index,:],:]
        return self

    def delimit_data_by_threshold(self, t_dict):
        off, self.dropped_sensors, self.dropped_data = {}, {}, {}
        for var, threshold in t_dict.items():
            if var!='Temperature':
                D = self.data.loc[var]['Value'].loc[self.data.loc[var]['Value']>0]
            else:
                D = self.data.loc[var]['Value']
            off[var] = []
            for i,si in enumerate(D.index.get_level_values(0).unique()):
                if D.loc[si].mean() > threshold:
                    off[var].append((var,si))
            self.dropped_sensors[var] = len(off[var])/i
            self.dropped_data[var] = 1 - (self.data.drop(off[var]).loc[var].shape[0]/self.data.loc[var].shape[0])
            self.data = self.data.drop(off[var])
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
    def __init__(self, configuration__, mode='load'):
        self.city = gpd.read_file(configuration__['SHAPE_PATH']+'')
        self.city = self.filter_by_label(configuration__['Geography__filter_column'],configuration__['Geography__filter_label'])
        self.city = self.city.to_crs(fiona.crs.from_epsg(4326))
        self.city.crs = {'init': 'epsg:4326', 'no_defs': True}
        self.city = gpd.GeoDataFrame(geometry=gpd.GeoSeries(shapely.ops.cascaded_union(self.city['geometry'])))
        self.load_OSM_objects(configuration__['OSM_FOLDER'])
        self.make_meshgrid(**configuration__['Geography__meshgrid'])
        if mode=='make':
            self.delimit_meshgrid_by_quantiles(configuration__['DATA_FOLDER']+'median_quantiles_osmfeatures.csv', {
                'Geography': self,
                'input_pointdf': self.meshgrid,
                'line_objs': configuration__['Features__osm_line_objs'],
                'point_objs': configuration__['Features__osm_point_objs']
            }).to_csv(configuration__['DATA_FOLDER']+'mesh_valid-regions.csv')
        elif mode=='load':
            self.load_meshgrid_csv(configuration__['DATA_FOLDER']+'mesh_valid-regions.csv')

    def filter_by_label(self, column, label):
        return self.city[self.city[column].str.contains(label)]

    def load_OSM_objects(self, OSM_FOLDER):
        self.lines = gpd.read_file(OSM_FOLDER+'newcastle_streets.shp')
        self.lines.crs = self.city.crs
        self.lines = gpd.sjoin(self.lines, self.city, how='inner', op='intersects')[self.lines.columns]
        self.points = gpd.read_file(OSM_FOLDER+'newcastle_points.shp')
        self.points.crs = self.city.crs
        self.points = gpd.sjoin(self.points, self.city, how='inner', op='intersects')[self.points.columns]
        return self

    def make_meshgrid(self, dimensions, longitude_range, latitude_range, delimit=False):
        self.meshdimensions = dimensions
        self.longitude_range = longitude_range
        self.latitude_range = latitude_range
        self.meshlonv, self.meshlatv = np.meshgrid(np.linspace(longitude_range[0], longitude_range[1], dimensions[0]),
                                 np.linspace(latitude_range[0], latitude_range[1], dimensions[1]))
        self.meshgrid = np.vstack([self.meshlonv.ravel(), self.meshlatv.ravel()]).T
        self.meshgrid = gpd.GeoDataFrame(self.meshgrid, geometry=[shapely.geometry.Point(xy) for xy in self.meshgrid],crs={'init': 'epsg:4326'})
        self.meshgrid.rename(columns={0:'lon',1:'lat'}, inplace=True)
        self.meshgrid.crs = self.city.crs
        self.meshgrid = gpd.sjoin(self.meshgrid, self.city, how='inner', op='intersects')[self.meshgrid.columns]
        return self

    def delimit_meshgrid_by_quantiles(self, QUANTILES_PATH, osm_args):
        idx = pd.IndexSlice
        osm_df = Features({},mode=None).make_osm_features(**osm_args)
        l = len(osm_df)
        quantiles = pd.read_csv(QUANTILES_PATH, index_col=0, header=-1, squeeze=True)
        osm_df = osm_df.loc[osm_df.apply(lambda x: True if np.any(x>quantiles) else False, axis=1)]
        self.meshgrid = self.meshgrid.loc[osm_df.index].join(osm_df)
        return self.meshgrid

    def load_meshgrid_csv(self, MESHGRID_PATH):
        self.meshgrid = pd.read_csv(MESHGRID_PATH, index_col=0)
        self.meshgrid['geometry'] = [shapely.geometry.Point(xy) for xy in zip(self.meshgrid['lon'], self.meshgrid['lat'])]
        self.meshgrid = gpd.GeoDataFrame(self.meshgrid)
        return self


class Features(object):
    """
    You can load or make meta-features matrixes (zx, zi) to after
    get your features for an specific variable with get_train_features(variable).
    Pay attention that if your meta-features matrixes aren't ready in
    DATA_FOLDER you will need to make it, and it can be very slow, taking even
    days of runnning in case of 1GB+ data_allsensors_8days.

    Examples:
        - to instantiate by making new features

        - to instantiate by loading zx and zi
            features = Features(configuration__)
        - having zx and zi, to get train features
            no2_X, no2_y = features.get_train_features('NO2')
            t_X, t_y = features.get_train_features('Temperature')
    """
    def __init__(self, configuration__, mode='load', Sensors=None, Geography=None):
        if mode == 'load':
            self.zx, self.zi = self.load_csv__old(configuration__['DATA_FOLDER'], configuration__['Sensors__frequency'])
            self.zx.rename({'PM2':'PM2.5','d_PM2':'d_PM2.5','PM1':'PM1.0','d_PM1':'d_PM1.0'},axis='columns',inplace=True)
            self.zx.dropna(axis='columns',inplace=True)
            self.zi = self.zi.set_index('Variable',append=True).swaplevel(1,2).swaplevel(0,1)
        elif mode == 'make':
            t0 = time.time()
            osm_features = self.make_osm_features(Geography, Sensors.sensors,
                                        configuration__['Features__osm_line_objs'],
                                        configuration__['Features__osm_point_objs'])
            osm_features.quantile(0.5).to_csv(configuration__['DATA_FOLDER']+'median_quantiles_osmfeatures.csv')
            self.zx, self.zi = sensingbee.utils.ingestion2(Sensors, configuration__['Sensors__variables'], k=5, osmf=osm_features)
            self.zx.dropna(axis='columns',inplace=True)
            self.zx.to_csv(configuration__['DATA_FOLDER']+'zx_{}.csv'.format(configuration__['Sensors__frequency']))
            self.zi.to_csv(configuration__['DATA_FOLDER']+'zi_{}.csv'.format(configuration__['Sensors__frequency']))
            print('Features ingested and saved in {} seconds',time.time()-t0)

    # should be updated when ingestion2 finished
    def load_csv__old(self, DATA_FOLDER, frequency):
        """
        This can only be done if ingestion have been executed a first time
        and zx.csv and zi.csv are available in DATA_FOLDER.
        """
        # self.zx = pd.read_csv(DATA_FOLDER+'zx.csv')
        self.zx = pd.read_csv(DATA_FOLDER+'zx_{}.csv'.format(frequency))
        if 'Sensor Name' not in self.zx.columns:
            self.zx.rename({'Unnamed: 0':'Sensor Name'}, axis='columns', inplace=True)
        self.zx['Timestamp'] = pd.to_datetime(self.zx['Timestamp'])
        self.zx.set_index(['Sensor Name','Timestamp'], inplace=True)
        self.zx.rename(columns={key:key.split('.')[0] for key in self.zx.columns if '.' in key}, inplace=True)
        # self.zi = pd.read_csv(DATA_FOLDER+'zi.csv')
        self.zi = pd.read_csv(DATA_FOLDER+'zi_{}.csv'.format(frequency))
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

    def get_train_features(self, variable):
        var_y = self.zi.loc[variable]
        var_x = self.zx.loc[var_y.index]
        return {'X':var_x, 'y': var_y}

    def mesh_ingestion(self, Sensors, Geography, variables, timestamp=None):
        if timestamp is None or timestamp=='*': #'take all period of Sensors.data'
            X_mesh = pd.DataFrame()
            for t in Sensors.data.index.get_level_values(2).unique():
                zmesh = sensingbee.utils.mesh_ingestion(Sensors, Geography.meshgrid, variables, t)
                zmesh['Timestamp'] = t
                zmesh = zmesh.set_index('Timestamp',append=True).swaplevel(0,1)
                X_mesh = X_mesh.append(zmesh)
        else:
            timestamp = pd.to_datetime(timestamp)
            zmesh = sensingbee.utils.mesh_ingestion(Sensors, Geography.meshgrid, variables, timestamp)
            zmesh['Timestamp'] = timestamp
            zmesh = zmesh.set_index('Timestamp',append=True).swaplevel(0,1)
            X_mesh = zmesh
        return X_mesh


class Model(object):
    """
    """
    def __init__(self, regressor):
        self.regressor = regressor

    def load_model(self, MODEL_FILE):
        return joblib.load(MODEL_FILE)

    def save_model(self, MODEL_FILE):
        joblib.dump(self.regressor, OUTPUT_FILE)

    def fit(self, X, y):
        X = MinMaxScaler().fit_transform(X)
        y = y.values.ravel()
        cv_r2, cv_mse = [], []
        for train, test in RepeatedKFold(n_splits=10, n_repeats=1).split(X):
            self.regressor.fit(X[train],y[train])
            X_pred = self.regressor.predict(X[test])
            cv_r2.append(r2_score(y[test],X_pred))
            cv_mse.append(mean_squared_error(y[test],X_pred))
        self.r2, self.r2_std = np.mean(cv_r2), np.std(cv_r2)
        self.mse, self.mse_std = np.mean(cv_mse), np.std(cv_mse)
        return self

    def predict(self, X_mesh, Geography, plot=False):
        y_pred = pd.DataFrame(self.regressor.predict(MinMaxScaler().fit_transform(X_mesh)),
                              index=Geography.meshgrid.index, columns=['pred'])
        y_pred['lat'] = Geography.meshgrid['lat']
        y_pred['lon'] = Geography.meshgrid['lon']
        if plot:
            if plot is True:
                plot = {'vmin':0, 'vmax':100}
            self.plot_interpolation(y_pred, Geography, plot['vmin'], plot['vmax'])
        return y_pred

    def plot_interpolation(self, y_pred, Geography, vmin=0, vmax=100):
        Z = np.zeros(Geography.meshlonv.shape[0]*Geography.meshlonv.shape[1]) - 9999
        Z[Geography.meshgrid.index.values] = y_pred['pred'].values.ravel()
        Z = Z.reshape(Geography.meshlonv.shape)
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(7.5,5.5))
        Geography.city.plot(ax=axes, color='white', edgecolor='black', linewidth=2)
        cs = plt.contourf(Geography.meshlonv, Geography.meshlatv, Z,
                          levels=np.linspace(0, Z.max(), 20), cmap=plt.cm.Spectral_r, alpha=1, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(cs)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, X, y):
        train_sizes, train_scores, test_scores = learning_curve(
                self.regressor, MinMaxScaler().fit_transform(X), y.values.ravel(), scoring='r2', n_jobs=2,
                cv=RepeatedKFold(n_splits=10, n_repeats=1),
                train_sizes=np.linspace(.1, 1.0, 5))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.3)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.3, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        # plt.title('Learning Curves', fontsize=18)
        plt.tight_layout()
        plt.show()


class Bee(object):
    """
    """
    def __init__(self, configuration__):
        self.configuration__ = configuration__

    def fit(self, mode, variables, regressor=None):
        self.geography = Geography(self.configuration__, mode)
        self.sensors = Sensors(self.configuration__, mode, delimit_geography=self.geography, delimit_quantiles=True)
        self.features = Features(self.configuration__, mode, Sensors=self.sensors, Geography=self.geography)
        self.models = {}
        self.scores = {}
        self.multiregressors = False
        if type(regressor)==list: # tuples with label as [('rf', RandomForest(params)), ('gb', GradiendBoost(params))]
            self.multiregressors = []
            for ri in regressor:
                self.multiregressors.append(ri[0])
                self.models[ri[0]] = {}
                self.scores[ri[0]] = {}
                for var in variables:
                    self.models[ri[0]][var] = Model(ri[1]).fit(**self.features.get_train_features(var))
                    self.scores[ri[0]][var] = (self.models[ri[0]][var].r2, self.models[ri[0]][var].mse)
        else:
            for var in variables:
                if regressor is None:
                    r = GradientBoostingRegressor(n_estimators=200, max_depth=5, max_features=0.5)
                self.models[var] = Model(r).fit(**self.features.get_train_features(var))
                self.scores[var] = (self.models[var].r2, self.models[var].mse)
        return self

    def interpolate(self, variables, data=None, timestamp=None):
        if data is None:
            data = self.sensors
        # else (TODO)
        X_mesh = self.features.mesh_ingestion(data, self.geography, self.configuration__['Sensors__variables'], timestamp=timestamp)
        self.z = {}
        for var in variables:
            y_pred = pd.DataFrame(index=X_mesh.index, columns=['pred','lat','lon'])
            if self.multiregressors:
                for ri in self.multiregressors:
                    self.z[ri] = {}
                    for t in X_mesh.index.get_level_values(0).unique():
                            y_pred.loc[t] = self.models[ri][var].predict(X_mesh.loc[t], self.geography).values
                    self.z[ri][var] = y_pred
            else:
                for t in X_mesh.index.get_level_values(0).unique():
                    y_pred.loc[t] = self.models[var].predict(X_mesh.loc[t], self.geography).values
                self.z[var] = y_pred
        return self

    def plot(self, variable, timestamp, vmin=0, vmax=100, regressor=None):
        if timestamp is None or timestamp=='*':
            if multiregressors:
                Z = self.z[regressor][variable].reset_index().astype({'pred':'int64','lat':'float64','lon':'float64'},errors='ignore').groupby('level_1').mean()
            else:
                Z = self.z[variable].reset_index().astype({'pred':'int64','lat':'float64','lon':'float64'},errors='ignore').groupby('level_1').mean()
        else:
            timestamp = pd.to_datetime(timestamp)
            Z = self.z[variable].loc[timestamp]
        if self.multiregressors:
            self.models[regressor][variable].plot_interpolation(Z, self.geography, vmin, vmax)
        else:
            self.models[variable].plot_interpolation(Z, self.geography, vmin, vmax)


if __name__=='__main__':
    configuration__ = {
        'DATA_FOLDER':'/home/adelsondias/Repos/newcastle/air-quality/data_30days/',
        'SHAPE_PATH':'/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp',
        'OSM_FOLDER':'/home/adelsondias/Downloads/newcastle_streets/',
        'VALIDREGIONS_FILE': '/home/adelsondias/Repos/newcastle/air-quality/data_30days/mesh_valid-regions.csv',
        'Sensors__frequency':'D',
        'Sensors__variables': ['NO2','Temperature','PM2.5'],
        'Geography__filter_column':'msoa11nm',
        'Geography__filter_label':'Newcastle upon Tyne',
        'Geography__meshgrid':{'dimensions':[50,50], 'longitude_range':[-1.8, -1.5], 'latitude_range':[54.95, 55.08]},
        'Features__osm_line_objs': ['primary','trunk','motorway','residential'],
        'Features__osm_point_objs': ['traffic_signals','crossing']
    }

    bee = Bee(configuration__).fit(mode='load', variables=['NO2','Temperature','PM2.5'])
    bee.interpolate(variables=['NO2'], timestamp='2018-07-01')
    bee.plot(variable='NO2', timestamp='2018-07-01', vmin=0, vmax=100)
