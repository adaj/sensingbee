"""
sensingbee
Author: Adelson Araujo Jr (adelsondias@gmail.com)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import shapely
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

import sensingbee.utils as utils


class Sensors(object):
    """
    Stores data and metadata about sensors. There are required a `configuration__` dictionary,
    containing at least information about a (1) data folder path to store metadata, (2) list
    of variables' names that will be considered, (3) the frequency that those variables should
    be aggregated ('D' for daily, 'W' for weekly etc) and (4) a threshold for not considering
    uncallibrated sensors (it will cut sensors with samples mean upper than the threshold).
    Besides the configuration, the instantiation parameter `mode` is required. It has the options
    of "make" a new object based on two csv files (data.csv and sensors.csv, containing the
    sensors samples and metainformation respectively), "load" for loading pre-maked data, but
    also "get", that can pull data from API, such as Urban Observatory open sensors API, that
    should use information on parameter `path` to make the request.
    """
    def __init__(self, configuration__, mode, path, delimit_geography=None, delimit_quantiles=True, delimit_data_by_threshold=True):
        idx = pd.IndexSlice
        if mode=='get':
            path['start_time'] = path['start_time'].replace('-','')
            path['end_time'] = path['end_time'].replace('-','')
            path['variable'] = '-and-'.join(configuration__['Sensors__variables']).lower()
            data = pd.read_csv('{}?start_time={}000000&end_time={}005959&variable={}'.format(
                                    path['url'], path['start_time'], path['end_time'], path['variable']))
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            self.sensors = data.loc[:,['type','active','lon','lat','name']].set_index('name')
            self.sensors = gpd.GeoDataFrame(self.sensors,
                                    geometry=[shapely.geometry.Point(xy) for xy in zip(self.sensors['lon'], self.sensors['lat'])],
                                    crs={'init': 'epsg:4326'}).to_crs(fiona.crs.from_epsg(4326))
            self.data = data.loc[:,['Variable','name','Timestamp','Value']].set_index(['Variable','name','Timestamp'])
            self.data = self.data.loc[idx[configuration__['Sensors__variables']],:]
            self.sensors = self.sensors.loc[self.data.index.get_level_values(1).unique()]
            self.sensors.drop_duplicates(['lon','lat'], inplace=True)
            self.data, self.sensors = self.resample_by_frequency(configuration__['Sensors__frequency'])
            self.data.index.names = ['Variable','Sensor Name','Timestamp']
            if delimit_geography is not None:
                self.delimit_sensors_by_geography(delimit_geography.city)
            if delimit_quantiles:
                self.delimit_sensors_by_osm_quantile(osm_args={
                    'Geography': delimit_geography,
                    'line_objs': configuration__['osm_line_objs'],
                    'point_objs': configuration__['osm_point_objs']
                })
            self.delimit_data_by_threshold(configuration__['Sensors__threshold_callibration'])
        elif mode=='make':
            self.sensors = pd.read_csv(path+'sensors.csv',index_col='name')
            self.sensors = gpd.GeoDataFrame(self.sensors[['type','active','lon','lat']],
                                    geometry=[shapely.geometry.Point(xy) for xy in zip(self.sensors['lon'], self.sensors['lat'])],
                                    crs={'init': 'epsg:4326'}).to_crs(fiona.crs.from_epsg(4326))
            self.data = pd.read_csv(path+'data.csv')
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            self.data = self.data.set_index(['Variable','Sensor Name','Timestamp'])
            self.data = self.data.loc[idx[configuration__['Sensors__variables']],:]
            self.sensors = self.sensors.loc[self.data.index.get_level_values(1).unique()]
            self.sensors.drop_duplicates(['lon','lat'], inplace=True)
            self.data, self.sensors = self.resample_by_frequency(configuration__['Sensors__frequency'])
            if delimit_geography is not None:
                self.delimit_sensors_by_geography(delimit_geography.city)
            if delimit_quantiles:
                self.delimit_sensors_by_osm_quantile(osm_args={
                    'Geography': delimit_geography,
                    'line_objs': configuration__['osm_line_objs'],
                    'point_objs': configuration__['osm_point_objs']
                })
            if delimit_data_by_threshold:
                self.delimit_data_by_threshold(configuration__['Sensors__threshold_callibration'])
            self.data.to_csv(configuration__['DATA_FOLDER']+'data__.csv')
            self.sensors.to_csv(configuration__['DATA_FOLDER']+'sensors__.csv')
        elif mode=='load':
            self.sensors = pd.read_csv(path+'sensors__.csv', index_col=0)
            self.sensors = gpd.GeoDataFrame(self.sensors, geometry=[shapely.geometry.Point(xy) for xy in zip(self.sensors['lon'], self.sensors['lat'])])
            self.data = pd.read_csv(path+'data__.csv')
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            self.data = self.data.set_index(['Variable','Sensor Name','Timestamp'])

    # Used for drop sensors outside a Geography object.
    def delimit_sensors_by_geography(self, geography_city):
        idx = pd.IndexSlice
        self.sensors.crs = geography_city.crs
        self.sensors = gpd.sjoin(self.sensors, geography_city, how='inner' ,op='intersects')[self.sensors.columns]
        self.data = self.data.loc[idx[:,self.sensors.index,:],:]
        return self

    # Used for drop sensors outside the reasonable area delimited by OpenStreetMaps features.
    # This is made to avoid bias on predicting/interpolating for zones which the urban configuration
    # is not properly similar to where data is collected.
    def delimit_sensors_by_osm_quantile(self, osm_args):
        idx = pd.IndexSlice
        osm_args['input_pointdf'] = self.sensors
        osm_df = Features({},mode=None).make_osm_features(**osm_args)
        quantiles = osm_df.quantile(0.5)
        osm_df = osm_df.loc[osm_df.apply(lambda x: True if np.any(x>quantiles) else False, axis=1)]
        self.sensors = self.sensors.loc[osm_df.index]
        self.data = self.data.loc[idx[:,osm_df.index,:],:]
        return self

    # Used for drop sensors considered uncallibrated due to their samples average
    # is beyond an reasonable threshold. A t_dict parameter should have as keys
    # the variables' names and as values the corresponding thresold.
    def delimit_data_by_threshold(self, t_dict):
        off, self.dropped_sensors, self.dropped_data = {}, {}, {}
        for var, threshold in t_dict.items():
            D = self.data.loc[var]['Value']
            off[var] = []
            for i,si in enumerate(D.index.get_level_values(0).unique()):
                if D.loc[si].mean() > threshold:
                    off[var].append((var,si))
            self.dropped_sensors[var] = len(off[var])/i
            self.dropped_data[var] = 1 - (self.data.drop(off[var]).loc[var].shape[0]/self.data.loc[var].shape[0])
            self.data = self.data.drop(off[var])
        return self

    # Used for resampling the data by a frequency parameter, i.e. 'D' for daily,
    # 'W' for weekly etc.
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
    Stores geospatial objects, such as city geometry (loading from a shapefile) and OpenStreetMaps
    highway objects, such as roads, crossings and traffic_signals (pulling from Overpass API). For
    proper use, it's required to pass a `configuration__` dictionary with (1) the city's shapefile local path,
    (and optionally a filter label to delimit the shapefile subobjects considered, e.g. if you are using a
    shapefile from a whole country but only want to work with a single city within it), the (2) OSM features
    parameters such as the bounding box, highway-line (primary, trunk ...) and highway-point (traffic_signals, ...)
    type names. Also, this class implements the creation of the meshgrid that the interpolation will be applied,
    and this meshgrid should be configured in `configuration__` dictionary with the (3) meshgrid's dimensions,
    and latitude/longitude ranges for rectangular boundaries, so that a file will be created with the meshgrid
    geometries in a (4) data folder. All those enumerate itens are required in `configuration__`. If it's the
    first time that Geography is instantiated, you have to call it with `mode`="make".
    """
    def __init__(self, configuration__, mode='load'):
        self.city = gpd.read_file(configuration__['SHAPE_PATH']+'')
        try:
            self.city = self.filter_by_label(configuration__['Geography__filter_column'],configuration__['Geography__filter_label'])
        except:
            pass
        self.city = self.city.to_crs(fiona.crs.from_epsg(4326))
        self.city.crs = {'init': 'epsg:4326', 'no_defs': True}
        self.city = gpd.GeoDataFrame(geometry=gpd.GeoSeries(shapely.ops.cascaded_union(self.city['geometry'])))
        self.lines, self.points = utils.pull_osm_objects(configuration__['osm_bbox'],
                            configuration__['osm_line_objs'], configuration__['osm_point_objs']) # OSM api
        self.delimit_osm_by_city()
        self.make_meshgrid(**configuration__['Geography__meshgrid'])
        if mode=='make':
            self.delimit_meshgrid_by_quantiles({
                'Geography': self,
                'input_pointdf': self.meshgrid,
                'line_objs': configuration__['osm_line_objs'],
                'point_objs': configuration__['osm_point_objs']
            }).to_csv(configuration__['DATA_FOLDER']+'meshgrid.csv')
        elif mode=='load':
            self.load_meshgrid_csv(configuration__['DATA_FOLDER']+'meshgrid.csv')

    # Used for filtering shapefile with a column and a label. Example: your shapefile is for
    # the whole UK, but you just want the geometries that have "name"=="London".
    def filter_by_label(self, column, label):
        return self.city[self.city[column].str.contains(label)]

    # Used for delimit the OpenStreetMaps geometries that are inside Geography.city,
    # corresponding to the shapefile object.
    def delimit_osm_by_city(self):
        self.lines.crs = self.city.crs
        self.lines = gpd.sjoin(self.lines, self.city, how='inner', op='intersects')[self.lines.columns]
        self.points.crs = self.city.crs
        self.points = gpd.sjoin(self.points, self.city, how='inner', op='intersects')[self.points.columns]
        return self

    # Used to produce the geospatial dataframe for the meshgrid collection of dimension[0]*dimension[1]
    # points, given the boundaries imposed by longitude_range and latitude_range.
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
        # if delimit:
        #     self.meshgrid = delimit_meshgrid_by_quantiles()
        return self

    # Used to filter the points in the meshgrid that are proper to have predictions/interpolation
    # in order to avoid regions with the urban configuration critically different from where
    # data is collected.
    def delimit_meshgrid_by_quantiles(self, osm_args):
        idx = pd.IndexSlice
        osm_df = Features({},mode=None).make_osm_features(**osm_args)
        quantiles = osm_df.quantile(0.5)
        osm_df = osm_df.loc[osm_df.apply(lambda x: True if np.any(x>quantiles) else False, axis=1)]
        self.meshgrid = self.meshgrid.loc[osm_df.index].join(osm_df)
        return self.meshgrid

    # Used for loading meshgrid from the already-made meshgrid object. This is only used
    # after a round of Geography(mode="make"), by Geography(mode="load") calls.
    def load_meshgrid_csv(self, MESHGRID_PATH):
        self.meshgrid = pd.read_csv(MESHGRID_PATH, index_col=0)
        self.meshgrid['geometry'] = [shapely.geometry.Point(xy) for xy in zip(self.meshgrid['lon'], self.meshgrid['lat'])]
        self.meshgrid = gpd.GeoDataFrame(self.meshgrid)
        return self


class Features(object):
    """
    Stores features for spatial interpolation of Sensors within a Geography. A Feature object
    contains training data `zx` (features matrix) and `zi` (target values) that are the
    information used for the regression made by the Model object. Basically, `zx` extract
    for each variable, k (default=5) neighbors samples and the distance between them. For
    instance, consider to extract features for NO2 and Temperature, you will have the following
    channels/columns:
    [NO2, NO2, NO2, NO2, NO2, d_NO2, d_NO2, d_NO2, d_NO2, d_NO2, Temperature, Temperature, Temperature
    Temperature, Temperature, d_Temperature, d_Temperature, d_Temperature, d_Temperature, d_Temperature],
    and their respective rows. In training sample, these rows are points where one sensor have data
    to be predicted. So, the training process is to use these features described to predict the
    value collected in the sensor for a particular variable (supervised learning). This process of
    pulling features from the Sensors object is made by the `ingestion` method*. Besides, when it comes
    to the prediction, it will be necessary to ingest features for the meshgrid, and the Feature object
    also returns a feature matrix for the meshgrid.

    *The ingestion method when making new features are a very slow method, so for reuse already
    made features, you can use the `mode`="load".

    Examples:
        - to instantiate by making new features
            features = Features(configuration__, mode='make', Sensors, Geography)
        - to instantiate by loading pre-made zx and zi
            features = Features(configuration__, mode='load')
        - having zx and zi, to get train features for a particular variable*
            no2_X, no2_y = features.get_train_features('NO2')
            t_X, t_y = features.get_train_features('Temperature')
        *these X and y are the matrixes used by Model
    """
    def __init__(self, configuration__, mode='load', Sensors=None, Geography=None, save=True):
        if mode == 'load':
            self.zx, self.zi = self.load_csv(configuration__['DATA_FOLDER'], configuration__['Sensors__frequency'])
            self.zx.rename({'PM2':'PM2.5','d_PM2':'d_PM2.5','PM1':'PM1.0','d_PM1':'d_PM1.0'},axis='columns',inplace=True)
            self.zx.dropna(axis=0,inplace=True)
            self.zi = self.zi.set_index('Variable',append=True).swaplevel(1,2).swaplevel(0,1)
        elif mode == 'make':
            t0 = time.time()
            osm_features = self.make_osm_features(Geography, Sensors.sensors,
                                        configuration__['osm_line_objs'],
                                        configuration__['osm_point_objs'])
            # try:
            deprivation_features = utils.pull_depr_sensors(Sensors)
            # except:
                # print('Deprivation features not extracted')
                # deprivation_features = None
            self.zx, self.zi = utils.ingestion3(Sensors, configuration__['Sensors__variables'], k=5, osmf=osm_features, deprf=deprivation_features, freq='D')
            self.zx.dropna(axis=0,inplace=True)
            if save:
                self.zx.to_csv(configuration__['DATA_FOLDER']+'zx_{}.csv'.format(configuration__['Sensors__frequency']))
                self.zi.to_csv(configuration__['DATA_FOLDER']+'zi_{}.csv'.format(configuration__['Sensors__frequency']))
            print('Features ingested and saved in {} seconds'.format(time.time()-t0))

    # Used for load already-made feature matrixes zx and zi
    def load_csv(self, DATA_FOLDER, frequency):
        self.zx = pd.read_csv(DATA_FOLDER+'zx_{}.csv'.format(frequency))
        if 'Sensor Name' not in self.zx.columns:
            self.zx.rename({'Unnamed: 0':'Sensor Name'}, axis='columns', inplace=True)
        self.zx['Timestamp'] = pd.to_datetime(self.zx['Timestamp'])
        self.zx.set_index(['Sensor Name','Timestamp'], inplace=True)
        self.zx.rename(columns={key:key.split('.')[0] for key in self.zx.columns if '.' in key}, inplace=True)
        self.zi = pd.read_csv(DATA_FOLDER+'zi_{}.csv'.format(frequency))
        self.zi['Timestamp'] = pd.to_datetime(self.zi['Timestamp'])
        self.zi.set_index(['Sensor Name','Timestamp'], inplace=True)
        return self.zx, self.zi

    # Used to resample feature matrix if wanted to train models in another frequency
    # different from the Sensors' frequency.
    def resample_by_frequency(self, frequency):
        idx = pd.IndexSlice
        level_values_zx = self.zx.index.get_level_values
        level_values_zi = self.zi.index.get_level_values
        self.zx = (self.zx.groupby([level_values_zx(i) for i in [0]]
                           +[pd.Grouper(freq=frequency, level=-1)]).median())
        self.zi = (self.zi.groupby([level_values_zi(i) for i in [0]]
                           +[pd.Grouper(freq=frequency, level=-1)]).median())
        return self

    # Used in other classes to produce the urban feature matrixes using
    # the OpenStreetMaps objects in Geography.
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

    # Used for pull features for a particular variable from the zx and zi
    def get_train_features(self, variable):
        var_y = self.zi.loc[variable]
        var_x = self.zx.loc[var_y.index]
        return {'X':var_x, 'y': var_y}

    # Used for get feature matrixes to feed a trained model in order
    # to predict/interpolate for the meshgrid
    def mesh_ingestion(self, Sensors, Geography, variables, timestamp=None):
        if timestamp is None or timestamp=='*': #'take all period of Sensors.data'
            X_mesh = pd.DataFrame()
            for t in Sensors.data.index.get_level_values(2).unique():
                zmesh = utils.mesh_ingestion(Sensors, Geography.meshgrid, variables, t)
                zmesh['Timestamp'] = t
                zmesh = zmesh.set_index('Timestamp',append=True).swaplevel(0,1)
                X_mesh = X_mesh.append(zmesh)
        else:
            timestamp = pd.to_datetime(timestamp)
            zmesh = utils.mesh_ingestion(Sensors, Geography.meshgrid, variables, timestamp)
            zmesh['Timestamp'] = timestamp
            zmesh = zmesh.set_index('Timestamp',append=True).swaplevel(0,1)
            X_mesh = zmesh
        return X_mesh


class Model(object):
    """
    For the interpolation, a model/regressor needs to be fitted with data for
    a specific variable. The way that this class is implemented is to support
    wrapping for sklearn objects, such as RandomForestRegressor. It also implements
    prediction for a meshgrid features matrix and provide a routine for contourplot
    to visualize the interpolation. At last, in order to give more information on
    training process, it has a visualization on the learning curve of the regressor.
    """
    def __init__(self, regressor):
        self.regressor = regressor

    # To reuse pretrained models
    def load_model(self, MODEL_FILEPATH):
        return joblib.load(MODEL_FILEPATH)

    # To save models for reusing
    def save_model(self, MODEL_FILEPATH):
        joblib.dump(self.regressor, OUTPUT_FILE)

    # It fits the regressor by scaling features and through a 10-fold CV training process
    # the results are stored in metrics attributes
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

    # Used for the prediction in a meshgrid of a Geography object after the mesh_ingestion
    # process. It also could be setted to plot the interpolation result
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

    # Used for generate a contour plot with the meshgrid and the predicted values.
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

    # Used to give further information on the training performance with different
    # samples of data. The question that this method answer is whether you are
    # training with enough data or not. Please refer to the web for learning curve
    # interpretation.
    def plot_learning_curve(self, X, y, title='', plot_path=False):
        train_sizes, train_scores, test_scores = learning_curve(
                self.regressor, MinMaxScaler().fit_transform(X), y.values.ravel(), scoring='r2', n_jobs=2,
                cv=RepeatedKFold(n_splits=10, n_repeats=1),
                train_sizes=np.linspace(.1, 1.0, 5))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.3, color='r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.3, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="CV")
        plt.legend(loc="best")
        plt.title(title, fontsize=18)
        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path+title+'.png')
        plt.show()


class Bee(object):
    """
    As a wrapper of the whole spatial interpolation pipeline, Bee can fit the pipeline,
    train different regressors and interpolate variables with new data, even data from
    a csv fileset (data.csv and sensors.csv) or from HTTP/GET requests. For visualization,
    the plot function wraps the Model.plot_interpolation but also plots aggregating a median
    for the whole period of data, instead of plotting the interpolation for a single timestamp.
    For usage examples to "make" a new Bee or to "load" an already made one, refer to /examples folder.
    """
    def __init__(self, configuration__):
        self.configuration__ = configuration__

    # It loads/makes its Geography, Sensors and Features objects for further usage
    def fit(self, mode, verbose=False):
        t0 = time.time()
        self.geography = Geography(self.configuration__, mode)
        if verbose:
            print('[Geography] {} seconds'.format(time.time()-t0))
            t0 = time.time()
        self.sensors = Sensors(self.configuration__, mode, self.configuration__['DATA_FOLDER'], delimit_geography=self.geography, delimit_quantiles=True)
        if verbose:
            print('[Sensors] {} seconds'.format(time.time()-t0))
            t0 = time.time()
        self.features = Features(self.configuration__, mode, Sensors=self.sensors, Geography=self.geography)
        if verbose:
            print('[Features] {} seconds'.format(time.time()-t0))
            t0 = time.time()
        return self

    # Wrapper for Model. Can fit multiple regressor given a list of tuples ("label", Regressor)
    def train(self, variables, regressor=None, X=None, y=None):
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
                    if X is None and y is None:
                        self.models[ri[0]][var] = Model(ri[1]).fit(**self.features.get_train_features(var))
                    else:
                        self.models[ri[0]][var] = Model(ri[1]).fit(X, y)
                    self.scores[ri[0]][var] = (self.models[ri[0]][var].r2, self.models[ri[0]][var].mse)
        else:
            for var in variables:
                if regressor is None:
                    r = GradientBoostingRegressor(n_estimators=200, max_depth=5, max_features=0.5)
                else:
                    r = regressor
                if X is None and y is None:
                    self.models[var] = Model(r).fit(**self.features.get_train_features(var))
                else:
                    self.models[var] = Model(r).fit(X, y)
                self.scores[var] = (self.models[var].r2, self.models[var].mse)
        return self

    # Wrapper for Model prediction applied to the Geography.meshgrid
    def interpolate(self, variables, data=None, timestamp=None):
        if data is None:
            data = self.sensors
        if timestamp is not None and timestamp!='*':
            timestamp = pd.to_datetime(timestamp)
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
                    print('bee.interpolate - ',t)
                    y_pred.loc[t] = self.models[var].predict(X_mesh.loc[t], self.geography).values
                self.z[var] = y_pred
        return self

    # Wrapper for contour plot of predicted values of a specific variable. You can
    # configure the color range for the contour plot with vmin and vmax parameters.
    def plot(self, variable, timestamp, vmin=0, vmax=100, regressor=None):
        if timestamp is None or timestamp=='*':
            if multiregressors:
                Z = self.z[regressor][variable].reset_index().astype({'pred':'int64','lat':'float64','lon':'float64'},errors='ignore').groupby('level_1').median()
            else:
                Z = self.z[variable].reset_index().astype({'pred':'int64','lat':'float64','lon':'float64'},errors='ignore').groupby('level_1').median()
        else:
            timestamp = pd.to_datetime(timestamp)
            Z = self.z[variable].loc[timestamp]
        if self.multiregressors:
            self.models[regressor][variable].plot_interpolation(Z, self.geography, vmin, vmax)
        else:
            self.models[variable].plot_interpolation(Z, self.geography, vmin, vmax)
