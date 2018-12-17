import pandas as pd
import numpy as np
import geopandas as gpd
import shapely, fiona
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler

import geofeatures as gf
import utils, __utils


class Sensors(object):

    def __init__(self, configuration, data, metadata):
        self.configuration = configuration
        self.data = data
        self.metadata = metadata
        self.data = utils.resample_sensors_data(data, configuration['sensors_frequency'])

    def fit_in_geo(self, geo):
        idx = pd.IndexSlice
        self.metadata = gpd.sjoin(self.metadata, geo, how='inner', op='intersects')[self.metadata.columns]
        self.data = self.data.loc[idx[:,self.metadata.index],:]
        return self


class Geography(object):

    def __init__(self, configuration, city):
        self.configuration = configuration
        self.city = city
        self.grid, self.lonv, self.latv = utils.make_grid(configuration['geography_bbox'], dimensions=configuration['grid_dimensions'])
        self.grid = gpd.sjoin(self.grid, self.city, how='inner', op='intersects')[self.grid.columns]


class Features(object):

    def __init__(self, configuration):
        self.configuration = configuration

    def ingest(self, data, metadata, method, param, label):
        idx = pd.IndexSlice

        # IDW/NN features
        if method=='NN':
            cols = []
            for var in data.index.get_level_values(0).unique():
                for n in range(param):
                    cols.append('{}_nn{}'.format(var,n+1))
            X = pd.DataFrame(index=data.loc[var].index, columns=cols)
            for var in data.index.get_level_values(0).unique():
                for time in data.index.get_level_values(2).unique():
                    variable_data = data.loc[idx[var, :, time]]
                    variable_data = variable_data.join(metadata, on='Sensor Name')[['Value','geometry']]
                    for sensor in data.index.get_level_values(1).unique():
                        nearest_samples_dist = variable_data['geometry'].apply(lambda x: x.distance(metadata.loc[sensor]['geometry'])).sort_values()[:param]
                        nearest_samples_values = variable_data.loc[idx[var,nearest_samples_dist.index.get_level_values(1),time],'Value']
                        X.loc[idx[sensor, time],X.columns.str.contains(var)] = nearest_samples_values.loc[nearest_samples_dist.index].values
        elif method=='IDW':
            cols = []
            for var in data.index.get_level_values(0).unique():
                    cols.append('{}_idw'.format(var))
            X = pd.DataFrame(index=data.loc[var].index, columns=[cols])
            for var in data.index.get_level_values(0).unique():
                for time in data.index.get_level_values(2).unique():
                    variable_data = data.loc[idx[var, :, time]]
                    variable_data = variable_data.join(metadata, on='Sensor Name')[['Value','geometry']]
                    for sensor in data.index.get_level_values(1).unique():
                        nearest_samples_dist = variable_data['geometry'].apply(lambda x: x.distance(metadata.loc[sensor]['geometry'])).sort_values()
                        nearest_samples_dist = nearest_samples_dist.loc[nearest_samples_dist<param] # limit of distance
                        nearest_samples_values = variable_data.loc[idx[var,nearest_samples_dist.index.get_level_values(1),time],'Value']
                        idw_value = sum((1-1000*nearest_samples_dist)*nearest_samples_values)/sum(1-1000*nearest_samples_dist)
                        X.loc[idx[sensor, time],'{}_idw'.format(var)] = idw_value

        # temporal features
        X['week'] = X.index.get_level_values(1).week
        X['month'] = X.index.get_level_values(1).month
        X['year'] = X.index.get_level_values(1).year

        # geographic features
        gt = gf.GeoTransformer(bbox=self.configuration['geography_bbox'], grid=metadata[['lat','lon','geometry']])
        for key in self.configuration['osm_points'].keys():
            kde_geof = gt.extract_from_points(key, self.configuration['osm_points'][key], 'kde', {'bandwidth':0.5/69, 'kernel':'gaussian'})
            X = X.join(kde_geof, on='Sensor Name')
        for key in self.configuration['osm_lines'].keys():
            kde_geof = gt.extract_from_lines(key, self.configuration['osm_lines'][key], 'kde', {'bandwidth':0.5/69, 'kernel':'gaussian'})
            X = X.join(kde_geof, on='Sensor Name')

        # saving features in volume
        X.to_csv('{}/X_{}_v{}.csv'.format(self.configuration['volume_folder'], method, label).replace('//','/'))
        y = {}
        for var in data.index.get_level_values(0).unique():
            y[var] = data.loc[var]
            y[var].to_csv('{}/y_{}_v{}.csv'.format(self.configuration['volume_folder'], var.replace('.','-'), label).replace('//','/'))
        return X, y

    def load(self, variables, method, label):
        X = pd.read_csv('{}/X_{}_v{}.csv'.format(self.configuration['volume_folder'], method, label).replace('//','/'), index_col=[0,1])
        y = {}
        for var in variables:
            y[var] = pd.read_csv('{}/y_{}_v{}.csv'.format(self.configuration['volume_folder'], var.replace('.','-'), label).replace('//','/'), index_col=[0,1])
        return X, y


class Model(object):

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y, cv_repetitions=2):
        self.xscaler = MinMaxScaler().fit(X)
        X_ = self.xscaler.transform(X)
        y_ = y.values.ravel()
        cv_r2, cv_mse, cv_medae, cv_ev = [], [], [], []
        for train, test in RepeatedKFold(n_splits=5, n_repeats=cv_repetitions).split(X_):
            self.regressor.fit(X_[train],y_[train])
            y_pred = self.regressor.predict(X_[test])
            cv_r2.append(r2_score(y.values[test].reshape(-1, 1),y_pred.reshape(-1, 1)))
            cv_mse.append(mean_squared_error(y.values[test].reshape(-1, 1),y_pred.reshape(-1, 1)))
            cv_medae.append(median_absolute_error(y.values[test].reshape(-1, 1),y_pred.reshape(-1, 1)))
            cv_ev.append(explained_variance_score(y.values[test].reshape(-1, 1),y_pred.reshape(-1, 1)))
        self.r2 = np.mean(cv_r2)
        self.mse = np.mean(cv_mse)
        self.medae = np.mean(cv_medae)
        self.ev = np.mean(cv_ev)
        return self

    def fit_randomizedsearchCV(self, X, y, param_grid, refit_scoring='r2', n_iter=5):
        self.xscaler = MinMaxScaler().fit(X)
        X_ = self.xscaler.transform(X)
        y_ = y.values.ravel()
        random_search = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions=param_grid,
                                   scoring=['r2','neg_mean_squared_error','neg_median_absolute_error','explained_variance'],
                                   refit=refit_scoring, n_iter=2, cv=5, return_train_score=True)
        random_search.fit(X_, y_)
        self.regressor = random_search.best_estimator_
        bix = random_search.best_index_
        self.r2 = random_search.cv_results_['mean_test_r2'][bix]
        self.mse = random_search.cv_results_['mean_test_neg_mean_squared_error'][bix]
        self.medae = random_search.cv_results_['mean_test_neg_median_absolute_error'][bix]
        self.ev = random_search.cv_results_['mean_test_explained_variance'][bix]
        return self

    def predict(self, X):
        X_ = self.xscaler.fit_transform(X)
        return pd.DataFrame(self.regressor.predict(X_).reshape(-1,1), index=X.index)


class Bee(object):

    def __init__(self, configuration):
        self.configuration = configuration

    def fit(self, mode):
        return self

    # def train_model(self, var, regressor, X, y):
    #     return self

    def interpolate(self, var, timestamp, data, plot=False):
        return self
