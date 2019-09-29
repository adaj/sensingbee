import pandas as pd
import geohunter

def inverse_distance_weighting(x, _sensors, _samples, p, threshold):
    d = _sensors.distance(x['geometry'])
    d = d.loc[(d<threshold) & (d>0)] # distance threshold
    wi = 1/(d**p)
    return (wi*_samples.loc[d.index, 'Value']).sum() / wi.sum()
def nearest_neighbor(x, _sensors, _samples):
    d = _sensors.distance(x['geometry'])
    d = d.loc[d>0].nsmallest(1)
    return _samples.loc[d.index, 'Value'].values
def spatial_averaging(x, _sensors, _samples, threshold):
    d = _sensors.distance(x['geometry'])
    d = d.loc[(d<threshold) & (d>0)]
    return _samples.loc[d.index, 'Value'].mean()
# def kriging():


class Features(object):

    def __init__(self, variable, method='nn', **kwargs): # TODO: to acept variables and methods
        self.variable = variable
        self.method = method
        if method=='idw':
            self.params = {'p': kwargs.pop('p', 2), 'threshold': kwargs.pop('threshold', 10/110)}
            self.method=inverse_distance_weighting # the function
        elif method=='nn':
            self.params = {}
            self.method=nearest_neighbor # the function
        elif method=='sa':
            self.params = {'threshold': kwargs.pop('threshold', 10/110)}
            self.method=spatial_averaging # the function

    def fit(self, data):
        return self

    def transform(self, samples, metadata, grid=None): # data must be `data_preparation.Data`
        idx = pd.IndexSlice
        y = samples.loc[self.variable]
        if grid is not None:
            X = pd.DataFrame(index=pd.MultiIndex.from_product([grid.index, samples.index.get_level_values('Timestamp')], names=['Sensor Name', 'Timestamp']))
            X = X.loc[~X.index.duplicated(keep='first')]
        else:
            X = pd.DataFrame(index=y.index)
        for var in samples.index.get_level_values('Variable').unique():
            for time in samples.index.get_level_values('Timestamp').unique():
                v_samples = samples.loc[idx[var, :, time]].reset_index()
                v_sensors = metadata.loc[v_samples['Sensor Name']]
                v_sensors = v_sensors.dropna()
                mask = y.loc[idx[:, time],:]
                if grid is not None:
                    _ = grid.apply(lambda x: self.method(x,
                                    v_sensors, v_samples.set_index('Sensor Name'), **self.params), axis=1)
                    _.index = X.loc[idx[:, time],:].index
                    X.loc[idx[:, time],var] = _.values
                else:
                    _ = y.loc[idx[:, time],:].reset_index()['Sensor Name']\
                            .apply(lambda x: self.method(metadata.loc[x],
                                    v_sensors, v_samples.set_index('Sensor Name'), **self.params))
                    _.index = y.loc[idx[:, time],:].index
                    X.loc[idx[:, time],var] = _.values
        return X, y


class GeoFeatures(object):

    def __init__(self, params='auto'):
        self.params = params

    def transform(self, geodata, places):
        print('@ GeoFeatures: Extracting...')
        return geohunter.features.KDEFeatures(geodata,
            kde_params=self.params).fit_transform(places)
