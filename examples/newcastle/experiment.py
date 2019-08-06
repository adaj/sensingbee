import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import sys, os, datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import sensingbee

class Experiment(object):

    def __init__(self, X, y, samples, models, t, scores):
        self.X = X
        self.y = y
        self.samples = samples
        self.models = models
        self.t = t
        self.scores = scores


estimator = RandomForestRegressor(max_depth=3)#RandomForestRegressor(max_depth=5)
tuning_conf = {'params':{'n_estimators':[10, 20]},
                    'iterations': 2, 'cv':5, 'scoring':['r2','neg_mean_squared_error']}


t0 = time.time()
data = sensingbee.data_preparation.Data(os.path.join(os.path.dirname(__file__), 'data', '{}'.format(sys.argv[1])),
                geodata={'points':{'shop':['*'], 'highway':['traffic_signals']},
                        'lines':{'highway':['residential','primary']}},
                grid_resolution=1)
print('Data .. {}s'.format(time.time()-t0))

method_params = {
    'sa': {'threshold': 10/69},
    'nn': {},
    'idw': {'threshold': 10/69, 'p': 2}
}

print('Start')
X, y = {}, {}
t = {}
scores = {}
samples = {}
models = {}
#
Xg = sensingbee.feature_engineering.GeoFeatures(params='auto').transform(data.geodata ,places=data.metadata)
#
for var in ['Temperature', 'Humidity', 'PM2.5', 'CO', 'NO2']:
    X[var], y[var] = {}, {}
    t[var] = {}
    scores[var] = {}
    samples[var] = data.samples.loc[var].shape[0]
    models[var] = {}
    print('+ {} - {} samples'.format(var, samples[var]))
    for method in ['sa', 'nn', 'idw']:
        for features in ['', 'gf']:
            t_ = time.time()
            X[var][method+'_'+features], y[var][method+'_'+features] = sensingbee.feature_engineering.Features(variable=var, method=method, **method_params[method]).transform(data)
            t[var][method+'_'+features] = time.time()-t_
            print('> {}-{} in {}s'.format(var, method, features, t[var][method+'_'+features]))
            if features == 'gf':
                X[var][method+'_'+features] = X[var][method+'_'+features].join(Xg, on='Sensor Name')
            X[var][method+'_'+features] = X[var][method+'_'+features].sample(frac=1)
            y[var][method+'_'+features] = y[var][method+'_'+features].loc[X[var][method+'_'+features].index]
            m = sensingbee.ml_modeling.Model(estimator, tuning_conf).fit(X[var][method+'_'+features], y[var][method+'_'+features])
            scores[var][method+'_'+features] = m.base_estimator.best_score_
            models[var][method+'_'+features] = m
print('Done .. {}s'.format(time.time()-t0))

joblib.dump(Experiment(X, y, samples, models, t, scores), 'experiment_{}.joblib'.format(str(datetime.datetime.now())[:10]))
