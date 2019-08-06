import sensingbee

#
# Data loading
#
data = sensingbee.data_preparation.Data('/home/adelsondias/Repos/sensingbee/examples/newcastle/data/daily',
                geodata={'points':{'shop':['*'], 'highway':['traffic_signals']},
                        'lines':{'highway':['residential','primary']}},
                grid_resolution=1)

#
# Feature Extraction
#
import time
t0 = time.time()
X_sa, y = sensingbee.feature_engineering.Features(variable='NO2',
                        method='sa', threshold=30/69).transform(data)
print(time.time()-t0)
t0 = time.time()
X_idw, y = sensingbee.feature_engineering.Features(variable='NO2',
                        method='idw', threshold=30/69).transform(data)
print(time.time()-t0)

X = X_idw.copy()
X
#
# Geographic features
#
Xg = sensingbee.feature_engineering.GeoFeatures(params='auto').transform(data.geodata ,places=data.metadata)
X = X.join(Xg, on='Sensor Name')
X = X.sample(frac=1)




#
# Model fitting
#
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
estimator = GradientBoostingRegressor(max_depth=4)#RandomForestRegressor()
tuning_conf = {'params':{'n_estimators':[10, 200]},
                    'iterations': 2, 'cv':5, 'scoring':'r2'}
m = sensingbee.ml_modeling.Model(estimator, tuning_conf).fit(X, y.loc[X.index])
print('R² = ',m.base_estimator.best_score_)
m.feature_importances_


#
# Temporal (STL) features prototyping
#
import pandas as pd
from stldecompose import decompose

# Global STL
ts = y.groupby('Timestamp').median()
ts.index = pd.to_datetime(ts.index)
stl = decompose(ts, period=7)
Xt = pd.DataFrame()
Xt['NO2_trend'] = stl.trend['Value']
Xt['NO2_seasonal'] = stl.seasonal['Value']
Xt['NO2_diff'] = ts.diff().fillna(0)['Value']
Xt.index = X.index.get_level_values('Timestamp').unique()

idx = pd.IndexSlice
for s in X.index.get_level_values('Sensor Name').unique():
    x = X.loc[idx[s,:],'NO2']
    ts = x.reset_index('Sensor Name',drop=True)
    ts.index = pd.to_datetime(ts.index)
    print(ts.shape, s)
    stl = decompose(ts, period=7)
    X.loc[x.index, 'NO2_trend'] = stl.trend
    X.loc[x.index, 'NO2_seasonal'] = stl.seasonal
    X.loc[x.index, 'NO2_diff'] = ts.diff()

X.head()
