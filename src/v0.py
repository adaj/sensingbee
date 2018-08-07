import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import fiona

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

def load_data(SHAPE_FOLDER, DATA_FOLDER):
    lsoa = gpd.read_file(SHAPE_FOLDER+'Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp')
    lsoa = lsoa[lsoa['msoa11nm'].str.contains('Newcastle upon Tyne')]
    lsoa = lsoa.to_crs(fiona.crs.from_epsg(4326))
    lsoa.crs = {'init': 'epsg:4326', 'no_defs': True}

    metadata = pd.read_csv(DATA_FOLDER+'sensors.csv',index_col='name')
    metadata = gpd.GeoDataFrame(metadata[['type','active','lon','lat']],
                            geometry=[shapely.geometry.Point(xy) for xy in zip(metadata['lon'], metadata['lat'])],
                            crs={'init': 'epsg:4326'}).to_crs(fiona.crs.from_epsg(4326))
    metadata = gpd.sjoin(metadata, lsoa, how='inner' ,op='intersects')[['type','active','lon','lat','geometry']]

    sfeat = pd.read_csv(DATA_FOLDER+'street_features_newcastle.csv',index_col=0)
    sfeat = sfeat.loc[metadata.index]

    sensors = pd.read_csv(DATA_FOLDER+'data.csv')
    sensors['Timestamp'] = pd.to_datetime(sensors['Timestamp'])
    sensors = sensors.set_index(['Variable','Sensor Name','Timestamp'])

    return metadata, sfeat, sensors

def resampling_sensors(sensors, metadata, variables, freq): # by region, variables and frequency
    idx = pd.IndexSlice
    level_values = sensors.index.get_level_values
    sensors = (sensors.groupby([level_values(i) for i in [0,1]]
                       +[pd.Grouper(freq=freq, level=-1)]).median())
    sensors = sensors.loc[idx[variables['sensors'],metadata.index,:],:]
    metadata = metadata.loc[sensors.index.get_level_values(1).unique()]
    return sensors, metadata

def ingestion(sensors, metadata, sfeat, variables, k, target, method):
    zxcols = []
    [zxcols.extend([[var]*k][0] + [["d_{}".format(var)]*k][0]) for var in variables['sensors']]
    idx = pd.IndexSlice
    zi = sensors.loc[idx[target,:,:],:]
    zx = pd.DataFrame(index=pd.MultiIndex.from_product([zi.index.get_level_values(1).unique(),zi.index.get_level_values(2).unique()],names=['Sensor Name','Timestamp']),
                      columns=zxcols)
    for s in zi.index.get_level_values(1).unique():
        si = metadata.loc[s]
        for t in zi.index.get_level_values(2).unique():
            for var in variables['sensors']:
                sdf = sensors.loc[idx[var,:,t],:] # sensors of the var variable at  time t
                mdf = metadata.loc[sdf.index.get_level_values(1).unique()] # metadata about them

                dij = mdf['geometry'].apply(lambda x: si['geometry'].distance(x)).sort_values() # nearest measures for (var,t)
                dij = dij.loc[(dij.index!=si.name) & (dij>0)] # excluding the sensor si
                if method=='nn':
                    dij = dij[:k]
                elif method=='randomized':
                    dij = dij[:10].sample(k, random_state=0)
                zj = sdf.loc[idx[:,dij.index,:],:].values.reshape(1,-1)[0]
                zx.loc[idx[si.name,t],'d_{}'.format(var)] = dij.values
                zx.loc[idx[si.name,t],var] = zj
    if 'hour' in variables['exogenous']:
        zx['hour'] = zx.index.get_level_values(1).hour
    if 'dow' in variables['exogenous']:
        zx['dow'] = zx.index.get_level_values(1).dayofweek
    if 'day' in variables['exogenous']:
        zx['day'] = zx.index.get_level_values(1).day
    if 'month' in variables['exogenous']:
        zx['month'] = zx.index.get_level_values(1).month
    if 'year' in variables['exogenous']:
        zx['year'] = zx.index.get_level_values(1).year

    if sfeat is not None:
        try:
            zx = zx.reset_index([1]).join(sfeat).set_index('index',append=True).fillna(0)
        except:
            print('Warning: \'street\' variables are not available in metadata')
            pass

    zi = zi.loc[target]
    zx = zx.loc[zi.index]
    zi = zi.loc[idx[zx.index.get_level_values(0).unique(),zx.index.get_level_values(1).unique()],:]

    return zx, zi

def osm_features(lines, points, metadata, conf):
    lines = lines[lines['highway'].isin(conf['lines'])]
    points = points[points['highway'].isin(conf['points'])]

    if conf['method'] == 'count_nn':
        ldf = pd.DataFrame(columns=conf['lines'])
        pdf = pd.DataFrame(columns=conf['points'])
        for m in range(metadata.shape[0]):
            i = metadata.iloc[m]
            # distance of 0.0011 = 100m radius
            lfeatures = lines[lines['geometry'].apply(lambda x: x.distance(i['geometry']))<conf['radius']]['highway'].value_counts().reindex(lines['highway'].unique(), fill_value=0)
            ldf.loc[i.name] = lfeatures[l]

            pfeatures = points[points['geometry'].apply(lambda x: x.distance(i['geometry']))<conf['radius']]['highway'].value_counts().reindex(points['highway'].unique(), fill_value=0)
            pdf.loc[i.name] = pfeatures[p]
        return pd.concat([ldf,pdf],axis=1)

    elif conf['method'] == 'distance':
        df = pd.DataFrame(columns=conf['lines']+conf['points'])

        for m in range(metadata.shape[0]):
            i = metadata.iloc[m]

            d = {}
            for key in conf['lines']:
                d[key] = 1/lines.loc[(lines['highway']==key) | (lines['highway']=='{}_link'.format(key)),'geometry'].apply(lambda x: x.distance(i['geometry'])).min()
            for key in conf['points']:
                d[key] = 1/points.loc[(points['highway']==key),'geometry'].apply(lambda x: x.distance(i['geometry'])).min()
            df.loc[i.name] = d
        return df


def idw_features(zx, sensor_variables):
    zidw = pd.DataFrame(index=zx.index)
    for var in sensor_variables:
        zidw['idw_{}'.format(var)] = (((zx['d_{}'.format(var)].values*zx[var].values).sum(axis=1))/zx['d_{}'.format(var)].values.sum(axis=1))
    return zidw

def spavg_features(zx, sensor_variables):
    zavg = pd.DataFrame(index=zx.index)
    for var in sensor_variables:
        zavg['spavg_{}'.format(var)] = zx[var].values.sum(axis=1)/zx[var].shape[1]
    return zavg

def nn_features(zx, sensor_variables):
    znn = pd.DataFrame(index=zx.index)
    for var in sensor_variables:
        znn['nn_{}'.format(var)] = zx[['d_{}'.format(var),var]].apply(lambda x: x[var].values[x['d_{}'.format(var)].values.argmin()], axis=1)
    return znn

def mlp(x, y, it, get_estimator):
    paramsmlp = {
        'hidden_layer_sizes':[(3,1),(3,5),(5,1),(5,5),(10,1),(10,3),(10,5),(20,3),(20,5),(20,10),(40,20),(40,40)],
        'activation':['identity','relu','logistic'],
        'alpha':np.logspace(0.0001,0.1,50)
    }

    grid = RandomizedSearchCV(MLPRegressor(max_iter=10000), param_distributions=paramsmlp,
                        n_iter=it, scoring='r2', n_jobs=-1, cv=5).fit(x, y) # zx.values, np.ravel(y.values)
    if get_estimator:
        return grid.best_score_, grid.best_estimator_
    else:
        return grid.best_score_

def rf(x, y, it, get_estimator):
    paramsrf = {
        'n_estimators':np.arange(5,300,5),
        'max_features':np.arange(0.1, 1.01, 0.05),
        'max_depth':np.arange(1,20,2)
    }
    grid = RandomizedSearchCV(RandomForestRegressor(), param_distributions=paramsrf,
                        n_iter=it, scoring='r2', n_jobs=-1, cv=5).fit(x, y) # zx.values, np.ravel(y.values)
    if get_estimator:
        return grid.best_score_, grid.best_estimator_
    else:
        return grid.best_score_

def gb(x, y, it, get_estimator):
    paramsab = {
        'n_estimators':np.arange(5,300,5),
        'learning_rate':np.arange(0.01, 10.01, 0.1),
        'loss': ['ls','lad','huber','quantile'],
    }
    grid = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions=paramsgb,
                        n_iter=it, scoring='r2', n_jobs=-1, cv=5).fit(x, y) # zx.values, np.ravel(y.values)
    if get_estimator:
        return grid.best_score_, grid.best_estimator_
    else:
        return grid.best_score_
