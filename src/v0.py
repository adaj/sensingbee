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

   
    osmf = pd.read_csv(DATA_FOLDER+'street_features_newcastle.csv',index_col=0)
    osmf = osmf.loc[metadata.index]
   
    sensors = pd.read_csv(DATA_FOLDER+'data.csv')
    sensors['Timestamp'] = pd.to_datetime(sensors['Timestamp'])
    sensors = sensors.set_index(['Variable','Sensor Name','Timestamp'])

    return metadata, osmf, sensors

def load_data__(DATA_FOLDER):
    zx = pd.read_csv(DATA_FOLDER+'zx.csv')
    zx['Timestamp'] = pd.to_datetime(zx['Timestamp'])
    zx.set_index(['Sensor Name','Timestamp'], inplace=True)
    zx.rename(columns={key:key.split('.')[0] for key in zx.columns if '.' in key}, inplace=True)
    zi = pd.read_csv(DATA_FOLDER+'zi.csv')
    zi['Timestamp'] = pd.to_datetime(zi['Timestamp'])
    zi.set_index(['Sensor Name','Timestamp'], inplace=True)
    return zx, zi

def resampling_sensors(sensors, metadata, variables, freq): # by region, variables and frequency
    idx = pd.IndexSlice
    level_values = sensors.index.get_level_values
    sensors = (sensors.groupby([level_values(i) for i in [0,1]]
                       +[pd.Grouper(freq=freq, level=-1)]).median())
    sensors = sensors.loc[idx[variables['sensors'],metadata.index,:],:]
    metadata = metadata.loc[sensors.index.get_level_values(1).unique()]
    return sensors, metadata

def resampling_sensors__(zx, zi, freq):
    if freq==zi.index.get_level_values(1).freq:
        return zx, zi
    idx = pd.IndexSlice
    level_values = zx.index.get_level_values
    zx = (zx.groupby([level_values(i) for i in [0]]
                       +[pd.Grouper(freq=freq, level=-1)]).median())
    zi = (zi.groupby([level_values(i) for i in [0]]
                       +[pd.Grouper(freq=freq, level=-1)]).median())
    if freq=='D' or freq=='W':
        zx.drop('hour',axis=1,inplace=True)
    return zx, zi

def ingestion(sensors, metadata, sfeat, variables, k, target, method):
    idx = pd.IndexSlice
    
    zxcols = []
    for var in variables:
        [zxcols.append(var) for i in range(k)]
        [zxcols.append('d_{}'.format(var)) for i in range(k)]
    zi = sensors.loc[idx[target,:,:],:]
    zx = pd.DataFrame(index=pd.MultiIndex.from_product([zi.index.get_level_values(1).unique(),zi.index.get_level_values(2).unique()],names=['Sensor Name','Timestamp']),
                      columns=zxcols)
    for s in zi.index.get_level_values(1).unique():
        si = metadata.loc[s]
        for t in zi.index.get_level_values(2).unique():
            for var in variables['sensors']:
                sdf = sensors.loc[idx[var,:,t],:] # sensors of the var variable at  time t
                mdf = metadata.loc[sdf.index.get_level_values(1).unique()] # metadata about them

                try:
                    dij = mdf['geometry'].apply(lambda x: si['geometry'].distance(x)).sort_values() # nearest measures for (var,t)
                    dij = dij.loc[(dij.index!=si.name)] # excluding the sensor si
                    if method=='nn':
                        dij = dij[:k]
                    elif method=='randomized':
                        dij = dij[:10].sample(k, random_state=0)
                except:
                    print('erro in ',s,t,var)
                    continue
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
            zx = zx.reset_index([1]).join(sfeat).set_index('Timestamp',append=True).fillna(0)
        except:
            print('Warning: \'street\' variables are not available in metadata')
            pass

    zi = zi.loc[target]
    zx = zx.loc[zi.index]
    zi = zi.loc[idx[zx.index.get_level_values(0).unique(),zx.index.get_level_values(1).unique()],:]

    return zx, zi

def ingestion2(sensors, metadata, variables, osmf=None):
    idx = pd.IndexSlice
    sens_names = sensors.index.get_level_values(1).unique()
    sens_times = sensors.index.get_level_values(2).unique()
    zxcols = []
    for var in variables:
        [zxcols.append(var) for i in range(k)]
        [zxcols.append('d_{}'.format(var)) for i in range(k)]
    zx = pd.DataFrame(index=pd.MultiIndex.from_product([sens_names,sens_times],names=['Sensor Name','Timestamp']),
                      columns=zxcols)
    for s in sens_names:
        si = metadata.loc[s]
        for t in sens_times:
            for var in variables:
                sdf = sensors.loc[idx[var,:,t]] # sensors of the var variable at  time t
                mdf = metadata.loc[sdf.index.get_level_values(1).unique()] # metadata about them
                try:
                    dij = mdf['geometry'].apply(lambda x: si['geometry'].distance(x)).sort_values()
                    dij = dij.loc[(dij.index!=si.name) & (dij<0.11)].sample(k, random_state=0)
                    dij = sdf.loc[idx[var,dij.index,t],:].join(dij)
                    zx.loc[idx[si.name,t],'d_{}'.format(var)] = dij['geometry'].values
                    zx.loc[idx[si.name,t],var] = dij['Value'].values
                except:
                    print('erro in ',s,t,var)
                    pass
    if sensors.index.get_level_values(2).freq == 'H':
        zx['hour'] = zx.index.get_level_values(1).hour
        zx['dow'] = zx.index.get_level_values(1).dayofweek
        zx['day'] = zx.index.get_level_values(1).day
    elif sensors.index.get_level_values(2).freq == 'D':
        zx['dow'] = zx.index.get_level_values(1).dayofweek
        zx['day'] = zx.index.get_level_values(1).day
    try:
        if osmf is not None:
            zx = zx.reset_index(level=1).join(osmf).set_index('Timestamp', append=True)         
    except:
        print('Warning: osm features are not available in metadata')
    return zx

def osm_features(STREET_SHAPE_FOLDER, LSOA_SHAPE_FOLDER, metadata, conf):
    lines = gpd.read_file(STREET_SHAPE_FOLDER+'newcastle_streets.shp')
    points = gpd.read_file(STREET_SHAPE_FOLDER+'newcastle_points.shp')

    lines = lines[lines['highway'].isin(conf['lines'])]
    points = points[points['highway'].isin(conf['points'])]

    lsoa = gpd.read_file(LSOA_SHAPE_FOLDER+'Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp')
    lsoa = lsoa[lsoa['msoa11nm'].str.contains('Newcastle upon Tyne')]
    lsoa = lsoa.to_crs(fiona.crs.from_epsg(4326))
    lsoa.crs = {'init': 'epsg:4326', 'no_defs': True}

    lines = gpd.sjoin(lines, lsoa, how='inner' ,op='intersects')[lines.columns]
    points = gpd.sjoin(points, lsoa, how='inner' ,op='intersects')[points.columns]

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

def rfe(zx, zi, n=None, step=1):
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor
    selector = RFE(RandomForestRegressor(), n_features_to_select=n, step=step)
    selector.fit(zx, zi['Value'])
    try:
        sel = pd.DataFrame(columns=zx.columns)
        sel.loc['support']= selector.support_
        sel.loc['ranking'] = selector.ranking_
        cols = sel.loc['support'].reset_index()['support']
        zxtmp = zx.iloc[:,cols[cols].index]
    except:
        zxtmp = selector.transform(zx)
    return zxtmp

def poly_rfe(zx, zi, n=30, step=50):
    from sklearn.preprocessing import PolynomialFeatures
    zxtmp = PolynomialFeatures(degree=2).fit_transform(zx)
    zxtmp = rfe(zxtmp,zi,n=n,step=step)
    return zxtmp2

def mlp(x, y, it, get_estimator):
    paramsmlp = {
        'hidden_layer_sizes':[(5,5),(10,5),(20,10),(40,20),(40,40),(10,5,5),(20,10,10),(50,30,20),(30,20,20,10),(15,10,5,5)],
        'activation':['relu'],
        'alpha':np.linspace(1e-7,1e-3,50)
    }

    grid = RandomizedSearchCV(MLPRegressor(max_iter=10000, solver='sgd'), param_distributions=paramsmlp,
                        n_iter=it, scoring='r2', n_jobs=-1, cv=5).fit(x, y)
    if get_estimator:
        return grid.best_score_, grid.best_estimator_
    else:
        return grid.best_score_

def rf(x, y, it, get_estimator):
    paramsrf = {
        'n_estimators':np.arange(50,300,5),
        'max_features':np.arange(0.1, 1.01, 0.05),
        'max_depth':np.arange(1,15,1)
    }
    grid = RandomizedSearchCV(RandomForestRegressor(), param_distributions=paramsrf,
                        n_iter=it, scoring='r2', n_jobs=-1, cv=5).fit(x, y)
    if get_estimator:
        return grid.best_score_, grid.best_estimator_
    else:
        return grid.best_score_

def gb(x, y, it, get_estimator):
    paramsgb = {
        'n_estimators':np.arange(50,300,5),
        'learning_rate':np.arange(0.01, 10.01, 0.1),
        'loss': ['ls','lad','huber','quantile'],
    }
    grid = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions=paramsgb,
                        n_iter=it, scoring='r2', n_jobs=-1, cv=5).fit(x, y)
    if get_estimator:
        return grid.best_score_, grid.best_estimator_
    else:
        return grid.best_score_
