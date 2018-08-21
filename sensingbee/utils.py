import pandas as pd
import numpy as np


def ingestion2(Sensors, variables, k=5, osmf=None):
    # this function is very slow and is needed to produce zx.csv and zi.csv
    # for the first time.
    idx = pd.IndexSlice
    sens_names = Sensors.data.index.get_level_values(1).unique()
    sens_times = Sensors.data.index.get_level_values(2).unique()
    zxcols = []
    for var in variables:
        [zxcols.append(var) for i in range(k)]
        [zxcols.append('d_{}'.format(var)) for i in range(k)]
    zx = pd.DataFrame(index=pd.MultiIndex.from_product([sens_names,sens_times],names=['Sensor Name','Timestamp']),
                      columns=zxcols)
    for s in sens_names:
        si = Sensors.sensors.loc[s]
        for t in sens_times:
            for var in variables:
                sdf = Sensors.data.loc[idx[var,:,t]] # data of the var variable at  time t
                mdf = Sensors.sensors.loc[sdf.index.get_level_values(1).unique()] # sensors about them
                try:
                    dij = mdf['geometry'].apply(lambda x: si['geometry'].distance(x)).sort_values()
                    dij = dij.loc[(dij.index!=si.name) & (dij<0.11)].sample(k, random_state=0) #0.11 = 10km
                    dij = sdf.loc[idx[var,dij.index,t],:].join(dij)
                    zx.loc[idx[si.name,t],'d_{}'.format(var)] = dij['geometry'].values
                    zx.loc[idx[si.name,t],var] = dij['Value'].values
                except:
                    print('erro in ',s,t,var)
                    pass
    if Sensors.data.index.get_level_values(2).freq == 'H':
        zx['hour'] = zx.index.get_level_values(1).hour
        zx['dow'] = zx.index.get_level_values(1).dayofweek
        zx['day'] = zx.index.get_level_values(1).day
    elif Sensors.data.index.get_level_values(2).freq == 'D':
        zx['dow'] = zx.index.get_level_values(1).dayofweek
        zx['day'] = zx.index.get_level_values(1).day
    if osmf is not None:
        zx = zx.reset_index(level=1).join(osmf).set_index('Timestamp', append=True)
    return zx, Sensors.data


def mesh_ingestion(Sensors, osm_features, variables, mesh, timestamp):
    idx = pd.IndexSlice
    zxcols = []
    for var in variables['sensors']:
        [zxcols.append(var) for i in range(5)]
        [zxcols.append('d_{}'.format(var)) for i in range(5)]
    zmesh = pd.DataFrame(index=mesh.index, columns=zxcols)
    timestamp = pd.to_datetime(timestamp)
    for m in range(mesh.shape[0]):
        i = mesh.iloc[m]
        for var in variables['sensors']:
            st = Sensors.data.loc[idx[var,:,timestamp]]
            closest = Sensors.sensors.loc[st.index.get_level_values(1),'geometry'].apply(lambda x: x.distance(i['geometry'])).nsmallest(5)
            closest = st.loc[idx[var,closest.index,timestamp],:].join(closest)
            zmesh.loc[i.name,var] = closest['Value'].values
            zmesh.loc[i.name,"d_{}".format(var)] = closest['geometry'].values
    zmesh['hour'] = timestamp.hour
    zmesh['day'] = timestamp.day
    zmesh['dow'] = timestamp.dayofweek
    zmesh[['primary','trunk','motorway','traffic_signals']] = osm_features
    return zmesh
