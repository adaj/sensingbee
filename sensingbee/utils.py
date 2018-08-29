import pandas as pd
import numpy as np


def ingestion2(Sensors, variables, k=5, osmf=None):
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

def mesh_ingestion(Sensors, meshgrid, variables, timestamp):
    idx = pd.IndexSlice
    timestamp = pd.to_datetime(timestamp)
    zxcols = []
    for var in variables:
        [zxcols.append(var.split('.')[0]) for i in range(5)]
        [zxcols.append('d_{}'.format(var.split('.')[0])) for i in range(5)]
    zmesh = pd.DataFrame(index=meshgrid.index, columns=zxcols)
    def loop_var(i):
        values = []
        for var in variables:
            st = Sensors.data.loc[idx[var,:,timestamp]]
            closest = Sensors.sensors.loc[st.index.get_level_values(1),'geometry'].apply(lambda x: x.distance(i['geometry'])).nsmallest(5)
            closest = st.loc[idx[var,closest.index,timestamp],:].join(closest, on='Sensor Name')

            values += list(closest['Value'].values)
            values += list(closest['geometry'].values)
        zmesh.loc[i.name] = values
    meshgrid.apply(lambda x: loop_var(x), axis=1)

    zmesh['dow'] = timestamp.dayofweek
    zmesh['day'] = timestamp.day
    zmesh = zmesh.join(meshgrid[meshgrid.columns[~meshgrid.columns.isin(['lat','lon','geometry'])]])
    return zmesh
