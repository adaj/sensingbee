import sys
# sys.path.insert(0,'~/Repos/newcastle/air-quality/src/')
sys.path.insert(0,'../src')
from v0 import *

SHAPE_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/'
DATA_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/data_7days/'
metadata0, sfeat, sensors0 = load_data(SHAPE_FOLDER, DATA_FOLDER)
print('data loaded')

ml_iterations = 10
variables =  {
        'sensors':['NO2','Temperature','O3','CO','PM10'],
        'exogenous':['street','day','dow']
}

scores = pd.DataFrame(index=pd.MultiIndex.from_product([['D','H'],['none','iwd','savg','nn'],['ys', 'ns', 'mys', 'mns']], names=['freq','transf','feat_conf']) ,columns=['rf','ab','mlp'])
for freq in ['D','H']:

    sensors, metadata = resampling_sensors(sensors0, metadata0, variables, freq)

    for feat_conf in ['ys', 'ns', 'mys', 'mns']:
        if feat_conf == 'ys':
            variables =  {
                'sensors':['NO2','Temperature','O3','CO','PM10'],
                'exogenous':['street','day','dow']
            }
        elif feat_conf == 'ns':
            variables =  {
                'sensors':['NO2','Temperature','O3','CO','PM10'],
                'exogenous':['street','day','dow']
            }
        elif feat_conf == 'mys':
            variables =  {
                'sensors':['NO2'],
                'exogenous':['street','day','dow']
            }
        elif feat_conf == 'mns':
            variables =  {
                'sensors':['NO2'],
                'exogenous':['day','dow']
            }
        if freq=='H':
            variables['exogenous'].append('hour')

        zx, zi = ingestion(sensors, metadata, sfeat, variables, 5, 'NO2', 'randomized')

        for transf in ['none', 'iwd', 'savg', 'nn']:
            print(freq, transf)

            if transf == 'iwd':
                zxtmp = iwd_features(zx, variables['sensors'])
                zxtmp = zxtmp.join(zx[zx.columns[-11:]])
            elif transf == 'savg':
                zxtmp = spavg_features(zx, variables['sensors'])
                zxtmp = zxtmp.join(zx[zx.columns[-11:]])
            elif transf == 'nn':
                zxtmp = nn_features(zx, variables['sensors'])
                zxtmp = zxtmp.join(zx[zx.columns[-11:]])
            else:
                zxtmp = zx

            best = {}
            best['rf'] = rf(zx.values, np.ravel(zi.values), ml_iterations, False)
            best['ab'] = ab(zx.values, np.ravel(zi.values), ml_iterations, False)
            best['mlp'] = mlp(zx.values, np.ravel(zi.values), int(ml_iterations), False)
            scores.loc[(freq,transf,feat_conf)] = best

scores.to_csv(DATA_FOLDER+'results_experiments_NO2_freq-transf-featconf-ml.csv')
