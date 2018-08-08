import sys
# sys.path.insert(0,'~/Repos/newcastle/air-quality/src/')
sys.path.insert(0,'../src')
from v0 import *
from sklearn.preprocessing import RobustScaler

SHAPE_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/'
DATA_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/data_allsensors_8days/'
metadata0, sfeat, sensors0 = load_data(SHAPE_FOLDER, DATA_FOLDER)
print('data loaded')

ml_iterations = 5
variables =  {
        'sensors':['NO2','Temperature','O3','CO','PM10'],
        'exogenous':['street','day','dow','hour'] #'street' = 'primary','trunk','motorway','traffic_signals'
}

scores = pd.DataFrame(index=pd.MultiIndex.from_product([['D','H'],['none','iwd','savg','nn'],['multi_exo', 'multi_endo', 'mono_exo', 'mono_endo']], names=['freq','transf','feat_conf']) ,columns=['rf','ab','mlp'])

# for freq in ['D','H']:
freq = 'H'
sensors, metadata = resampling_sensors(sensors0, metadata0, variables, freq)
zx, zi = ingestion(sensors, metadata, sfeat, variables, 5, 'NO2', 'randomized')

zx.head()


for feat_conf in ['multi_exo', 'multi_endo', 'mono_exo', 'mono_endo']:

    if feat_conf == 'multi_exo':
        variables =  {
                'sensors':['NO2','Temperature','O3','CO','PM10'],
                'exogenous':['primary','trunk','motorway','traffic_signals','day','dow']
        }
    elif feat_conf == 'multi_endo':
        variables =  {
                'sensors':['NO2','Temperature','O3','CO','PM10'],
                'exogenous':['day','dow']
        }
    elif feat_conf == 'mono_exo':
        variables =  {
                'sensors':['NO2'],
                'exogenous':['primary','trunk','motorway','traffic_signals','day','dow']
        }
    elif feat_conf == 'mono_endo':
        variables =  {
                'sensors':['NO2'],
                'exogenous':['day','dow']
        }
    if freq=='H':
        variables['exogenous'].append('hour')

    for transf in ['none', 'iwd', 'savg', 'nn']:
        if transf == 'iwd':
            zxtmp = iwd_features(zx, variables['sensors'])
            zxtmp = zxtmp.join(zx[variables['exogenous']])
        elif transf == 'savg':
            zxtmp = spavg_features(zx, variables['sensors'])
            zxtmp = zxtmp.join(zx[variables['exogenous']])
        elif transf == 'nn':
            zxtmp = nn_features(zx, variables['sensors'])
            zxtmp = zxtmp.join(zx[variables['exogenous']])
        else:
            zxtmp = zx
        zxtmp = RobustScaler().fit_transform(zxtmp)

        best = {}
        best['rf'] = rf(zx.values[:200], np.ravel(zi.values)[:200], ml_iterations, False)
        print(feat_conf, transf, 'rf', best['rf'])
        best['ab'] = ab(zx.values[:200], np.ravel(zi.values)[:200], ml_iterations, False)
        print(feat_conf, transf, 'ab', best['ab'])
        best['mlp'] = mlp(zx.values[:200], np.ravel(zi.values)[:200], ml_iterations, False)
        print(feat_conf, transf, 'mlp', best['mlp'])

# scores.loc[(freq,transf,feat_conf)] = best

# scores.to_csv(DATA_FOLDER+'test.csv')