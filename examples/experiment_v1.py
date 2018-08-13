import sys
# sys.path.insert(0,'~/Repos/newcastle/air-quality/src/')
sys.path.insert(0,'../src')
from v0 import *
from sklearn.preprocessing import RobustScaler

SHAPE_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/'
DATA_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/data_30days/'
zx, zi = load_data__(DATA_FOLDER)
print('data loaded')

ml_iterations = 500

scores = pd.DataFrame(index=pd.MultiIndex.from_product([['H','D'],['Nan','pca_idw','idw','savg','nn']], names=['freq','transf']) , columns=['rf','gb','mlp'])

for freq in ['H','D']:
    print(freq)
    variables =  {
	   'sensors':['NO2','Temperature','O3','CO'],
        'exogenous':['primary','trunk','motorway','traffic_signals','day','dow','hour']
    }

    zx, zi = resampling_sensors__(zx, zi, freq)

    for transf in ['Nan','pca_idw','idw', 'savg', 'nn']:
        print(transf)
        if transf == 'idw' or transf == 'pca_idw':
            zxtmp = idw_features(zx, variables['sensors'])
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
        zitmp = np.sqrt(zi['Value'].values)

        if transf == 'pca_idw':
            pca = PCA(n_components=2)
            zxtmp = pca.fit_transform(zxtmp)
            #print('pca: ',pca.explained_variance_ratio_)

        best = {}
        best['rf'] = rf(zxtmp, zitmp, ml_iterations, False)
        #print(best['rf'])
        best['gb'] = gb(zxtmp, zitmp, ml_iterations, False)
        #print(best['gb'])
        best['mlp'] = mlp(zxtmp, zitmp, ml_iterations, False)
        #print(best['mlp'])
        scores.loc[(freq,transf)] = best

scores.to_csv(DATA_FOLDER+'results_NO2_freq-transf-ml.csv')
