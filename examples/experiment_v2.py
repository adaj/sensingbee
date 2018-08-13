import sys
# sys.path.insert(0,'~/Repos/newcastle/air-quality/src')
sys.path.insert(0,'../src')
from v0 import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pylab as plt

SHAPE_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/'
DATA_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/data_30days/'
zx, zi = load_data__(DATA_FOLDER)
print('data loaded')


grid = {
    'freq': ['H','D'],
    'max_features': [0.25, 0.5, 0.75, 0.9],
    'feature_configuration': ['all','iwd']
}
variables =  {
   'sensors':['NO2','Temperature','O3','CO']
}

for freq in grid['freq']:
    zx, zi = resampling_sensors__(zx, zi, freq)

    if freq == 'H':
        variables['exogenous'] = ['primary','trunk','motorway','traffic_signals','hour','day','dow']
    else:
        variables['exogenous'] = ['primary','trunk','motorway','traffic_signals','day','dow']

    for feature_configuration in grid['feature_configuration']:

        if feature_configuration=='iwd':
            zxtmp = idw_features(zx, variables['sensors'])
            zxtmp = zxtmp.join(zx[variables['exogenous']])

        zxtmp = MinMaxScaler().fit_transform(zxtmp)
        zitmp = np.log(zi['Value'].values)

        for max_features in grid['max_features']:
            model = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=max_features)
            train_sizes, train_scores, test_scores = learning_curve(
                                                        model, zxtmp, zitmp, scoring='r2', n_jobs=-1,
                                                        cv=ShuffleSplit(n_splits=10, test_size=0.1, random_state=0),
                                                        train_sizes=np.linspace(.1, 1.0, 4))
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.3,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.3, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
            plt.legend(loc="best")
            plt.title('Random Forest Learning Curves', fontsize=18)
            plt.xlabel('n_estimators={} max_depth={}\nfeature_configuration={}'.format(200,5,'0.75-None'))
            plt.tight_layout()
            plt.savefig('{}/rf/rf-200_5--{}-{}-{}.png'.format(DATA_FOLDER,freq,int(max_features*100),feature_configuration)
            plt.show()
