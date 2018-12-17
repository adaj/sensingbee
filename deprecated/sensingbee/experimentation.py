import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import source as sb

class Experimentation(object):
    def __init__(self, variables, features_conf, regression_alg):
        self.variables = variables
        self.features_conf = features_conf
        self.regression_alg = regression_alg

    def fit(self, configuration__, mode):
        self.configuration__ = configuration__
        # declaring Bee
        self.geography = sb.Geography(configuration__, mode='load')
        self.sensors = sb.Sensors(configuration__, mode='load', path=configuration__['DATA_FOLDER'], delimit_geography=self.geography)
        self.features = sb.Features(configuration__, mode, Sensors=self.sensors, Geography=self.geography)

    def initiate(self, iterations, path_to_results):
        self.scores = []
        for var in self.variables:
            for fc in self.features_conf:
                for ra in self.regression_alg:
                    features = ['dow','day','week']
                    f = self.features.get_train_features(var)
                    if 'p' in fc:
                        features += [var]
                        features += ['d_'+var]
                    if 'u' in fc:
                        features += self.configuration__['osm_line_objs']+self.configuration__['osm_point_objs']
                    if 's' in fc:
                        features += [x for x in self.configuration__['Sensors__variables'] if x!= var]
                        features += ['d_'+x for x in self.configuration__['Sensors__variables'] if x!= var]
                    if 'd' in fc:
                        features += ['Index of Multiple Deprivation (IMD) Score', 'Income Score (rate)',
                                       'Employment Score (rate)', 'Education, Skills and Training Score',
                                       'Crime Score', 'Barriers to Housing and Services Score',
                                       'Living Environment Score',
                                       'Total population: mid 2012 (excluding prisoners)',
                                       'Population aged 16-59: mid 2012 (excluding prisoners)']

                    if ra == 'rf':
                        regressor = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=1)
                    elif ra == 'gb':
                        regressor = GradientBoostingRegressor(n_estimators=200, max_depth=5, max_features=1)

                    r2,mse = [],[]
                    for i in range(1,iterations+1):
                        m = sb.Model(regressor).fit(f['X'][features], f['y'])
                        r2.append(m.r2)
                        mse.append(m.mse)
                    self.scores.append([var,fc,ra,np.mean(r2),np.std(r2),np.mean(np.sqrt(mse)),np.std(np.sqrt(mse))])
                    # print('[{} - {} - {}] r2: {} ~ std: {}'.format(var, fc, ra, np.mean(r2), np.std(r2))
        self.scores = pd.DataFrame(self.scores, columns=['variable','feature_configuration','regressor','r2','r2_std','mse','mse_std'])
        self.scores.to_csv(path_to_results)
        return self

if __name__=='__main__':
    configuration__ = {
        'DATA_FOLDER':'/home/adelsondias/Repos/newcastle/air-quality/data_6m/',
        'SHAPE_PATH':'/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp',
        'Sensors__frequency':'D',
        'Sensors__variables': ['NO2','Temperature','PM2.5'],
        'Sensors__threshold_callibration': {'Temperature':25, 'NO2':80, 'PM2.5':15},
        'Geography__filter_column':'msoa11nm',
        'Geography__filter_label':'Newcastle upon Tyne',
        'Geography__meshgrid':{'dimensions':[50,50], 'longitude_range':[-1.8, -1.51], 'latitude_range':[54.96, 55.05]},
        'osm_bbox': '(54.96,-1.8,55.05,-1.51)',
        'osm_line_objs': ['primary','trunk','motorway','residential'],
        'osm_point_objs': ['traffic_signals','crossing']
    }
    e = Experimentation(variables=['NO2','Temperature','PM2.5'], features_conf=['p','ps','pu','pus','pusd'], regression_alg=['gb','rf'])

    e.fit(configuration__, mode='load')
    e.initiate(iterations=3,path_to_results='/home/adelsondias/Repos/sensingbee/examples/experiments/e_cd_6m.csv')
