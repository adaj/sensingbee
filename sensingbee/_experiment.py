"""
Author: Adelson Araujo jr
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sensingbee.source as sbee

configuration__ = {
    'DATA_FOLDER':'/home/adelsondias/Repos/newcastle/air-quality/data_3m/',
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


bee = sbee.Bee(configuration__).fit(mode='load', verbose=True)

scores = pd.DataFrame(columns=['r2','mse','variable','model','features'])
for i in range(10):
    for var in ['NO2', 'Temperature', 'PM2.5']:
        print(var,i)
        f = bee.features.get_train_features(var)
        bee.train(variables=[var],
                  regressor=GradientBoostingRegressor(n_estimators=500, max_depth=5, max_features=None),
                  X=f['X'][['{}'.format(var),'d_{}'.format(var),'primary','trunk','motorway','residential','traffic_signals','crossing']], y=f['y'])
        scores.loc[var+'_gb_pe'+str(i)] = bee.scores[var]+(var,'gb','pe',)
        bee.train(variables=[var],
                  regressor=GradientBoostingRegressor(n_estimators=500, max_depth=5, max_features=None),
                  X=f['X'][['{}'.format(var),'d_{}'.format(var)]], y=f['y'])
        scores.loc[var+'_gb_p'+str(i)] = bee.scores[var]+(var,'gb','p',)
        bee.train(variables=[var],
                  regressor=GradientBoostingRegressor(n_estimators=500, max_depth=5, max_features=None),
                  X=f['X'], y=f['y'])
        scores.loc[var+'_gb_pes'+str(i)] = bee.scores[var]+(var,'gb','pes',)
        #
        bee.train(variables=[var],
                  regressor=RandomForestRegressor(n_estimators=500, max_depth=5, max_features=None),
                  X=f['X'][['{}'.format(var),'d_{}'.format(var),'primary','trunk','motorway','residential','traffic_signals','crossing']], y=f['y'])
        scores.loc[var+'_rf_pe'+str(i)] = bee.scores[var]+(var,'rf','pe',)
        bee.train(variables=[var],
                  regressor=RandomForestRegressor(n_estimators=500, max_depth=5, max_features=None),
                  X=f['X'][['{}'.format(var),'d_{}'.format(var)]], y=f['y'])
        scores.loc[var+'_rf_p'+str(i)] = bee.scores[var]+(var,'rf','p',)
        bee.train(variables=[var],
                  regressor=RandomForestRegressor(n_estimators=500, max_depth=5, max_features=None),
                  X=f['X'], y=f['y'])
        scores.loc[var+'_rf_pes'+str(i)] = bee.scores[var]+(var,'rf','pes',)

scores.to_csv(configuration__['DATA_FOLDER']+'results/experiment0.csv')
