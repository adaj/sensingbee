import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sys
sys.path.append('/home/adelsondias/Repos/sensingbee/src/')

import source, utils
import __utils
configuration = json.load(open('conf.json'))


geography = source.Geography(configuration, __utils.load_newcastle_city_shapefile(configuration['data_shapefile_path']))

data, metadata = __utils.load_newcastle_sensors_csv(configuration['data_csv_path'])
sensors = source.Sensors(configuration, data, metadata).fit_in_geo(geography.city)

features = source.Features(configuration)
X, y = features.ingest(sensors.data, sensors.metadata, 'NN', 5, 0)
# X, y = features.load(['NO2', 'PM2.5', 'Temperature'],' NN', 0)

model = source.Model(RandomForestRegressor())
model.fit(X.loc[y['NO2'].index], y['NO2'], cv_repetitions=10)
print(model.r2, model.mse)
