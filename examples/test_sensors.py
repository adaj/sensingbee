import sensingbee.source as sb
import sensingbee.utils as sbutils
import pandas as pd

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

bee = sb.Bee(configuration__).fit(mode='load', verbose=True)
bee.train(variables=['NO2'])
bee.scores

w = sb.Sensors(
    configuration__,
    mode='get', path={
        'start_time':'2018-05-10',
        'end_time':'2018-05-10',
        'url': 'https://api.newcastle.urbanobservatory.ac.uk/api/v1/sensors/data/raw.csv'
    }, delimit_geography=bee.geography, delimit_quantiles=True
)
wf = sb.Features(configuration__, mode='make',
            Sensors=w, Geography=bee.geography, save=False)

# f and g for the same timestamp should be the same
f = wf.get_train_features('NO2') # from get
g = bee.features.get_train_features('NO2') # from training data

idx = pd.IndexSlice
f['y'].head()
g['y'].loc[idx[:,'2018-05-10'],:].head()

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
p = bee.models['NO2'].regressor.predict(X=MinMaxScaler().fit_transform(f['X']))
r2_score(f['y']['Value'].ravel(), p)
