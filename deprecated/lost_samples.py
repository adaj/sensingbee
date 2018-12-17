# import sensingbee as sb
import sys
sys.path.append('/home/adelsondias/Repos/sensingbee/sensingbee')
import source as sb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


bsensors = sb.Sensors(configuration__, mode='make', path=configuration__['DATA_FOLDER'], 
                      delimit_quantiles=True, delimit_data_by_threshold=True)
"""
print('Quantity of sensors')
print(bsensors.sensors.shape)
print('NO2:',bsensors.data.loc['NO2'].shape, bsensors.sensors.loc[bsensors.data.loc['NO2'].index.get_level_values(0)].drop_duplicates(['lat','lon']).shape)
print('Temperature:',bsensors.data.loc['Temperature'].shape, bsensors.sensors.loc[bsensors.data.loc['Temperature'].index.get_level_values(0)].drop_duplicates(['lat','lon']).shape)
print('PM2.5:',bsensors.data.loc['PM2.5'].shape, bsensors.sensors.loc[bsensors.data.loc['PM2.5'].index.get_level_values(0)].drop_duplicates(['lat','lon']).shape)

bsensors = sb.Sensors(configuration__, mode='make', path=configuration__['DATA_FOLDER'], 
                      delimit_quantiles=True, delimit_data_by_threshold=False)
print('Quantity of sensors with valid urbanicity')
print(bsensors.sensors.shape)



bsensors = sb.Sensors(configuration__, mode='make', path=configuration__['DATA_FOLDER'], 
                      delimit_quantiles=False, delimit_data_by_threshold=True)
print('Removing uncalibrated sensors (threshold: {\'NO2\':80, \'Temperature\':25, \'PM2.5\':15})')
print('NO2:',bsensors.data.loc['NO2'].shape, bsensors.sensors.loc[bsensors.data.loc['NO2'].index.get_level_values(0)].drop_duplicates(['lat','lon']).shape)
print('Temperature:',bsensors.data.loc['Temperature'].shape, bsensors.sensors.loc[bsensors.data.loc['Temperature'].index.get_level_values(0)].drop_duplicates(['lat','lon']).shape)
print('PM2.5:',bsensors.data.loc['PM2.5'].shape, bsensors.sensors.loc[bsensors.data.loc['PM2.5'].index.get_level_values(0)].drop_duplicates(['lat','lon']).shape)
"""
