import sensingbee.source as sb

configuration__ = {
    'DATA_FOLDER':'/home/adelsondias/Repos/newcastle/air-quality/data_6m/',
    'SHAPE_PATH':'/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp',
    'neighborhoods_path': '/home/adelsondias/Repos/newcastle/air-quality/shape/Lower_Layer_Super_Output_Areas_December_2011_Full_Extent__Boundaries_in_England_and_Wales/Lower_Layer_Super_Output_Areas_December_2011_Full_Extent__Boundaries_in_England_and_Wales.shp',
    'deprivation_path':'/home/adelsondias/Downloads/deprivation.csv',
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
import time
t0 = time.time()

sensors = sb.Sensors(configuration__, mode='load', path=configuration__['DATA_FOLDER'])
print('sensors ok {}'.format(time.time()-t0))
geography = sb.Geography(configuration__, 'load', sensors)
print('geography ok {}'.format(time.time()-t0))
sensors.delimit_sensors_by_geography(geography.city)
features = sb.Features(configuration__, mode='make', Sensors=sensors, Geography=geography)
print('features ok {}'.format(time.time()-t0))
#bee = sb.Bee(configuration__).fit(mode='make', verbose=True)
