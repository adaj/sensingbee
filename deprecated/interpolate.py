import sensingbee.source as sb

configuration__ = {
    'DATA_FOLDER':'/home/adelsondias/Repos/newcastle/air-quality/data_3m/',
    'SHAPE_PATH':'/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp',
    'Sensors__frequency':'W',
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

bee.interpolate(variables=['NO2'], data=sb.Sensors(
    configuration__,
    mode='get', path={
        'start_time':'2018-09-01',
        'end_time':'2018-09-01',
        'url': 'https://api.newcastle.urbanobservatory.ac.uk/api/v1/sensors/data/raw.csv'
    }, delimit_geography=bee.geography, delimit_quantiles=True
))
bee.plot(variable='NO2', timestamp='2018-09-02',vmin=0, vmax=150)
