import sensingbee.source as sb

import matplotlib.pyplot as plt
import numpy as np

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

geography = sb.Geography(configuration__, mode='load')

sensors = sb.Sensors(configuration__, mode='load', path='/home/adelsondias/Repos/newcastle/air-quality/data_3m/',
                delimit_data_by_threshold=False)

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,8))
sensors.data.loc[('NO2','new_new_emote_2902')].hist(bins=10, ax=axes[0][0])
axes[0][0].set_title('new_new_emote_2902')
sensors.data.loc[('NO2','new_new_emote_2904')].hist(bins=10, ax=axes[0][1])
axes[0][1].set_title('new_new_emote_2904')
sensors.data.loc[('NO2','new_new_emote_2903')].hist(bins=10, ax=axes[1][0])
axes[1][0].set_title('new_new_emote_2903')
sensors.data.loc['NO2'].hist(bins=10, ax=axes[1][1])
axes[1][1].set_title('All NO2 sensors')
plt.tight_layout()
plt.savefig('no2_histograms.svg', eps=1000)

fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(7,5))
geography.city.plot(ax=axes, color='white', edgecolor='black', linewidth=2)
geography.lines.plot(ax=axes, linewidth=0.5)
sensors.sensors.plot(ax=axes, linewidth=0.5, color='red')
plt.axis('off')
plt.tight_layout()
plt.savefig('sensors.svg', eps=1000)
plt.show()
