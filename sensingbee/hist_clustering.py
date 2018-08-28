import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import fiona

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import seaborn as sns

configuration__ = {
    'DATA_FOLDER':'/home/adelsondias/Repos/newcastle/air-quality/data_30days/',
    'SHAPE_PATH':'/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp',
    'OSM_FOLDER':'/home/adelsondias/Downloads/newcastle_streets/',
    'VALIDREGIONS_FILE': '/home/adelsondias/Repos/newcastle/air-quality/data_30days/mesh_valid-regions.csv',
    'Sensors__frequency':'D',
    'Geography__filter_column':'msoa11nm',
    'Geography__filter_label':'Newcastle upon Tyne',
    'Geography__meshgrid':{'dimensions':[50,50], 'longitude_range':[-1.8, -1.5], 'latitude_range':[54.95, 55.08]},
    'variables_sensors': ['NO2','Temperature','O3','PM2.5','NO','Pressure'],
    'osm_line_objs': ['primary','trunk','motorway','residential'],
    'osm_point_objs': ['traffic_signals','crossing']
}
idx = pd.IndexSlice
sensors = pd.read_csv(configuration__['DATA_FOLDER']+'sensors.csv',index_col='name')
sensors = gpd.GeoDataFrame(sensors[['type','active','lon','lat']],
                        geometry=[shapely.geometry.Point(xy) for xy in zip(sensors['lon'], sensors['lat'])],
                        crs={'init': 'epsg:4326'}).to_crs(fiona.crs.from_epsg(4326))
data = pd.read_csv(configuration__['DATA_FOLDER']+'data.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data = data.set_index(['Variable','Sensor Name','Timestamp'])
data = data.loc[idx[configuration__['variables_sensors']],:]
sensors = sensors.loc[data.index.get_level_values(1).unique()]

def resample_by_frequency(frequency, data, sensors):
    idx = pd.IndexSlice
    level_values = data.index.get_level_values
    data = (data.groupby([level_values(i) for i in [0,1]]
                       +[pd.Grouper(freq=frequency, level=-1)]).median())
    data = data.loc[idx[:,sensors.index,:],:]
    sensors = sensors.loc[data.index.get_level_values(1).unique()]
    return data, sensors
data, sensors = resample_by_frequency('D', data, sensors)

data.head()
vdic = {
    'O3':50,
    'Temperature':25,
    'NO2':100,
    'PM2.5':15
}
off, dropped_sensors, dropped_data = {}, {}, {}
for var, threshold in vdic.items():
    if var!='Temperature':
        D = data.loc[var]['Value'].loc[data.loc[var]['Value']>0]
    else:
        D = data.loc[var]['Value']
    off[var] = []
    for i,si in enumerate(D.index.get_level_values(0).unique()):
        if D.loc[si].mean() > threshold:
            off[var].append((var,si))
    dropped_sensors[var] = len(off[var])/i
    dropped_data[var] = 1 - (data.drop(off[var]).loc[var].shape[0]/data.loc[var].shape[0])
    data = data.drop(off[var])


data2.shape
data.shape
data.drop(data.index[0][:2]).head()
data.loc['NO'].head()

data.index[0]
data.head()
