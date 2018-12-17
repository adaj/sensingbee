import pandas as pd
import numpy as np
import geopandas as gpd
import shapely, fiona

def load_newcastle_sensors_csv(DATA_FOLDER):
    data = pd.read_csv('{}/data.csv'.format(DATA_FOLDER).replace('//','/'))
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index(['Variable','Sensor Name','Timestamp'], inplace=True)
    metadata = pd.read_csv('{}/sensors.csv'.format(DATA_FOLDER))
    metadata['geometry'] = metadata.apply(lambda x: shapely.geometry.Point(x['lon'],x['lat']),axis=1)
    metadata = gpd.GeoDataFrame(metadata, crs={'init': 'epsg:4326'})
    metadata.set_index('name', inplace=True)
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    metadata = metadata.to_crs(fiona.crs.from_epsg(4326))
    return data, metadata
# data, metadata = load_newcastle_sensors_csv('data/')

def load_newcastle_city_shapefile(DATA_PATH):
    city = gpd.read_file(DATA_PATH)
    city = city.loc[city[city.columns[2]].str.contains('Newcastle upon Tyne')]
    city = city.to_crs(fiona.crs.from_epsg(4326))
    return city
# path = '/home/adelsondias/Repos/sensingbee/sensingbee2/data/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales.shp'
# city = load_newcastle_city_shapefile(path)
