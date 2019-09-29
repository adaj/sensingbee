import pandas as pd
import os
import shapely.geometry
import geopandas as gpd


DATA_FOLDER = '/home/adelsondias/Repos/sensingbee/examples/newcastle/data/01jan-31mar'
OUTPUT_FOLDER = '/home/adelsondias/Repos/sensingbee/examples/newcastle/data/weekly'
SHAPEFILE = '/home/adelsondias/Downloads/Major_Towns_and_Cities_December_2015_Boundaries.geojson'
SAMPLE_FREQUENCY = 'W'
load = False

if load:
    data = pd.read_csv(os.path.join(OUTPUT_FOLDER,'samples.csv'), index_col=0)
else:
    data = pd.DataFrame()
    for file_name in os.listdir(DATA_FOLDER):
        if file_name.split('.')[-1]=='csv' and file_name!='sensors.csv':
            x = pd.read_csv(os.path.join(DATA_FOLDER,file_name))
            x = x.loc[x['Flagged as Suspect Reading']==False]
            x = x.loc[x['Flagged as Suspect Reading']==False]
            x.loc[:,'Timestamp'] = pd.to_datetime(x['Timestamp'])
            x = x.set_index(['Variable','Sensor Name','Timestamp'])['Value']
            level_values = x.index.get_level_values
            result = (x.groupby([level_values(i) for i in [0,1]]
                                  +[pd.Grouper(freq=SAMPLE_FREQUENCY, level=-1)]).median())
            data = data.append(result.reset_index())

meta = pd.read_csv(os.path.join(DATA_FOLDER,'sensors.csv'))
meta['geometry'] = meta.apply(lambda x: shapely.geometry.Point(x[['Sensor Centroid Longitude','Sensor Centroid Latitude']].values),axis=1)
meta = gpd.GeoDataFrame(meta)

shape = gpd.read_file(SHAPEFILE)
shape = shape.loc[shape['tcity15nm'].str.contains('Newcastle upon Tyne')]
shape.to_file(os.path.join(OUTPUT_FOLDER,'city.geojson'), driver='GeoJSON')

meta = gpd.sjoin(meta, shape, op='within')[meta.columns]
meta[['Sensor Name', 'Ground Height Above Sea Level',
    'Sensor Height Above Ground',
    'Broker Name', 'geometry']].to_file(os.path.join(OUTPUT_FOLDER,'sensors'))

data.loc[data['Sensor Name'].isin(meta['Sensor Name'].unique())].to_csv(os.path.join(OUTPUT_FOLDER,'samples.csv'))
