import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

import geohunter


class Data(object):
    def __init__(self, folder_path, geodata=False, grid_resolution=1):
        self.samples = pd.read_csv(os.path.join(folder_path,'samples.csv'),
                            index_col=0).set_index(['Variable','Sensor Name','Timestamp'])
        self.metadata = gpd.read_file(os.path.join(folder_path,'sensors')).set_index('Sensor Nam')
        self.metadata['lon'] = self.metadata.geometry.x
        self.metadata['lat'] = self.metadata.geometry.y
        self.city = gpd.read_file(os.path.join(folder_path,'city.geojson'))
        bbox = {'north':self.city.bounds.max().values[3],
                'east':self.city.bounds.max().values[2],
                'south':self.city.bounds.min().values[1],
                'west':self.city.bounds.min().values[0]}
        self.geodata = {}
        if geodata:
            print('@ Data: Requesting geodata from OSM ...')
            self.geodata = geohunter.features.Landmarks(self.city,
                            osm_folder=os.path.join(folder_path,'osm')).fit(**geodata)
            print('@ Data: geodata loaded!')
        self.grid = geohunter.features.Grid(grid_resolution).fit(self.city) #make_grid(self.city, resolution=)
