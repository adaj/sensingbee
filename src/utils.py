import pandas as pd
import numpy as np
import geopandas as gpd
import shapely, fiona

def resample_sensors_data(data, frequency):
    idx = pd.IndexSlice
    level_values = data.index.get_level_values
    data = (data.groupby([level_values(i) for i in [0,1]]
                       +[pd.Grouper(freq=frequency, level=-1)]).median())
    return data

def make_grid(bbox, dimensions=[30,30]):
        lon = np.linspace(float(bbox['west']), float(bbox['east']), dimensions[0])
        lat = np.linspace(float(bbox['south']), float(bbox['north']), dimensions[1])
        lonv, latv = np.meshgrid(lon,lat)
        grid = np.vstack([lonv.ravel(), latv.ravel()]).T
        grid = gpd.GeoDataFrame(grid, geometry=[shapely.geometry.Point(xy) for xy in grid],crs={'init': 'epsg:4326'})
        grid.rename(columns={0:'lon',1:'lat'}, inplace=True)
        grid = grid.to_crs(fiona.crs.from_epsg(4326))
        return grid, lonv, latv
