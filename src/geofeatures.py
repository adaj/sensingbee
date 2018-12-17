import json
import urllib
import fiona, shapely
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import KernelDensity


def make_grid(bbox, dimensions=[30,30]):
        lon = np.linspace(float(bbox['west']), float(bbox['east']), dimensions[0])
        lat = np.linspace(float(bbox['south']), float(bbox['north']), dimensions[1])
        lonv, latv = np.meshgrid(lon,lat)
        grid = np.vstack([lonv.ravel(), latv.ravel()]).T
        grid = gpd.GeoDataFrame(grid, geometry=[shapely.geometry.Point(xy) for xy in grid],crs={'init': 'epsg:4326'})
        grid.rename(columns={0:'lon',1:'lat'}, inplace=True)
        return grid, lonv, latv


class OSMLeech(object):

    def __init__(self, bbox):
        self.bbox = '({},{},{},{})'.format(bbox['south'], bbox['west'], bbox['north'], bbox['east'])

    def points_by_key(self, key, items): #keys: 'amenity', 'building', 'highway', 'tourism', 'historic'
        query_string = ''.join(["node[\"{}\"=\"{}\"]{};way[\"{}\"=\"{}\"]{};relation[\"{}\"=\"{}\"]{};".format(key, i, self.bbox, key, i, self.bbox, key, i, self.bbox) for i in items]).replace("=\"*\"",'')
        query_string = "[out:json][timeout:50];({});out+geom;".format(query_string)
        result = json.loads(urllib.request.urlopen('http://overpass-api.de/api/interpreter?data='+query_string).read())
        points = []
        for x in result['elements']:
            elem = {}
            if x['type']=='node':
                lon, lat = x['lon'],x['lat']
            else:
                lon, lat = x['bounds']['minlon'], x['bounds']['minlat']
            elem['geometry'] = shapely.geometry.Point([lon,lat])
            elem['id'] = x['id']
            elem['tag'] = x['tags'][key]
            points.append(elem)
        points = gpd.GeoDataFrame(points,crs={'init': 'epsg:4326'}).to_crs(fiona.crs.from_epsg(4326))
        points.set_index('id', inplace=True)
        if '*' in items:
            points['tag'] = ['*']*points.shape[0]
        return points

    def lines_by_key(self, key, items, aspoints=False):
        query_string = ''.join(["way[\"{}\"=\"{}\"]{};".format(key, i, self.bbox) for i in items])
        query_string = "[out:json][timeout:50];({});out+geom;".format(query_string)
        result = json.loads(urllib.request.urlopen('http://overpass-api.de/api/interpreter?data='+query_string).read())
        lines = []
        for x in result['elements']:
            elem = {}
            try: # it must have at least ['lon', 'lat', 'id', 'tags']
                elem['geometry'] = shapely.geometry.LineString([(i['lon'],i['lat']) for i in x['geometry']])
                elem['id'] = x['id']
                elem['tag'] = x['tags'][key]
            except:
                continue
            lines.append(elem)
        lines = gpd.GeoDataFrame(lines)
        lines.set_index('id', inplace=True)
        if aspoints:
            l1, l2 = [], []
            for x in range(lines.shape[0]):
                tmp = [shapely.geometry.Point(i) for i in list(lines.iloc[x]['geometry'].coords)]
                l1 += tmp
                l2 += [lines.iloc[x]['tag']]*len(tmp)
            lines = gpd.GeoDataFrame({'geometry':l1, 'tag':l2})
        return lines


class GeoTransformer(object):

    def __init__(self, bbox, grid=None):
        self.leech = OSMLeech(bbox)
        if grid is None:
            self.grid = make_grid(bbox, dimensions=[30,30])[0]
        else:
            self.grid = grid

    def make_kde(self, key, points, bandwidth, kernel):
        points = points.copy(deep=True)
        points['lon'] = points['geometry'].apply(lambda x: x.coords.xy[0][0])
        points['lat'] = points['geometry'].apply(lambda x: x.coords.xy[1][0])
        kde_ = pd.DataFrame()
        for x in points['tag'].unique():
            px = points[points['tag']==x]
            kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(px[['lon','lat']]) # 1 mile
            kde_['kde_'+key+'_'+x] = pd.Series(np.exp(kde.score_samples(self.grid[['lon','lat']])),index=self.grid.index)
        kde_.index.name = 'place'
        return kde_

    def make_dist(self, key, objects):
        def min_dist(grid_point, objects):
            dis = []
            for j in range(len(px)):
                dis.append(grid_point.distance(objects.iloc[j]['geometry']))
            m = min(dis)
            return m
        dist = pd.DataFrame(index=self.grid.index)
        for x in objects['tag'].unique():
            px = objects[objects['tag']==x]
            dist['dist_'+key+'_'+x] = self.grid.apply(lambda x: min_dist(x['geometry'], px), axis=1)
        dist.index.name = 'place'
        return dist

    def extract_from_points(self, key, items, method, params):
        objs = self.leech.points_by_key(key, items)
        if method=='kde':
            return self.make_kde(key, objs, **params)
        elif method=='dist':
            return self.make_dist(key, objs)

    def extract_from_lines(self, key, items, method, params):
        if method=='kde':
            return self.make_kde(key, self.leech.lines_by_key(key, items, aspoints=True), **params)
        elif method=='dist':
            return self.make_dist(key, self.leech.lines_by_key(key, items, aspoints=False))
