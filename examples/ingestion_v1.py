import sys
sys.path.insert(0,'../src')
#from v0 import *
from cyv0 import *
import pandas as pd
import time

t0 = time.time()
SHAPE_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/shape/Middle_Layer_Super_Output_Areas_December_2011_Full_Extent_Boundaries_in_England_and_Wales/'
DATA_FOLDER = '/home/adelsondias/Repos/newcastle/air-quality/data_30days/'
metadata, osmf, sensors = load_data(SHAPE_FOLDER, DATA_FOLDER)
print('data loaded -',time.time()-t0)

variables =  {
	#'sensors':['NO2','Temperature','CO']
        'sensors':['NO2','Temperature','O3','PM2.5','NO','Pressure','Wind Direction'],
        'exogenous':['primary','trunk','motorway','construction','residential','traffic_signals','crossing','bus_stop','day','dow','hour']
}

sensors, metadata = resampling_sensors(sensors, metadata, variables, 'H')
print('sensors resampled -',time.time()-t0)
print(sensors.shape)
zx = ingestion2(sensors, metadata, variables['sensors'], 5, osmf)
zx.to_csv(DATA_FOLDER+'zx_extended0.csv')
sensors.to_csv(DATA_FOLDER+'zi_extended0.csv')
print('ingestion completed -',time.time()-t0)
