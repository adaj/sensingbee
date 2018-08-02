# Spatial interpolation on open sensors data

Recently, IoT systems have been implemented to absorb city's varaibles, such as traffic, air quality etc. To do so, the deployment of the sensors becomes a fundamental requirement, and the cost for scaling up becomes an obstacle.

The [Urban Observatory](http://uoweb1.ncl.ac.uk), in Newcastle University, has the biggest open sensor data in the world, but they still are not capable to describe the variables they collect across a large spatiality. Look at the figure above, these are the places with sensors in Newcastle upon Tyne.

![All sensors](https://github.com/adaj/air-quality/blob/master/data_7days/plots/all_sensors_newcastle.png?raw=true)

That is a huge amount of sensors spread around the city. Now, let's filter it on "air quality" ones.

![Air quality sensors](https://github.com/adaj/air-quality/blob/master/data_7days/plots/airquality_sensors_newcastle.png?raw=true)

When it comes to specific variables, less and less sensors are available due to the deployment of cost and maintance of such data gathering systems.

Thus, thic project aims to apply and develop models for spatial interpolation to spread the knowledge obtained by the sensors, through estimates, around regions where data is not collected. In air quality context, for instance, from the multiple variables collected, such as NO2, O3, temperature and humidity (and more), it's possible to build multivariate predictive models to estimate them in such zones and then expand the potential of the IoT infrastructure in the city. 

If you want to help or get in touch, contact me at adelsondias@live.com.


