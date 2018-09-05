# sensingbee

Spatial interpolation for sensors data :earth_americas:

## Purpose

Recently, IoT systems have been implemented to absorb city's variables, such as traffic, air quality etc. To do so, the deployment of a network of sensors becomes a fundamental requirement, and the cost for scaling up becomes an obstacle.

The [Urban Observatory](http://uoweb1.ncl.ac.uk), from the Newcastle University, has the biggest open sensor dataset available online in the world. However, to represent such variables across larger areas, much more sensors are needed. To illustrate that, look at the figure above, these are the places with sensors in Newcastle upon Tyne, UK.

![](https://github.com/adaj/sensingbee/blob/master/examples/plots/sensors.png)

When it comes to specific variables, less sensors are available due to the cost and maintance of such data collection systems.

Thus, this project aims to apply and implement spatial interpolation models to spread data obtained by the sensors around regions where data isn't collected, and then expand the potential of the city's IoT infrastructure.

## How to use

### Installing

Clone this repository in a virtualenv and install sensingbee.

```bash
git clone https://github.com/adaj/sensingbee
cd sensingbee
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install -e . --user
```

If you have any issues on installing, open an issue or get in contact with us.

### Usage

Having data distributed into "data.csv" and "sensors.csv" files (samples and sensors metadata respectively), you will have to set a `configuration__` dictionary (as described in examples/). Then, you can instantiate a Bee to make up your analysis. The first time you used the same `configuration__` use mode="make", and after then you can use mode="load" to speed things up, because the basic settings will be cached for you.

```bash
import sensingbee.source as sb

#configuration__ = {...}
bee = sb.Bee(configuration__).fit(mode='make')
```

You can train models for variables you have data. Let's say we want to train a model for NO2. Then, we would call an api of sensors, such as they have in Urban Observatory, passing the timestamp we want to interpolate.

```bash
bee.train(variables=['NO2'])
bee.interpolate(variables=['NO2'], data=sb.Sensors(
    configuration__,
    mode='get', path={
        'start_time':'2018-09-01',
        'end_time':'2018-09-01',
        'url': 'https://api.newcastle.urbanobservatory.ac.uk/api/v1/sensors/data/raw.csv'
    }, delimit_geography=bee.geography))
bee.plot(variable='NO2', timestamp='2018-09-01',vmin=0, vmax=150)
```

Result:

![](https://github.com/adaj/sensingbee/blob/master/examples/plots/interpolation_no2.png?raw=true)


## Bug reports and Feature request


You can help by reporting bugs, suggesting features, reviewing feature specifications or just by sharing your opinion. Open [issues](https://github.com/adaj/sensingbee/issues) for all of that.

If you want to help or get in touch, please contact me at adelsondias@live.com.

## Contributing

1. Fork the project.
2. Create a branch for your new feature.
3. Test your code.
4. Submit a pull request.

All pull requests are welcome !


## Authors

This project was developed by [Adelson Araujo Jr](https://adaj.github.io).

## License

This project uses the MIT license. See [LICENSE](https://github.com/adaj/sensingbee/blob/master/LICENSE.md) for more details.
