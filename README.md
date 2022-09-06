# sensingbee

Spatial interpolation for sensors data :earth_americas:

## Purpose

Recently, IoT systems have been implemented to measure the city's variables (such as weather, traffic, air quality, sound etc). To do so, the deployment of a (big) network of sensors becomes a fundamental requirement, and the high cost for spatially scaling up the sensor network becomes an obstacle.

One great example comes from the [Urban Observatory](http://uoweb1.ncl.ac.uk), from the Newcastle University, which has the biggest open sensor dataset available online in the world. Even with a huge amount of sensors, much more sensors are needed to represent such variables across larger (and marginal) areas in a continuous surface. To illustrate that, look at the figure above, these are the places with sensors in Newcastle upon Tyne, UK.

![](https://github.com/adaj/sensingbee/blob/master/examples/plots/sensors.png)

One one hand, more sensors are expensive to deploy and maintain. On the other hand, more sensors mean higher reliability for such variables. Smart cities have to think this trade-off carefully.

This project aims to apply and implement reliable spatial interpolation with data obtained by the sensors around regions where data isn't collected, and (potentially) expand the potential of the city's IoT infrastructure. A visualization of such spatial interpolation (of one variable) can be illustrated as below.

![](https://github.com/adaj/sensingbee/blob/master/examples/plots/interpolation_no2.png?raw=true)


## Know more

This project is only a PoC, so you should not rely on it as is. Please contribute!


### Usage

Please look in the examples folder, a notebook will certainly guide you to use this tool.


## Contributing

1. Fork the project.
2. Create a branch for your new feature.
3. Test your code.
4. Submit a pull request.

Pull requests are welcome !


## Authors

This project was developed by [Adelson de Araujo](https://adaj.github.io).

## License

This project uses the MIT license. See [LICENSE](https://github.com/adaj/sensingbee/blob/master/LICENSE.md) for more details.
