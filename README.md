# DGOPT

[ikki407/DGOPT](https://github.com/ikki407/DGOPT) - Flexible Optimization Project of DG Allocation Problem, written in Python.

## Description

Two-stage stochastic programming is used to formulate an allocation problem of distributed generation(DG).  
__DGOPT__ helps to perform numerical simulations under various conditions.  
User can set and change any parameters, which include distribution systems, demand & weather data, system conditions, costs, and more.  
__DGOPT__ has mainly used Pyomo, which provides an effective framework for stochastic programming.

## Environment

Linux or Mac OS/X

Python 2.7

Gurobi 7.0.0 (or 6.5.0)

## Install

This library is recommended to be installed by using virtualenv, but if you do not use virtualenv, just type `git clone`.

```
git clone git@github.com:ikki407/DGOPT.git
```

### Virtualenv (Recommended)

Make virtual environment.

```
mkvirtualenv DGOPT
```

Then, work on `workon DGOPT` and move to the root directory of virtualenv `cdvirtualenv` and clone DGOPT. 

```
git clone git@github.com:ikki407/DGOPT.git
```

### Requirement

Move to __DGOPT__ directory, and type,

```
pip install -r requirements.txt
```

### Remark

If you had a import problem of matplotlib under virtualenv, change the backend in matplotlibrc as follows:

```
backend : Tkagg
```

The path of your matplotlibrc can be found by

```
python -c "import matplotlib;print(matplotlib.matplotlib_fname())"
```


## Usage

Optimization will start by running the following command in `src` directory.

```
sh all_run.sh
```


## Files

[ikki407/DGOPT/data](https://github.com/ikki407/DGOPT/data) - Directory for demand and weather data.

[ikki407/DGOPT/src](https://github.com/ikki407/DGOPT/src) - Source directory.

[ikki407/DGOPT/src/all\_run.sh](https://github.com/ikki407/DGOPT/src/all_run.sh) - Main source script.

[ikki407/DGOPT/src/config](https://github.com/ikki407/DGOPT/src/config) - Directory for config files of general settings and parameters.

[ikki407/DGOPT/src/concrete](https://github.com/ikki407/DGOPT/src/concrete) - Directory for concrete optimization models.

[ikki407/DGOPT/src/scenario\_generation](https://github.com/ikki407/DGOPT/src/scenario_generation) - Directory for scripts of scenario generation.

[ikki407/DGOPT/src/system\_data](https://github.com/ikki407/DGOPT/src/system_data) - Directory for distribution system data.

[ikki407/DGOPT/src/arrange.py](https://github.com/ikki407/DGOPT/src/arrange.py) - Script for arranging simulation results.

[ikki407/DGOPT/src/postprocessing.py](https://github.com/ikki407/DGOPT/src/postprocessing.py) - Postprocessing script for summarizing all results.

## Contribution

1. Fork ([https://github.com/ikki407/DGOPT/fork](https://github.com/ikki407/DGOPT/fork))
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create new Pull Request

If you had a problem or suggestion, please feel free to contact me.

## Reference

[1][Pyomo](https://github.com/Pyomo)

[2][Gurobi Optimization](http://www.gurobi.com/)


## Licence

[MIT](https://github.com/ikki407/DGOPT/blob/master/LICENSE)

## Author

[ikki407](https://github.com/ikki407)
