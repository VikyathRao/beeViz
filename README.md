# beeViz: Visualizing the honeybee society
__________________________________________


## Goal
-------

To create a Bokheh dashboard to visualize data for honeybee trajectories and behaviours inside a 2-d glass-walled hive.


Challenges:

* trajectory datasets are too large to be loaded into memory
* datasets for different behaviours have to be integrated and synchronized
* metadata about individual bees are of different types (categorical, discrete, continuous, positive definite, etc.) -- but should all be displayed in an informative way


## Demo
-------

![beeViz Demo GIF](/images/beeviz_demo.gif?raw=true "beeViz demo")


## Features
--------

Currently available:

* Begin or pause movie using Play/Pause button.
* Adjust framerate using the slider to set the time between two subsequent frames in the movie.
* Scroll to a particular time in the dataset using either the timestamp slider or the text box above it.
* Trophallaxis events are displayed by a red line joining a pair of bees.
* Egg-laying events are displayed using a circle around the bee.
* Colouring by arbitrary metadata (categorical or continuous).
* Hover above an individual bee to see its metadata.
* Track an individual bee by entering its ID into the text box -- a trail should appear.
* When an individual bee is tracked, behaviour data over the course of its lifespan appears at the resolution of individual events.
* When an individual bee is tracked, its kinetic energy timeseries is displayed (if available).




## Installation and Usage
-------------------------

The most convenient setup is to install [Anaconda with Python 3](https://www.anaconda.com/download/) and create a virtual environment for beeViz.


After installing Anaconda, open a terminal (use Anaconda terminal in Windows), navigate to the parent beeViz directory and follow the instructions below. Don't forget to specify the input data files in 'config.yaml' before attempting to run beeViz.


### Linux

Create the virtual environment using:

```
conda env create -f beeviz/environment.yaml
```

To use beeViz, enter

```
source activate beeviz
bokeh serve --show beeviz/
```

to activate the virtual environment and run beeViz.


### Windows

To set up the environment, run the following sequence of commands:

```
conda create --name beeviz
activate beeviz
conda install pip
pip install -r beeviz/requirements.txt
```

To use beeViz, enter

```
activate beeviz
bokeh serve --show beeviz/
```

to activate the virtual environment and run beeViz.



## TODO
-------

Upcoming features:

* accumulating statistics of tracked bee's trail
* integrate entrance camera data
* add kinetic energy metadata
* ability to deal with time-varying metadata

____________________________________________________________
