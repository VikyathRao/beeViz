# -*- coding: utf-8 -*-
""" main.py

Main script for the beeViz Bokeh app.
"""
# ==============================================================================
# Imports
import colorcet as cc
import numpy as np
import pandas as pd
import sys
import plotters as plt  # custom display classes
import utils  # custom helper classes/functions

from bokeh.io import curdoc
from bokeh.layouts import layout, row, column

# ------------------------------------------------------------------------------
# Load and set up the data for plotting

d = utils.Dataset("beeviz/config.yaml")
d.load()
d.load_activity_timeseries()

data = d.data
t_first, t_last, t0 = d.t_first, d.t_last, d.t0


# ------------------------------------------------------------------------------
# Set up the plotting canvas

# Screen that displays bee movements/behaviours
screen = plt.Screen(clock=t_first - 1000, metadata_fields=d.metadata_fields)

# Plots of activity timeseries
timeseries = plt.TimeseriesPlots(screen)
timeseries.make_plots(d)

# Plots of individual bee timeseries
beeplots = plt.IndividualBeePlots(screen, d)
beeplots.setup_plots()


# ------------------------------------------------------------------------------
# Define the animation protocols

def animate_update():
    """ This function is called at each callback.
    """
    time = screen.clock + 1000  # increment time

    # If the end of the slider is reached
    if time > t_last:  # loop back to start
        time = t_first
        data['tracking'].seek_to_time(t_first)

    # Check where slider is relative to currently-loaded image
    delta_t = data['tracking'].time - time
        
    # Go to next image in tracking file
    if delta_t < -500:
        data['tracking'].__next__()
        delta_t = data['tracking'].time - time

    # Update slider
    if abs(delta_t) < 500:  # corrects for time drift in data
        screen.clock = data['tracking'].time
        timedisplay.slider.value = int(round(screen.clock / 1000.0))
    else:  # missing data
        screen.reset_renderers()
        screen.clock = time
        print("WARNING: possibly missing data at {} ({})".format(
            time, utils.time_readable(time)))


        
def slider_time_update(attrname, old, new):
    """ This function defines actions to take when slider is updated.
    """
    time = timedisplay.slider.value * 1000
    timedisplay.textinput.value = utils.time_readable(time)

    # Update red lines marking the time
    val = utils.time_ms_to_day(time - t0)
    timedisplay.t_marker.location = val
    timeseries.update_t_markers(val)
    beeplots.update_t_markers(val)


    # If slider was manually changed rather than via callback
    if abs(time - screen.clock) >= 1000:
        data['tracking'].seek_to_time(time)
        data['tracking'].__next__()
        screen.clock = time
        screen.reset_renderers()

    # Update screen
    if abs(screen.clock - data['tracking'].time) < 1000:
        screen.clock = data['tracking'].time
        screen.update(data['tracking'].image, d, time)


def colour_update(attrs, old, new):
    """ Called when colour dropdown is changed.
    """
    val = controls.dropdown_clr.value
    controls.dropdown_clr.label = "Colouring by '{}'".format(val)
    screen.colourby(val, d.metadata_cat.get(val, 'cat'))

    
def pen_update(attrs, old, new):
    """ Called when pen value is changed.
    """
    screen.reset_pen(controls.textinput_pen.value)
    if screen.pen.is_down:
        beeplots.update_plots(screen.pen.bee)
    else:
        beeplots.reset_renderers()
        

def button_set_time_click():
    """ Called when time Go! button is clicked.
    """
    timedisplay.slider.value = int(round(utils.time_str_to_timestamp(
        timedisplay.textinput.value)/1000))

def reset_trail():
    """ Called when pen reset button is clicked.
    """
    screen.reset_pen("")
    screen.reset_pen(controls.textinput_pen.value)


def animate():
    """ Called when Play button is activated.
    """
    if controls.play_button.label == 'Play':
        controls.play_button.label = 'Pause'
        curdoc().add_periodic_callback(
            animate_update, controls.framerate_slider.value)
    else:
        controls.play_button.label = 'Play'
        curdoc().remove_periodic_callback(animate_update)


# ------------------------------------------------------------------------------
# Animation controls
controls = plt.Controls(screen, d.metadata_fields)
controls.play_button.on_click(animate)
controls.track_button.on_click(reset_trail)
controls.textinput_pen.on_change('value', pen_update)
controls.dropdown_clr.on_change('value', colour_update)
    
# Time controls
timedisplay = plt.TimeDisplay(d, screen)
timedisplay.slider.on_change('value', slider_time_update)
timedisplay.button.on_click(button_set_time_click)

# Layout of the different elements
pagelayout = layout([row(timeseries.layout, screen.plot),
                     row(controls.layout, timedisplay.layout),
                     beeplots.layout],
                    sizing_mode='scale_width')

curdoc().add_root(pagelayout)
curdoc().title = "beeViz"

# ==============================================================================
