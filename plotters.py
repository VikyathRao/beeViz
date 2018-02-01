# -*- coding: utf-8 -*-
""" plotters.py

Classes and functions for beeViz plot elements.
"""
# ==============================================================================
# Imports
import colorcet as cc
import numpy as np
import pandas as pd
import utils

from bokeh.io import curdoc
from bokeh import palettes
from bokeh.layouts import layout, row, column, widgetbox
from bokeh.models import (ColumnDataSource, HoverTool, FuncTickFormatter,
                          Slider, Button, Label, Range1d, Span)
from bokeh.models.mappers import LinearColorMapper, CategoricalColorMapper
from bokeh.models.tickers import FixedTicker
from bokeh.models.widgets import TextInput, Dropdown, PreText
from bokeh.plotting import figure, show

# ------------------------------------------------------------------------------
# Parameters (hard-coded here for now)

alpha = 0.75  # opacity of bee ellipses
alpha_daylight = 0.25  # opacity of yellow shade marking daylight
image_scale = 9.0  # scale factor by which image is shrunk
xpad = np.array([100, 100])  # horizontal padding of image 
ypad = np.array([100, 100])  # vertical padding of image 
max_seek_iters = 10
max_pendown_lines = 60*60
pi_over_two = np.pi / 2.0
two_pi = 2 * np.pi

# ------------------------------------------------------------------------------
# Colour specs and colour maps

colours = { 'bar_bg': "#021C29",
            'daylight' : "yellow",
            'door' : "#403B2B",
            'frame' : '#403B2B',
            'laying' : 'red',
            'trophallaxis' : 'red' }

colours_behav = { 'trophallaxis' : 'black',
                  'laying' : '#8E3063',  # magenta
                  'entrance' : 'blue' }

cmap = { 'cat' : LinearColorMapper(palette=palettes.Set2_5, # alt: Accent5
                                   low=0, high=4),
         'bounded1' : LinearColorMapper(palette=cc.palette['kbc'][::-1],
                                         low=0, high=1),
         'pos_neg' : LinearColorMapper(palette=cc.palette['gwv'],
                                       low=-1, high=1) }

# ------------------------------------------------------------------------------
# Plot classes

class Screen:
    """ This class contains all plot objects for the main screen that displays
    bee movements and behaviours.
    """
    def __init__(self, clock, metadata_fields={}):
        """ Sets up all elements for plotting on the screen.
        The initial clock reading may be specified.
        """
        self.clock = clock
        self.metadata_fields = metadata_fields
        # Plotting area
        image_size = utils.image_size
        frame_size = np.array(list(image_size)) + np.array(
            [xpad.sum(), ypad.sum()])
        self.plot_width, self.plot_height = list(
            map(int, map(lambda x: x/image_scale, image_size)))
        self.plot = figure(plot_width=self.plot_width,
                           plot_height=self.plot_height,
                           tools="pan,box_zoom,save,reset")
        self.plot.axis.visible = False
        self.plot.x_range = Range1d(-xpad[0], image_size[0] + xpad[1])
        self.plot.y_range = Range1d(image_size[1] + ypad[0], -ypad[1])  # nb: inverted!
        self.plot.xgrid.grid_line_color = None
        self.plot.ygrid.grid_line_color = None

        # Hover tool
        self.hover = HoverTool(tooltips=[
            ("(x, y)", "($x{int}, $y{int})"), ("bee", "@b")
        ] + [(mtitle, "@"+metadata_fields[mtitle])
             for mtitle in sorted(metadata_fields.keys())])
                                          # ("specialist score", "@spec_score"),
                                          # ("generalist score", "@gen_score"),
                                          # ("source colony", "@source_colony"),
                                          # ("sequenced?", "@sequenced")])
        self.plot.add_tools(self.hover)

        # "Pen" that marks the trail of an individual bee
        self.pen = Pen(bee=0, x=-1000, y=-1000)

        # Draw the hive frame
        self.plot.segment(x0=[0, 0, image_size[0], image_size[0]],
                          y0=[0, 0, 0, image_size[1]],
                          x1=[0, image_size[0], image_size[0], 0],
                          y1=[image_size[1], 0, image_size[1], image_size[1]],
                          color=colours['frame'], line_width=2, alpha=0.4)
        self.plot.quad(left=image_size[0], right=image_size[0]+50,
                       bottom=image_size[1], top=utils.image_cutout_y,
                       color=colours['door'], alpha=0.4)

        # Plot renderers (these are updated at each callback)...
        # each bee is represented by both an ellipse and a triangle;
        # the others are for behaviours
        self.src = ColumnDataSource({ attr : [] for attr in [
            'x', 'y', 'angle', 'color_value',
            'x_s', 'y_s', 'angle_s', 'b',
            'spec_score', 'gen_score', 'source_colony'] })
        self.renderers = {
            'ellipse' : self.plot.oval(
                x='x', y='y', angle='angle', source=self.src,
                fill_color={'field' : 'color_value',
                            'transform' : cmap['cat']},
                width=200, height=50, alpha=alpha),
            'triangle' : self.plot.triangle(
                x='x_s', y='y_s', angle='angle_s', source=self.src,
                fill_color={'field' : 'color_value',
                            'transform' : cmap['cat']},
                size=5, line_color='black', line_width=0.5),
            'trophallaxis' : self.plot.segment(
                x0=[], y0=[], x1=[], y1=[],
                color=colours['trophallaxis'], line_width=3),
            'laying' : self.plot.ellipse(
                x=[], y=[], width=250, height=250,
                color=colours['laying'], fill_alpha=0, line_width=1.5),
            'pen' : self.plot.segment(
                x0=[], y0=[], x1=[], y1=[], line_dash=[],
                color='red', line_width=1.5, alpha=0.35),
            'hilite' : self.plot.rect(
                x=[], y=[], width=225, height=225,
                fill_color='red', line_color='red', fill_alpha=0.1) }

        self.colour_fld = 'uniform'

    def colourby(self, fld, fld_type):
        self.colour_fld = fld
        for x in ['ellipse', 'triangle']:
            self.renderers[x].glyph.fill_color = {
                'field' : 'color_value', 'transform' : cmap[fld_type] }


    def reset_pen(self, text_input):
        bee = int(text_input) if text_input is not "" else -1
        self.renderers['pen'].data_source.data = {
            attr : [] for attr in ['x0', 'x1', 'y0', 'y1'] }
        self.renderers['hilite'].data_source.data = {'x' : [], 'y' : []}
        self.pen.x, self.pen.y = (-1000, -1000)
        if bee < 0:
            self.pen.is_down = False
        else:
            self.pen.is_down = True
            self.pen.bee = bee


    def reset_renderers(self):
        """ Set all renderers to null.
        """
        for x in ['ellipse', 'triangle']:
            self.renderers[x].data_source.data = { attr : [] for attr in [
                'x','y','angle','color_value','x_s','y_s','angle_s'] }
        self.renderers['trophallaxis'].data_source.data = { 
            attr : []
            for attr in ['x0', 'y0', 'x1', 'y1'] }
        self.renderers['laying'].data_source.data = { 'x' : [],  'y' : [] }
        self.renderers['pen'].data_source.data = {
            attr : [] for attr in ['x0', 'x1', 'y0', 'y1'] }
        self.renderers['hilite'].data_source.data = { 'x' : [],  'y' : [] }


    def update(self, image, dset, time):
        """ Updates screen using data from the latest image.

        Args
        ----
        image : tracking image (array of DataLines)
        dset : Dataset object
        time : current time
        """
        # Get new bee positions/orientations
        attrs = ['b','x','y','angle']
        dat = dict(zip(attrs,
                       np.transpose([[pt[x] for x in attrs]
                                     for pt in image])))
        dat['angle'] *= -1  # needed because of inverted y-axis

        for _, fld in self.metadata_fields.items():
            dat[fld] = dset.metadata[fld].loc[dat['b']]
        dat['color_value'] = dset.metadata_norm[self.colour_fld].loc[dat['b']]

        
        dat['x_s'] = dat['x'] + 60*np.cos(dat['angle'])
        dat['y_s'] = dat['y'] - 60*np.sin(dat['angle'])  # minus <- inverted Y
        dat['angle_s'] = np.mod(dat['angle'] - pi_over_two, two_pi)

        df = pd.DataFrame(dat).set_index('b', drop=True)

        # Get behaviours observed at this time
        dat_behav = {
            'trophallaxis' : utils.get_trophallaxis_events(
                dset.data['trophallaxis'], df, time),
            'laying': utils.get_laying_events(dset.data['laying'], df, time) }

        # Pen position
        dat_pen_new = self.pen.update_position(df)
        # self.pen, dat_pen_new = utils.get_pen_displacement(self.pen, df)

        # Update renderers
        self.update_pen_renderer(dat_pen_new)
        self.update_bee_positions(dat)
        self.update_behaviours(dat_behav)

    def update_bee_positions(self, dat):
        """ Updates ellipse and triangle renderers with new bee position,
        orientation and meta data.
        """
        self.renderers['ellipse'].data_source.data = dat.copy()
        self.renderers['triangle'].data_source.data = dat

    def update_behaviours(self, dat):
        """ Updates behaviour renderers with new data.
        """
        for x, coords in dat.items():
            self.renderers[x].data_source.data = coords

    def update_pen_renderer(self, dat_new):
        """ Updates pen renderer with new data.
        """
        if self.pen.is_down:
            self.renderers['hilite'].data_source.data = {
                'x':[self.pen.x], 'y':[self.pen.y]}
            self.renderers['pen'].data_source.stream(
                dat_new, rollover=max_pendown_lines)


class TimeseriesPlots:
    """ Class containing plot objects for the set of timeseries plots.
    """
    def __init__(self, screen):
        self.plot = {
            x : figure(plot_width=int(screen.plot_width/3.),
                       plot_height=int(screen.plot_height/3.),
                       toolbar_location="right", toolbar_sticky=False,
                       tools="pan,box_zoom,save,reset")
            for x in ['trophallaxis', 'laying', 'entrance'] }

        self.t_markers = { x : Span(location=0, dimension='height',
                                   line_color='red', line_width=1)
                          for x in self.plot }

        self.layout = column(*[self.plot[x] for x in self.plot])
        
    def make_plots(self, dset):
        """ Plots each timeseries, plus yellow rectangles to indicate
        daylight hours.
        """
        # Sunrise/sunset rect positions
        Xw = utils.time_ms_to_day(np.asarray(dset.ss.tf - dset.ss.ti))
        X = utils.time_ms_to_day(np.asarray(dset.ss.ti) - dset.t0) + (Xw / 2.0)

        # Plot timeseries and sunrise/sunset rects
        for x, p in self.plot.items():
            Y = dset.activity[x]
            Ymax = Y.max()
            p.rect(x=X, y=0.5*Ymax*np.ones(dset.ss.shape[0]),
                   width=Xw, height=Ymax, color=colours['daylight'],
                   alpha=alpha_daylight)
            p.line(dset.activity['t'], Y, color=colours_behav[x],
                   line_width=1)
            p.add_layout(self.t_markers[x])
            p.yaxis.axis_label = x

    def update_t_markers(self, val):
        """ Updates red lines marking current time.
        """
        # val = utils.time_ms_to_day(time - t0)
        val_mod1 = np.mod(val, 1)
        for x, mark in self.t_markers.items():
            mark.location = val



class IndividualBeePlots:
    """ Class containing plot objects for the set of individual-level
    bee behaviour plots.
    """
    def __init__(self, screen, dset):
        self.dset = dset
        
        # Data table for bee
        self.table = PreText(text="",
                             width=int(0.5*screen.plot_width),
                             height=int(0.45*screen.plot_height))

        # Main plot objects
        self.plot = {
            'behaviours' : figure(
                plot_width=int(0.95*screen.plot_width),
                plot_height=int(0.8*screen.plot_height),
                toolbar_location="right", toolbar_sticky=False,
                tools="pan,box_zoom,save,reset") }

        self.plot['ke'] = figure(
            plot_width=int(0.5*screen.plot_width),
            plot_height=int(0.3*screen.plot_height),
            toolbar_location="right", toolbar_sticky=False,
            tools="pan,box_zoom,save,reset")

        self.plot['ke'].yaxis.axis_label = "kinetic energy"

        # Red lines to mark current time
        self.t_markers = {
            'ke' : Span(location=0, dimension='height',
                                       line_color='red', line_width=1) }
        self.t_markers['behaviours'] = Span(
            location=0, dimension='height', line_color='red',
            line_width=1, line_dash='dashed', line_alpha=0.7)
        self.t_markers['behaviours_short'] = self.plot['behaviours'].segment(
            x0=[], y0=[], x1=[], y1=[], color='red', line_width=1)

        for x in self.plot:
            self.plot[x].add_layout(self.t_markers[x])


        # Layout
        self.layout = row([self.plot['behaviours'],
                           column([widgetbox(self.table),
                                   self.plot['ke']])])

        # Initial setup
        self.setup_plots()

        
    def setup_plots(self):
        """ Sets up timeseries plots of behaviours and kinetic energy for an
        individual bee.
        """
        # Sunrise/sunset rect positions
        Xw = utils.time_ms_to_day(np.asarray(self.dset.ss.tf - self.dset.ss.ti))
        X = utils.time_ms_to_day(np.asarray(
            self.dset.ss.ti) - self.dset.t0) + (Xw / 2.0)
        self.daylight_data = { 'x' : X, 'width' : Xw }
        self.marker_height = 0.18

        # Setup axes of behaviours plot
        yticker = FixedTicker(ticks=np.arange(0, self.dset.num_days))
        self.plot['behaviours'].yaxis[0].ticker = yticker
        self.plot['behaviours'].ygrid[0].ticker = yticker
        
        self.plot['behaviours'].x_range = Range1d(0, 1)
        xticker = FixedTicker(ticks=np.linspace(0, 1, 12 + 1))
        self.plot['behaviours'].xaxis[0].ticker = xticker
        self.plot['behaviours'].xgrid[0].ticker = xticker
        self.plot['behaviours'].xaxis.formatter = FuncTickFormatter(
            code="""
            function ticker(tick) {
              var seconds = tick*24*3600.0;
              var h = Math.floor(seconds / 3600);
              return '' + h + 'h'
            }
            return ticker(tick);
            """)
        

        # Main data renderers
        self.renderers = {
            'daylight_ke' : self.plot['ke'].rect(
                x=[], y=[], width=[], height=[],
                color=colours['daylight'],
                alpha=alpha_daylight),
            'daylight_behaviours' : self.plot['behaviours'].rect(
                x=np.mod(X, 1), y=(np.arange(self.dset.num_days)),
                width=Xw, height=np.ones(self.dset.num_days),
                color=colours['daylight'], alpha=0.1),
            'ke' : self.plot['ke'].line(x=[], y=[],
                                        color='#5B2B75', line_width=1) }


        for behav in ['trophallaxis', 'laying', 'entrance']:
            self.renderers[behav] = self.plot['behaviours'].rect(
                x=[], y=[], width=[], height=0.90*self.marker_height,
                color=colours_behav[behav])
            

    def reset_renderers(self):
        """ Resets all renderers associated with this object.
        """
        for rend in ['daylight_behaviours', 'daylight_ke']:
            self.renderers[rend].data_source.data = {
                'x' : [], 'y' : [], 'width' : [], 'height' : [] }

        self.renderers['ke'].data_source.data = { 'x' : [], 'y' : [] }

        for behav in ['trophallaxis', 'laying', 'entrance']:
            self.renderers[behav].data_source.data = {
                'x' : [], 'y' : [], 'width' : [] }

        self.table.text = ""


    def update_plots(self, bee):
        """ Plots timeseries of behaviours and kinetic energy for an
        individual bee.
        """
        self.reset_renderers()
        
        # Update KE timeseries
        ke_vals = self.dset.data['ke'].loc[bee].values
        self.renderers['ke'].data_source.data = { 'x' : self.dset.ke_t,
                                                  'y' : ke_vals }
        
        # Add correctly-scaled daylight rects to KE plot
        daylight_data = self.daylight_data.copy()
        daylight_data['height'] = np.nanmax(ke_vals)*np.ones(self.dset.num_days)
        daylight_data['y'] = daylight_data['height'] / 2.0
        self.renderers['daylight_ke'].data_source.data = daylight_data

        # Add trophallaxis events
        for k, behav in enumerate(['laying', 'trophallaxis', 'entrance']):
            bdat = self.dset.get_individual_behaviours(behav, bee)

            if behav=='entrance':
                Xi = utils.time_ms_to_day(bdat['t'].values - self.dset.t0)
                Xw = (1e-4) * np.ones(Xi.shape[0])
            else:
                Xw = utils.time_ms_to_day(
                    np.asarray(bdat['tf'] - bdat['ti']))
                Xi = utils.time_ms_to_day(bdat['ti'].values - self.dset.t0)

            Ximod1 = np.mod(Xi, 1)
            Y = Xi - Ximod1 + ((k - 1)*self.marker_height)
            X = Ximod1 + (Xw / 2.0)
            self.renderers[behav].data_source.data = {
                'x' : X, 'y' : Y, 'width' : Xw }

        # Update data table
        bee_metadata = self.dset.metadata.loc[bee][
            sorted(self.dset.metadata.loc[bee].index)]
        self.table.text = bee_metadata.to_string()


    def update_t_markers(self, val):
        """ Updates red lines marking current time.
        """
        # val = utils.time_ms_to_day(time - self.dset.t0)
        val_mod1 = np.mod(val, 1)
        self.t_markers['ke'].location = val
        self.t_markers['behaviours'].location = np.mod(val, 1)
        self.t_markers['behaviours_short'].data_source.data = {
            'x0':[val_mod1], 'x1':[val_mod1], 'y0':[val - val_mod1 - 0.5],
            'y1':[val - val_mod1 + 0.5] }

        
class Pen:
    """ A simple pen class that is used to draw line segments that track
    an individual in the Bokeh plotting screen.
    """
    def __init__(self, bee, x, y, is_down=False):
        self.is_down = is_down
        self.x = x
        self.y = y
        self.bee = bee

    def update_position(self, data_positions):
        """ If pen is down to track a particular bee, this returns a
        line segment corresponding to its displacement from the previous
        timestep and updates the pen position.
        
        Args
        ----
        data_positions : pandas DataFrame for bee positions, indexed by bee
        
        Returns
        -------
        disp :  dict with keys 'x0', 'y0', 'x1', 'y1' specifying the marked
        bee's displacement
        """
        disp = { 'x0' : [self.x], 'y0' : [self.y],
                 'x1' : [self.x], 'y1' : [self.y] }
        
        if self.is_down and (self.bee in data_positions.index):
            x1, y1 = list(map(int, data_positions.loc[self.bee][['x', 'y']]))
            disp['x1'], disp['y1'] = ([x1], [y1])
            if (self.x < 0) or (self.y < 0):  # pen has just been set down
                disp['x0'], disp['y0'] = ([x1], [y1])
            self.x, self.y = (x1, y1)

        return disp


class Controls:
    """ A class for the objects used to control the animation,
    namely: the framerate slider, the play/pause button, the
    track bee input box and the colour options.
    """
    def __init__(self, screen, metadata_fields={}):
        self.play_button = Button(label='Play', width=60)
        self.track_button = Button(label='Reset pen', width=60)
        self.framerate_slider = Slider(
            start=1, end = 1000, value=500, step=1,
            title='Frame time (ms)',
            bar_color=colours['bar_bg'],
            width=int(0.40*screen.plot_width))
        self.textinput_pen = TextInput(
            placeholder="Track bee...", value="", width=80)
        self.dropdown_clr_menu = [
            ("uniform", "uniform")] + [
                (mtitle, metadata_fields[mtitle])
                for mtitle in sorted(metadata_fields.keys())]
        self.dropdown_clr = Dropdown(
            label='Colour by...', button_type="warning",
            width=int(0.40*screen.plot_width),
            menu=self.dropdown_clr_menu)
        
        self.layout = widgetbox(children=[self.dropdown_clr,
                                          self.framerate_slider,
                                          self.play_button,
                                          self.textinput_pen,
                                          self.track_button],
                                width=int(0.5*screen.plot_width),
                                sizing_mode='scale_width')



class TimeDisplay:
    """ A class for the objects used to display and control
     the time: text box (and corresponding button), slider,
    and slider annotation.
    """
    def __init__(self, dset, screen):
        self.dset = dset
        self.screen = screen

        # Slider for time
        self.slider = Slider(
            start=int(self.dset.t_first / 1000 - 1),
            end=int(self.dset.t_last / 1000),
            value=int(self.dset.t_first / 1000 - 1),
            step=1, bar_color=colours['bar_bg'],
            title="Timestamp", width=int(0.70*self.screen.plot_width))

        self.t_marker = Span(location=0, dimension='height',
                           line_color='red', line_width=1)

        # Annotation below slider
        self.slider_annot = self.make_slider_annot()

        # Time text input and corresponding button to activate
        self.textinput = TextInput(
            placeholder=utils.time_readable(dset.t_first - 1000),
            value="", width=80)
        self.button = Button(label="Go!", width=60)

        # Layout of all elements
        self.layout = column([widgetbox(children=[self.textinput,
                                                 self.button,
                                                 self.slider],
                                       width=int(0.75*screen.plot_width),
                                       sizing_mode='scale_width'), 
                             self.slider_annot])


    def make_slider_annot(self):
        """ Creates plot that serves as time-annotation below slider.
        """
        s = figure(plot_width=int(0.74*self.screen.plot_width),
                   plot_height=45,
                   toolbar_location=None)
        xmin, xmax = map(
            utils.time_ms_to_day,
            [self.dset.t_first - self.dset.t0,
             self.dset.t_last - self.dset.t0])
        s.x_range = Range1d(xmin, xmax)
        s.y_range = Range1d(0, 1)
        s.yaxis.visible = False
        s.ygrid.grid_line_color = None
        s.toolbar.logo = None
        s.toolbar_location = None

        Xw = utils.time_ms_to_day(
            np.asarray(self.dset.ss.tf - self.dset.ss.ti))
        X = utils.time_ms_to_day(
            np.asarray(self.dset.ss.ti) - self.dset.t0) + (Xw / 2.0)
        s.rect(x=X, y=0.5*np.ones(self.dset.ss.shape[0]),
               width=Xw, height=1, color=colours['daylight'],
               alpha=alpha_daylight)
        s.segment(x0=xmin, x1=xmax, y0=1, y1=1, color='black')
        s.add_layout(self.t_marker)
        return s


