# -*- coding: utf-8 -*-
""" utils.py

Classes and functions for data processing.
"""
import calendar
import json
import numpy as np
import os
import pandas as pd
import yaml
from datetime import datetime
from file_read_backwards import FileReadBackwards


# ==============================================================================
# Static data

image_size = (6576, 4384)
image_cutout_y = 3370

# ==============================================================================
# Data loader

class Dataset:
    """ Objects of this class hold all tracking and behaviour data.
    """
    def __init__(self, config_file):
        """ Files corresponding to the dataset are specified
        in a YAML config file.
        """
        self.config_file = config_file

        with open(config_file) as f:
            config = yaml.load(f)
            
        self.fnames = { 'tracking' : config['file.tracking'],
                        'index' : config['file.index'],
                        'trophallaxis' : config['file.trophallaxis'],
                        'laying' : config['file.laying'],
                        'entrance' : config['file.entrance'],
                        'solar' : config['file.solar_events'],
                        'metadata' : config['file.metadata'],
                        'ke' : config['file.kinetic_energy'] }
        for x in self.fnames:
            self.fnames[x] = os.path.join(config['directory.data'],
                                          self.fnames[x])

        self.metadata_fields = config['metadata']
        _ = self.metadata_fields.pop("bee")  # don't need 'bee' field
        self.config = config

        # Record first and last times in tracking file
        self.t_first = get_first_time(self.fnames['tracking'])
        self.t_last = get_last_time(self.fnames['tracking'])

            

    def load(self):
        self.data = {
            'tracking' : TrajectoryFile(self.fnames['tracking'],
                                        self.fnames['index']),
            'trophallaxis' : (pd.read_csv(self.fnames['trophallaxis'], header=None)
                              ) if os.path.isfile(self.fnames['trophallaxis']) else (
                                  pd.DataFrame(columns=np.arange(4))),
            'laying' : (pd.read_csv(self.fnames['laying'], header=None)
                        ) if os.path.isfile(self.fnames['laying']) else (
                            pd.DataFrame(columns=np.arange(3))),
            'entrance' : (pd.read_csv(self.fnames['entrance'], header=None)
                      ) if os.path.isfile(self.fnames['entrance']) else (
                          pd.DataFrame(columns=np.arange(9)))
        }

        self.data['trophallaxis'].columns = [
            'b1','b2','ti','tf','direction'][:self.data['trophallaxis'].shape[1]]
        self.data['laying'].columns = ['ti', 'tf', 'b']

        # if os.path.isfile(self.fnames['entrance']):
        #     # self.data['entrance'] = read_bcode_data(self.fnames['entrance'])
        #     self.data['entrance'] = pd.read_csv(self.fnames['entrance'], header=None)
        self.data['entrance'].columns = [
            't', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'b', 'a']

        # Load solar times, metadata and KE
        if os.path.isfile(self.fnames['solar']):
            self.load_solar_events()

        if os.path.isfile(self.fnames['metadata']):
            self.load_metadata()


        self.load_kinetic_energy()


    def load_solar_events(self):
        """ Creates a DataFrame of sunrise/sunset times during the 
        period of the experiment.
        """
        solar_events = np.loadtxt(self.fnames['solar'],
                                  delimiter=',', dtype='i8')
        sel1 = (solar_events[:,1] >= self.t_first - 18*3600*1000)
        sel2 = (solar_events[:,0] <= self.t_last + 18*3600*1000)
        self.ss = pd.DataFrame(solar_events[np.logical_and(sel1, sel2), :],
                               columns=['ti', 'tf'])
        self.ss = self.ss.assign(
            Ti=pd.to_datetime(self.ss.ti, unit='ms', utc=True),
            Tf=pd.to_datetime(self.ss.tf, unit='ms', utc=True))

        # Make a list of days and record first midnight of recording
        self.num_days = self.ss.shape[0]
        self.days = pd.DatetimeIndex(
            [t.normalize() for t in pd.DatetimeIndex(self.ss.Ti)], freq='D')
        self.t0 = self.days.astype(np.int64)[1] / 1e6 # first midnight


    def load_metadata(self):
        """ Loads metadata into a pandas DataFrame and also creates a
        DataFrame for colouring bees.
        """
        self.metadata = pd.read_csv(self.fnames['metadata'], index_col='bee')
        self.bees = self.metadata.index.values

        # Normalize metadata depending on category
        self.metadata_norm = self.metadata.copy() # normalized data
        self.metadata_cat = {}  # category
        for col in self.metadata_norm.columns:
            nvals = np.unique(self.metadata_norm[col]).size
            if nvals <= 5:  # categorical data
                self.metadata_cat[col] = 'cat'
                self.metadata_norm[col] = self.metadata[col].astype(
                    'category').cat.codes
            else:  # numerical data
                signs = np.sign(self.metadata[col])
                signs_uniq = np.unique(signs[signs != 0])
                if signs_uniq.size > 1: # both positive and negative
                    self.metadata_cat[col] = 'pos_neg'
                    sel = (self.metadata_norm[col] >= 0)
                    vmax = np.max(self.metadata_norm[col])
                    vmin = abs(np.min(self.metadata_norm[col]))
                    # Normalize in [-1, 1]
                    self.metadata_norm.loc[sel, col] /= vmax
                    self.metadata_norm.loc[~sel, col] /= vmin
                else:  # only positive or only negative
                    self.metadata_cat[col] = 'bounded1'
                    self.metadata_norm[col] *= signs_uniq[0] # all positive
                    self.metadata_norm[col] -= self.metadata_norm[col].min()
                    self.metadata_norm[col] /= self.metadata_norm[col].max()

        self.metadata_norm['uniform'] = np.ones(self.bees.size, dtype=int)
        self.metadata_cat['uniform'] = 'cat'


    def load_activity_timeseries(self, t_incr=600000):
        """ Creates timeseries for the level of activity, for different
        behaviours.
        """
        tbins = np.arange(self.t0, self.t_last + t_incr, t_incr)
        tbins_shifted = time_ms_to_day(tbins - self.t0)
        norm = 1.0 / 10  # counts per 10 minutes
        self.activity = { 't' : (tbins_shifted[1:] + tbins_shifted[:-1])/2.0 }
        for x, attr in zip(['trophallaxis', 'laying', 'entrance'],
                           ['ti', 'ti', 't']):
            counts, _ = np.histogram(self.data[x][attr], bins=tbins)
            self.activity[x] = norm*counts

    def load_kinetic_energy(self):
        """ Loads kinetic energy JSON file.
        """
        if not os.path.isfile(self.fnames['ke']):
            self.data['ke'] = pd.DataFrame(self.bees,
                                           columns=['b'])
            self.data['ke']['t'] = 0
            self.data['ke'] = self.data['ke'].set_index('b')
            self.ke_t = np.array([0])
            return None

        with open(self.fnames['ke']) as f:
            kedat = json.load(f)
            
        self.data['ke'] = pd.DataFrame(
            {int(b) : kedat['ke'][b] for b in kedat['ke']}).transpose()
        num_t = self.data['ke'].shape[1]
        self.data['ke'].index.name = 'b'
        self.data['ke'].columns = pd.DatetimeIndex(
            start=kedat['t_start']*1e6, freq=str(int(kedat['t_incr']))+'S',
            periods=num_t)
        self.ke_t = time_ms_to_day(
            (kedat['t_start'] - self.t0) + kedat['t_incr']*np.arange(num_t)
        ) # measured in days since start
        
    def get_individual_behaviours(self, behav, bee):
        """ Returns a pandas DataFrame containing behaviour data for a
        single specified bee.
        """
        if behav=='trophallaxis':
            sel = ((self.data[behav].b1==bee)|(self.data[behav].b2==bee))
        else:
            sel = (self.data[behav].b==bee)
            
        return self.data[behav][sel].reset_index(drop=True)


# ==============================================================================
# Classes for reading data conveniently

        
def trajectoryDataLine(in_line):
    t, X1, Y1, X2, Y2, X3, Y3, b, _ = list(map(
        int, in_line.strip().split(',')))
    ctr, orient = bcode_decoder(X1, Y1, X2, Y2, X3, Y3)
    return {'in_line' : in_line,
            't' : t,
            'b' : b,
            'x' : ctr[0],
            'y' : ctr[1],
            'orient' : orient,
            'dx' : orient.real,
            'dy' : orient.imag,
            'angle' : np.angle(orient)}


class TrajectoryFile:
    """ Reads a trajectory file consisting of lines of bCode data
    of form 't,X1,Y2,X2,Y2,X3,Y3,b,a', and returns an iterator object
    that gives the data points for each timestep of the file.

    Args
    ----
    fname : filename of trajectory file
    fname_idx : filename of the index file (default: None)
    """
    def __init__(self, fname, fname_idx=None):
        self.fname = fname  # file name
        self.fname_idx = fname_idx  # filename of index file
        self.EOF = False  # reached end of file?

        self.__file_obj = open(self.fname)
        self.line = trajectoryDataLine(self.__file_obj.readline())

        self.image_next = [self.line]  # start making image
        self.time_next = self.line['t']  # time of next image
        self.image = []
        self.time = self.time_next - 1000 # placeholder

        if os.path.isfile(fname_idx):
            self.idx = np.loadtxt(fname_idx, delimiter=',',
                                  dtype=np.int64)

    def __iter__(self):
        return self

    def __next__(self):
        if self.EOF:
            raise StopIteration()

        while True:
            l = self.__file_obj.readline()

            # Check if EOF
            if l=="":
                self.__file_obj.close()
                self.EOF = True
                self.time = self.time_next
                self.image = list(self.image_next)
                return self.image  # last image

            self.line = trajectoryDataLine(l)
            # Check if new time
            if self.line['t'] != self.time_next:
                self.image = list(self.image_next)
                self.time = self.time_next
                self.time_next = self.line['t']
                self.image_next = [self.line] # new image
                return self.image

            # If none of the above, we are on same image
            self.image_next.append(self.line)

    def seek_to_time(self, t):
        """ If an index file is provided, this method can
        be used to skip to a given time in the
        trajectory file.
        """
        assert os.path.isfile(self.fname_idx), "No index file!"

        i = np.searchsorted(self.idx[:,0], t, side='left')
        if self.idx[i,0] > t:
            i = max(i-1, 0)
        self.__file_obj.seek(self.idx[i,1])

        self.line = trajectoryDataLine(self.__file_obj.readline())
        self.image_next = [self.line]  # start making image
        self.time_next = self.line['t']  # time of next image


# ==============================================================================
# Functions to help with processing data

rot_phase = np.exp(-1j * np.pi / 4.0)  # needed in function below
cmplx_unit = np.array([1, 1j])  # needed in function below

def bcode_decoder(X1, Y1, X2, Y2, X3, Y3):
    """Given raw bCode coords, this computes
    the spatial coordinates and orientation of the bCode.
    
    Args
    ----
    X1, Y1 : top-left
    X2, Y2 : top-right
    X3, Y3 : bottom-left

    Returns
    -------
    ctr : coordinates of the centre (array of 2 ints)
    orient : orientation (complex)
    """
    # t, X1, Y1, X2, Y2, X3, Y3, b, _ = list(map(
    #     int, line.strip().split(',')))

    # Calculate vectors between bCode corners
    tr = np.array([X2, Y2], dtype=int)  # top right
    bl = np.array([X3, Y3], dtype=int)  # bottom left
    bl_to_tr = np.array(tr - bl, dtype=float)

    # Coordinates of the centre of the bCode
    ctr = np.array(np.round(bl + (bl_to_tr / 2.0)),
                   dtype=int)

    # Orientation of the bCode
    orient = bl_to_tr.dot(cmplx_unit) * rot_phase # rotate 45 deg towards X
    orient /= abs(orient)  # normalization

    return (ctr, orient)


def get_first_time(traj_file):
    """ Returns first timestamp in trajectory file."""
    with open(traj_file, 'r') as f:
        pt = trajectoryDataLine(f.readline())
    return pt['t']

def get_last_time(traj_file):
    """ Returns last timestamp in trajectory file."""
    with FileReadBackwards(traj_file) as f:
        for l in f:
            pt = trajectoryDataLine(l)
            break
    return pt['t']


def get_laying_events(data_laying, data_positions, time):
    """ Given data about laying events and instantaneous bee positions,
    this returns the positions of all bees engaged in laying at that time.

    trophallaxis at that time as a dictionary.

    Args
    ----
    data_laying : pandas DataFrame for laying events
    data_positions : pandas DataFrame for bee positions, indexed by bee
    time : current time

    Returns
    -------
    dict with keys 'x' and 'y' listing positions
    """
    sel = (time >= data_laying.ti) & (time <= data_laying.tf)
    if np.sum(sel) > 0:
        dat_laying_sel = data_laying[sel].reset_index(drop=True)
        b = np.asarray(dat_laying_sel['b'])
        pos = np.asarray(data_positions.loc[b][['x', 'y']])

        return { 'x' : pos[:,0], 'y' : pos[:,1] }
    return {'x' : [], 'y': [] }


def get_pen_displacement(pen, data_positions):
    """ If pen is down to track a particular bee, this returns a
    line segment corresponding to its displacement from the previous
    timestep.

    Args
    ----
    pen : pen object (defined in plotters)
    data_positions : pandas DataFrame for bee positions, indexed by bee

    Returns
    -------
    pen : with updated position
    newline :  dict with keys 'x0', 'y0', 'x1', 'y1' specifying the marked
               bee's displacement
    """
    newline = { 'x0' : [pen.x], 'y0' : [pen.y],
                'x1' : [pen.x], 'y1' : [pen.y] }

    if pen.is_down and (pen.bee in data_positions.index):
        x1, y1 = list(map(int, data_positions.loc[pen.bee][['x', 'y']]))
        newline['x1'], newline['y1'] = ([x1], [y1])
        if (pen.x < 0) or (pen.y < 0):  # pen has just been set down
            newline['x0'], newline['y0'] = ([x1], [y1])
        pen.x, pen.y = (x1, y1)

    return (pen, newline)



def get_trophallaxis_events(data_troph, data_positions, time):
    """ Given data about trophallaxis and bee positions at a given instant
    in time, this returns the positions of all pairs of bees engaged in

    Args
    ----
    data_troph : pandas DataFrame for trophallaxis events
    data_positions : pandas DataFrame for bee positions, indexed by bee
    time : current time

    Returns
    -------
    dict with keys 'x0', 'y0', 'x1' and 'y1', listing positions of pairs
    of bees engaged in trophallaxis.

    """
    attrs = ['x0','y0','x1','y1']
    coords = [[], [], [], []]
    sel = (time >= data_troph.ti) & (time <= data_troph.tf)
    if np.sum(sel) > 0:
        dat_troph_sel = data_troph[sel].reset_index(drop=True)
        b1 = np.asarray(dat_troph_sel['b1'])
        b2 = np.asarray(dat_troph_sel['b2'])
        r0 = np.asarray(data_positions.loc[b1][['x', 'y']])
        r1 = np.asarray(data_positions.loc[b2][['x', 'y']])

        coords = [ r0[:,0], r0[:,1], r1[:,0], r1[:,1] ]

    return dict(zip(attrs, coords))


def read_bcode_data(fname):
    """ Reads a file of bCode tracking data and returns a pandas
    DataFrame for (t, b, x, y, orient). Make sure the file isn't too big!
    """
    dat = []
    with open(fname) as f:
        for l in f:
            t, X1, Y1, X2, Y2, X3, Y3, b, _ = list(map(
                int, l.strip().split(',')))
            ctr, orient = bcode_decoder(X1, Y1, X2, Y2, X3, Y3)

            dat.append([t, b, ctr[0], ctr[1], orient])
    return pd.DataFrame(dat, columns=['t','b','x','y','orient'])

def time_ms_to_day(t):
    """ Convert time in milliseconds to days.
    """
    return t / (1000.0*3600*24)

def time_str_to_timestamp(t_str):
    """ Converts timestamp string to Unix time in milliseconds.
    """
    t = pd.DatetimeIndex([t_str])
    return np.int64(t.astype(np.int64)/1e6)[0]


def time_readable(unix_time, utc=True):
    """ Convert a Unix timestamp (in milliseconds) to a
    readable string.
    """
    t = datetime.utcfromtimestamp(unix_time/1000.) if (utc
        ) else datetime.fromtimestamp(unix_time/1000.)
    t_str = '{0}-{1:02}-{2:02} {3:02}:{4:02}:{5:02}'.format(
        t.year, t.month, t.day, t.hour, t.minute, t.second)
    return t_str

# ==============================================================================

