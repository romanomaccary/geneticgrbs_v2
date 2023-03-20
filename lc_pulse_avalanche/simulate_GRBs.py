################################################################################
# IMPORT LIBRARIES
################################################################################

import os
import sys
import yaml, h5py
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

SEED=42
np.random.seed(SEED)

# set the username for the path of the files:
user='LB'
#user='AF'
if user=='LB':
    sys.path.append('/home/lorenzo/git/lc_pulse_avalanche/statistical_test')
    sys.path.append('/home/lorenzo/git/lc_pulse_avalanche/lc_pulse_avalanche')
elif user=='AF':
    sys.path.append('......WRITE_HERE....../lc_pulse_avalanche/statistical_test')
    sys.path.append('......WRITE_HERE....../lc_pulse_avalanche/lc_pulse_avalanche')
else:
    raise ValueError('Assign to the variable "user" a correct username!')

from statistical_test import *
from avalanche import LC, Restored_LC

export_path='../simulations/'

################################################################################
# SET PARAMETERS
################################################################################

# The values of the 7 parameters from the paper [Stern & Svensson, 1996] are
mu=1.2
mu0=1
alpha=4
delta1=-0.5
delta2=0
tau_min=0.02
tau_max=26


t_i=0   # [s]
t_f=150 # [s]
t90_threshold=2 # [s] --> used to select only _long_ GRBs

N_grb=10
n_cut=2000
with_bg=False 


instrument = 'batse'
#instrument = 'swift'
#instrument = 'sax'
if instrument=='batse':
    res          = 0.064 # time resolution of the light curves [ms]
    eff_area     = 3600  # effective area of instrument [cm2]
    bg_level     = 10.67 # background level [cnt/cm2/s]
    sn_threshold = 70
elif instrument=='swift':
    res          = 0.064            # time resolution of the light curves [ms]
    eff_area     = 1400             # effective area of instrument [cm2]
    bg_level     = (10000/eff_area) # background level [cnt/cm2/s]
    sn_threshold = 15
elif instrument=='sax':
    res          = 0.0078125       # time resolution of the light curves [s]
    eff_area     = 420             # effective area of instrument [cm2]
    bg_level     = (1000/eff_area) # background level [cnt/cm2/s]
    sn_threshold = 10
else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax".')


################################################################################
################################################################################

_ = generate_GRBs(# number of simulated GRBs to produce
                  N_grb=N_grb, 
                  # 7 parameters
                  mu=mu, 
                  mu0=mu0, 
                  alpha=alpha, 
                  delta1=delta1, 
                  delta2=delta2,  
                  tau_min=tau_min, 
                  tau_max=tau_max, 
                  # instrument parameters
                  instrument=instrument, 
                  bin_time=res, 
                  eff_area=eff_area,
                  bg_level=bg_level, 
                  # constraint parameters
                  t90_threshold=t90_threshold,
                  sn_threshold=sn_threshold, 
                  t_f=t_f, 
                  # other parameters
                  export_files=True, 
                  export_path=export_path, 
                  n_cut=2000, 
                  with_bg=False)

################################################################################
################################################################################