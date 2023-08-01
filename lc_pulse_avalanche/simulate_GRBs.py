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

#seed=42
#np.random.seed(SEED)

# set the username for the path of the files:
user='LB'
#user='AF'
#user='bach'
if user=='bach':
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
elif user=='LB':
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/statistical_test')
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/lc_pulse_avalanche')
elif user=='AF':
    sys.path.append('C:/Users/Lisa/Documents/GitHub/lc_pulse_avalanche/statistical_test')
    sys.path.append('C:/Users/Lisa/Documents/GitHub/lc_pulse_avalanche/lc_pulse_avalanche')
else:
    raise ValueError('Assign to the variable "user" a correct username!')

from statistical_test import *
from avalanche import LC 

export_path='../simulations/'

################################################################################
# SET PARAMETERS
################################################################################

# The values of the 7 parameters from the paper [Stern & Svensson, 1996] are
# mu      = 1.2 
# mu0     = 1
# alpha   = 4
# delta1  = -0.5
# delta2  = 0
# tau_min = 0.02
# tau_max = 26

# The 7 values obtained from v1 optimization are
# (3 loss)
# mu      = 1.3712230777324108
# mu0     = 1.292056879500315
# alpha   = 6.238631180118012
# delta1  = -0.5895371604462968
# delta2  = 0.21749228991192124
# tau_min = 0.06234108759332604
# tau_max = 23.443421866972386

# The 7 values obtained from v2 optimization are
# (4 loss)
# mu      = 1.3824946258409123
# mu0     = 1.15547634120758
# alpha   = 5.240511090395332
# delta1  = -0.45579705811174676
# delta2  = 0.1341616114704469
# tau_min = 0.003487215483012309
# tau_max = 32.858056193896196

# The 7 values obtained from v3 optimization are
# (4 loss)
# mu      = 1.3980875041410008
# mu0     = 1.5997385641936739
# alpha   = 3.8373579048667117
# delta1  = -0.5497159353657516
# delta2  = 0.12206808487464499
# tau_min = 0.00047431784713861797
# tau_max = 39.313297221735766

# The 7 values obtained from v4 optimization are
# (4 loss)
# mu      = 1.7377495777582268
# mu0     = 1.2674137674116688
# alpha   = 6.56892665444723
# delta1  = -0.5989803252226719
# delta2  = 0.02306881143876948
# tau_min = 6.478038929262871e-06
# tau_max = 45.936383095147605

# The 7 values obtained from v5 optimization are
# (4 loss)
# mu      = 1.8642165398675894
# mu0     = 0.9460684332226531
# alpha   = 6.539055496753974
# delta1  = -0.7805636907606287
# delta2  = 0.07414591188731365
# tau_min = 6.350848178629759e-06
# tau_max = 52.41492789344243

# The 7 values obtained from v6 optimization are
# (4 loss)
# mu      = 1.5355877552761932
# mu0     = 1.534168123065679
# alpha   = 3.1200524011794863
# delta1  = -0.7655182486991188
# delta2  = 0.2206237762670341
# tau_min = 0.0018477209878527603
# tau_max = 50.124910976218175

# The 7 values obtained from v7 optimization are
# (5 loss)
# mu      = 1.5197492009322398
# mu0     = 1.5588763589949317
# alpha   = 2.7027204695213194
# delta1  = -0.7741267250062283
# delta2  = 0.20809088491524874
# tau_min = 0.025098559904990592
# tau_max = 53.18239761751395

# The 7 values obtained from v8 optimization are
# (5 loss, Poisson)
# mu      = 1.264930172400689
# mu0     = 1.9545413768529967
# alpha   = 1.9791552702076287
# delta1  = -0.5633113305426156
# delta2  = 0.21883657826929145
# tau_min = 0.023468723791192015
# tau_max = 37.80588105257772

# The 7 values obtained from v9 optimization are
# (4 loss, Poisson)
# mu      = 1.3329447024643284
# mu0     = 1.2263029893911657
# alpha   = 3.273629312229965
# delta1  = -0.49675439969345714
# delta2  = 0.12694034684654393
# tau_min = 0.00011096965026213441
# tau_max = 41.172294302312366

# The 7 values obtained from v10 optimization are
# (one loss (<F/F_p>), Poisson)
# mu      = 0.9475251603474704
# mu0     = 0.9798380943908988
# alpha   = 3.5671437449135976
# delta1  = -0.7198120480484348
# delta2  = 0.14634415982397875
# tau_min = 5.535018240950338e-06
# tau_max = 27.408501454687663

# The 7 values obtained from v11 optimization are
# (one loss (<(F/F_p)**3>), Poisson)
# mu      = 1.275006111708082
# mu0     = 0.9241332593921731
# alpha   = 3.9560667766550814
# delta1  = -0.6039275951949501
# delta2  = 0.04714747706984365
# tau_min = 3.1160832088232976e-06
# tau_max = 27.685109594537607

# The 7 values obtained from v12 optimization are
# (one loss (<ACF>), Poisson)
# mu      = 0.9446841808077004
# mu0     = 0.9681754054935662
# alpha   = 1.757960546967845
# delta1  = -0.32164512051516003
# delta2  = 0.09051446260540383
# tau_min = 0.0009609765738533678
# tau_max = 31.1900145774892

# The 7 values obtained from v13 optimization are
# (one loss (duration distribution), Poisson)
# mu      = 1.423219221170363
# mu0     = 1.8486292768207997
# alpha   = 2.607690092406438
# delta1  = -0.6637684243758021
# delta2  = 0.20456507843804805
# tau_min = 0.007777873403883386
# tau_max = 47.147406633040376

# The 7 values obtained from v14 optimization are
# (4 loss, Poisson, correct weight, keep_elitism=0)
mu      = 1.3400110297200563
mu0     = 1.890238938249485
alpha   = 3.1950527492978287
delta1  = -0.4600048148857563
delta2  = 0.12234358572102152
tau_min = 5.563858305836977e-05
tau_max = 40.49163583715167


#------------------------------------------------------------------------------#

t_i=0   # [s]
t_f=150 # [s]

N_grb=1000

instrument = 'batse'
#instrument = 'swift'
#instrument = 'sax'
#instrument = 'sax_lr'

if instrument=='batse':
    res           = instr_batse['res']
    eff_area      = instr_batse['eff_area']
    bg_level      = instr_batse['bg_level']
    t90_threshold = instr_batse['t90_threshold']
    sn_threshold  = instr_batse['sn_threshold']
elif instrument=='swift':
    res           = instr_swift['res']
    eff_area      = instr_swift['eff_area']
    bg_level      = instr_swift['bg_level']
    t90_threshold = instr_swift['t90_threshold']
    sn_threshold  = instr_swift['sn_threshold']
elif instrument=='sax':
    res           = instr_sax['res']
    eff_area      = instr_sax['eff_area']
    bg_level      = instr_sax['bg_level']
    t90_threshold = instr_sax['t90_threshold']
    sn_threshold  = instr_sax['sn_threshold']
elif instrument=='sax_lr':
    res           = instr_sax_lr['res']
    eff_area      = instr_sax_lr['eff_area']
    bg_level      = instr_sax_lr['bg_level']
    t90_threshold = instr_sax_lr['t90_threshold']
    sn_threshold  = instr_sax_lr['sn_threshold']
else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax", "sax_lr".')


################################################################################
################################################################################
from datetime import datetime
start = datetime.now()

test_pulse_distr = True
test  = generate_GRBs(# number of simulated GRBs to produce
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
                      filter=True,
                      # other parameters
                      export_files=False, 
                      export_path=export_path, 
                      n_cut=2000, 
                      with_bg=False,
                      test_pulse_distr=test_pulse_distr)

if test_pulse_distr:
    pulse_out_file=open('./n_of_pulses.txt', 'w')
    for grb in test:
        pulse_out_file.write('{0}\n'.format(grb.num_of_sig_pulses))
    pulse_out_file.close()

if test_pulse_distr:
    n_of_pulses = [ grb.num_of_sig_pulses for grb in test ]

print('Time elapsed: ', (datetime.now() - start).seconds)
################################################################################
################################################################################
