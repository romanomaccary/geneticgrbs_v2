
import sys
import numpy as np

from statistical_test import *
    

sys.path.append('/home/bazzanini/PYTHON/genetic3_5metrics/statistical_test')
sys.path.append('/home/bazzanini/PYTHON/genetic3_5metrics/lc_pulse_avalanche')
sys.path.append('/home/ferro/lc_pulse_avalance/statistical_test')
sys.path.append('/home/ferro/lc_pulse_avalance/lc_pulse_avalanche')
sys.path.append('/home/maistrello/geneticgrbs/statistical_test')
sys.path.append('/home/maistrello/geneticgrbs/lc_pulse_avalanche')

batse_path = '/astrodata/guidorzi/CGRO_BATSE/'
swift_path = '/astrodata/guidorzi/Swift_BAT/'
sax_path   = '/astrodata/guidorzi/BeppoSAX_GRBM/'
fermi_path = '/astrodata/romain/GBM_LC_repository/' 



### Load the FERMI/GBM GRBs

# load all data
grb_list_fermi = load_lc_fermi(path=fermi_path) 

# apply constraints

t_i       = 0   # [s]
t_f       = 150 # [s]
t_f_sax   = 50  # [s]
t_f_fermi = 50  # [s]

bin_time_fermi     = instr_fermi['res'] # temporal resolution (bins) for BATSE [s]
test_times_fermi   = np.linspace(t_i, t_f_fermi, int((t_f_fermi-t_i)/bin_time_fermi))
sn_threshold_fermi = instr_fermi['sn_threshold']
t90_threshold      = instr_fermi['t90_threshold'] 
t90_frac           = 15

grb_list_fermi = apply_constraints(grb_list=grb_list_fermi, 
                                    bin_time=bin_time_fermi, 
                                    t90_threshold=t90_threshold, 
                                    t90_frac=t90_frac,
                                    sn_threshold=sn_threshold_fermi, 
                                    t_f=t_f_fermi, 
                                    filter=True)

