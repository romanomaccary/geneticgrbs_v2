# -*- coding: utf-8 -*-
################################################################################
# # IMPORT LiBRARIES
################################################################################

#import os


from ast import If
import sys
import time
import pygad
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
#comment2

print(1)
### Increase the recursion limit to avoid: "RecursionError: maximum recursion depth exceeded in comparison"
rec_lim=50000
if sys.getrecursionlimit()<rec_lim:
    sys.setrecursionlimit(rec_lim)

### Suppress some warnings
# import warnings
# warnings.filterwarnings("ignore", message="p-value capped")
# warnings.filterwarnings("ignore", message="p-value floored")

### Plots
#import seaborn as sns
#sns.set_style('darkgrid')
#from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
save_plot=0

save_folder='/astrodata/romain/sde_GA/geneticgrbs_v2/genetic_algorithm/RESULT/result_sde_new_sde_formulation_x0_v2/'

random_seed=777
print(random_seed)
def fix_all_seeds(seed):
    #Fix randomness. Usage: fix_all_seeds(42)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = True

fix_all_seeds(random_seed)

print_time=True

################################################################################
# SET PATHS
################################################################################

### Set the username for the path of the files:
#user='LB'
#user='AF'
#user='bach'
#user='gravity'

user = 'romano'
#user='pleiadi'
#user = 'MM'

if user=='bach':
    # library paths
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = '/astrodata/guidorzi/CGRO_BATSE/'
    swift_path = '/astrodata/guidorzi/Swift_BAT/'
    sax_path   = '/astrodata/guidorzi/BeppoSAX_GRBM/'
elif user=='gravity':
    # library paths
    sys.path.append('/home/bazzanini/PYTHON/genetic3_5metrics/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic3_5metrics/lc_pulse_avalanche')
    sys.path.append('/home/ferro/lc_pulse_avalance/statistical_test')
    sys.path.append('/home/ferro/lc_pulse_avalance/lc_pulse_avalanche')
    sys.path.append('/home/maistrello/geneticgrbs/statistical_test')
    sys.path.append('/home/maistrello/geneticgrbs/lc_pulse_avalanche')
    # real data
    batse_path = '/astrodata/guidorzi/CGRO_BATSE/'
    swift_path = '/astrodata/guidorzi/Swift_BAT/'
    sax_path   = '/astrodata/guidorzi/BeppoSAX_GRBM/'
    fermi_path = '/astrodata/romain/GBM_LC_repository/data/' 
elif user=='pleiadi':
    # library paths
    sys.path.append('/beegfs/mbulla/genetic_grbs/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/beegfs/mbulla/genetic_grbs/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = '/beegfs/mbulla/genetic_grbs/CGRO_BATSE/'
    swift_path = '/beegfs/mbulla/genetic_grbs/Swift_BAT/'
    sax_path   = '/beegfs/mbulla/genetic_grbs/BeppoSAX_GRBM/'
elif user=='LB':
    # library paths
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/statistical_test')
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = '/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/DATA/CGRO_BATSE/'
    swift_path = '/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/DATA/Swift_BAT/'
    sax_path   = '/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/DATA/BeppoSAX_GRBM/'
elif user=='AF':
    # libraries
    sys.path.append('C:/Users/lisaf/Desktop/GitHub/lc_pulse_avalanche/statistical_test')
    sys.path.append('C:/Users/lisaf/Desktop/GitHub/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = 'D:/grb_to_test/CGRO_BATSE/'
    swift_path = 'D:/grb_to_test/Swift_BAT/'
    sax_path   = 'D:/grb_to_test/BeppoSAX_GRBM/'
elif user == 'MM':
    sys.path.append('/home/manuele/Documents/university/grbs/geneticgrbs/statistical_test')
    sys.path.append('/home/manuele/Documents/university/grbs/geneticgrbs/lc_pulse_avalanche')
    # real data
    batse_path = '/home/manuele/Documents/university/grbs/geneticgrbs_data/CGRO_BATSE/'
    swift_path = '/home/manuele/Documents/university/grbs/geneticgrbs_data/Swift_BAT/'
    sax_path   = '/home/manuele/Documents/university/grbs/geneticgrbs_data/BeppoSAX_GRBM/'
elif user == 'romano':
    # library paths
    sys.path.append('/astrodata/romain/sde_GA/geneticgrbs_v2/statistical_test')
    sys.path.append('/astrodata/romain/sde_GA/geneticgrbs_v2/lc_pulse_avalanche')
    # real data
    batse_path = '/astrodata/guidorzi/CGRO_BATSE/'
    swift_path = '/astrodata/guidorzi/Swift_BAT/'
    sax_path   = '/astrodata/guidorzi/BeppoSAX_GRBM/'
    fermi_path = '/astrodata/romain/GBM_LC_repository/data/' 
else:
    raise ValueError('Assign to the variable "user" a correct username!')

#from statistical_test_test import *
from statistical_test import *
from sde import LC




################################################################################
# SET PARAMETERS
################################################################################

### Choose the instrument
instrument = 'batse'
#instrument = 'swift'
#instrument = 'sax'
#instrument = 'fermi'

#------------------------------------------------------------------------------#

if instrument=='batse':
    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_batse['eff_area']      # 2025  # effective area of instrument [cm2]
    bg_level      = instr_batse['bg_level']      # 2.8   # background level [cnt/cm2/s]
    t90_threshold = instr_batse['t90_threshold'] # 2     # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_batse['sn_threshold']  # 70    # signal-to-noise ratio
    sn_threshold_sup = 1385.5025634765625*5 # Two times the maximum value of the sn_distr_real
    
    bin_time      = instr_batse['res']           # 0.064 # [s] temporal bins for BATSE (time resolution)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='swift':
    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_swift['eff_area']      # 1400 # effective area of instrument [cm2]
    bg_level      = instr_swift['bg_level']      # (10000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = instr_swift['t90_threshold'] # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_swift['sn_threshold']  # 15 # signal-to-noise ratio
    bin_time      = instr_swift['res']           # 0.064 # [s] temporal bins for Swift (time resolution)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='sax':
    t_i           = 0                          # [s]
    t_f           = 50                         # [s] (HR)
    eff_area      = instr_sax['eff_area']      # 420 # effective area of instrument [cm2]
    bg_level      = instr_sax['bg_level']      # (1000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = instr_sax['t90_threshold'] # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_sax['sn_threshold']  # 10 # signal-to-noise ratio
    bin_time      = instr_sax['res']           # 0.0078125 # [s] temporal bins for BeppoSAX (HR)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='sax_lr':
    t_i           = 0                              # [s]
    t_f           = 150                            # [s] (LR)
    eff_area      = instr_sax_lr['eff_area']       # 420 # effective area of instrument [cm2]
    bg_level      = instr_sax_lr['bg_level']       # (1000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = instr_sax_lr['t90_threshold']  # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_sax_lr['sn_threshold']   # 10   # signal-to-noise ratio
    bin_time      = instr_sax_lr['res']            # 1.0  # [s] temporal bins for BeppoSAX (LR)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='fermi':
    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_fermi['eff_area']      # 100 # effective area of instrument [cm2]
    bg_level      = instr_fermi['bg_level']      # 39.4 # background level [cnt/cm2/s]
    t90_threshold = instr_fermi['t90_threshold'] # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_fermi['sn_threshold']  # 5 # signal-to-noise ratio
    bin_time      = instr_fermi['res']           # 0.064 # [s] temporal bins for Fermi
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
# elif instrument=='batse_old': # it is actually for 'galileo'
#     t_f           = 150   # [s]
#     eff_area      = 3600  # effective area of instrument [cm2]
#     bg_level      = 10.67 # background level [cnt/cm2/s]
#     t90_threshold = 2     # [s] --> used to select only _long_ GRBs
#     t90_frac      = 15
#     sn_threshold  = 70    # signal-to-noise ratio
#     bin_time      = 0.064 # [s] temporal bins for BATSE (time resolution)
#     test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax", "sax_lr", and "fermi".')

#------------------------------------------------------------------------------#

# Genetic Algorithm parameters
parent_selection_type = "tournament" 
crossover_probability = 1                      # 'None' means couples parent k with parent k+1, otherwise it selects from the parents candidate list each one of them with probability 'crossover_probability', and then it takes two of them at random
initial_population    = None                   # if 'None', the initial population is randomly chosen using the 'sol_per_pop; and 'num_genes' parameters
mutation_type         = "random"
crossover_type        = "scattered"
num_generations       = 30                     # Number of generations.
sol_per_pop           = 5000                # Number of solutions in the population (i.e., number of different sets per generation).
num_parents_mating    = int(0.15*sol_per_pop)  # Number of solutions to be selected as parents in the mating pool.
keep_parents          = 100                      # if 0, keep NO parents (the ones selected for mating in the current population) in the next population
keep_elitism          = 0                      # keep in the next generation the best N solution of the current generation
mutation_probability  = 0.04                   # by default is 'None', otherwise it selects a value randomly from the current gene's space (each gene is changed with probability 'mutation_probability')
#mutation_probability=0.1

# Other parameters
N_grb            = 2000   # number of simulated GRBs to produce per set of parameters
test_sn_distr    = True   # add a fifth metric regarding  the S/N distribution (set True by default)
test_pulse_distr = False  # add a sixth metric regarding the distribution of number of pulses per GRB (set False by default)

# Options for parallelization
if user=='pleiadi':
    n_processes = int(os.environ['OMP_NUM_THREADS'])
else:
    n_processes = 100
parallel_processing  = ["process", n_processes]  # USE THIS ONE!  
#parallel_processing = ["thread", n_processes]   # this is slower
#parallel_processing = None                      # single thread

# Name of the pkl file where to save the GA instance at the end of the run
filename_model = save_folder+'geneticGRB_sde'

epsilon = 1.e-6

# full range of the parameters!
# range_q      = {"low": np.log10(1e3),            "high": np.log10(1e1)} #log scale
# range_a     = {"low": np.log10(1e-3)-np.log10(2),            "high": np.log10(1e1)-np.log10(2)} #log scale
# range_alpha   = {"low": 1,               "high": 4} # line scale
# range_t_0  = {"low": np.log10(1e-3),               "high": np.log10(1e3)}
# range_k  = {"low": np.log10(1e-3),            "high":np.log10(10)} #log scale
# range_norm_A     = {"low": np.log10(1e0),             "high": np.log10(1e8)} # sample norm


# range_q      = {"low": np.log10(1e-2),            "high": np.log10(1e0)} #log scale
# range_a     = {"low": np.log10(1e-3)-np.log10(2),            "high": np.log10(1e1)-np.log10(2)} #log scale
# range_alpha   = {"low": 1,               "high": 4} # line scale
# range_t_0  = {"low": np.log10(5e-2),               "high": np.log10(5e1)}
# range_k  = {"low": np.log10(5e-2),            "high":np.log10(5)} #log scale
# range_norm_A     = {"low": np.log10(1e4),             "high": np.log10(1e8)} # sample norm


### THIS ONES WORK WITHOUT BLOCKING!

# range_q      = {"low": np.log10(1e-2),            "high": np.log10(1e0)} #log scale
# range_a     = {"low": np.log10(1e-3)-np.log10(2),            "high": np.log10(1e1)-np.log10(2)} #log scale
# range_alpha   = {"low": 1,               "high": 4} # line scale
# range_t_0  = {"low": np.log10(6e-2),               "high": np.log10(5e1)}
# range_k  = {"low": np.log10(5e-2),            "high":np.log10(4)} #log scale
# range_norm_A     = {"low": np.log10(1e2),             "high": np.log10(1e8)} # sample norm

## NEW FORMULATION OF THE SDE

# range_tau_i      = {"low": np.log10(1e0),            "high": np.log10(1e1)} #log scale
# range_tau_d     = {"low": np.log10(1e0),            "high": np.log10(1e1)} #log scale
# range_alpha      = {"low": 1,                        "high": 4}  # linear scale
# range_tau_se     = {"low": np.log10(1e-2),             "high": np.log10(1e2)}
# range_x_min      = {"low":np.log10(1e2),           "high":np.log10(1e8)}
# range_alpha_pl   = {"low":1,                         "high":4}


## NEW FORMULATION OF THE SDE
## NEW RUN WITH BROADER RANGE OF PARAMS (last time some params reach upper bounds)

## first run with x0

# range_tau_i      = {"low": np.log10(1e0),            "high": np.log10(1e2)} #log scale
# range_tau_d     = {"low": np.log10(1e0),            "high": np.log10(1e2)} #log scale
# range_alpha      = {"low": 1,                        "high": 8}  # linear scale
# range_tau_se     = {"low": np.log10(1e-2),             "high": np.log10(1e3)}
# range_x0      = {"low":np.log10(1e0),           "high":np.log10(1e4)}

## second run with x0

range_tau_i      = {"low": np.log10(1e0),            "high": np.log10(1e2)} #log scale
range_tau_d     = {"low": np.log10(1e0),            "high": np.log10(1e2)} #log scale
range_alpha      = {"low": 0,                        "high": 8}  # linear scale
range_tau_se     = {"low": np.log10(1e-2),             "high": np.log10(1e3)}
range_x0      = {"low":np.log10(1e-1),           "high":np.log10(1e2)}

range_constraints = [range_tau_i,range_tau_d,range_alpha,range_tau_se,range_x0]

num_genes = len(range_constraints) 

nparams=5 # we fixed a=q/2, so the number of free parameters is 4

save_model = True

print('\n\n')
print('################################################################################')
print('START')
print('################################################################################')
print('\n\n')


################################################################################
# LOAD REAL DATA
################################################################################

init_load_time = time.perf_counter()

### Load the BATSE GRBs
if instrument=='batse':
    # load all data
    grb_list_real = load_lc_batse(path=batse_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real,
                                      bin_time=bin_time,
                                      t90_threshold=t90_threshold,
                                      t90_frac=t90_frac,
                                      sn_threshold=sn_threshold,
                                      sn_threshold_sup=sn_threshold_sup,
                                      t_f=t_f,
                                      zero_padding=True)
    # Load MEPSA results on BATSE (ONLY those that satisfy the constraint!)
    mepsa_out_file_list_temp = []
    for i in range(len(grb_list_real)):
        name = grb_list_real[i].name
        mepsa_out_file_list_temp.append(name)
    reb_factor          = np.inf
    peak_sn_level       = 10
    mepsa_out_file_list = [ batse_path+'PEAKS_ALL/peaks_'+el+'_all_bs_2.txt' for el in mepsa_out_file_list_temp ]
    if test_pulse_distr:
        n_of_pulses_real = readMEPSAres(mepsa_out_file_list=mepsa_out_file_list, # mepsa results on BATSE data
                                        maximum_reb_factor=reb_factor, 
                                        sn_level=peak_sn_level)
    else:
        n_of_pulses_real = None

### Load the Swift GRBs
elif instrument=='swift': 
    # load all data
    grb_list_real = load_lc_swift(path=swift_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold,
                                      t90_frac=t90_frac, 
                                      sn_threshold=sn_threshold, 
                                      sn_threshold_sup=sn_threshold_sup,
                                      t_f=t_f,
                                      zero_padding=True)
    n_of_pulses_real = None

### Load the BeppoSAX GRBs
elif instrument=='sax': 
    # load all (HR) data
    grb_list_real = load_lc_sax_hr(path=sax_path) 
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold, 
                                      t90_frac=t90_frac,
                                      sn_threshold=sn_threshold, 
                                      sn_threshold_sup=sn_threshold_sup,
                                      t_f=t_f, 
                                      zero_padding=True)
    n_of_pulses_real = None

### Load the Fermi/GBM GRBs
elif instrument=='fermi':
    # load all data
    grb_list_real = load_lc_fermi(path=fermi_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold, 
                                      t90_frac=t90_frac,
                                      sn_threshold=sn_threshold, 
                                      sn_threshold_sup=sn_threshold_sup,
                                      t_f=t_f, 
                                      zero_padding=True)
    n_of_pulses_real = None

### Load the Fermi GRBs
elif instrument=='fermi': 
    # load all data
    grb_list_real = load_lc_fermi(path=fermi_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real,
                                      bin_time=bin_time,
                                      t90_threshold=t90_threshold,
                                      t90_frac=t90_frac,
                                      sn_threshold=sn_threshold,
                                      sn_threshold_sup=sn_threshold_sup,
                                      t_f=t_f,
                                      zero_padding=True)    
    n_of_pulses_real = None    

else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax", "fermi".')

end_load_time = time.perf_counter()
print('\n')
print('--------------------------------------------------------------------------------')
print('* {} data loaded in {:0.0f} sec'.format(instrument,(end_load_time-init_load_time)))
print('--------------------------------------------------------------------------------')

################################################################################
# COMPUTE AVERAGE QUANTITIES OF REAL DATA
################################################################################

### TEST 1&2: Average Peak-Aligned Profiles
averaged_fluxes_real, \
averaged_fluxes_cube_real, \
averaged_fluxes_rms_real = compute_average_quantities(grb_list=grb_list_real,
                                                      t_f=t_f,
                                                      bin_time=bin_time,
                                                      filter=True)
### TEST 3: Autocorrelation
# For the REAL LCs we use the Link+93 formula to compute the autocorrelation,
# whereas for the simulated LCs instead we use the scipy.signal.correlate()
# function on the model curve, i.e., the one before adding the Poisson noise.
# N_lim = np.min( [N_grb, len(grb_list_real)] )
steps_real, acf_real = compute_autocorrelation(grb_list=grb_list_real,
                                               N_lim=len(grb_list_real),
                                               t_max=t_f,
                                               bin_time=bin_time,
                                               mode='link93',
                                               compute_rms=False)

### TEST 4: Duration
duration_real = [ evaluateDuration20(times=grb.times,
                                     counts=grb.counts,
                                     filter=True,
                                     t90=grb.t90,
                                     bin_time=bin_time)[0] for grb in grb_list_real ]
duration_distr_real = compute_kde_log_duration(duration_list=duration_real)

### TEST 5: S2N distribution
if test_sn_distr:
    sn_distr_real = [evaluateGRB_SN(grb.times, 
                                    grb.counts, 
                                    grb.errs, 
                                    grb.t90, 
                                    t90_frac, 
                                    bin_time,
                                    filter=True)[0] for grb in grb_list_real]
    sn_distr_real = np.array(sn_distr_real)
else:
    sn_distr_real = []

################################################################################
# DEFINE FITNESS FUNCTION OF THE GENETIC ALGORITHM
################################################################################

# pygad 3.X
def fitness_func(ga_instance, solution, solution_idx=None):
# pygad 2.X
#def fitness_func(solution, solution_idx=None):
    #--------------------------------------------------------------------------#
    # Generate the GRBs
    #--------------------------------------------------------------------------#
    
    ## if the solution extremely violates the condition tau_d <= 2tau_i, (like tau_d> 3*tau_i), we assign to this solution a fitness of 1e-9.
    if 10**solution[1] > 3*(10**solution[0]):
        print('conditioln not satisfied')
        return 1e-9
    
    if nparams==5:
        try:
            grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                                    N_grb=N_grb,
                                    # 5 parameters:
                                    tau_i=10**solution[0],
                                    tau_d=10**solution[1],
                                    alpha=solution[2],
                                    tau_se=10**solution[3],
                                    x0 = 10**solution[4],
                                    # instrument parameters:
                                    instrument=instrument,
                                    bin_time=bin_time,
                                    eff_area=eff_area,
                                    bg_level=bg_level,
                                    # constraint parameters:
                                    sn_threshold=sn_threshold,
                                    sn_threshold_sup=sn_threshold_sup,
                                    t90_threshold=t90_threshold,
                                    t90_frac=t90_frac,
                                    t_f=t_f,
                                    filter=True,
                                    # other parameters:
                                    export_files=False,
                                    with_bg=False,
                                    test_pulse_distr=test_pulse_distr
                                 )
        except Exception:
            #n_discarded += 1
            print("we can't generate good GRBs with these parameters",["%1.3f"%10**solution[0],"%1.3f"%10**solution[1],"%1.3f"%solution[2],"%1.3f"%10**solution[3],"%1.3f"%10**solution[4]])
            #print("fufa")
            return 1e-9

                                    
    elif nparams==4:
        try:
            grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                                    N_grb=N_grb,
                                    # 5 parameters:
                                    tau_i=10**solution[0],
                                    tau_d=0.5*(10**solution[0]),
                                    alpha=solution[2],
                                    tau_se=10**solution[3],
                                    x0=10**solution[4],
                                    # instrument parameters:
                                    instrument=instrument,
                                    bin_time=bin_time,
                                    eff_area=eff_area,
                                    bg_level=bg_level,
                                    # constraint parameters:
                                    sn_threshold=sn_threshold,
                                    sn_threshold_sup=sn_threshold_sup,
                                    t90_threshold=t90_threshold,
                                    t90_frac=t90_frac,
                                    t_f=t_f,
                                    filter=True,
                                    # other parameters:
                                    export_files=False,
                                    with_bg=False,
                                    test_pulse_distr=test_pulse_distr
                                    )
        except Exception:
            #n_discarded += 1
            print("we can't generate good GRBs with these parameters",solution)
            print('fitness')
            return 1e-9
    
    
    if test_pulse_distr:
        n_of_pulses_sim = [ grb.num_of_sig_pulses for grb in grb_list_sim ]
    else:
        n_of_pulses_sim = None

    #--------------------------------------------------------------------------#
    # COMPUTE AVERAGE QUANTITIES OF SIMULATED DATA
    #--------------------------------------------------------------------------#
    ### TEST 1&2: Average Peak-Aligned Profiles
    averaged_fluxes_sim, \
    averaged_fluxes_cube_sim, \
    averaged_fluxes_rms_sim = compute_average_quantities(grb_list=grb_list_sim,
                                                         t_f=t_f,
                                                         bin_time=bin_time,
                                                         filter=True)
    ### TEST 3: Autocorrelation
    # For the REAL LCs we use the Link+93 formula to compute the autocorrelation,
    # whereas for the simulated LCs instead we use the scipy.signal.correlate
    # function on the model curve, i.e., the one before adding the Poisson noise.
    steps_sim, acf_sim = compute_autocorrelation(grb_list=grb_list_sim,
                                                 N_lim=N_grb,
                                                 t_max=t_f,
                                                 bin_time=bin_time,
                                                 mode='scipy',
                                                 compute_rms=False)
    ### TEST 4: Duration
    #duration_sim = [ evaluateDuration20(times=grb.times, 
    #                                    counts=grb.counts,
    #                                    filter=True,
    #                                    t90=grb.t90,
    #                                    bin_time=bin_time)[0] for grb in grb_list_sim ]
    duration_sim       = np.array( [ grb.t20 for grb in grb_list_sim ] )
    duration_distr_sim = compute_kde_log_duration(duration_list=duration_sim)

    ### TEST 5: S2N distribution
    if test_sn_distr:
        sn_distr_sim = [evaluateGRB_SN(grb.times, 
                                        grb.counts, 
                                        grb.errs, 
                                        grb.t90, 
                                        t90_frac, 
                                        bin_time,
                                        filter=True)[0] for grb in grb_list_sim]
        sn_distr_sim = np.array(sn_distr_sim)
    else:
        sn_distr_sim = []

    #--------------------------------------------------------------------------#
    # COMPUTE LOSS
    #--------------------------------------------------------------------------#
    l2_loss = compute_loss(averaged_fluxes=averaged_fluxes_real,
                           averaged_fluxes_sim=averaged_fluxes_sim,
                           averaged_fluxes_cube=averaged_fluxes_cube_real,
                           averaged_fluxes_cube_sim=averaged_fluxes_cube_sim,
                           acf=acf_real,
                           acf_sim=acf_sim,
                           duration=duration_distr_real,
                           duration_sim=duration_distr_sim,
                           n_of_pulses=n_of_pulses_real,
                           n_of_pulses_sim=n_of_pulses_sim,
                           sn_distrib_real=sn_distr_real,
                           sn_distrib_sim=sn_distr_sim,
                           test_pulse_distr=test_pulse_distr,
                           test_sn_distr=test_sn_distr)
    fitness = 1.0 / (l2_loss + 1.e-9)
    print('fitness=',"%1.3f"%fitness)
    return fitness

################################################################################
# DEFINE AUXILIARY FUNCTION
################################################################################

def write_best_par_per_epoch(solution,loss, filename=save_folder+'best_par_per_epoch.txt'):
    """
    Function to write the best parameters of each generation in a file. The file
    is opened in append mode, so that we can append the results of eacch generation
    at the end of the file at each epoch.
    Parameters:
    - solution: array containing the best solution (7 params) of a generation.
    - filename: The name of the file to open in append mode. Default is 'output.txt'.
    """
    with open(filename, 'a') as file:
        file.write("tau_i        = {solution}".format(solution=10**solution[0])+'\n')
        file.write("tau_d       = {solution}".format(solution=10**solution[1])+'\n')
        file.write("alpha     = {solution}".format(solution=solution[2])+'\n')
        file.write("tau_se    = {solution}".format(solution=10**solution[3])+'\n')
        file.write("x0    = {solution}".format(solution=10**solution[4])+'\n')

        file.write(loss+'\n')

        file.write('\n')


last_fitness, last_loss, current_fitness, current_loss = 0, 0, 0, 0
def on_generation(ga_instance):
    """
    This function is executed after each generation. It prints useful 
    information of the current epoch, in particular, the details of the best
    solution in the current generation.
    """
    global last_fitness, last_loss, current_fitness, current_loss
    print('--------------------------------------------------------------------------------')
    print("Generation     = {generation}".format(generation=ga_instance.generations_completed))
    current_fitness       = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    current_loss          = current_fitness**(-1)                
    print("Best Loss      = {solution_loss}".format(solution_loss="%1.3f"%current_loss))
    print("Averaged Loss  = {avgd_loss}".format(avgd_loss="%1.3f"%np.median(ga_instance.last_generation_fitness**(-1))))
    print("Best Fitness   = {fitness}".format(fitness="%1.3f"%current_fitness))
    print("Fitness Change = {change}".format(change=current_fitness-last_fitness))
    last_fitness          = current_fitness
    last_loss             = current_loss
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    # Print the best solution of the current generation on TERMINAL
    print("Parameters of the best solution in the current generation:")
    print("    - tau_i        = {solution}".format(solution=10**solution[0]))
    print("    - tau_d        = {solution}".format(solution=10**solution[1]))
    print("    - alpha        = {solution}".format(solution=solution[2]))
    print("    - tau_se       = {solution}".format(solution=10**solution[3]))
    print("    - x0        = {solution}".format(solution=10**solution[4]))
    #print("    - x_min        = {solution}".format(solution=10**solution[4]))
        
    fitness_values = ga_instance.last_generation_fitness
    loss_values = fitness_values**(-1)
    #print('fitness values length',len(loss_values))
    #print('fitness_values',fitness_values)
    print('loss values below threshold',len( np.where(loss_values >1e8)[0] ))
    num_below_threshold = len( np.where(loss_values>1e8)[0] )/len(loss_values)
    print("Percentage of discarded solutions: ","%1.3f"%(100*num_below_threshold),"%")
    #print("Number of solutions that were discarded",n_discarded)
    #
    # n_discarded = 0 # reinitialize the counter of discarded solutions

    # Print the best solution of the current generation on FILE
    write_best_par_per_epoch(solution,"Best Loss      = {solution_loss}".format(solution_loss=current_loss))


def on_start(ga_instance):
    print("ga started")

def on_fitness(ga_instance, population_fitness):
    print("computing fitness...")

def on_parents(ga_instance, selected_parents):
    print("selecting parents...")

def on_crossover(ga_instance, offspring_crossover):
    print("do crossover ...")

def on_mutation(ga_instance, offspring_mutation):
    print("do mutation ...")

#def on_generation(ga_instance):
#    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

################################################################################
# INSTANTIATE THE 'GENETIC ALGORITHM' CLASS
################################################################################

if __name__ == '__main__':

    #global n_discarded
    #n_discarded=0

    #k= 3.891 t0= 0.145 q= 0.093 alpha= 1.461 norm 1.970e+03
    #solution = np.array([np.log10(0.093),0.1,1.461,np.log10(3.891),np.log10(0.145),np.log10(1.970e+03)])
    #print('fitness sol=',fitness_func(solution))

    # Usage: 
    #     - to run it the first time, type in the terminal: `python testGA.py`
    #     - to continue the run, type in the terminal:      `python testGA.py -c`


    # print(f"Arguments count: {len(sys.argv)}")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")

    # if len(sys.argv)==1: # python testGA.py
    #     print(f"\n[INFO] Running '{sys.argv[0]}'...\n")
    #     MODE = 'first'
    # elif '-c' in sys.argv: # python testGA.py -c
    #     print(f"\n[INFO] Resuming '{sys.argv[0]}' from checkpoint...\n")
    #     MODE = 'resume'
    # else:
    #     raise ValueError(f"Unrecognized command line argument(s): {sys.argv[1:]}")  
      
   # GRBs = generate_GRBs(10,                                                              # number of simulated GRBs to produce
    #               q, a, alpha, k, t_0,                                                # 7 parameters
    #               instrument, bin_time, eff_area, bg_level,                           # instrument parameters
    #               sn_threshold, t_f,                                                  # constraint parameters 
    #               t90_threshold, t90_frac=15, filter=True,                            # constraint parameters
    #               export_files=False, export_path='None',                             # other parameters
    #               n_cut=2500, with_bg=False, seed=None,                               # other parameters
    #               remove_instrument_path=False, test_pulse_distr=False,               # other parameters          
    #               )
    # GRBs
    MODE ='first'
    print("MODE=",MODE)

    print('seed=',random_seed)
    if MODE=='first':
        ga_GRB = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          sol_per_pop=sol_per_pop,
                          num_genes=num_genes,
                          gene_type=float,
                          initial_population=initial_population,
                          on_start=on_start,
                          on_fitness=on_fitness,
                          on_parents=on_parents,
                          on_crossover=on_crossover,
                          on_mutation=on_mutation,
                          on_generation=on_generation,
                          on_stop=on_stop,
                          ### fitness function:
                          fitness_func=fitness_func,
                          ### parent selection:
                          parent_selection_type=parent_selection_type,
                          keep_parents=keep_parents,           
                          keep_elitism=keep_elitism,           
                          ### crossover:
                          crossover_probability=crossover_probability,
                          crossover_type=crossover_type,
                          ### mutation:
                          mutation_type=mutation_type,
                          mutation_probability=mutation_probability,     
                          ### set range of parameters:
                          gene_space=range_constraints,
                          ### other stuff:
                          save_best_solutions=True,
                          save_solutions=True,
                          parallel_processing=parallel_processing,
                          random_seed=random_seed)
    
    elif MODE=='resume':
        # Load the saved GA instance
        ga_GRB = pygad.load(filename=filename_model)

        # Reload the fitness function (otherwise it will raise an error, I don't know why...)
        # pygad 3.X
        def fitness_func_reloaded(ga_instance, solution, solution_idx=None):
        # pygad 2.X
        # def fitness_func_reloaded(solution, solution_idx=None):
            # global loss_list
            #--------------------------------------------------------------------------#
            # Generate the GRBs
            #--------------------------------------------------------------------------#
            if nparams == 4 :
                try:
                    grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                                            N_grb=N_grb,
                                            # 5 parameters:
                                            tau_i=10**solution[0],
                                            tau_d=0.5*(10**solution[0]),
                                            alpha=solution[2],
                                            tau_se=10**solution[3],
                                            x0=10**solution[4],
                                            # instrument parameters:
                                            instrument=instrument,
                                            bin_time=bin_time,
                                            eff_area=eff_area,
                                            bg_level=bg_level,
                                            # constraint parameters:
                                            sn_threshold=sn_threshold,
                                            sn_threshold_sup=sn_threshold_sup,
                                            t90_threshold=t90_threshold,
                                            t90_frac=t90_frac,
                                            t_f=t_f,
                                            filter=True,
                                            # other parameters:
                                            export_files=False,
                                            with_bg=False,
                                            test_pulse_distr=test_pulse_distr)
                except Exception:
                    print("we can't generate good GRBs with these parameters",solution)
                    return 1e-9
            elif nparams == 5:
                try:
                    grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                                            N_grb=N_grb,
                                            # 5 parameters:
                                            tau_i=10**solution[0],
                                            tau_d=10**solution[1],
                                            alpha=solution[2],
                                            tau_se=10**solution[3],
                                            x0=10**solution[4],
                                            # instrument parameters:
                                            instrument=instrument,
                                            bin_time=bin_time,
                                            eff_area=eff_area,
                                            bg_level=bg_level,
                                            # constraint parameters:
                                            sn_threshold=sn_threshold,
                                            sn_threshold_sup=sn_threshold_sup,
                                            t90_threshold=t90_threshold,
                                            t90_frac=t90_frac,
                                            t_f=t_f,
                                            filter=True,
                                            # other parameters:
                                            export_files=False,
                                            with_bg=False,
                                            test_pulse_distr=test_pulse_distr)
                except Exception:
                    print("we can't generate good GRBs with these parameters",solution)
                    return 1e-9
            
        
            if test_pulse_distr:
                n_of_pulses_sim = [ grb.num_of_sig_pulses for grb in grb_list_sim ]
            else:
                n_of_pulses_sim = None

            #--------------------------------------------------------------------------#
            # Compute average quantities of simulated data needed for the loss function
            #--------------------------------------------------------------------------#
            ### TEST 1&2: Average Peak-Aligned Profiles
            averaged_fluxes_sim, \
            averaged_fluxes_cube_sim, \
            averaged_fluxes_rms_sim = compute_average_quantities(grb_list=grb_list_sim,
                                                                 t_f=t_f,
                                                                 bin_time=bin_time,
                                                                 filter=True)
            ### TEST 3: Autocorrelation
            # For the REAL LCs we use the Link+93 formula to compute the autocorrelation,
            # whereas for the simulated LCs instead we use the scipy.signal.correlate
            # function on the model curve, i.e., the one before adding the Poisson noise.
            steps_sim, acf_sim = compute_autocorrelation(grb_list=grb_list_sim,
                                                         N_lim=N_grb,
                                                         t_max=t_f,
                                                         bin_time=bin_time,
                                                         mode='scipy',
                                                         compute_rms=False)
            ### TEST 4: Duration
            #duration_sim = [ evaluateDuration20(times=grb.times, 
            #                                    counts=grb.counts,
            #                                    filter=True,
            #                                    t90=grb.t90,
            #                                    bin_time=bin_time)[0] for grb in grb_list_sim ]
            duration_sim       = np.array( [ grb.t20 for grb in grb_list_sim ] )
            duration_distr_sim = compute_kde_log_duration(duration_list=duration_sim)

            ### TEST 5: S2N distribution
            if test_sn_distr:
                sn_distr_sim = [evaluateGRB_SN(grb.times, 
                                                grb.counts, 
                                                grb.errs, 
                                                grb.t90, 
                                                t90_frac, 
                                                bin_time,
                                                filter=True)[0] for grb in grb_list_sim]
                sn_distr_sim = np.array(sn_distr_sim)
            else:
                sn_distr_sim = []


            #--------------------------------------------------------------------------#
            # Compute loss
            #--------------------------------------------------------------------------#
            l2_loss = compute_loss(averaged_fluxes=averaged_fluxes_real,
                                   averaged_fluxes_sim=averaged_fluxes_sim,
                                   averaged_fluxes_cube=averaged_fluxes_cube_real,
                                   averaged_fluxes_cube_sim=averaged_fluxes_cube_sim,
                                   acf=acf_real,
                                   acf_sim=acf_sim,
                                   duration=duration_distr_real,
                                   duration_sim=duration_distr_sim,
                                   n_of_pulses=n_of_pulses_real,
                                   n_of_pulses_sim=n_of_pulses_sim,
                                   test_pulse_distr=test_pulse_distr)
            fitness = 1.0 / (l2_loss + 1.e-9)
            return fitness


        ga_GRB.fitness_func = fitness_func_reloaded

    # print summary of the GA
    ga_GRB.summary() 

    ############################################################################
    # RUN THE GENETIC ALGORITHM
    ############################################################################

    init_run_time = time.perf_counter()
    print('\nStarting the GA...\n')
    ga_GRB.run()
    #ga_GRB.plot_fitness()
    end_run_time = time.perf_counter()

    print('\n')
    print('--------------------------------------------------------------------------------')
    print('* Model run in {:0.0f} sec'.format((end_run_time-init_run_time)))
    print('--------------------------------------------------------------------------------')


    ############################################################################
    # SAVE THE MODEL
    ############################################################################

    ### Save the GA instance
    if save_model:
        ga_GRB.save(filename=filename_model)


    ############################################################################
    # PRINT FINAL RESULTS
    ############################################################################

    #--------------------------------------------------------------------------#
    # Print on terminal
    #--------------------------------------------------------------------------#
    solution, solution_fitness, solution_idx = ga_GRB.best_solution(ga_GRB.last_generation_fitness)
    print('\n################################################################################')
    print('################################################################################')
    print("* Parameters of the BEST solution:")
    print("    - tau_i        = {solution}".format(solution=10**solution[0]))
    print("    - tau_d        = {solution}".format(solution=10**solution[1]))
    print("    - alpha        = {solution}".format(solution=solution[2]))
    print("    - tau_se       = {solution}".format(solution=10**solution[3]))
    print("    - x0       = {solution}".format(solution=10**solution[4]))

    
    print("* Loss value of the best solution    : {solution_loss}".format(solution_loss=solution_fitness**(-1)))
    print("* Fitness value of the best solution : {solution_fitness}".format(solution_fitness=solution_fitness))
    #print("Index of the best solution          : {solution_idx}".format(solution_idx=solution_idx))
    if ga_GRB.best_solution_generation != -1:
        print("* Best fitness value reached after N={best_solution_generation} generations.".format(best_solution_generation=ga_GRB.best_solution_generation))
    print('################################################################################')
    print('################################################################################')
    #--------------------------------------------------------------------------#
    # Print on file
    #--------------------------------------------------------------------------#
    file = open(save_folder+"simulation_info.txt", "w")
    file.write('################################################################################')
    file.write('\n')
    file.write("INPUT")
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('\n')
    file.write('N_GRBs_per_set       = {}'.format(N_grb))
    file.write('\n')
    file.write('num_generations      = {}'.format(num_generations))
    file.write('\n')
    file.write('sol_per_pop          = {}'.format(sol_per_pop))
    file.write('\n')
    file.write('num_parents_mating   = {}'.format(num_parents_mating))
    file.write('\n')
    file.write('keep_parents         = {}'.format(keep_parents))
    file.write('\n')
    file.write('keep_elitism         = {}'.format(keep_elitism))
    file.write('\n')
    file.write('mutation_probability = {}'.format(mutation_probability))
    file.write('\n')
    file.write('\n')
    file.write('range_tau_i             = {}'.format(range_tau_i))
    file.write('\n')
    file.write('range_tau_d           = {}'.format(range_tau_d))
    file.write('\n')
    file.write('range_alpha          = {}'.format(range_alpha))
    file.write('\n')
    file.write('range_tau_se         = {}'.format(range_tau_se))
    file.write('\n')
    file.write('range_x0         = {}'.format(range_x0))
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write("OUTPUT")
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('\n')
    file.write("* Parameters of the BEST solution:")
    file.write('\n')
    file.write("    - tau_i      = {solution}".format(solution=10**solution[0]))
    file.write('\n')
    file.write("    - tau_d     = {solution}".format(solution=(10**solution[1])))
    file.write('\n')
    file.write("    - alpha   = {solution}".format(solution=solution[2]))
    file.write('\n')
    file.write("    - tau_se  = {solution}".format(solution=10**solution[3]))
    file.write('\n')
    file.write("    - x0  = {solution}".format(solution=10**solution[4]))
    file.write('\n')
   
    file.write("* Loss value of the best solution    : {solution_loss}".format(solution_loss=solution_fitness**(-1)))
    file.write('\n')
    file.write("* Fitness value of the best solution : {solution_fitness}".format(solution_fitness=solution_fitness))
    file.write('\n')
    #print("Index of the best solution          : {solution_idx}".format(solution_idx=solution_idx))
    if ga_GRB.best_solution_generation != -1:
        file.write("* Best fitness value reached after N = {best_solution_generation} generations.".format(best_solution_generation=ga_GRB.best_solution_generation))
    file.write('\n')
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('################################################################################')
    file.close()
    #--------------------------------------------------------------------------#


    ############################################################################
    # EXPORT DATA FOR THE PLOT 1
    ############################################################################

    if MODE=='first':
        best_loss = np.array(ga_GRB.best_solutions_fitness)**(-1)
        loss_list = np.array(ga_GRB.solutions_fitness)**(-1)
        loss_list = loss_list[np.where(loss_list<1e7)[0]]
        avg_loss  = np.zeros(len(best_loss))
        std_loss  = np.zeros(len(best_loss))
        for i in range(len(best_loss)):
            avg_loss[i] = np.mean( loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
            std_loss[i] = np.std(  loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
        #print('best_loss[-1] =', best_loss[-1])

        datafile = save_folder+'datafile.txt'
        file = open(datafile, 'w')
        file.write('# generation\t best_loss\t avg_loss\t std_loss\t std_loss/sqrt(sol_per_pop)\n')
        for i in range(len(best_loss)):
            file.write('{0} {1} {2} {3} {4}\n'.format(i, best_loss[i], avg_loss[i], std_loss[i], std_loss[i]/np.sqrt(sol_per_pop)))
        file.close()

    elif MODE=='resume':
        best_loss = np.array(ga_GRB.best_solutions_fitness)**(-1)
        loss_list = np.array(ga_GRB.solutions_fitness)**(-1)
        avg_loss  = np.zeros(len(best_loss))
        std_loss  = np.zeros(len(best_loss))
        best_loss_print = []
        avg_loss_print  = []
        std_loss_print  = []
        for i in range(len(best_loss)):
            avg_loss[i] = np.mean( loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
            std_loss[i] = np.std(  loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
        #print('best_loss[-1] =', best_loss[-1])
        for i in range(len(best_loss)):
            if i%(num_generations+1)==0 and i!=0:
                pass
            else:
                best_loss_print.append(best_loss[i])
                avg_loss_print.append(avg_loss[i])
                std_loss_print.append(std_loss[i])              

        datafile = save_folder+'datafile.txt'
        file = open(datafile, 'w')
        file.write('# generation\t best_loss\t avg_loss\t std_loss\t std_loss/sqrt(sol_per_pop)\n')
        for i in range(len(best_loss_print)):
            file.write('{0} {1} {2} {3} {4}\n'.format(i, best_loss_print[i], avg_loss_print[i], std_loss_print[i], std_loss_print[i]/np.sqrt(sol_per_pop)))
        file.close()

    ############################################################################
    # PLOT THE RESULTS
    ############################################################################

    if save_plot:
        if MODE=='first':
            plt.plot(best_loss, ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Best Loss')
            plt.savefig(save_folder+'fig01.pdf')
            plt.clf()

            plt.errorbar(np.arange(len(best_loss)), avg_loss, yerr=std_loss/np.sqrt(sol_per_pop), ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Average Loss')
            plt.savefig(save_folder+'fig02.pdf')
            plt.clf()

            plt.plot(std_loss, ls='-', lw=2, c='b')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Standard Deviation of the loss')
            plt.savefig(save_folder+'fig03.pdf')
            plt.clf()

        elif MODE=='resume':
            plt.plot(np.array(best_loss_print), ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Best Loss')
            plt.savefig(save_folder+'fig01.pdf')
            plt.clf()

            plt.errorbar(np.arange(len(best_loss_print)), np.array(avg_loss_print), yerr=np.array(std_loss_print)/np.sqrt(sol_per_pop), ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Average Loss')
            plt.savefig(save_folder+'fig02.pdf')
            plt.clf()

            plt.plot(np.array(std_loss_print), ls='-', lw=2, c='b')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Standard Deviation of the loss')
            plt.savefig(save_folder+'fig03.pdf')
            plt.clf()
    

    ############################################################################
    # EXPORT DATA FOR THE PLOT 2
    ############################################################################
    # Here we save the parameters of ALL the individuals in ALL generations,
    # along with their associated fitness.

    # all fitness values in the ALL epochs:
    all_gen_fitness = np.array(ga_GRB.solutions_fitness[:])

    # all solutions in the ALL epochs:
    all_gen_sol       = np.array(ga_GRB.solutions[:])
    all_gen_tau_i        = 10**np.array(all_gen_sol[:,0])       # array with all the mu      of the ALL generations 
    all_gen_tau_d       = 10**np.array(all_gen_sol[:,1])       # array with all the mu0     of the ALL generations
    all_gen_alpha     = np.array(all_gen_sol[:,2])       # array with all the alpha   of the ALL generations
    all_gen_tau_se    = 10**np.array(all_gen_sol[:,3])       # array with all the delta1  of the ALL generations
    all_gen_x0    = 10**np.array(all_gen_sol[:,4])       # array with all the delta1  of the ALL generations

    data_all_gen = {
        'tau_i':        all_gen_tau_i,
        'tau_d':       all_gen_tau_d,
        'alpha':     all_gen_alpha,
        'tau_se':    all_gen_tau_se,
        'x0': all_gen_x0,
        'fitness':   all_gen_fitness
    }
    df_all_gen = pd.DataFrame(data_all_gen)
    df_all_gen.to_csv(save_folder+'df_all_gen.csv', index=False)    
    
    ############################################################################
    ############################################################################

    print('\n')
    print('################################################################################')
    print('END')
    print('################################################################################')
