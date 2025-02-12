################################################################################
# IMPORT LIBRARIES
################################################################################
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')

from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

# Increase the recursion limit to avoid: 
# "RecursionError: maximum recursion depth exceeded in comparison"
rec_lim=50000
if sys.getrecursionlimit()<rec_lim:
    sys.setrecursionlimit(rec_lim)

# seed=42
# np.random.seed(SEED)

#------------------------------------------------------------------------------#
# Set the username for the path of the files:
#------------------------------------------------------------------------------#
#user='external_user'
#user='LB'
#user='AF'
#user='bach'
#user='gravity'
#user='pleiadi'
user = 'romano'
if user=='bach':
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
elif user=='gravity':
    sys.path.append('/home/bazzanini/PYTHON/genetic3/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic3/lc_pulse_avalanche')
elif user=='pleiadi':
    sys.path.append('/beegfs/mbulla/genetic_grbs/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/beegfs/mbulla/genetic_grbs/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
elif user=='LB':
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/statistical_test')
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/lc_pulse_avalanche')
    export_path='../simulations/'
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
elif user=='AF':
    sys.path.append('C:/Users/lisaf/Desktop/GitHub/lc_pulse_avalanche/statistical_test')
    sys.path.append('C:/Users/lisaf/Desktop/GitHub/lc_pulse_avalanche/lc_pulse_avalanche')
    export_path='C:/Users/lisaf/Desktop/'
elif user=='MM':
    sys.path.append('/home/manuele/Documents/university/grbs/geneticgrbs/statistical_test')
    sys.path.append('/home/manuele/Documents/university/grbs/geneticgrbs/lc_pulse_avalanche')
    export_path='/home/manuele/Documents/university/grbs/geneticgrbs_simulations/'
elif user == 'romano':
    sys.path.append('/astrodata/romain/sde_GA/geneticgrbs_v2/statistical_test')
    sys.path.append('/astrodata/romain/sde_GA/geneticgrbs_v2/lc_pulse_avalanche')
    export_path='/astrodata/romain/GA_SIMULATIONS/geneticgrbs_simulations_new_formulation_x0/'
elif user=='external_user':
    sys.path.append('../statistical_test')
    sys.path.append('../lc_pulse_avalanche')
    export_path=''
else:
    raise ValueError('Assign to the variable "user" a correct username!')

from statistical_test import *
from sde import LC 
    
if __name__ == '__main__':

    #--------------------------------------------------------------------------#
    # SET PARAMETERS External user: read params from config file
    #--------------------------------------------------------------------------#
    if user=='external_user':
        if len(sys.argv) != 2:
            print("\nERROR\n")
            print("Usage: python simulate_GRBs.py <config_file>")
            sys.exit(1)

        remove_instrument_path = True
        # Read argument from command line, and create the dictionary
        config_file = sys.argv[1]
        variables   = read_values(filename=config_file)
        # Create the folder where to store the simulated GRB LCs
        sim_dir, lcs_dir, timestamp = create_dir(variables=variables)
        variables['dir'] = str(lcs_dir)
        # Assign the variables to the corresponding values
        instrument  = variables['instrument']
        N_grb       = variables['N_grb']
        tau_i          = variables['tau_i']
        tau_d         = variables['tau_d']
        alpha       = variables['alpha']
        tau_se          = variables['tau_se']
        x0       = variables['x0']
        export_path = variables['dir']
        # Print the variables and their values
        print('==============================================')
        print('Generating LCs with the following properties:')
        print('==============================================')
        save_config_path = str(sim_dir)+'/config_'+timestamp+'.txt'
        save_config(variables=variables, file_name=save_config_path)
        print('==============================================') 

    #--------------------------------------------------------------------------#
    # SET PARAMETERS Read params from down below
    #--------------------------------------------------------------------------#
    else:
        remove_instrument_path = False
        #----------------------------------------------------------------------#
        instrument = 'batse'
        #instrument = 'swift'
        #instrument = 'sax'
        #instrument = 'sax_lr'
        #instrument = 'fermi'
        #----------------------------------------------------------------------#
        N_grb = 5000 # number of simulated GRBs to produce per set of parameters
        #----------------------------------------------------------------------#
        # BATSE
        #----------------------------------------------------------------------#
        if instrument=='batse':
            # Train best loss: 
            # Train average loss (last gen): 
            # sde Best parameters 10 gen
            #----------------------------------------------------------------------#
            
            tau_i       = 3.52  
            tau_d       = 8.05   
            alpha     = 1.48   
            tau_se    = 22.61  
            x0    = 3.67  
         #----------------------------------------------------------------------#
        # Swift 
        #----------------------------------------------------------------------#
        elif instrument=='swift':
            # Train best loss: 0.428
            # Train average loss (last gen): 0.892
            # SS96 parameters
            mu        = 1.06
            mu0       = 1.24 
            alpha     = 7.03
            delta1    = -1.37
            delta2    = 0.04
            tau_min   = 0.02
            tau_max   = 62.46
            # Peak flux distribution parameters
            alpha_bpl = 1.89
            beta_bpl  = 2.53
            F_break   = 3.44e-07
            F_min     = 1.41e-08
        #----------------------------------------------------------------------#


        #----------------------------------------------------------------------#
        # Fermi
        #----------------------------------------------------------------------#
        elif instrument=='fermi':
            # SS96 parameters
            # Train best loss: 0.537
            # Train average loss (last gen): 0.937
            mu        = 0.97
            mu0       = 1.55
            alpha     = 3.85
            delta1    = -0.99
            delta2    = 0.03
            tau_min   = 0.03
            tau_max   = 35.84
            # Peak flux distribution parameters
            alpha_bpl = 1.88
            beta_bpl  = 2.58
            F_break   = 2.88e-07
            F_min     = 6.04e-08
        #----------------------------------------------------------------------# 
        
        else:
            raise ValueError('Assign to the variable "instrument" a correct name!')

    # other parameters
    t_i   = 0    # [s]
    t_f   = 150  # [s]


    if instrument=='batse':
        res           = instr_batse['res']
        eff_area      = instr_batse['eff_area']
        bg_level      = instr_batse['bg_level']
        t90_threshold = instr_batse['t90_threshold']
        sn_threshold  = instr_batse['sn_threshold']
        sn_threshold_sup = instr_batse['sn_threshold_sup']
    elif instrument=='swift':
        res           = instr_swift['res']
        eff_area      = instr_swift['eff_area']
        bg_level      = instr_swift['bg_level']
        t90_threshold = instr_swift['t90_threshold']
        sn_threshold  = instr_swift['sn_threshold']
    # elif instrument=='sax':
    #     res           = instr_sax['res']
    #     eff_area      = instr_sax['eff_area']
    #     bg_level      = instr_sax['bg_level']
    #     t90_threshold = instr_sax['t90_threshold']
    #     sn_threshold  = instr_sax['sn_threshold']
    #     t_f           = 50 # s
    # elif instrument=='sax_lr':
    #     res           = instr_sax_lr['res']
    #     eff_area      = instr_sax_lr['eff_area']
    #     bg_level      = instr_sax_lr['bg_level']
    #     t90_threshold = instr_sax_lr['t90_threshold']
    #     sn_threshold  = instr_sax_lr['sn_threshold']
    elif instrument=='fermi':
        res           = instr_fermi['res']
        eff_area      = instr_fermi['eff_area']
        bg_level      = instr_fermi['bg_level']
        t90_threshold = instr_fermi['t90_threshold']
        sn_threshold  = instr_fermi['sn_threshold']
    else:
        raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", or "fermi".')


    ################################################################################
    ################################################################################
    from datetime import datetime
    start = datetime.now()

    test_pulse_distr = False # True
    test  = generate_GRBs(# number of simulated GRBs to produce
                          N_grb=N_grb,
                          # 7 parameters
                          tau_i=tau_i,
                          tau_d=tau_d,
                          alpha=alpha,
                          tau_se=tau_se,
                          x0=x0,
                          # instrument parameters
                          instrument=instrument,
                          bin_time=res,
                          eff_area=eff_area,
                          bg_level=bg_level,
                          # constraint parameters
                          t90_threshold=t90_threshold,
                          sn_threshold=sn_threshold,
                          sn_threshold_sup=sn_threshold_sup,
                          t_f=t_f,
                          filter=True,
                          # other parameters
                          export_files=True,
                          export_path=export_path,
                          with_bg=False,
                          remove_instrument_path=remove_instrument_path,
                          test_pulse_distr=test_pulse_distr,
                          ### 4 parameters of BPL
                          )

    if test_pulse_distr:
        pulse_out_file=open('./n_of_pulses.txt', 'w')
        for grb in test:
            pulse_out_file.write('{0}\n'.format(grb.num_of_sig_pulses))
        pulse_out_file.close()

    if test_pulse_distr:
        n_of_pulses = [ grb.num_of_sig_pulses for grb in test ]

    print('Time elapsed: ', (datetime.now() - start))
################################################################################
################################################################################

# ###############################################################################
# The 6 values obtained from v1 optimization are
# (5 loss)
#Parameters of the BEST solution:
# q      = 0.33
# a      = 0.165c
# alpha  = 3.847
# k      = 0.129
# t_0    = 7.792
# norm_A = 1.157e+06 
################################################################################
################################################################################

################################################################################
################################################################################
# The 6 values obtained from v2 optimization are (after having change the lower bound on A from 1e4 to 1e2)
# * Parameters of the BEST solution:
#     - q      = 0.30526957851320435
#     -  a     = 0.15263478925660218
#     - alpha   = 2.7080590009905228
#     - k  = 3.235686888523361
#     - t_0  = 32.73763035698295
#     - norm_A  = 164.98492999980897
# * Loss value of the best solution    : 1.521938741334827
# * Fitness value of the best solution : 0.6570566691291023
# * Best fitness value reached after N = 9 generations.
################################################################################
################################################################################

