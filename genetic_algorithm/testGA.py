################################################################################
# IMPORT LIBRARIES
################################################################################

import os
import sys
import time
import pygad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Increase the recursion limit
rec_lim=40000
if sys.getrecursionlimit()<rec_lim:
    sys.setrecursionlimit(rec_lim)

### Suppress some warnings
import warnings
warnings.filterwarnings("ignore", message="p-value capped")
warnings.filterwarnings("ignore", message="p-value floored")

### Plots
#import seaborn as sns
#sns.set_style('darkgrid')
#from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
save_plot=0

random_seed=None
#random_seed=42
#np.random.seed(random_seed)

print_time=True

################################################################################
# SET PATHS
################################################################################

### Set the username for the path of the files:
#user='LB'
#user='AF'
user='bach'
if user=='bach':
    # library paths
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/statistical_test')
    sys.path.append('/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = '/astrodata/guidorzi/CGRO_BATSE/'
    swift_path = '/astrodata/guidorzi/Swift_BAT/'
    sax_path   = '/astrodata/guidorzi/BeppoSAX_GRBM/'
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
    sys.path.append('C:/Users/Lisa/Documents/GitHub/lc_pulse_avalanche/statistical_test')
    sys.path.append('C:/Users/Lisa/Documents/GitHub/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = 'E:/grb_to_test/CGRO_BATSE/'
    swift_path = 'E:/grb_to_test/Swift_BAT/'
    sax_path   = 'E:/grb_to_test/BeppoSAX_GRBM/'
else:
    raise ValueError('Assign to the variable "user" a correct username!')

from statistical_test import *
from avalanche import LC


################################################################################
# SET PARAMETERS
################################################################################

### Choose the instrument
#instrument = 'batse'
instrument = 'swift'
#instrument = 'sax'

#------------------------------------------------------------------------------#

if instrument=='batse':
    t_i           = 0     # [s]
    t_f           = 150   # [s]
    eff_area      = 3600  # effective area of instrument [cm2]
    bg_level      = 10.67 # background level [cnt/cm2/s]
    t90_threshold = 2     # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = 70    # signal-to-noise ratio
    bin_time      = 0.064 # [s] temporal bins for BATSE (its time resolution)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='swift':
    t_i           = 0                # [s]
    t_f           = 150              # [s]
    eff_area      = 1400             # effective area of instrument [cm2]
    bg_level      = (10000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = 2                # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = 20               # signal-to-noise ratio
    bin_time      = 0.064            # [s] temporal bins for Swift (its time resolution)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='sax':
    t_i           = 0               # [s]
    #t_f          = 150             # [s]
    t_f           = 50              # [s] (HR)
    eff_area      = 420             # effective area of instrument [cm2]
    bg_level      = (1000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = 2               # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = 10              # signal-to-noise ratio
    #bin_time     = 1.0             # [s] temporal bins for BeppoSAX
    bin_time      = 0.0078125       # [s] temporal bins for BeppoSAX (HR)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax".')

#------------------------------------------------------------------------------#

# parent_selection_type = "sss"
# crossover_probability = 0.5   # 'None' means couples parent k with parent k+1, otherwise it selects from the parents candidate list each one of them with probability 'crossover_probability', and then it takes two of them at random
# initial_population    = None  # if 'None', the initial population is randomly chosen using the 'sol_per_pop; and 'num_genes' parameters
# mutation_type         = "random"
# crossover_type        = "single_point"
# num_generations       = 20                     # Number of generations.
# sol_per_pop           = 500                    # Number of solutions in the population (i.e., number of different sets per generation).
# num_parents_mating    = int(0.20*sol_per_pop)  # Number of solutions to be selected as parents in the mating pool.
# keep_parents          = 0                      # if 0, keep NO parents (the ones selected for mating in the current population) in the next population
# keep_elitism          = int(sol_per_pop*0.005) # keep in the next generation the best N solution of the current generation
# mutation_probability  = 0.03                   # by default is 'None', otherwise it selects a value randomly from the current gene's space (each gene is changed with probability 'mutation_probability')

parent_selection_type = "tournament" 
crossover_probability = 1     # 'None' means couples parent k with parent k+1, otherwise it selects from the parents candidate list each one of them with probability 'crossover_probability', and then it takes two of them at random
initial_population    = None  # if 'None', the initial population is randomly chosen using the 'sol_per_pop; and 'num_genes' parameters
mutation_type         = "random"
crossover_type        = "scattered"
num_generations       = 15                     # Number of generations.
sol_per_pop           = 2000                   # Number of solutions in the population (i.e., number of different sets per generation).
num_parents_mating    = int(0.20*sol_per_pop)  # Number of solutions to be selected as parents in the mating pool.
keep_parents          = 0                      # if 0, keep NO parents (the ones selected for mating in the current population) in the next population
keep_elitism          = 0                      # keep in the next generation the best N solution of the current generation
mutation_probability  = 0.04                   # by default is 'None', otherwise it selects a value randomly from the current gene's space (each gene is changed with probability 'mutation_probability')

N_grb                 = 2000                   # number of simulated GRBs to produce per set of parameters
test_pulse_distr      = False                  # add a fifth metric regarding the distribution of number of pulses per GRB (set False by default)

# Options for parallelization:
parallel_processing  = ["process", 50]      
#parallel_processing = ["thread", 50]       
#parallel_processing = None

# We impose constraints on the range of values that the 7 parameter can assume
range_mu      = {"low": 0.80,            "high": 1.7}
range_mu0     = {"low": 0.80,            "high": 1.9} 
range_alpha   = {"low": 1,               "high": 15} 
range_delta1  = {"low": -1.5,            "high": -0.25-1.e-6} 
range_delta2  = {"low": 0,               "high": 0.25}
range_tau_min = {"low": np.log10(1.e-6), "high": np.log10(bin_time-1.e-6)} # sample uniformly in log space
range_tau_max = {"low": bin_time+10,     "high": 55} 
# The values of the 7 parameters from the paper [Stern & Svensson, 1996] are:
# mu=1.2
# mu0=1
# alpha=4
# delta1=-0.5
# delta2=0
# tau_min=0.02
# tau_max=26

range_constraints = [range_mu, 
                     range_mu0,
                     range_alpha,
                     range_delta1, 
                     range_delta2, 
                     range_tau_min, 
                     range_tau_max]

num_genes = len(range_constraints) # 7

save_model = 1

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
                                      t_f=t_f)
    # load MEPSA results on BATSE (only those that satisfy the constraint!)
    mepsa_out_file_list_temp = []
    for i in range(len(grb_list_real)):
        name = grb_list_real[i].name
        mepsa_out_file_list_temp.append(name)
    reb_factor          = np.inf
    peak_sn_level       = 10
    mepsa_out_file_list = [ batse_path+'PEAKS_ALL/peaks_'+el+'_all_bs_2.txt' for el in mepsa_out_file_list_temp ]
    n_of_pulses_real    = readMEPSAres(mepsa_out_file_list=mepsa_out_file_list, # mepsa results on BATSE data
                                       maximum_reb_factor=reb_factor, 
                                       sn_level=peak_sn_level)
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
                                      t_f=t_f)
    n_of_pulses_real=None
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
                                      t_f=t_f)
    n_of_pulses_real=None
else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax".')

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
# whereas for the simulated LCs instead we use the scipy.signal.correlate
# function on the model curve, i.e., the one before adding the Poisson noise.
N_lim = np.min( [N_grb, len(grb_list_real)] )
steps_real, acf_real = compute_autocorrelation(grb_list=grb_list_real,
                                               N_lim=N_lim,
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


################################################################################
# DEFINE FITNESS FUNCTION OF THE GENETIC ALGORITHM
################################################################################

def fitness_func(solution, solution_idx=None):
    global loss_list
    #--------------------------------------------------------------------------#
    # Generate the GRBs
    #--------------------------------------------------------------------------#
    grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                                 N_grb=N_grb,
                                 # 7 parameters:
                                 mu=solution[0],
                                 mu0=solution[1],
                                 alpha=solution[2],
                                 delta1=solution[3],
                                 delta2=solution[4],
                                 tau_min=10**solution[5],  # sample uniformly in log space
                                 tau_max=solution[6],
                                 # instrument parameters:
                                 instrument=instrument,
                                 bin_time=bin_time,
                                 eff_area=eff_area,
                                 bg_level=bg_level,
                                 # constraint parameters:
                                 sn_threshold=sn_threshold,
                                 t90_threshold=t90_threshold,
                                 t90_frac=t90_frac,
                                 t_f=t_f,
                                 filter=True,
                                 # other parameters:
                                 export_files=False,
                                 n_cut=2500,
                                 with_bg=False,
                                 test_pulse_distr=test_pulse_distr)
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
                                                 N_lim=N_lim,
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


################################################################################
# DEFINE AUXILIARY FUNCTION
################################################################################

last_fitness, last_loss, current_fitness, current_loss = 0, 0, 0, 0
def on_generation(ga_instance):
    """
    This function is executed after _each_ generation. It prints useful 
    information of the current epoch, in particular, the details of the best
    solution in the current generation.
    """
    global last_fitness, last_loss, current_fitness, current_loss
    print('--------------------------------------------------------------------------------')
    print("Generation     = {generation}".format(generation=ga_instance.generations_completed))
    current_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    current_loss    = current_fitness**(-1)                
    print("Best Loss      = {solution_loss}".format(solution_loss=current_loss))
    print("Best Fitness   = {fitness}".format(fitness=current_fitness))
    print("Fitness Change = {change}".format(change=current_fitness-last_fitness))
    last_fitness = current_fitness
    last_loss    = current_loss
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Parameters of the best solution in the current generation:")
    print("    - mu      = {solution}".format(solution=solution[0]))
    print("    - mu0     = {solution}".format(solution=solution[1]))
    print("    - alpha   = {solution}".format(solution=solution[2]))
    print("    - delta1  = {solution}".format(solution=solution[3]))
    print("    - delta2  = {solution}".format(solution=solution[4]))
    print("    - tau_min = {solution}".format(solution=10**(solution[5])))
    print("    - tau_max = {solution}".format(solution=solution[6]))

    # scrivere codice per salvare i risultati intermedi in un file; aprire il 
    # file in append mode!

################################################################################
# INSTANTIATE THE 'GENETIC ALGORITHM' CLASS
################################################################################

if __name__ == '__main__':

    ga_GRB = pygad.GA(num_generations=num_generations,
                      num_parents_mating=num_parents_mating,
                      sol_per_pop=sol_per_pop,
                      num_genes=num_genes,
                      gene_type=float,
                      initial_population=initial_population,
                      on_generation=on_generation,
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

    # print summary of the GA
    ga_GRB.summary()


    ############################################################################
    # RUN THE GENETIC ALGORITHM
    ############################################################################

    init_run_time = time.perf_counter()
    #print('Starting the GA...\n')
    # Run the GA to optimize the parameters of the function.
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
        filename_model = 'geneticGRB'
        ga_GRB.save(filename=filename_model)

    ### Load the saved GA instance
    # loaded_ga_instance = pygad.load(filename=filename)
    # loaded_ga_instance.plot_fitness()


    ############################################################################
    # PRINT FINAL RESULTS
    ############################################################################

    #--------------------------------------------------------------------------#
    # Print on terminal
    #--------------------------------------------------------------------------#
    solution, solution_fitness, solution_idx = ga_GRB.best_solution(ga_GRB.last_generation_fitness)
    print('\n')
    print('################################################################################')
    print('################################################################################')
    print("* Parameters of the BEST solution:")
    print("    - mu      = {solution}".format(solution=solution[0]))
    print("    - mu0     = {solution}".format(solution=solution[1]))
    print("    - alpha   = {solution}".format(solution=solution[2]))
    print("    - delta1  = {solution}".format(solution=solution[3]))
    print("    - delta2  = {solution}".format(solution=solution[4]))
    print("    - tau_min = {solution}".format(solution=10**(solution[5])))
    print("    - tau_max = {solution}".format(solution=solution[6]))
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
    file = open("./simulation_info.txt", "w")
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
    file.write('range_mu             = {}'.format(range_mu))
    file.write('\n')
    file.write('range_mu0            = {}'.format(range_mu0))
    file.write('\n')
    file.write('range_alpha          = {}'.format(range_alpha))
    file.write('\n')
    file.write('range_delta1         = {}'.format(range_delta1))
    file.write('\n')
    file.write('range_delta2         = {}'.format(range_delta2))
    file.write('\n')
    file.write('range_tau_min        = {}'.format(range_tau_min))
    file.write('\n')
    file.write('range_tau_max        = {}'.format(range_tau_max))
    file.write('\n')
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
    file.write("    - mu      = {solution}".format(solution=solution[0]))
    file.write('\n')
    file.write("    - mu0     = {solution}".format(solution=solution[1]))
    file.write('\n')
    file.write("    - alpha   = {solution}".format(solution=solution[2]))
    file.write('\n')
    file.write("    - delta1  = {solution}".format(solution=solution[3]))
    file.write('\n')
    file.write("    - delta2  = {solution}".format(solution=solution[4]))
    file.write('\n')
    file.write("    - tau_min = {solution}".format(solution=10**(solution[5])))
    file.write('\n')
    file.write("    - tau_max = {solution}".format(solution=solution[6]))
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

    best_loss = np.array(ga_GRB.best_solutions_fitness)**(-1)
    loss_list = np.array(ga_GRB.solutions_fitness)**(-1)
    avg_loss  = np.zeros(num_generations+1)
    std_loss  = np.zeros(num_generations+1)
    for i in range(num_generations+1):
        avg_loss[i] = np.mean( loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
        std_loss[i] = np.std(  loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
    #print('best_loss[-1] =', best_loss[-1])

    datafile = './datafile.txt'
    file = open(datafile, 'w')
    file.write('# generation\t best_loss\t avg_loss\t std_loss\t std_loss/sqrt(sol_per_pop)\n')
    for i in range(num_generations+1):
        file.write('{0} {1} {2} {3} {4}\n'.format(i, best_loss[i], avg_loss[i], std_loss[i], std_loss[i]/np.sqrt(sol_per_pop)))
    file.close()


    ############################################################################
    # EXPORT PLOT 1
    ############################################################################

    if save_plot:
        plt.plot(best_loss, ls='-', lw=2, c='b')
        #plt.yscale('log')
        plt.xlabel(r'Generation')
        plt.ylabel(r'Best Loss')
        plt.savefig('fig01.pdf')
        plt.clf()

        plt.errorbar(np.arange(num_generations+1), avg_loss, yerr=std_loss/np.sqrt(sol_per_pop), ls='-', lw=2, c='b')
        #plt.yscale('log')
        plt.xlabel(r'Generation')
        plt.ylabel(r'Average Loss')
        plt.savefig('fig02.pdf')
        plt.clf()

        plt.plot(std_loss, ls='-', lw=2, c='b')
        plt.xlabel(r'Generation')
        plt.ylabel(r'Standard Deviation of the loss')
        plt.savefig('fig03.pdf')
        plt.clf()
    

    ############################################################################
    # EXPORT DATA FOR THE PLOT 2
    ############################################################################
    # Here we save the parameters of ALL the individuals in ALL generations,
    # along with their associated fitness.

    # all fitness values in the ALL epochs:
    all_gen_fitness = np.array(ga_GRB.solutions_fitness[:])

    # all solutions in the ALL epochs:
    all_gen_sol     = np.array(ga_GRB.solutions[:])
    all_gen_mu      = np.array(all_gen_sol[:,0])       # array with all the mu      of the ALL generations 
    all_gen_mu0     = np.array(all_gen_sol[:,1])       # array with all the mu0     of the ALL generations
    all_gen_alpha   = np.array(all_gen_sol[:,2])       # array with all the alpha   of the ALL generations
    all_gen_delta1  = np.array(all_gen_sol[:,3])       # array with all the delta1  of the ALL generations
    all_gen_delta2  = np.array(all_gen_sol[:,4])       # array with all the delta1  of the ALL generations
    all_gen_tau_min = 10**(np.array(all_gen_sol[:,5])) # array with all the tau_min of the ALL generations
    all_gen_tau_max = np.array(all_gen_sol[:,6])       # array with all the tau_max of the ALL generations

    data_all_gen = {
        'mu':      all_gen_mu,
        'mu0':     all_gen_mu0,
        'alpha':   all_gen_alpha,
        'delta1':  all_gen_delta1,
        'delta2':  all_gen_delta2,
        'tau_min': all_gen_tau_min,
        'tau_max': all_gen_tau_max,
        'fitness': all_gen_fitness
    }
    df_all_gen = pd.DataFrame(data_all_gen)
    df_all_gen.to_csv('./df_all_gen.csv', index=False)    
    
    ############################################################################
    ############################################################################

    print('\n\n')
    print('################################################################################')
    print('END')
    print('################################################################################')
    print('\n\n')

