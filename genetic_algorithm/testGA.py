################################################################################
# IMPORT LIBRARIES
################################################################################

import os
import sys
import pygad
#import yaml, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

seed=42
np.random.seed(seed)


################################################################################
# SET PATHS
################################################################################

# set the username for the path of the files:
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
    sys.path.append('/home/lorenzo/git/lc_pulse_avalanche/statistical_test')
    sys.path.append('/home/lorenzo/git/lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = '/home/lorenzo/Desktop/Astrophysics/PYTHON/DATA/CGRO_BATSE/'
    swift_path = '/home/lorenzo/Desktop/Astrophysics/PYTHON/DATA/Swift_BAT/'
    sax_path   = '/home/lorenzo/Desktop/Astrophysics/PYTHON/DATA/BeppoSAX_GRBM/'
elif user=='AF':
    # libraries
    sys.path.append('......WRITE_HERE....../lc_pulse_avalanche/statistical_test')
    sys.path.append('......WRITE_HERE....../lc_pulse_avalanche/lc_pulse_avalanche')
    # real data
    batse_path = 'E:/grb_to_test/CGRO_BATSE/'
    swift_path = 'E:/grb_to_test/Swift_BAT/'
    sax_path   = 'E:/grb_to_test/BeppoSAX_GRBM/'
else:
    raise ValueError('Assign to the variable "user" a correct username!')

from statistical_test import *
from avalanche import LC, Restored_LC


################################################################################
# SET PARAMETERS
################################################################################

# choose the instrument
instrument = 'batse'
#instrument = 'swift'
#instrument = 'sax'

#------------------------------------------------------------------------------#

if instrument=='batse':
    t_i           = 0     # [s]
    t_f           = 150   # [s]
    eff_area      = 3600  # effective area of instrument [cm2]
    bg_level      = 10.67 # background level [cnt/cm2/s]
    t90_threshold = 2     # [s] --> used to select only _long_ GRBs
    sn_threshold  = 70    # signal-to-noise ratio
    bin_time      = 0.064 # [s] temporal bins for BATSE (its time resolution)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
elif instrument=='swift':
    t_i           = 0                # [s]
    t_f           = 150              # [s]
    eff_area      = 1400             # effective area of instrument [cm2]
    bg_level      = (10000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = 2                # [s] --> used to select only _long_ GRBs
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
    sn_threshold  = 10              # signal-to-noise ratio
    #bin_time     = 1.0             # [s] temporal bins for BeppoSAX
    bin_time      = 0.0078125       # [s] temporal bins for BeppoSAX (HR)
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax".')

#------------------------------------------------------------------------------#

num_generations      = 6 # 10                 # Number of generations.
sol_per_pop          = 2000                   # Number of solutions in the population.
num_parents_mating   = int(0.20*sol_per_pop)  # Number of solutions to be selected as parents in the mating pool.
keep_parents         = 0                      # if 0, keep NO parents (the ones selected for mating in the current population) in the next population
keep_elitism         = int(sol_per_pop*0.005) # keep in the next generation the best N solution of the current generation
mutation_probability = 0.01                   # by default is 'None', otherwise it selects a value randomly from the current gene's space (each gene is changed with probability 'mutation_probability')

# The values of the 7 parameters from the paper [Stern & Svensson, 1996] are
# mu=1.2
# mu0=1
# alpha=4
# delta1=-0.5
# delta2=0
# tau_min=0.02
# tau_max=26

# We impose constraints on the range of values that the 7 parameter can assume
range_mu      = {"low": 0.75,            "high": 1.5}
range_mu0     = {"low": 0.75,            "high": 1.5} 
range_alpha   = {"low": 1,               "high": 10} 
range_delta1  = {"low": -1.5,            "high": -0.25-1.e-9} 
range_delta2  = {"low": np.log10(1.e-9), "high": np.log10(0.25)}           # sample uniformly in log space
range_tau_min = {"low": np.log10(1.e-9), "high": np.log10(bin_time-1.e-9)} # sample uniformly in log space
range_tau_max = {"low": bin_time+15,     "high": 35} 

range_constraints = [range_mu, 
                     range_mu0,
                     range_alpha,
                     range_delta1, 
                     range_delta2, 
                     range_tau_min, 
                     range_tau_max]

num_genes = len(range_constraints) # 7

save_model = 0 

#------------------------------------------------------------------------------#


################################################################################
# LOAD REAL DATA
################################################################################

### Load the BATSE GRBs
if instrument=='batse': 
    # load all data
    grb_list_real = load_lc_batse(path=batse_path) 
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold, 
                                      sn_threshold=sn_threshold, 
                                      t_f=t_f)
### Load the Swift GRBs
elif instrument=='swift': 
    # load all data
    grb_list_real = load_lc_swift(path=swift_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold, 
                                      sn_threshold=sn_threshold, 
                                      t_f=t_f)
### Load the BeppoSAX GRBs
elif instrument=='sax': 
    # load all (HR) data
    grb_list_real = load_lc_sax_hr(path=sax_path) 
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold, 
                                      sn_threshold=sn_threshold, 
                                      t_f=t_f)
else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax".')


# Set the number of simulated GRBs to produce equal to the number of real GRBs
# that passed the constraint selection
N_grb=1000 #len(grb_list_real)


################################################################################
# COMPUTE AVERAGE QUANTITIES OF REAL DATA
################################################################################

#------------------------------------------------------------------------------#
### TEST 1&2: Average Peak-Aligned Profiles
averaged_fluxes_real, \
averaged_fluxes_cube_real, \
averaged_fluxes_rms_real = compute_average_quantities(grb_list=grb_list_real, 
                                                      t_f=t_f, 
                                                      bin_time=bin_time,
                                                      filter=True)
#------------------------------------------------------------------------------#
### TEST 3: Autocorrelation
N_lim = np.min( [N_grb, len(grb_list_real)] ) 
steps_real, acf_real = compute_autocorrelation(grb_list=grb_list_real,
                                               N_lim=N_lim,
                                               t_max=t_f,
                                               bin_time=bin_time,
                                               mode='scipy')
#------------------------------------------------------------------------------#
### TEST 4: Duration
duration_real = [ evaluateDuration20(times=grb.times, 
                                     counts=grb.counts,
                                     filter=True,
                                     t90=grb.t90,
                                     bin_time=bin_time)[0] for grb in grb_list_real ]
duration_distr_real = compute_kde_duration(duration_list=duration_real)
#------------------------------------------------------------------------------#


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
                                 delta2=10**solution[4],   # sample uniformly in log space
                                 tau_min=10**solution[5],  # sample uniformly in log space
                                 tau_max=solution[6],
                                 # instrument parameters:
                                 instrument=instrument,
                                 bin_time=bin_time,
                                 eff_area=eff_area,
                                 bg_level=bg_level,
                                 # constraint parameters:
                                 t90_threshold=t90_threshold,
                                 sn_threshold=sn_threshold,
                                 t_f=t_f,
                                 # other parameters:
                                 export_files=False,
                                 n_cut=2000,
                                 with_bg=False)
    #--------------------------------------------------------------------------#
    # Compute average quantities of simulated data needed for the loss function
    #--------------------------------------------------------------------------#
    # TEST 1&2: Average Peak-Aligned Profiles
    averaged_fluxes_sim, \
    averaged_fluxes_cube_sim, \
    averaged_fluxes_rms_sim = compute_average_quantities(grb_list=grb_list_sim,
                                                         t_f=t_f,
                                                         bin_time=bin_time,
                                                         filter=True)
    # TEST 3: Autocorrelation
    steps_sim, acf_sim      = compute_autocorrelation(grb_list=grb_list_sim,
                                                      N_lim=N_lim,
                                                      t_max=t_f,
                                                      bin_time=bin_time,
                                                      mode='scipy')
    # TEST 4: Duration
    duration_sim = [ evaluateDuration20(times=grb.times, 
                                         counts=grb.counts,
                                         filter=True,
                                         t90=grb.t90,
                                         bin_time=bin_time)[0] for grb in grb_list_sim ]
    duration_distr_sim = compute_kde_duration(duration_list=duration_sim)
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
                           duration_sim=duration_distr_sim)
    fitness = 1.0 / (l2_loss + 1.e-9)
    return fitness


################################################################################
# DEFINE AUXILIARY FUNCTION
################################################################################

last_fitness, last_loss, current_fitness, current_loss = 0, 0, 0, 0
def on_generation(ga_instance):
    """
    This function is executed after _each_ generation. It prints useful 
    information of the currect epoch.
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
    # Returning the details of the best solution in the current generation.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Parameters of the best solution in the current generation:")
    print("    - mu      = {solution}".format(solution=solution[0]))
    print("    - mu0     = {solution}".format(solution=solution[1]))
    print("    - alpha   = {solution}".format(solution=solution[2]))
    print("    - delta1  = {solution}".format(solution=solution[3]))
    print("    - delta2  = {solution}".format(solution=10**(solution[4])))
    print("    - tau_min = {solution}".format(solution=10**(solution[5])))
    print("    - tau_max = {solution}".format(solution=solution[6]))


################################################################################
# INSTANTIATE THE 'GENETIC ALGORITHM' CLASS
################################################################################

ga_GRB = pygad.GA(num_generations=num_generations,
                  num_parents_mating=num_parents_mating,
                  sol_per_pop=sol_per_pop,
                  num_genes=num_genes,
                  gene_type=float,
                  initial_population=None,      # if 'None', the initial population is randomly chosen using the 'sol_per_pop; and 'num_genes' parameters
                  on_generation=on_generation,
                  ### fitness function:
                  fitness_func=fitness_func,
                  ### parent selection:
                  parent_selection_type='sss',
                  keep_parents=keep_parents,           
                  keep_elitism=keep_elitism,           
                  ### crossover:
                  crossover_probability=0.5,    # 'None' means couples parent k with parent k+1, otherwise it selects from the parents candidate list each one of them with probability 'crossover_probability', and then it takes two of them at random
                  crossover_type="scattered",
                  ### mutation:
                  mutation_type="random",
                  mutation_probability=mutation_probability,     # by default is 'None', otherwise it selects a value randomly from the current gene's space (each gene is changed with probability 'mutation_probability')
                  ### set range of parameters:
                  gene_space=range_constraints,
                  ### other stuff:
                  save_best_solutions=True,
                  save_solutions=True,
                  parallel_processing=["process", 50], # =None,
                  random_seed=seed)

# print summary of the GA
ga_GRB.summary()


################################################################################
# RUN THE GENETIC ALGORITHM
################################################################################

# Run the GA to optimize the parameters of the function.
ga_GRB.run()
#ga_GRB.plot_fitness()


################################################################################
# SAVE THE MODEL
################################################################################

# Save the GA instance.
if save_model:
    filename_model = 'geneticGRB'
    ga_instance.save(filename=filename_model)

# # Load the saved GA instance.
# loaded_ga_instance = pygad.load(filename=filename)
# loaded_ga_instance.plot_fitness()


################################################################################
# PRINT FINAL RESULTS
################################################################################

#------------------------------------------------------------------------------#
# Print on terminal
#------------------------------------------------------------------------------#
# Return the details of the best solution.
solution, solution_fitness, solution_idx = ga_GRB.best_solution(ga_GRB.last_generation_fitness)
print('\n')
print('\n')
print('################################################################################')
print('################################################################################')
print("* Parameters of the BEST solution:")
print("    - mu      = {solution}".format(solution=solution[0]))
print("    - mu0     = {solution}".format(solution=solution[1]))
print("    - alpha   = {solution}".format(solution=solution[2]))
print("    - delta1  = {solution}".format(solution=solution[3]))
print("    - delta2  = {solution}".format(solution=10**(solution[4])))
print("    - tau_min = {solution}".format(solution=10**(solution[5])))
print("    - tau_max = {solution}".format(solution=solution[6]))
print("* Loss value of the best solution    : {solution_loss}".format(solution_loss=solution_fitness**(-1)))
print("* Fitness value of the best solution : {solution_fitness}".format(solution_fitness=solution_fitness))
#print("Index of the best solution          : {solution_idx}".format(solution_idx=solution_idx))
if ga_GRB.best_solution_generation != -1:
    print("* Best fitness value reached after N={best_solution_generation} generations.".format(best_solution_generation=ga_GRB.best_solution_generation))
print('################################################################################')
print('################################################################################')
#------------------------------------------------------------------------------#
# Print on file
#------------------------------------------------------------------------------#
file = open("./results.txt", "w")
file.write('################################################################################')
file.write('\n')
file.write("INPUT")
file.write('\n')
file.write('################################################################################')
file.write('\n')
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
file.write("* Parameters of the best solution:")
file.write('\n')
file.write("    - mu      = {solution}".format(solution=solution[0]))
file.write('\n')
file.write("    - mu0     = {solution}".format(solution=solution[1]))
file.write('\n')
file.write("    - alpha   = {solution}".format(solution=solution[2]))
file.write('\n')
file.write("    - delta1  = {solution}".format(solution=solution[3]))
file.write('\n')
file.write("    - delta2  = {solution}".format(solution=10**(solution[4])))
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
#------------------------------------------------------------------------------#


################################################################################
# EXPORT PLOT
################################################################################

inv_fit = (np.array(ga_GRB.best_solutions_fitness))**(-1)
print('inv_fit[-1] =', inv_fit[-1])
plt.plot(inv_fit, ls='-', lw=2, c='b')
#plt.yscale('log')
plt.xlabel(r'Generation')
plt.ylabel(r'Best Loss')
plt.savefig('fig01.pdf')
plt.clf()


fitness_list = np.array(ga_GRB.solutions_fitness)
loss_list    = 1. / (fitness_list + 1.e-9)
avg_loss     = np.zeros(num_generations+1)
std_loss     = np.zeros(num_generations+1)
for i in range(num_generations+1):
        avg_loss[i] = np.mean( loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
        std_loss[i] = np.std(  loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
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

################################################################################
################################################################################
