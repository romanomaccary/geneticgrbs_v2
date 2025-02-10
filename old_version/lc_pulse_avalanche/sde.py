import inspect
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import exponential, lognormal, normal, uniform
from scipy.stats import poisson
import os, h5py
from functools import partial
from scipy import stats
from statistical_test_test import *

SEED=None


#SEED=42
#np.random.seed(SEED)

#==============================================================================#
#==============================================================================#

import time

def slow_function():
    #print("Starting...")
    time.sleep(2)  # Pause for 2 seconds
    #print("Finished!")

slow_function()


def evaluateDuration20(times, counts, t90=None, t90_frac=15, bin_time=None, filter=True):
    """
    Compute the duration of the GRB event as described in [Stern et al., 1996].
    We define the starting time when the signal reaches the 20% of the value of
    the peak, and analogously for the ending time. The difference of those two
    times is taken as definition of the duration of the GRBs (T20%).
    If filter==True, we smooth the signal using a Savitzky-Golay filter on the
    light curves before computing the T20%.
    Inputs:
      - times: time values of the bins of the light-curve;
      - counts: counts per bin of the GRB;
      - t90: T90 duration of the GRB;
      - bin_time: temporal bin size of the instrument [s];
      - t90_frac: fraction of T90 to be used as window length;
      - filter: boolean variable. If True, it activates the smoothing savgol
                filter before computing the T20% duration;
    Output:
      - duration: T20%, that is, the duration at 20% level;
    """
    if filter:
        t90_frac = t90_frac
        window   = int(t90/t90_frac/bin_time)+2
        window   = window if window%2==1 else window+1

        try:
            counts = savgol_filter(x=counts,
                                   window_length=window,
                                   polyorder=2)
        except:
            #print('window_length =', window)
            print('Error in "evaluateDuration20()" during the "savgol_filter()"...')
            sys.exit()

    threshold_level = 0.20
    c_max           = np.max(counts)
    c_threshold     = c_max * threshold_level
    selected_times  = times[ np.where(counts>=c_threshold)[0] ]
    #selected_times = times[counts >= c_threshold]
    tstart          = selected_times[ 0]
    tstop           = selected_times[-1]
    duration        = tstop - tstart # T20%
    assert duration>0

    return np.array( [duration, tstart, tstop] )

def evaluateGRB_SN(times, counts, errs, t90, t90_frac, bin_time, filter, 
                   return_cnts=False):
    """
    Compute the S/N ratio between the total signal from a GRB and the background
    in a time interval equal to the GRB duration, as defined in Stern+96, i.e.,
    the time interval between the first and the last moments in which the signal
    reaches the 20% of the peak (T20%). The S2N ratio is defined in the 
    following way: we sum of the signal inside the time window defined by the 
    T20%, and we divide it by the square root of the squared sum of the errors
    in the same time interval.
    Input:
     - times: array of times;
     - counts: counts of the event;
     - errs: errors over the counts;
     - t90: T90 of the GRB;
     - bin_time: temporal bin size of the instrument [s];
     - filter: if True, apply savgol filter;
     - return_cnts: if True, return also the total counts inside the T20% interval;
    Output:
     - s2n: signal to noise ratio;
     - T20: duration of the GRB at 20% level;
     - T20_start: start time of the T20% interval;
     - T20_stop: stop time of the T20% interval; 
     - sum_grb_counts: total counts inside the T20% interval;
    """
    T20, tstart, tstop = evaluateDuration20(times=times, 
                                            counts=counts,
                                            t90=t90, 
                                            t90_frac=t90_frac, 
                                            bin_time=bin_time,
                                            filter=filter)
    
    event_times_mask = np.logical_and(times>=tstart, times<=tstop)
    sum_grb_counts   = np.sum( counts[event_times_mask] )
    sum_errs         = np.sqrt( np.sum(errs[event_times_mask]**2) )
    s2n              = np.abs( sum_grb_counts/sum_errs )
    if not(return_cnts):
        return s2n, T20, tstart, tstop
    else:
        return s2n, T20, tstart, tstop, sum_grb_counts

def generate_fluence(p, alpha, beta, F_break, F_min):
    """This function returns a fluence value F from the broken-power-law (BPL)
    distribution: 
    
    p(F) = (F_break/F_min)**(-alpha) * (F/F_break)**(-alpha) if F <  F_break;
           (F_break/F_min)**(-alpha) * (F/F_break)**(-beta)  if F >= F_break;
    
    where alpha and beta are BPL indices, F_break is the break fluence, and 
    F_min is minimum fluence. p(F) is the so-called "survival function" (SF), 
    which is the analogous of the log(N)-log(S) of the GRBs but for individual
    pulses. 
    The idea is sampling the SF, whose values are found between 0 and 1 by 
    definition, and turn them into the corresponding fluence values.

    Args:
        p       (float): sampled value from the SF;
        alpha   (float): first BPL index;
        beta    (float): second BPL index;
        F_break (float): break fluence;
        F_min   (float): minimum fluence.

    Returns:
        float: the corresponding fluence value F.
    """
    R_min = F_min/F_break
    f_0   = (F_break*((1-R_min**(1-alpha))/(1-alpha)-1/(1-beta)))**(-1)
    p_0   = f_0*F_break*(1-R_min**(1-alpha))/(1-alpha)
    return np.piecewise(p, [p < p_0, p >= p_0], [lambda p: F_break*(p*(1-alpha)/(f_0*F_break)+R_min**(1-alpha))**(1/(1-alpha)), 
                                                 lambda p: F_break*((p-1)*(1-beta)/(f_0*F_break))**(1/(1-beta))])

def generate_peak_counts(generated_fluence, k_values):
    """This function turn fluence into peak counts through a conversion factor
    that depends on the selected instrument. Each time, the function randomly 
    pick a value from the list of conversion factors in order to take into
    account the spectral diversity of the GRBs.

    Args:
        generated_fluence (float): fluence value generated through the function
                                   generate_fluence;
        k_values          (float): list of conversion factors.

    Returns:
        float: the corresponding peak counts.
    """
    fluence   = generated_fluence(np.random.rand())
    k_sampled = np.random.choice(k_values)
    counts    = (10.**(-k_sampled))*fluence
    return counts 

path_k_values_file_batse = "../lc_pulse_avalanche/log10_fluence_over_counts_CGRO_BATSE.txt"
k_values_batse = np.loadtxt(path_k_values_file_batse, unpack = True)

path_k_values_file_swift = "../lc_pulse_avalanche/log10_fluence_over_counts_Swift_BAT.txt"
k_values_swift = np.loadtxt(path_k_values_file_swift, unpack = True)

path_k_values_file_fermi = "../lc_pulse_avalanche/log10_fluence_over_counts_Fermi_GBM.txt"
k_values_fermi = np.loadtxt(path_k_values_file_fermi, unpack = True)

# path_k_values_file_batse = "../lc_pulse_avalanche/log10_fluence_over_counts_CGRO_BATSE.txt"
# k_values_batse = np.loadtxt(path_k_values_file_batse, unpack = True)

# path_k_values_file_swift = "../lc_pulse_avalanche/log10_fluence_over_counts_Swift_BAT.txt"
# k_values_swift = np.loadtxt(path_k_values_file_swift, unpack = True)

# path_k_values_file_fermi = "../lc_pulse_avalanche/log10_fluence_over_counts_Fermi_GBM.txt"
# k_values_fermi = np.loadtxt(path_k_values_file_fermi, unpack = True)

#==============================================================================#
#==============================================================================#


### Load the (gaussian) errors of the Swift GRBs
#path_swift_errs = '../lc_pulse_avalanche/'                                                # LB
#path_swift_errs = '/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche/' # bach
path_swift_errs = '/home/bazzanini/PYTHON/genetic3/lc_pulse_avalanche/'                    # gravity
path_swift_errs = '/astrodata/romain/sde_ga/swift_err/'    
#
# bins_swift_errs = np.array([  0.1, 0.21544347, 0.46415888, 1., 2.15443469, 4.64158883, 10. , 21.5443469 , 46.41588834, 100. ])
bins_swift_errs = np.array([  0.1, 0.21544347, 0.46415888, 1., 2.15443469, 4.64158883, 10.])
dict_errs_swift = {}
for i in range(1, len(bins_swift_errs)+1):
    with open(path_swift_errs+'swift_errs_'+str(i)+'.txt', 'r') as f:
        dict_errs_swift[str(i)] = f.readlines()
for key in dict_errs_swift.keys():
    for i, line in enumerate(dict_errs_swift[key]):
        line       = line.rstrip(' \n')
        errs_split = list(map(float, line.split(' ')))
        dict_errs_swift[key][i] = errs_split

#==============================================================================#
# Define the class LC describing the light curve.                              #
#==============================================================================#

class LC(object):
    """
    A class to generate gamma-ray burst light curves (GRB LCs) using a pulse
    avalanche model ('chain reaction') proposed by Stern & Svensson, ApJ, 
    469: L109 (1996).
    
    :mu: average value of the Poisson distribution that samples the 
         number of child pulses, which is mu_b; average number of baby pulses
    :mu0: average value of the Poisson distribution that samples the 
          number of primary pulses, which is mu_s; average number of spontaneous
          (initial) pulses per GRB
    :alpha: delay parameter
    :delta1: lower boundary of log-normal probability distribution of tau
             (time constant of baby pulse)
    :delta2: upper boundary of log-normal probability distribution of tau
    :tau_min: lower boundary of log-normal probability distribution of tau_0 
              (time constant of spontaneous pulse); should be smaller than res
    :tau_max: upper boundary of log-normal probability distribution of tau_0
    :t_min: GRB LC start time
    :t_max: GRB LC stop time
    :res: GRB LC time resolution (s) (i.e., bin time)
    :eff_area: effective area of instrument (cm2)
    :bg_level: background level rate per unit area of detector (cnt/cm2/s)
    :min_photon_rate: left  boundary of -3/2 log N - log S distribution (ph/cm2/s)
    :max_photon_rate: right boundary of -3/2 log N - log S distribution (ph/cm2/s)
    :sigma: signal above background level
    :n_cut: maximum number of pulses in avalanche (useful to speed up the 
            simulations but in odds with the "classic" approach)
    :with_bg: boolean flag for keeping or removing the background level at the 
              end of the generation
    :use_poisson: boolean flag for using the Poisson or the (rounded) 
                  exponential for sampling the number of initial pulses and child
    """
    
    def __init__(self, q,a,alpha, k, t_0,norm_A,t_min=+0.1, t_max=1000, res=0.256, 
                 eff_area=3600, bg_level=10.67, with_bg=True, use_poisson=True,
                 min_photon_rate=1.3, max_photon_rate=1300, sigma=5, 
                 n_cut=None, instrument='batse', verbose=False): #New parameters of the BPL count distrib
        
        #self._mu      = mu # mu~1 --> critical runaway regime
        self._q =q
        self._a = a
        self._alpha = alpha
        self._k = k
        self._t_0 = t_0
        self._norm_A = norm_A
        self._eff_area = eff_area 
        self._bg = bg_level * self._eff_area # cnt/s
        self._min_photon_rate = min_photon_rate  
        self._max_photon_rate = max_photon_rate 
        self._verbose = verbose
        self._res = res # s
        self._n = int(np.ceil((t_max - t_min)/self._res)) + 1 # time steps
        self._t_min = t_min # ms
        self._t_max = (self._n - 1) * self._res + self._t_min # ms
        self._times, self._step = np.linspace(self._t_min, self._t_max, self._n, retstep=True)
        # Arrays of COUNT RATES
        self._rates       = np.zeros(len(self._times))
        self._sp_pulse    = np.zeros(len(self._times))
        self._total_rates = np.zeros(len(self._times))
        
        # Arrays of COUNTS
        #self._child_counts  = np.zeros(len(self._times))
        #self._parent_counts = np.zeros(len(self._times))
        
        # Other parameters
        self._lc_params   = list()
        
        self._sigma       = sigma
        self._n_cut       = n_cut
        self._n_pulses    = 0
        self._with_bg     = with_bg
        self._use_poisson = use_poisson
        self._instrument  = instrument

        if self._instrument == 'batse':
            #self._peak_count_rate_sample = peak_count_rate_batse_sample
            self.k_values_path           = path_k_values_file_batse
            self.k_values                = k_values_batse
        elif self._instrument == 'swift':
            #self._peak_count_rate_sample = peak_count_rate_swift_sample
            self.k_values_path           = path_k_values_file_swift
            self.k_values                = k_values_swift
        elif self._instrument == 'sax_lr':
            pass
            #self._peak_count_rate_sample = peak_count_rate_sax_lr_sample
        elif self._instrument == 'sax':
            pass
            #self._peak_count_rate_sample = peak_count_rate_sax_sample
        elif self._instrument == 'fermi':
            self.k_values_path           = path_k_values_file_fermi
            self.k_values                = k_values_fermi
        else:
            raise ValueError("Instrument not recognized...")
        
        # self.alpha_bpl = alpha_bpl
        # self.beta_bpl  = beta_bpl
        # self.F_break   = F_break
        # self.F_min     = F_min
        # self.generated_fluence = partial(generate_fluence, alpha = alpha_bpl, 
        #                                 beta = beta_bpl, F_break = F_break, 
        #                                 F_min = F_min)
        
        if self._verbose:
             print("Time resolution: ", self._step)
    #--------------------------------------------------------------------------#

    def generate_LC_from_sde(self,q,a,alpha,k,t_0,norm_A):

        # def sde_euler_maruyama(times, mu, sigma, n_paths):
        #     N = len(times)
        #     dt = times[1]-times[0]
        #     X = np.zeros((N, n_paths))
        #     x0 = np.random.random() # first point of the sde is set randomly
        #     X[0, :] = x0

        #     dW = np.random.normal(0, np.sqrt(dt), (N-1, n_paths))

        #     for i in range(1, N):
        #         X[i, :] = X[i-1, :] + mu(X[i-1, :], times[i-1]) * dt + sigma(X[i-1, :], times[i-1]) * dW[i-1, :]

        #     return X

        def brownian(n, dt, q):
            # For each element of x0, generate a sample of n numbers from a
            # normal distribution.
            r = stats.norm.rvs(size=n, scale=np.sqrt(q*dt))
            # This computes the Brownian motion by forming the cumulative sum of
            # the random samples. 
            beta= np.cumsum(r, axis=0)
            return beta

        # def generale_lc_from_solution_SDE(q,a,alpha,k,t_0,times):
        #     x00 = 1000
        #     print(x00)
        #     n=len(times)
        #     dt=times[1]-times[0]
        #     beta = brownian(n,dt,q)
        #     return x00*times**(alpha)*np.exp(-a*times)*np.exp(-k*(times/t_0)**(1./3.))*np.exp(beta)
        
        def generale_lc_from_solution_SDE(q,a,alpha,k,t_0,norm_A,times):
            #norm_A = generate_peak_counts(self.generated_fluence, self.k_values)
            #norm_A = np.float64(norm_A)
            #print("norm=",norm_B)
            #print("q=",q,"a=",a,"alpha=",alpha,"k=",k)
            n=len(times)
            dt=times[1]-times[0]
            beta = brownian(n,dt,q)
            return norm_A*(times/t_0)**(alpha)*np.exp(-a*times-k*(times/t_0)**(1./3.)+beta)  
        
        self._rates = generale_lc_from_solution_SDE(q,a,alpha,k,t_0,norm_A,self._times)
        self._max_raw_pc = self._rates.max()
        self._peak_value = self._max_raw_pc

        #print(self._max_raw_pc)
        #print('max_lc',self._peak_value)
        if (self._max_raw_pc<1.e-12):
            print('LC with 0 value')
            # check that we have generated a lc with non-zero values; otherwise,
            # exit and set the flag 'self.check=0', which indicates that this
            # lc has to be skipped
            self.check=0
            return 0
        else:
            self.check=1
        
        if (self._max_raw_pc>1.e17):
            print('PEAK>1e16')
            #return 0

        #if (self._max_raw_pc<1.e16) and (self._max_raw_pc>1e-12):
        #    print('PEAK=',self._max_raw_pc)
        
        #print("INSTRUMENT=",self._instrument)
         # lc from avalanche scaled + Poissonian bg added (for BATSE and Fermi)
         # For BATSE, the variable `_plot_lc` contains the COUNTS (and not the count RATES!)
        if self._instrument == 'batse' or self._instrument == 'fermi':
            self._model           = self._rates                                 # model COUNTS 
            self._modelbkg        = self._model + (self._bg * self._res)                # model COUNTS + constant bgk counts
            #print('condition for poisson',np.any(self._modelbkg > 1e17))
            if np.any(self._modelbkg > 1e17):
                #print('HIGH LAMBDA')
                self._plot_lc=np.random.normal(loc=self._modelbkg, scale=np.sqrt(self._modelbkg))
                #print('p=',np.random.normal(loc=self._modelbkg, scale=np.sqrt(self._modelbkg)))
            else:
                self._plot_lc = np.random.poisson(self._modelbkg).astype('float')   # total COUNTS (signal+noise) with Poisson
            
            self._plot_lc_with_bg = self._plot_lc  
            self._err_lc          = np.sqrt(self._plot_lc)
            if self._with_bg: # lc with background
                pass
            else: # background-subtracted lc
                self._plot_lc = self._plot_lc - (self._bg * self._res)   # total COUNTS (removed the constant bkg level)
        
        # For Swift, the variable `_plot_lc` contains the COUNTS RATES (and not the counts!)
        elif self._instrument == 'swift':
            self._model           = self._rates      # model COUNTS 
            self._model_rate      = self._model / self._res  # model COUNT RATES
            self._modelbkg        = self._model              # bkg 0 in Swift
            self._modelbkg_rate   = self._model_rate         # bkg 0 in Swift
            #
            if np.max(self._model_rate)<bins_swift_errs[1]:
                errs_swift_list = dict_errs_swift['1']
            elif np.max(self._model_rate)<bins_swift_errs[2]: 
                errs_swift_list = dict_errs_swift['2']
            elif np.max(self._model_rate)<bins_swift_errs[3]: 
                errs_swift_list = dict_errs_swift['3']
            elif np.max(self._model_rate)<bins_swift_errs[4]: 
                errs_swift_list = dict_errs_swift['4']
            elif np.max(self._model_rate)<bins_swift_errs[5]: 
                errs_swift_list = dict_errs_swift['5']
            elif np.max(self._model_rate)<bins_swift_errs[6]: 
                errs_swift_list = dict_errs_swift['6']
            else: 
                errs_swift_list = dict_errs_swift['7']
            #
            grb_index             = np.random.randint(len(errs_swift_list))
            errors_to_apply       = np.array(errs_swift_list[grb_index])
            max_err_index         = len(errors_to_apply)
            std_bkg               = np.array([errors_to_apply[np.random.randint(0,max_err_index)] for val in self._model_rate])
            self._plot_lc         = np.random.normal(loc=self._model_rate, scale=std_bkg) # total COUNTS RATE (signal+noise) with Gauss
            self._plot_lc_with_bg = self._plot_lc  
            self._err_lc          = std_bkg 
            # Since we set the noise as gaussian using std from real data, we
            # don't need to add the background level to the model
            # if self._with_bg: # lc with background
            #     pass
            # else: # background-subtracted lc
            #     self._plot_lc = self._plot_lc - (self._bg * self._res)   # total COUNTS (removed the constant bkg level)

        self._get_lc_properties()
        #print('S/N=',evaluateGRB_SN(self._times,self._plot_lc,self._err_lc,self._t90,15,0.064,True,return_cnts=False)[0],'k=',"%1.2f"%self._k,'t0=',"%1.2f"%self._t_0)
        #SNoi=evaluateGRB_SN(self._times,self._plot_lc,self._err_lc,self._t90,15,0.064,True,return_cnts=False)[0]
        #if SNoi < 5:
            #print('S/N=',"%1.3f"%SNoi,'k=',"%1.3f"%self._k,'t0=',"%1.3f"%self._t_0,"q=","%1.3f"%self._q,"alpha=","%1.3f"%self._alpha,"norm","%1.3e"%self._norm_A)
            #slow_function()
   
        #else:
        #    return self._lc_params
        return self._lc_params
    #--------------------------------------------------------------------------#

    def plot_lc(self, rescale=True, save=False, name="./plot_lc.pdf", show_duration=False):
        """
        Plots GRB light curve (COUNTS vs time)
        
        :rescale: to rescale the x-axis plotting only lc around T100
        :save: to save the plot to file
        :name: filename (including path) to save the plot
        """

        plt.figure(figsize=(9,6))
        plt.xlabel(r'$T-T_0$ [s]', size=16)
        if self._instrument == 'batse':
            plt.ylabel('Counts', size=16)
        elif self._instrument == 'swift':
            plt.ylabel('Count Rates', size=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        #self._restore_lc()
        
        plt.step(self._times, self._plot_lc, where='post', color='k')

        if self._with_bg:
            plt.plot(np.linspace(self._t_min, self._t_max, num=2, endpoint=True), [self._bg, self._bg], 'r--')
        else:
            pass
            #plt.plot(np.linspace(self._t_min, self._t_max, num=2, endpoint=True), [       0,        0], 'r--')
        
        if rescale:
            t_i = max(self._t_start - 0.5*self._t100, self._t_min)
            t_f = self._t_stop + 0.5*self._t100
            plt.xlim([t_i, t_f])
            
        if show_duration:
                plt.axvline(x=self._t_start, alpha=0.7, color='blue', label=r'$T_{100}$')
                plt.axvline(x=self._t_stop,  alpha=0.7, color='blue')
                plt.axvline(x=self._t90_i,   alpha=0.7, color='red',  label=r'$T_{90}$')
                plt.axvline(x=self._t90_f,   alpha=0.7, color='red')
                plt.legend(prop={'size': 16})
        
        if save:
            plt.savefig(name)
        
        plt.show()

    #--------------------------------------------------------------------------#
    
    def _get_lc_properties(self):
        """
        Calculates T90 and T100 durations along with their start and stop times, 
        total number of counts per T100, mean, max, and background count rates.
        """
#   #--------------------------------------------------------------------------#
#   # V1
#   #--------------------------------------------------------------------------#
#        self._aux_index = np.where(self._raw_lc>self._raw_lc.max()*1e-4)
#        #self._aux_index = np.where((self._plot_lc - self._bg) * self._res / (self._bg * self._res)**0.5 >= self._sigma)
#        self._max_snr   = ((self._plot_lc/self._res - self._bg) * self._res / (self._bg * self._res)**0.5).max()
#        self._aux_times = self._times[self._aux_index[0][0]:self._aux_index[0][-1]] # +1 in the index
#        self._aux_lc    = self._plot_lc[self._aux_index[0][0]:self._aux_index[0][-1]] / self._res
#
#        self._t_start = self._times[self._aux_index[0][0]]
#        #self._t_stop = self._times[self._aux_index[0][-1]+1]
#        self._t_stop  = self._times[self._aux_index[0][-1]]
#            
#        self._t100 = self._t_stop - self._t_start
#        
#        self._total_cnts = np.sum(self._aux_lc - self._bg*np.ones(len(self._aux_lc))) * self._res
#                
#        try:
#            # compute T90_i
#            sum_cnt = 0
#            i = 0
#            while sum_cnt < 0.05 * self._total_cnts:
#                sum_cnt += (self._aux_lc[i] - self._bg) * self._res
#                i += 1
#            self._t90_i = self._aux_times[i]
#                                     
#            # compute T90_f
#            sum_cnt = 0
#            j = -1
#            while sum_cnt < 0.05 * self._total_cnts:
#                sum_cnt += (self._aux_lc[j] - self._bg) * self._res
#                j += -1
#            self._t90_f = self._aux_times[j]      
#
#            # Define T90 as the difference between T90_f and T90_i
#            self._t90 = self._t90_f - self._t90_i            
#            self._t90_cnts = np.sum(self._aux_lc[i:j+1] - self._bg) * self._res
#            assert self._t90_i < self._t90_f
#            
#        except:
#            self._t90      = self._t100
#            self._t90_i    = self._t_start
#            self._t90_f    = self._t_stop
#            self._t90_cnts = self._total_cnts
           
#    #--------------------------------------------------------------------------#
#    # V2
#    #--------------------------------------------------------------------------#
#        # LB: I think that `self._plot_lc` should be the background-subtracted
#        self._aux_index = np.where(self._plot_lc>self._plot_lc.max()*1e-4)
#        #self._aux_index = np.where((self._plot_lc - self._bg) * self._res / (self._bg * self._res)**0.5 >= self._sigma)
#        self._max_snr   = ((self._plot_lc_with_bg/self._res - self._bg) * self._res / (self._bg * self._res)**0.5).max()
#        self._aux_times = self._times[self._aux_index[0][0]:self._aux_index[0][-1]] # +1 in the index
#        self._aux_lc    = self._plot_lc_with_bg[self._aux_index[0][0]:self._aux_index[0][-1]] / self._res
#
#        self._t_start = self._times[self._aux_index[0][0]]
#        #self._t_stop = self._times[self._aux_index[0][-1]+1]
#        self._t_stop  = self._times[self._aux_index[0][-1]]
#        self._t100    = self._t_stop - self._t_start
#        
#        self._total_cnts = np.sum(self._aux_lc - self._bg*np.ones(len(self._aux_lc))) * self._res
#        
#        try:
#            # compute T90_i
#            sum_cnt = 0
#            i = 0
#            while sum_cnt < 0.05 * self._total_cnts:
#                sum_cnt += (self._aux_lc[i] - self._bg) * self._res
#                i += 1
#            self._t90_i = self._aux_times[i]
#                                     
#            # compute T90_f
#            sum_cnt = 0
#            j = -1
#            while sum_cnt < 0.05 * self._total_cnts:
#                sum_cnt += (self._aux_lc[j] - self._bg) * self._res
#                j += -1
#            self._t90_f = self._aux_times[j]      
#
#            # Define T90 as the difference between T90_f and T90_i
#            self._t90      = self._t90_f - self._t90_i
#            self._t90_cnts = np.sum(self._aux_lc[i:j+1] - self._bg) * self._res
#            assert self._t90 > 0
#            
#        except:
#            print('Weird stuff happened...')
#            self._t90      = self._t100
#            self._t90_i    = self._t_start
#            self._t90_f    = self._t_stop
#            self._t90_cnts = self._total_cnts
           

    #--------------------------------------------------------------------------#
    # V3
    #--------------------------------------------------------------------------#
       
        self._aux_index = np.where(self._model>self._model.max()*1e-4)
        #self._aux_index = np.where((self._plot_lc - self._bg) * self._res / (self._bg * self._res)**0.5 >= self._sigma)
        self._max_snr   = ((self._plot_lc_with_bg/self._res - self._bg) * self._res / (self._bg * self._res)**0.5).max()
        self._aux_times = self._times[self._aux_index[0][0]:self._aux_index[0][-1]] # +1 in the index
        # as `self._aux_lc` lc, we use the 'model' one, which is just the sum of the single pulses
        self._aux_lc    = self._model[self._aux_index[0][0]:self._aux_index[0][-1]] / self._res  # count RATES

        self._t_start = self._times[self._aux_index[0][0]]
        #self._t_stop = self._times[self._aux_index[0][-1]+1]
        self._t_stop  = self._times[self._aux_index[0][-1]]
        self._t100    = self._t_stop - self._t_start
        
        self._total_cnts = np.sum(self._aux_lc) * self._res
        #self._total_cnts = np.sum(self._aux_lc - self._bg*np.ones(len(self._aux_lc))) * self._res
                
        try:
            # compute T90_i
            sum_cnt = 0
            i = 0
            while sum_cnt < 0.05 * self._total_cnts:
                sum_cnt += (self._aux_lc[i]) * self._res
                #sum_cnt += (self._aux_lc[i] - self._bg) * self._res
                i += 1
            if i!=0:
                i-=1
            self._t90_i = self._aux_times[i]
                                     
            # compute T90_f
            sum_cnt = 0
            j = -1
            while sum_cnt < 0.05 * self._total_cnts:
                sum_cnt += (self._aux_lc[j]) * self._res
                #sum_cnt += (self._aux_lc[j] - self._bg) * self._res
                j += -1
            if j!=-1:
                j+=1
            self._t90_f = self._aux_times[j]      

            # Define T90 as the difference between T90_f and T90_i
            self._t90      = self._t90_f - self._t90_i
            self._t90_cnts = np.sum(self._aux_lc[i:j+1]) * self._res
            #self._t90_cnts = np.sum(self._aux_lc[i:j+1] - self._bg) * self._res
            #if(self._t90<=0):
            #    print(self._t90)
            assert self._t90 > 0
 
        except:
            #print('Weird stuff happened...')
            self._t90      = self._t100
            self._t90_i    = self._t_start
            self._t90_f    = self._t_stop
            self._t90_cnts = self._total_cnts
    #--------------------------------------------------------------------------#

    @property
    def T90(self):
        return "{:0.3f}".format(self._t90), "{:0.3f}".format(self._t90_i), "{:0.3f}".format(self._t90_f)
    
    @property
    def T100(self):
        return "{:0.3f}".format(self._t100), "{:0.3f}".format(self._t_start), "{:0.3f}".format(self._t_stop)
    
    @property
    def total_counts(self):
        return "{:0.2f}".format(self._total_cnts)
    
    @property
    def max_rate(self):
        return "{:0.2f}".format(self._aux_lc.max())
    
    @property
    def mean_rate(self):
        return "{:0.2f}".format(np.mean(self._aux_lc))
    
    @property
    def bg_rate(self):
        return "{:0.2f}".format(self._bg)
    
    @property
    def max_snr(self):
        return "{:0.2f}".format(self._max_snr)
    
    #--------------------------------------------------------------------------#
#    #TODO LB: THIS METHOD HAS TO BE UPDATED TO THE NEW VERSION OF THE CODE!
#    def _restore_lc(self):
#        """Restores GRB LC from avalanche parameters.
#        Here we are plotting the count RATES, not the counts!"""
#        
#        self._raw_lc = np.zeros(len(self._times))
#        
#        for par in self._lc_params:
#            norm          = par['norm']
#            t_delay       = par['t_delay']
#            tau           = par['tau']
#            tau_r         = par['tau_r']
#            self._raw_lc += self.norris_pulse(norm, t_delay, tau, tau_r)
#
#        if self._with_bg:
#            self._plot_lc = (self._raw_lc * self._ampl * self._eff_area) + self._bg # total count rates (signal+bkg)
#            self._plot_lc = np.random.poisson( self._res * self._plot_lc )          # total count (signal+bkg) with Poisson
#            self._plot_lc = self._plot_lc / self._res                               # total count rates (signal+bkg) with Poisson
#        else:
#            self._plot_lc = (self._raw_lc * self._ampl * self._eff_area) + self._bg # total count rates (signal+bkg)
#            self._plot_lc = np.random.poisson( self._res * self._plot_lc )          # total count (signal+bkg) with Poisson
#            self._plot_lc = self._plot_lc / self._res                               # total count rates (signal+bkg) with Poisson
#            self._plot_lc = self._plot_lc - self._bg                                # total count rates (signal) with Poisson
#
#        self._get_lc_properties()
        
    #--------------------------------------------------------------------------#

    def hdf5_lc_generation(self, outfile, overwrite=False, seed=SEED):
        """
        Generates a new avalanche and writes it to an hdf5 file
        
        :n_lcs: number of light curves we want to simulate
        :outfile: file name
        :overwrite: overwrite existing file
        :seed: random seed for the avalanche generation, int or list
        """
        
        if overwrite == False:
            assert os.path.isfile(outfile), 'ERROR: file already exists!'

        self._f = h5py.File(outfile, 'w')

        
        self._f.create_group('GRB_PARAMETERS')
        self._f['GRB_PARAMETERS'].attrs['PARAMETER_ORDER'] = '[K, t_start, t_rise, t_decay]'

        self._grb_counter = 1
            
        if isinstance(seed, list):
            for sd in seed:
                self.aux_hdf5(seed=sd)
                
        else:
            self.aux_hdf5(seed=seed)

        self._f.close()
        
    #--------------------------------------------------------------------------#
        
    def aux_hdf5(self, seed):
        norms, t_delays, taus, tau_rs, peak_value = self.generate_avalanche(seed=seed, return_array=True)
        n_pulses = norms.size

        grb_array = np.concatenate((
                    norms.reshape(n_pulses,1),
                    t_delays.reshape(n_pulses,1),
                    tau_rs.reshape(n_pulses,1),
                    taus.reshape(n_pulses,1)),
                    axis=1
                    )

        self._f.create_dataset(f'GRB_PARAMETERS/GRB_{self._grb_counter}', data=grb_array)
        self._f[f'GRB_PARAMETERS/GRB_{self._grb_counter}'].attrs['PEAK_VALUE'] = peak_value
        self._f[f'GRB_PARAMETERS/GRB_{self._grb_counter}'].attrs['N_PULSES']   = n_pulses
        self._grb_counter += 1


#==============================================================================#
# Define the class Restore_LC
#==============================================================================#
class Restored_LC(LC):
    """
    Class to restore an avalanche from yaml file
    
    :res: GRB LC time resolution
    """
    
    def __init__(self, par_list, res=0.256, t_min=-10, t_max=1000, sigma=5):
        
        super(Restored_LC, self).__init__(res=res, t_min=t_min, t_max=t_max, sigma=sigma)

        if not par_list:
            raise TypeError("Avalanche parameters should be given")
        elif not isinstance(par_list, list):
            raise TypeError("The avalanche parameters should be a list of dictionaries")
        else:
            self._lc_params = par_list
 
        self._restore_lc()
#==============================================================================#
#==============================================================================#