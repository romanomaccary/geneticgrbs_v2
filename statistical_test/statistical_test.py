################################################################################
# IMPORT LIBRARIES
################################################################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from scipy.signal import savgol_filter
from scipy import signal
from scipy import stats
from sklearn.utils import resample
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

import seaborn as sns
sns.set_style('darkgrid')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

#SEED=42
#np.random.seed(SEED)


user='LB'
#user='AF'
#user='bach
if user=='bach':
    sys.path.append('/home/')
elif user=='LB':
    sys.path.append('/Users/lorenzo/Documents/UNIVERSITA/Astrophysics/PYTHON/GRBs/lc_pulse_avalanche/lc_pulse_avalanche')
elif user=='AF':
    sys.path.append('C:/Users/Lisa/Documents/GitHub/lc_pulse_avalanche/lc_pulse_avalanche')
else:
    raise ValueError('Assign to the variable "user" a correct username!')
from avalanche import LC

################################################################################

class GRB:
    """
    Class for GRBs to save their properties.
    """
    def __init__(self, 
                 grb_name, 
                 times, 
                 counts, 
                 errs, 
                 t90, 
                 grb_data_file_path, 
                 num_of_sig_pulses=0):
                 
        self.name   = grb_name
        self.times  = times
        self.counts = counts
        self.errs   = errs
        self.t90    = t90
        self.data_file_path    = grb_data_file_path
        self.num_of_sig_pulses = num_of_sig_pulses

    def copy(self):
        copy_grb = GRB(self.name, self.times, self.counts, self.errs, self.t90, self.data_file_path)
        return copy_grb

################################################################################

# Dictionaries where the properties of instruments are stored:
# - name         : name of the instrument
# - res          : time resolution of the instrument [s]
# - eff_area     : effective area of instrument [cm2]
# - bg_level     : background level [cnt/cm2/s]
# - t90_threshold: used to select only _long_ GRBs [s]
# - sn_threshold : used to select only lc with high S2N    

name_batse          = 'batse'
res_batse           = 0.064
eff_area_batse      = 3600
bg_level_batse      = 10.67
t90_threshold_batse = 2
sn_threshold_batse  = 70
instr_batse         = {
    "name"         : name_batse,
    "res"          : res_batse,
    "eff_area"     : eff_area_batse,
    "bg_level"     : bg_level_batse,
    "t90_threshold": t90_threshold_batse,
    "sn_threshold" : sn_threshold_batse
}

name_swift          = 'swift'
res_swift           = 0.064
eff_area_swift      = 1400
bg_level_swift      = (10000/eff_area_swift)
t90_threshold_swift = 2
sn_threshold_swift  = 15
instr_swift         = {
    "name"         : name_swift,
    "res"          : res_swift,
    "eff_area"     : eff_area_swift,
    "bg_level"     : bg_level_swift,
    "t90_threshold": t90_threshold_swift,
    "sn_threshold" : sn_threshold_swift
}

name_sax          = 'sax'
res_sax           = 0.0078125
eff_area_sax      = 420
bg_level_sax      = (1000/eff_area_sax)
t90_threshold_sax = 2
sn_threshold_sax  = 10
instr_sax         = {
    "name"         : name_sax,
    "res"          : res_sax,
    "eff_area"     : eff_area_sax,
    "bg_level"     : bg_level_sax,
    "t90_threshold": t90_threshold_sax,
    "sn_threshold" : sn_threshold_sax
}

name_sax_lr          = 'sax_lr'
res_sax_lr           = 1
eff_area_sax_lr      = eff_area_sax
bg_level_sax_lr      = bg_level_sax
t90_threshold_sax_lr = t90_threshold_sax
sn_threshold_sax_lr  = sn_threshold_sax
instr_sax_lr         = {
    "name"         : name_sax_lr,
    "res"          : res_sax_lr,
    "eff_area"     : eff_area_sax_lr,
    "bg_level"     : bg_level_sax_lr,
    "t90_threshold": t90_threshold_sax_lr,
    "sn_threshold" : sn_threshold_sax_lr
}


################################################################################

# def evaluateT90(times, counts):
#     """
#     Compute the T90 of a GRB, i.e., the 90% duration of the burst in seconds.
#     T90 measures the duration of the time interval during which 90% of the 
#     total observed counts have been detected. The start of the T90 interval
#     is defined by the time at which 5% of the total counts have been detected,
#     and the end of the T90 interval is defined by the time at which 95% of the
#     total counts have been detected (definition from: 
#     https://heasarc.gsfc.nasa.gov/grbcat/notes.html).
#     - for BATSE, the T90 of the GRBs have been already evaluated, see here:
#         /astrodata/guidorzi/CGRO_BATSE/T90_full.dat
#     - for Swift, the T90 of the GRBs have been already evaluated, see here:
#         /astrodata/guidorzi/Swift_BAT/merged_lien16-GCN_long_noshortEE_t90.dat
#     - for BeppoSAX, the T90 of the GRBs have been already evaluated, see here:
#         /astrodata/guidorzi/BeppoSAX_GRBM/saxgrbm_t90.dat
#     Inputs:
#       - times: time values of the bins of the light-curve;
#       - counts: counts per bin of the GRB;
#     Output:
#       - t90: T90 of the GRB
#     """
#     cumulative_counts = np.cumsum(counts)
#     total_counts = cumulative_counts[-1]
#     t_5_counts   = 0.05 * total_counts
#     t_95_counts  = 0.95 * total_counts
#     t_5_index    = np.where(cumulative_counts <=  t_5_counts )[0][-1]
#     t_95_index   = np.where(cumulative_counts >=  t_95_counts)[0][ 0]
#     t_5  = times[t_5_index]
#     t_95 = times[t_95_index]
#     t90  = t_95-t_5
#     assert t90>0
#     return t90

################################################################################

def evaluateDuration20(times, counts, t90=None, bin_time=None, filter=True):
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
      - filter: boolean variable. If True, it activates the smoothing savgol
                filter before computing the T20% duration;
    Output:
      - duration: T20%, that is, the duration at 20% level;
    """
    if filter:
        t90_frac = 5.
        window   = int(t90/t90_frac/bin_time)
        window   = window if window%2==1 else window+1

        try:
            counts = savgol_filter(x=counts,
                                   window_length=window,
                                   polyorder=2)
        except:
            #print('window_length =', window)
            print('Error in "evaluateDuration20()" during the "savgol_filter()"...')
            exit()


    threshold_level = 0.2
    c_max           = np.max(counts)
    c_threshold     = c_max * threshold_level
    selected_times  = times[counts >= c_threshold]
    tstart          = selected_times[ 0]
    tstop           = selected_times[-1]
    duration        = tstop - tstart
    assert duration>0

    return np.array( [duration, tstart, tstop] )

################################################################################

def evaluateGRB_SN(times, counts, errs, t90, bin_time, filter):
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
    Output:
     - s2n: signal to noise ratio;
    """
    _, tstart, tstop = evaluateDuration20(times=times, 
                                          counts=counts,
                                          t90=t90, 
                                          bin_time=bin_time,
                                          filter=filter)
    
    event_times_mask = np.logical_and(times>=tstart, times<=tstop)
    sum_grb_counts   = np.sum( counts[event_times_mask] )
    sum_errs         = np.sqrt( np.sum(errs[event_times_mask]**2) )
    s2n              = np.abs( sum_grb_counts/sum_errs )
    return s2n


def evaluateGRB_SN_peak(counts, errs):
    """
    Compute the S/N ratio of the peak of the GRB.
    Input:
     - counts: counts of the event;
     - errs: errors over the counts;
    Output:
     - s2n: signal to noise ratio of the peak;
    """
    c_max    = np.max(counts)
    i_c_max  = np.argmax(counts)
    s_n_peak = c_max / errs[i_c_max]
    return s_n_peak

################################################################################

def load_lc_batse(path):
    """
    Load the BATSE light curves, and put each of them in an object inside
    a list. Since in the analysis we consider only the long GRBs, we load 
    only the light curves listed in the 'alltrig_long.list' file.
    Input:
    - path: path to the folder that contains a file for each BATSE GRB and the
            file containing all the T90s;
    Output:
    - grb_list_batse: list of GRB objects;
    """
    # load only the GRBs that are already classified as 'long'
    long_list_file     = 'alltrig_long.list'
    all_grb_list_batse = [grb_num.rstrip('\n') for grb_num in open(path+long_list_file).readlines()]
    # load T90s
    t90data = np.loadtxt(path+'T90_full.dat')

    grb_list_batse = []
    grb_not_found  = []
    for grb_name in tqdm(all_grb_list_batse):
        try:
            times, counts, errs = np.loadtxt(path+grb_name+'_all_bs.out', unpack=True)
        except:
            # print(grb_name, ' not found!')
            grb_not_found.append(grb_name)
            continue
        t90    = t90data[t90data[:,0] == float(grb_name),1]
        times  = np.float32(times)
        counts = np.float32(counts)
        errs   = np.float32(errs)
        t90    = np.float32(t90)
        grb    = GRB(grb_name, times, counts, errs, t90, path+grb_name+'_all_bs.out')
        grb_list_batse.append(grb)

    print("Total number of _long_ GRBs in BATSE catalogue: ", len(all_grb_list_batse))
    print("GRBs in the catalogue which are NOT present in the data folder: ", len(grb_not_found))
    print("Loaded GRBs: ", len(grb_list_batse))
    return grb_list_batse

################################################################################

def load_lc_swift(path):
    """
    Load the Swift light curves, and put each of them in an object inside
    a list. Since in the analysis we consider only the _long_ GRBs, we load 
    only the light curves listed in the 'merged_lien16-GCN_long_noshortEE_t90.dat'
    file.
    Input:
    - path: path to the folder that contains a folder for each Swift GRB named
            with the name of the GRB, and the file containing all the T90s;
    Output:
    - grb_list_swift:  list of GRB objects;
    """

    # load only the GRBs that are already classified as 'long'
    long_list_file     = 'merged_lien16-GCN_long_noshortEE_t90.dat'
    all_grb_list_swift = []
    t90_dic            = {}
    with open(path+long_list_file) as f:
        for line in f:
            grb_name = line.split()[0]
            t90      = line.split()[1]
            all_grb_list_swift.append(grb_name)
            t90_dic[grb_name] = np.float32(t90)

    grb_list_swift = []
    grb_not_found  = []
    for grb_name in tqdm(all_grb_list_swift):
        try:
            times, counts, errs = np.loadtxt(path+grb_name+'/'+'all_3col.out', unpack=True)
        except:
            # print(grb_name, ' not found!')
            grb_not_found.append(grb_name)
            continue
        t90    = t90_dic[grb_name]
        times  = np.float32(times)
        counts = np.float32(counts)
        errs   = np.float32(errs)
        t90    = np.float32(t90)
        grb    = GRB(grb_name, times, counts, errs, t90, path+grb_name+'/'+'all_3col.out')
        grb_list_swift.append(grb)

    print("Total number of GRBs in Swift catalogue: ", len(all_grb_list_swift))
    print("GRBs in the catalogue which are NOT present in the data folder: ", len(grb_not_found))
    print("Loaded GRBs: ", len(grb_list_swift))
    return grb_list_swift

################################################################################

def load_lc_sax_hr(path):
    """
    Load the HIGH RESOLUTION BeppoSAX light curves, and put each of them in an 
    object inside a list. The GRBs are listed in the file:
    'formatted_catalogue_GRBM_paper.txt'. We discard GBRs which are not fully
    covered by the HR data; since the high res data last for 106 sec, we discard
    the GRBs which have a nominal T90 greater than 106 sec (we discard also the
    GRBs which do not have a nominal T90).
    Input:
    - path: path to the folder that contains a folder for each Sax GRB named
            with the name of the GRB, and the file containing all the T90s;
    Output:
    - grb_list_sax: list of GRB objects;
    """
    #-------------------------------------------------------------------------#
    # def find_file_OLD(path, grb_name):
    #     """
    #     Given the name of a SAX GRB, find the name of the file containing the
    #     light curve with the following properties:
    #     - high-res ('h');
    #     - channel zero ('0');
    #     - background-subtracted ('bs');
    #     Input:
    #     - path: path containing the SAX GRBs;
    #     - grb_name: name of the GBR;
    #     Output:
    #     - name of the file with the selected lc;
    #     """
    #     import fnmatch
    #     list_h0_bs = []
    #     for grb_filename in os.listdir(path+grb_name):
    #         # select only lc with: high-res, channel 0, background-subtracted
    #         if fnmatch.fnmatch(grb_filename, 'grb*_h0_bs.out'):
    #             list_h0_bs.append(grb_filename) 
    #     assert len(list_h0_bs)==1, 'The GRB "'+grb_name+'" does not have the high-res 0th channel light-curve...'
    #     return list_h0_bs[0]
    #-------------------------------------------------------------------------#

    def find_file(path, grb_name):
        """
        Given the name of a SAX GRB, find the name of the file containing the
        light curve with the following properties:
        - high-res ('h');
        - background-subtracted ('bs');
        - the channel is the one written in the 'best_grbm_snr.txt' file; if the
            file with the 'best' channel is not present, we select the HR lc with 
            the mixture of two channel (e.g., 12, 23, etc.);
        Input:
        - path: path containing the SAX GRBs;
        - grb_name: name of the GBR;
        Output:
        - name of the file with the selected lc;
        """
        import fnmatch

        # check if there is the 'best_grbm_snr.txt', and extract the number of
        # the corresponding channel
        try:
            best_idx=int(np.loadtxt(path+grb_name+'/best_grbm_snr.txt', unpack=True))
        except:
            raise FileNotFoundError

        list_hr_bs_best     = []
        list_hr_bs_not_best = []
        for grb_filename in os.listdir(path+grb_name):
            # check if the "best" GRB has the high-res lc
            if fnmatch.fnmatch(grb_filename, 'grb*_h'+str(best_idx)+'_bs.out'):
                list_hr_bs_best.append(grb_filename)
        # if the 'best' HR channel is not present, take the mixture of 2 channels
        if (len(list_hr_bs_best)==0):
            list_h_bs = []
            for grb_filename in os.listdir(path+grb_name):
                if fnmatch.fnmatch(grb_filename, 'grb*_h*_bs.out'):
                    list_h_bs.append(grb_filename)
            pairs = ['h12', 'h13', 'h14', 'h23', 'h24', 'h34', 'h21', 'h31', 'h41', 'h32', 'h42', 'h43']
            for el in pairs:        
                list_hr_bs_best.append( [ grb for grb in list_h_bs if el in grb] )
            list_hr_bs_best = [x for x in list_hr_bs_best if x] # remove empty elements
            assert len(list_hr_bs_best)==1, 'The GRB "'+grb_name+'" has more than 1 mixed HR channel...'
            list_hr_bs_best = list_hr_bs_best[0]

        ## if the HR lc corresponding the the "best" channel is not found,
        ## print all the _other_ HR lc in the folder
        #if(len(list_hr_bs_best)==0):
        #    for grb_filename in os.listdir(path+grb_name):
        #            if fnmatch.fnmatch(grb_filename, 'grb*_h*_bs.out'):
        #                list_hr_bs_not_best.append(grb_filename)

        #print(list_hr_bs_not_best)
        assert len(list_hr_bs_best)==1, 'The GRB "'+grb_name+'" does not have the high-res light-curve...'
        return list_hr_bs_best[0]

    #--------------------------------------------------------------------------#

    # load all the GRBs
    list_file = 'formatted_catalogue_GRBM_paper.txt'
    all_grb_list_sax     = []
    all_grb_cat_list_sax = []
    with open(path+list_file) as f:
        for line in f:
            grb_name     = line.split()[1]
            grb_cat_name = line.split()[2]
            all_grb_list_sax.append(grb_name)
            all_grb_cat_list_sax.append(grb_cat_name)

    # load T90
    t90data = np.loadtxt(path+'saxgrbm_t90.dat', dtype='str', skiprows=1)

    grb_no_hr=0
    grb_list_sax  = []
    grb_not_found = []
    grb_no_t90    = []
    grb_not_full  = []
    for grb_name, grb_cat_name in zip(all_grb_list_sax, all_grb_cat_list_sax):
        #
        try:
            hr_grb = find_file(path, grb_name)
            #print(hr_grb)
        except AssertionError:
            grb_no_hr+=1
            continue
        except FileNotFoundError:
            grb_no_hr+=1
            continue
        except:
            print('ERROR: do not know what happened; check manually...')
            exit()
        #
        try:
            times, counts, errs = np.loadtxt(path+grb_name+'/'+hr_grb, unpack=True)  
            grb_cat_name        = grb_cat_name.replace("GRB", "")
        except:
            grb_not_found.append(grb_name)
            continue
        # for some GRBs we don't have the T90 in the file
        t90 = t90data[t90data[:,0]==grb_cat_name][0][1]
        if t90=='n.a.':
            grb_no_t90.append(grb_name)
            continue
        t90    = t90.astype('float32')
        times  = np.float32(times)
        counts = np.float32(counts)
        errs   = np.float32(errs)
        t90    = np.float32(t90)
        if (t90>106): # discard GBRs which are not fully covered by the HR data
            grb_not_full.append(grb_name)
            continue
        grb = GRB(grb_name, times, counts, errs, t90, path+grb_name+'/'+hr_grb)
        grb_list_sax.append(grb)

    print("Total number of GRBs in BeppoSAX catalogue: ", len(all_grb_list_sax))
    print('GRBs that have an high-res "best" (or 2-mixed) channel lc:', len(all_grb_list_sax)-grb_no_hr)
    print("GRBs in the catalogue which are NOT present in the data folder: ", len(grb_not_found))
    print("GRBs in the catalogue which have a T90 greater than 106s: ", len(grb_not_full))
    print("GRBs in the catalogue which are present in the data folder, but with no T90: ", len(grb_no_t90))
    print("Loaded GRBs: ", len(grb_list_sax))
    return grb_list_sax

################################################################################

def load_lc_sax_lr(path):
    """
    Load the LOW RESOLUTION BeppoSAX light curves, and put each of them in an 
    object inside a list. The GRBs are listed in the file:
    'formatted_catalogue_GRBM_paper.txt'. 
    Input:
    - path: path to the folder that contains a folder for each Sax GRB named
            with the name of the GRB, and the file containing all the T90s;
    Output:
    - grb_list_sax: list of GRB objects;
    """

    list_file = 'formatted_catalogue_GRBM_paper.txt'
    all_grb_list_sax     = []
    all_grb_cat_list_sax = []
    with open(path+list_file) as f:
        for line in f:
            grb_name     = line.split()[1]
            grb_cat_name = line.split()[2]
            all_grb_list_sax.append(grb_name)
            all_grb_cat_list_sax.append(grb_cat_name)

    # load T90
    t90data = np.loadtxt(path+'saxgrbm_t90.dat', dtype='str', skiprows=1)

    grb_list_sax  = []
    grb_not_found = []
    grb_no_t90    = []

    for grb_name, grb_cat_name in zip(all_grb_list_sax, all_grb_cat_list_sax):
        try:
            #times, counts, errs = np.loadtxt(path+grb_name+'/'+hr_grb, unpack=True) 
            try: 
                times, counts, errs = np.loadtxt(path+grb_name+'/'+grb_name+'_grbm0_bs_nospk.out', unpack=True)  
                grb_path_name = path+grb_name+'_grbm0_bs_nospk.out'
            except FileNotFoundError:
                times, counts, errs = np.loadtxt(path+grb_name+'/'+grb_name +'_grbm0_bs.out', unpack=True)  
                grb_path_name = path+grb_name+'_grbm0_bs.out'
            grb_cat_name        = grb_cat_name.replace("GRB", "")
        except:
            grb_not_found.append(grb_name)
            continue
        # for some GRBs we don't have the T90 in the file
        t90 = t90data[t90data[:,0]==grb_cat_name][0][1]
        if t90=='n.a.':
            grb_no_t90.append(grb_name)
            continue
        t90    = t90.astype('float32')
        times  = np.float32(times)
        counts = np.float32(counts)
        errs   = np.float32(errs)
        t90    = np.float32(t90)

        grb = GRB(grb_name, times, counts, errs, t90, grb_path_name)
        grb_list_sax.append(grb)

    print("Total number of GRBs in BeppoSAX catalogue: ", len(all_grb_list_sax))
    print("GRBs in the catalogue which are NOT present in the data folder: ", len(grb_not_found))
    print("GRBs in the catalogue which are present in the data folder, but with no T90: ", len(grb_no_t90))
    print("Loaded GRBs: ", len(grb_list_sax))
    return grb_list_sax

################################################################################

def load_lc_sim(path):
    """
    Load the simulated light curves, which were previously generated and saved
    as files, named 'lcXXX.txt', one file for each simulated GRB ("XXX" is the 
    index of the GRB generated). The columns in the files are: 'times', 
    'counts', 'errs', 't90'. We put each light curve in a 'GRB' object inside
    a list. 
    Input:
    - path: path to the folder that contains a file for each simulated GRB;
    Output: 
    - grb_list_sim: list of GRB objects;
    """
    grb_sim_names = os.listdir(path)
    grb_list_sim  = []
    for grb_file in tqdm(grb_sim_names):
        left_idx  = grb_file.find('lc') + len('lc')
        right_idx = grb_file.find('.txt')
        grb_name  = grb_file[left_idx:right_idx] # extract the ID of the GRB as string
        times, counts, errs, t90 = np.loadtxt(path+grb_file, unpack=True)
        times  = np.float32(times)
        counts = np.float32(counts)
        errs   = np.float32(errs)
        t90    = np.float32(t90)
        grb    = GRB(grb_name, times, counts, errs, t90[0], path+grb_file)
        grb_list_sim.append(grb)

    print("Total number of simulated GRBs: ", len(grb_sim_names))
    return grb_list_sim

################################################################################

def apply_constraints(grb_list, t90_threshold, sn_threshold, bin_time, t_f, 
                      sn_distr=False, filter=True, verbose=True):
    """
    Given as input a list of GBR objects, the function outputs a list containing
    only the GRBs that satisfy the following constraint:
    - T90 > t90_threshold (2 sec);
    - GRB signal S2N > sn_threshold;
    - the measurement lasts at least for t_f (150 sec) after the peak;
    Input:
    - t90_threshold [s];
    - sn_threshold;
    - bin_time: temporal bin size of the instrument [s];
    - t_f: time after the peak that we need the signal to last [s];
    - sn_distr: if True, return also the distribution of the s2n of all the 
                GRBs in input;
    - filter: if True, is applies a savgol filter before computing the S2N;
    Output:
    - good_grb_list: list of GRB objects, where each one is a GRB that satisfies
                     the 3 constraints described above;
    - sn_levels: list containing the s2n ratio of _all_ the input GRBs (not only
                 of those selected); 
    """
    good_grb_list = []
    sn_levels     = []
    grb_with_neg_t20 = 0
    for grb in grb_list:
        times   = np.float32(grb.times)
        counts  = np.float32(grb.counts)
        errs    = np.float32(grb.errs)
        t90     = np.float32(grb.t90)
        i_c_max = np.argmax(counts)
        try:
            s_n = evaluateGRB_SN(times=times, 
                                 counts=counts, 
                                 errs=errs, 
                                 t90=t90,
                                 bin_time=bin_time,
                                 filter=filter)
        except AssertionError:
            #remove GRB if the t20% is negative
            grb_with_neg_t20 += 1
            s_n = 0
        if sn_distr:
            sn_levels.append(s_n)
        cond_1 = t90>t90_threshold
        cond_2 = s_n>sn_threshold
        #cond_2 = s_n_peak>sn_threshold
        cond_3 = len(counts[i_c_max:])>=(t_f/bin_time)
        if ( cond_1 and cond_2 and cond_3 ):
            good_grb_list.append(grb)

    if verbose:
        print("Total number of input GRBs: ", len(grb_list))
        print("GRBs with negative duration: ", grb_with_neg_t20)
        print("GRBs that satisfy the constraints: ", len(good_grb_list)) 

    if sn_distr:
        return good_grb_list, sn_levels
    else:
        return good_grb_list

################################################################################

# def rebinFunction(x, y, erry, s_n_threshold=5, bin_reb_max=100):
#     """
#     Rebins the arrays x, y and erry (errors on the y) with the constraint that 
#     all the bins of the rebinned vectors must have a S/N bigger than a given 
#     threshold.
#     Input:
#       - x: x-array of the data
#       - y: y-array of the data
#       - erry: errors on the y
#       - s_n_threshold: acceptance threshold on the S/N ratio.
#       - bin_reb_max: maximum number of bins that can be rebinned together. We 
#                      define a limit on the maximum number of bins which can be 
#                      rebinned together to stop the algorithm from looping 
#                      forever if we reach a region in which the background 
#                      dominates the signal completely;
#     Output:
#       - reb_x: rebinned x-array
#       - reb_y: rebinned y-array
#       - reb_err: rebinned errors on the y
#     """
#     n_bins = 1
#     reb_x, reb_y, reb_err = [x[0]], [y[0]], [erry[0]]
#     for i in range(1,len(y)):
#         new_x       = x[i]
#         bin_sum     = y[i]
#         err_bin_sum = erry[i]
#         sn = bin_sum/err_bin_sum
#         while(sn < s_n_threshold and n_bins <= bin_reb_max):
#             n_bins += 1
#             new_x       = np.mean(x[i:i+n_bins])
#             bin_sum     = np.sum(y[i:i+n_bins])
#             err_bin_sum = np.sqrt(np.sum(erry[i:i+n_bins])**2)
#             sn = np.abs(bin_sum/err_bin_sum)
#         reb_x.append(new_x)
#         reb_y.append(bin_sum/n_bins)
#         reb_err.append(err_bin_sum)
#         i += n_bins
#         n_bins = 1
#     #shift_x = reb_x[0] - x[0]
#     #reb_x -= shift_x
#     #shift_y = reb_y[0] - y[0]
#     #reb_y -= shift_y
#     return np.array(reb_x), np.array(reb_y), np.array(reb_err)


# def roughRebin(vec, reb_factor, with_mean = True):
#     """
#     Rebins a vector of a given _constant_ factor.
#     Input:
#       - vec: vector to rebin
#       - reb_factor: rebin factor, i. e. the number of points to rebin together
#       - with_mean: boolean value. If true, the value of the points of the 
#                    rebinned vector are evaluated as the average of the points
#                    of the original vector that where to be rebinned together.  
#     Output:
#       - reb_vec: rebinned vector
#     """
#     if with_mean:
#         reb_vec = np.array( [np.mean(vec[i:i+reb_factor]) for i in np.arange(0,len(vec),reb_factor)] )
#     else:
#         reb_vec = np.array( [np.sum(vec[i:i+reb_factor])  for i in np.arange(0,len(vec),reb_factor)] )
#     return reb_vec

################################################################################

def compute_average_quantities(grb_list, t_f=150, bin_time=0.064, 
                               filter=True, filter_window=21,
                               compute_rms=False):
    """
    Compute the averaged peak-aligned fluxes of the GRBs, following the 
    technique described in [Mitrofanov et al., 1996]. We need only the signal
    _after_ the peak, so we cut the lc at his maximum value, neglecting the lc
    in previous times. After extracting the peak-aligned curves, we average over 
    all the light curves in 'grb_list'. Finally, we cut these averages at
    t_f = 150 sec, as shown in the plot by [Stern et al., 1996].
    If filter==True, we smooth the final curve using a Savitzky-Golay filter.
    Input:
    - grb_list: list containing each GRB as an object;
    - t_f: range of time over which we compute the averaged fluxes;
    - bin_time: temporal bin size of the instrument [s] (0.064 is BATSE);
    Output: 
    - averaged_fluxes:        <(F/F_p)>
    - averaged_fluxes_cube:   <(F/F_p)^3>
    - averaged_fluxes_rms : ( <(F/F_p)^2> - <F/F_p>^2 )^(1/2)
    - averaged_fluxes_rms : ( <(F/F_p)^6> - <(F/F_p)^3>^2 )^(1/2) (optional)
    """
    n_steps                = int(t_f/bin_time)
    averaged_fluxes        = np.zeros(n_steps)
    averaged_fluxes_square = np.zeros(n_steps)
    averaged_fluxes_cube   = np.zeros(n_steps)
    if compute_rms:
        averaged_fluxes_cube_square = np.zeros(n_steps)

    for grb in grb_list:
        c_max         = np.max(grb.counts)
        i_c_max       = np.argmax(grb.counts)
        fluxes_to_sum = grb.counts[i_c_max:i_c_max+n_steps] / c_max
        assert np.isclose(fluxes_to_sum[0], 1, atol=1e-06), "ERROR: The peak is not aligned correctly..."
        
        averaged_fluxes        += fluxes_to_sum
        averaged_fluxes_square += fluxes_to_sum**2
        averaged_fluxes_cube   += fluxes_to_sum**3
        if compute_rms:
            averaged_fluxes_cube_square += fluxes_to_sum**6

    averaged_fluxes        /= len(grb_list)
    averaged_fluxes_square /= len(grb_list)
    averaged_fluxes_cube   /= len(grb_list)
    if compute_rms:
        averaged_fluxes_cube_square /= len(grb_list)

    averaged_fluxes_rms = np.sqrt(averaged_fluxes_square - averaged_fluxes**2)
    if compute_rms:
        averaged_fluxes_cube_rms = np.sqrt(averaged_fluxes_cube_square - averaged_fluxes_cube**2)

    if filter:
        averaged_fluxes      = savgol_filter(x=averaged_fluxes,      
                                             window_length=filter_window, 
                                             polyorder=2)
        averaged_fluxes_rms  = savgol_filter(x=averaged_fluxes_rms,  
                                             window_length=filter_window, 
                                             polyorder=2)
        averaged_fluxes_rms[0] = 0 # if we smooth, we loose the fact that the rms has to be 0 in zero                                     
        averaged_fluxes_cube = savgol_filter(x=averaged_fluxes_cube, 
                                             window_length=filter_window, 
                                             polyorder=2)
        if compute_rms:
            averaged_fluxes_cube_rms = savgol_filter(x=averaged_fluxes_rms,  
                                                      window_length=filter_window, 
                                                      polyorder=2)
            averaged_fluxes_cube_rms[0] = 0 # if we smooth, we loose the fact that the rms has to be 0 in zero 
        
    if compute_rms:
        return averaged_fluxes, averaged_fluxes_cube, averaged_fluxes_rms, averaged_fluxes_cube_rms
    else:
        return averaged_fluxes, averaged_fluxes_cube, averaged_fluxes_rms 

################################################################################

def compute_autocorrelation(grb_list, N_lim, t_min=0, t_max=150, 
                            bin_time=0.064, mode='scipy',
                            compute_rms=False):
    """
    Compute the autocorrelation (ACF) of the GRBs. The ACF is computed up to
    a shift of the light curve of t_max = 150 seconds. 
    Inputs:
    - grb_list: list of GRB objects;
    - N_lim: max number of GRBs with which we compute the average ACF;
    - t_min: min time lag for the autocorrelation [s], set by default to zero;
    - t_max: max time lag for the autocorrelation [s];
    - bin_time: temporal bin size of the instrument [s];
    - mode: choose the method to compute the ACF between:
            'scipy': use the scipy.signal.correlate() function. method='auto' 
            automatically chooses direct or Fourier method based on an estimate
            of which is faster.
            'link93': use the method described in Link et al., 1993;
    - compute_rms: if True, the function computes and returns also the rms of 
                   the autocorrelation;
    Outputs:
    - steps: time lags of the autocorrelation;
    - acf: autocorrelation;
    """

    steps   = int((t_max-t_min)/bin_time) # number of steps for ACF
    acf_sum = np.zeros(steps)
    if compute_rms:
        acf_sum_square = np.zeros(steps)

    # Evaluate ACF
    for grb in grb_list[:N_lim]:
        counts = np.array(grb.counts)
        errs   = np.array(grb.errs)
        if mode=='scipy':
            acf   = signal.correlate(in1=counts, in2=counts, method='auto')
            acf   = acf / np.max(acf)  # np.max(acf) is equal to np.sum(counts**2)
            lags  = signal.correlation_lags(in1_len=len(counts), in2_len=len(counts))
            idx_i = np.where(lags*bin_time==t_min)[0][ 0] # select the index corresponding to t =   0 s
            idx_f = np.where(lags*bin_time<=t_max)[0][-1] # select the index corresponding to t = 150 s
            assert lags[idx_i]==t_min, "ERROR: The left limit of the autocorrelation is not computed correctly..."
            assert np.isclose(lags[idx_f]*bin_time, t_max, atol=1e-1), "ERROR: The right limit of the autocorrelation is not computed correctly..."         
            acf = acf[idx_i:idx_f] # select only the autocorrelation up to a shift of t_max = 150 s
        elif mode=='link93':
            # errs=0
            acf = [np.sum((np.roll(counts, u) * counts)[u:]) / np.sum(counts**2 - errs**2) for u in range(steps)]
        acf_sum += acf
        if compute_rms:
            acf_sum_square += acf**2

    acf = acf_sum/N_lim
    if compute_rms:
        acf_square = acf_sum_square/N_lim
        acf_rms    = np.sqrt(acf_square - acf**2)

    if mode=='scipy':
        steps  = lags[idx_i:idx_f]
    elif mode=='link93':
        acf[0] = 1
        steps  = np.arange(steps) 

    if compute_rms:
        return steps, acf, acf_rms
    else:
        return steps, acf

################################################################################

def compute_kde_log_duration(duration_list, x_left=-2, x_right=5, h_opt=0.09):
    """
    Compute the kernel density estimate of the distribution of the (log10) of
    the duration of the selected GRBs;
    Input:
    - duration_list: list containing all the T20% durations of the selected GRBs
                     obtained as output of the function evaluateDuration20();
    - x_left:   left endpoint for the array onto which we compute the sum of gaussians;
    - x_right: right endpoint for the array onto which we compute the sum of gaussians;
    - h_opt: optimal sigma of the gaussian; this value has been obtained with
             GridSearch optimization (see the notebook in DEBUG section);
    Output:
    - dur_distr: kernel density estimate of the log of the duration of the selected GRBs;
    """ 
    duration_list = np.log10(duration_list)
    # Apply kernel density estimation to distribution of durations:
    x_grid     = np.linspace(x_left, x_right, 1000)
    dur_distr  = stats.norm.pdf(x_grid, duration_list[:, None], h_opt) # (x=, loc=, scale=)
    dur_distr /= len(duration_list)
    dur_distr  = dur_distr.sum(0)
    return dur_distr

################################################################################

def compute_loss(averaged_fluxes,      averaged_fluxes_sim,
                 averaged_fluxes_cube, averaged_fluxes_cube_sim,
                 acf,                  acf_sim,
                 duration,             duration_sim,
                 log=False,            verbose=False):
    """
    Compute the loss to be used for the optimization in the Genetic Algorithm.
    Input:
    -
    Output:
    - l2_loss: L2 loss;
    """
    if log:
        averaged_fluxes          = np.log10(averaged_fluxes)
        averaged_fluxes_sim      = np.log10(averaged_fluxes_sim)
        averaged_fluxes_cube     = np.log10(averaged_fluxes_cube)
        averaged_fluxes_cube_sim = np.log10(averaged_fluxes_cube_sim)
        acf                      = np.log10(acf)
        acf_sim                  = np.log10(acf_sim)
        # 'duration'     is already in log scale, since it is the output of compute_kde_log_duration()
        # 'duration_sim' is already in log scale, since it is the output of compute_kde_log_duration()

    w1 = 1.
    w2 = 1.
    w3 = 1.
    w4 = 1.

    l2_loss_fluxes      = np.sqrt( np.sum(np.power((averaged_fluxes-averaged_fluxes_sim),2)) )
    l2_loss_fluxes_cube = np.sqrt( np.sum(np.power((averaged_fluxes_cube-averaged_fluxes_cube_sim),2)) )
    l2_loss_acf         = np.sqrt( np.sum(np.power((acf-acf_sim),2)) )
    l2_loss_duration    = np.sqrt( np.sum(np.power((duration-duration_sim),2)) )
    l2_loss             = w1 * (1./4) * l2_loss_fluxes      + \
                          w2 * (1./4) * l2_loss_fluxes_cube + \
                          w3 * (1./4) * l2_loss_acf         + \
                          w4 * (1./4) * l2_loss_duration
    if verbose:
        # WE SHOULD CHECK WHAT IS THE ORDER OF MAGNITUDE OF EACH LOSS, SO THAT
        # WE KNOW HOW MUCH THEY CONTRIBUTE TO THE TOTAL!
        # Incidentally, there is one combination of log/no-log that makes the 
        # loss functions all in the range ~ np.abs( [0,1] ), which is the one
        # obtained by NOT choosing the log on averaged_fluxes, averaged_fluxes_cube,
        # and acf, while choosing the log for duration (which is automatically
        # in log scale, since it is the output of the function compute_kde_log_duration())
        pass
                          
    return l2_loss

################################################################################

def make_plot(instrument, test_times, 
              # plot 1
              averaged_fluxes,      
              averaged_fluxes_sim,
              averaged_fluxes_rms,  
              averaged_fluxes_rms_sim,
              # plot 2
              averaged_fluxes_cube, 
              averaged_fluxes_cube_sim,
              # plot 3
              steps, 
              steps_sim, 
              bin_time, 
              acf, 
              acf_sim,
              # plot 4
              duration, 
              duration_sim,
              # mode
              log=True, 
              hist=False, 
              # error bars
              err_bars=False, 
              sigma=1,
              averaged_fluxes_cube_rms=None, 
              averaged_fluxes_cube_rms_sim=None,
              acf_rms=None,                  
              acf_rms_sim=None,
              n_grb_real=None,               
              n_grb_sim=None, 
              # save plot
              save_fig=False, 
              name_fig='fig.pdf'):
    """
    Make plot as in Stern et al., 1996.
    """
    fig, ax = plt.subplots(2, 2, figsize=(14,12))

    if instrument=='batse':
        label_instr='BATSE'
        n_grb_real=578
    elif instrument=='swift':
        label_instr='Swift'
        n_grb_real=561
    elif instrument=='sax':
        label_instr='BeppoSAX'
        n_grb_real=121
    else:
        raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax".')

    averaged_fluxes          = np.array(averaged_fluxes)
    averaged_fluxes_sim      = np.array(averaged_fluxes_sim)
    averaged_fluxes_rms      = np.array(averaged_fluxes_rms)
    averaged_fluxes_rms_sim  = np.array(averaged_fluxes_rms_sim)
    averaged_fluxes_cube     = np.array(averaged_fluxes_cube)
    averaged_fluxes_cube_sim = np.array(averaged_fluxes_cube_sim)
    acf                      = np.array(acf)
    acf_sim                  = np.array(acf_sim)
    duration                 = np.array(duration)
    duration_sim             = np.array(duration_sim)

    #--------------------------------------------------------------------------#
    # <(F/F_p)>
    #--------------------------------------------------------------------------#

    # plots
    ax[0,0].plot(test_times**(1/3),     averaged_fluxes,             color='b', alpha=1.00, label = label_instr)
    ax[0,0].plot(test_times**(1/3),     averaged_fluxes_sim,         color='r', alpha=0.75, label = r'Simulated')
    ax[0,0].plot(test_times[1:]**(1/3), averaged_fluxes_rms[1:],     color='b', alpha=1.00)
    ax[0,0].plot(test_times[1:]**(1/3), averaged_fluxes_rms_sim[1:], color='r', alpha=0.75)
    # error bars
    if err_bars:
        errs     = averaged_fluxes_rms     / np.sqrt(n_grb_real)
        errs_sim = averaged_fluxes_rms_sim / np.sqrt(n_grb_sim)
        #
        ax[0,0].fill_between(test_times**(1/3),
                             averaged_fluxes-sigma*errs,
                             averaged_fluxes+sigma*errs,
                             color='b',
                             alpha=0.25) 
        ax[0,0].fill_between(test_times**(1/3),
                             averaged_fluxes_sim-sigma*errs_sim,
                             averaged_fluxes_sim+sigma*errs_sim,
                             color='r',
                             alpha=0.25)
    # set scale
    if log:
        ax[0,0].set_yscale('log', base=10) 
        #ax[0,0].set_xlim(0,test_times[-1]**(1/3))
        if err_bars:
            ax[0,0].set_ylim(1.e-3, 1.2)
    else:
        pass
        #ax[0,0].set_xlim(0,test_times[-1]**(1/3))
    # set labels
    ax[0,0].set_xlabel(r'$(\mathrm{time}\ [s])^{1/3}$',                       size=18)
    if log:
        ax[0,0].set_ylabel(r'$\log F_{rms},\quad \log \langle F/F_p\rangle$', size=18)
    else:
        ax[0,0].set_ylabel(r'$F_{rms},\quad \langle F/F_p\rangle$',           size=18)
    #
    ax[0,0].text(3,   10**(-0.7), r'$F_{rms}$',              fontsize=20)
    ax[0,0].text(2.2, 10**(-1.7), r'$\langle F/F_p\rangle$', fontsize=20)
    #
    ax[0,0].grid(True, which="major", lw=1.0, ls="-")
    ax[0,0].grid(True, which="minor", lw=0.3, ls="-")
    ax[0,0].xaxis.set_tick_params(labelsize=14)
    ax[0,0].yaxis.set_tick_params(labelsize=14)
    ax[0,0].legend(prop={'size':15}, loc="lower left", facecolor='white', framealpha=0.5)

    #--------------------------------------------------------------------------#
    # <(F/F_p)^3>
    #--------------------------------------------------------------------------#

    # plots
    ax[0,1].plot(test_times**(1/3), averaged_fluxes_cube,     color='b', label=label_instr)
    ax[0,1].plot(test_times**(1/3), averaged_fluxes_cube_sim, color='r', label='Simulated', alpha=0.75)
    # error bars
    if err_bars:
        errs     = averaged_fluxes_cube_rms     / np.sqrt(n_grb_real)
        errs_sim = averaged_fluxes_cube_rms_sim / np.sqrt(n_grb_sim)
        #
        ax[0,1].fill_between(test_times**(1/3),
                             averaged_fluxes_cube-sigma*errs,
                             averaged_fluxes_cube+sigma*errs,
                             color='b',
                             alpha=0.25) 
        ax[0,1].fill_between(test_times**(1/3),
                             averaged_fluxes_cube_sim-sigma*errs_sim,
                             averaged_fluxes_cube_sim+sigma*errs_sim,
                             color='r',
                             alpha=0.25)

    # set scale
    if log:
        ax[0,1].set_yscale('log', base=10)
        #ax[0,1].set_xlim(0,test_times[-1]**(1/3))
        if err_bars:
            ax[0,1].set_ylim(7.e-5, 1)
    else:
        pass
        #ax[0,1].set_xlim(0,test_times[-1]**(1/3))
    # set labels
    ax[0,1].set_xlabel(r'$(\mathrm{time}\ [s])^{1/3}$',         size=18)
    if log:
        ax[0,1].set_ylabel(r'$\log \langle (F/F_p)^3 \rangle$', size=18)
    else:
        ax[0,1].set_ylabel(r'$\langle (F/F_p)^3 \rangle$',      size=18)
    #
    ax[0,1].grid(True, which="major", lw=1.0, ls="-")
    ax[0,1].grid(True, which="minor", lw=0.3, ls="-")
    ax[0,1].xaxis.set_tick_params(labelsize=14)
    ax[0,1].yaxis.set_tick_params(labelsize=14)
    ax[0,1].legend(prop={'size':15}, loc="lower left", facecolor='white', framealpha=0.5)

    #--------------------------------------------------------------------------#
    # AUTOCORRELATION
    #--------------------------------------------------------------------------#

    # plots
    ax[1,0].plot((steps    *bin_time)**(1/3), acf,     color='b', label=label_instr)
    ax[1,0].plot((steps_sim*bin_time)**(1/3), acf_sim, color='r', label='Simulated', alpha=0.75)
    # error bars
    if err_bars:
        errs     = acf_rms     / np.sqrt(n_grb_real)
        errs_sim = acf_rms_sim / np.sqrt(n_grb_sim)
        #
        ax[1,0].fill_between((steps*bin_time)**(1/3),
                             acf-sigma*errs,
                             acf+sigma*errs,
                             color='b',
                             alpha=0.25) 
        ax[1,0].fill_between((steps_sim*bin_time)**(1/3),
                             acf_sim-sigma*errs_sim,
                             acf_sim+sigma*errs_sim,
                             color='r',
                             alpha=0.25)
    # set scale
    if log:
        ax[1,0].set_yscale('log', base=10)
    else:
        pass
    # set labels
    ax[1,0].set_xlabel(r'$(\mathrm{timelag}\ [s])^{1/3}$', size=18)
    if log:
        ax[1,0].set_ylabel(r'$\log \langle ACF \rangle$',  size=18)
    else:
        ax[1,0].set_ylabel(r'$\langle ACF \rangle$',       size=18)
    #
    ax[1,0].grid(True, which="major", lw=1.0, ls="-")
    ax[1,0].grid(True, which="minor", lw=0.3, ls="-")
    ax[1,0].xaxis.set_tick_params(labelsize=14)
    ax[1,0].yaxis.set_tick_params(labelsize=14)
    ax[1,0].legend(prop={'size':15}, loc="lower left", facecolor='white', framealpha=0.5)

    #--------------------------------------------------------------------------#
    # DISTRIBUTION OF DURATIONS
    #--------------------------------------------------------------------------#

    if log:
        duration     = np.log10(duration)
        duration_sim = np.log10(duration_sim)
    if log:
        range_hist = [-1.0, 3.5]
    else:
        range_hist = None

    if hist:
        # histogram
        n_bins=30
        #n1, bins, patches = ax[1,1].hist(x=duration,
        #                                 bins=n_bins,
        #                                 alpha=1.00,
        #                                 label=label_instr, 
        #                                 color='b',
        #                                 histtype='step',
        #                                 linewidth=4,
        #                                 range=range_hist,
        #                                 density=False)
        #n2, bins, patches = ax[1,1].hist(x=duration_sim,
        #                                 bins=n_bins,
        #                                 alpha=0.75,
        #                                 label='Simulated', 
        #                                 color='r',
        #                                 histtype='step',
        #                                 linewidth=4,
        #                                 range=range_hist,
        #                                 density=False)
        n1, bins = np.histogram(a=duration,     bins=n_bins, range=range_hist)
        n2, bins = np.histogram(a=duration_sim, bins=n_bins, range=range_hist)

        bin_centres = 0.5 * (bins[:-1] + bins[1:])

        ax[1,1].bar(x=bins[:-1], 
                    height=n1/(np.diff(bins)[0]*len(duration)),     
                    width=np.diff(bins), 
                    align='edge',
                    #facecolor='None',
                    #edgecolor='b',
                    #linewidth=2,
                    alpha=0.6,
                    color='b',
                    label=label_instr)
        ax[1,1].bar(x=bins[:-1], 
                    height=n2/(np.diff(bins)[0]*len(duration_sim)), 
                    width=np.diff(bins), 
                    align='edge',
                    #facecolor='None',
                    edgecolor='r',
                    linewidth=2,
                    alpha=0.4,
                    color='r',
                    label='Simulated')

        ax[1,1].set_ylim(-0.025,1.0)

        if err_bars:
            # Plot the error bars, centred on (bin_centre, bin_count), with length y_error
            ax[1,1].errorbar(x=bin_centres, 
                             y=n1/(np.diff(bins)[0]*len(duration)),
                             yerr=sigma*np.sqrt(n1)/(np.diff(bins)[0]*len(duration)), 
                             fmt='.', 
                             color='b',
                             capsize=3,
                             elinewidth=1.5)
            ax[1,1].errorbar(x=bin_centres, 
                             y=n2/(np.diff(bins)[0]*len(duration_sim)),
                             yerr=sigma*np.sqrt(n2)/(np.diff(bins)[0]*len(duration_sim)), 
                             fmt='.', 
                             color='r',
                             capsize=3,
                             elinewidth=1.5)


    else: 
        # kernel density estimation
        h_opt = 0.09 # values obtained with GridSearch optimization (see the notebook in DEBUG section)
        if log:
            x_grid = np.linspace(-2,    5, 1000)
        else:
            x_grid = np.linspace(-2, 1000, 1000)
        y_plot_real  = stats.norm.pdf(x_grid, duration[:, None],     h_opt)
        y_plot_sim   = stats.norm.pdf(x_grid, duration_sim[:, None], h_opt)
        y_plot_real /= (len(duration))
        y_plot_sim  /= (len(duration_sim))
        kde_real     = y_plot_real.sum(0)
        kde_sim      = y_plot_sim.sum(0)
        # plot
        ax[1,1].plot(x_grid, kde_real, c='b', lw=3, label=label_instr, zorder=5)
        ax[1,1].plot(x_grid, kde_sim,  c='r', lw=3, label='Simulated', zorder=6)
        # errors
        if err_bars:
            n_resample=500
            kde_real_r_stack     = np.zeros([len(kde_real),n_resample])
            kde_real_r_stack_sim = np.zeros([len(kde_sim), n_resample])
            for i in range(n_resample):
                dur_resampled_real = resample(duration,     replace=True)
                dur_resampled_sim  = resample(duration_sim, replace=True)
                y_plot_real_r      = stats.norm.pdf(x_grid, dur_resampled_real[:, None], h_opt)
                y_plot_sim_r       = stats.norm.pdf(x_grid, dur_resampled_sim[:, None],  h_opt)
                y_plot_real_r     /= (len(dur_resampled_real))
                y_plot_sim_r      /= (len(dur_resampled_sim))
                kde_real_r         = y_plot_real_r.sum(0)
                kde_sim_r          = y_plot_sim_r.sum(0)
                kde_real_r_stack[:,i]     = kde_real_r
                kde_real_r_stack_sim[:,i] = kde_sim_r
                # plot
                # ax[1,1].plot(x_grid, kde_real_r, c='cyan',   lw=1, alpha=0.05, zorder=3)
                # ax[1,1].plot(x_grid, kde_sim_r,  c='orange', lw=1, alpha=0.05, zorder=4)
            rms     = np.std(kde_real_r_stack,     axis=1)
            rms_sim = np.std(kde_real_r_stack_sim, axis=1)
            errs     = rms     #/ np.sqrt(n_resample)
            errs_sim = rms_sim #/ np.sqrt(n_resample)
            ax[1,1].fill_between(x_grid,
                                 kde_real-sigma*errs,
                                 kde_real+sigma*errs,
                                 color='b',
                                 alpha=0.25,
                                 zorder=1) 
            ax[1,1].fill_between(x_grid,
                                 kde_sim-sigma*errs_sim,
                                 kde_sim+sigma*errs_sim,
                                 color='r',
                                 alpha=0.25,
                                 zorder=2)

    # set scale
    if log:
        ax[1,1].set_xlim(-1.0,3.5)
    else:
        pass
    # set label
    ax[1,1].set_ylabel('(Normalized) Number of events',    size=18)
    if log:
        ax[1,1].set_xlabel(r'$\log\mathrm{duration}$ [s]', size=18)
    else:
        ax[1,1].set_xlabel(r'$\mathrm{duration}$ [s]',     size=18)
    #
    ax[1,1].grid(True, which="major", lw=1.0, ls="-")
    ax[1,1].grid(True, which="minor", lw=0.3, ls="-")
    ax[1,1].xaxis.set_tick_params(labelsize=14)
    ax[1,1].yaxis.set_tick_params(labelsize=14)
    ax[1,1].legend(prop={'size':15}, loc="upper left", facecolor='white', framealpha=0.5)

    #from scipy.stats import ks_2samp
    #ks_test_res = ks_2samp(n1, n2)
    #print(ks_test_res)
    #from scipy.stats import anderson_ksamp
    #ad_res = anderson_ksamp([n1, n2])
    #print(ad_res)

    #--------------------------------------------------------------------------#
    #--------------------------------------------------------------------------#

    if(save_fig):
        plt.savefig(name_fig)

    plt.show()

################################################################################

def make_plot_errs(test_times, 
                   # plot 1
                   averaged_fluxes,      
                   averaged_fluxes_sim,
                   averaged_fluxes_rms,  
                   averaged_fluxes_rms_sim,
                   # plot 2
                   averaged_fluxes_cube, 
                   averaged_fluxes_cube_sim,
                   # plot 3
                   steps, 
                   bin_time, 
                   acf,                  
                   acf_sim,
                   # 
                   averaged_fluxes_cube_rms, 
                   averaged_fluxes_cube_rms_sim,
                   acf_rms,
                   acf_rms_sim,
                   n_grb_real,
                   n_grb_sim, 
                   # save plot
                   save_fig=False, 
                   name_fig='fig_errs.pdf'):
    """
    Make plot of delta value over error.
    """

    sigma  = 1.96 # 97.5th percentile point
    sigma3 = 3 

    test_times                   = np.array(test_times)
    averaged_fluxes              = np.array(averaged_fluxes)
    averaged_fluxes_sim          = np.array(averaged_fluxes_sim)
    averaged_fluxes_rms          = np.array(averaged_fluxes_rms)
    averaged_fluxes_rms_sim      = np.array(averaged_fluxes_rms_sim)
    averaged_fluxes_cube         = np.array(averaged_fluxes_cube)
    averaged_fluxes_cube_sim     = np.array(averaged_fluxes_cube_sim)
    averaged_fluxes_cube_rms     = np.array(averaged_fluxes_cube_rms)
    averaged_fluxes_cube_rms_sim = np.array(averaged_fluxes_cube_rms_sim)
    acf                          = np.array(acf)
    acf_sim                      = np.array(acf_sim)

    fig, ax = plt.subplots(2, 2, figsize=(14,12))

    #--------------------------------------------------------------------------#
    # <(F/F_p)>
    #--------------------------------------------------------------------------#

    label = r'$\frac{\langle F/F_p \rangle_{sim}-\langle F/F_p \rangle_{real}}{\sqrt{\sigma^2_{sim}+\sigma^2_{real}}}$'
    # plots
    errs     = averaged_fluxes_rms[1:]     / np.sqrt(n_grb_real)
    errs_sim = averaged_fluxes_rms_sim[1:] / np.sqrt(n_grb_sim)
    ax[0,0].plot(test_times[1:]**(1/3), (averaged_fluxes_sim-averaged_fluxes)[1:]/np.sqrt(errs_sim**2+errs**2), c='b', label=label)
    # sigma
    ax[0,0].axhline(y=+sigma,  xmin=test_times[0]**(1/3), xmax=test_times[-1]**(1/3), c='k', ls='--', label=r'$1.96\sigma$')
    ax[0,0].axhline(y=-sigma,  xmin=test_times[0]**(1/3), xmax=test_times[-1]**(1/3), c='k', ls='--')
    ax[0,0].axhline(y=+sigma3, xmin=test_times[0]**(1/3), xmax=test_times[-1]**(1/3), c='r', ls='--', label=r'$3\sigma$')
    ax[0,0].axhline(y=-sigma3, xmin=test_times[0]**(1/3), xmax=test_times[-1]**(1/3), c='r', ls='--')
    # set labels
    ax[0,0].set_xlabel(r'$(\mathrm{time}\ [s])^{1/3}$',         size=18)
    ax[0,0].set_ylabel(r'$ \langle F/F_p\rangle$', size=18)
    # set limits
    ax[0,0].set_ylim(-5,5)
    # other
    ax[0,0].grid(True, which="major", lw=1.0, ls="-")
    ax[0,0].grid(True, which="minor", lw=0.3, ls="-")
    ax[0,0].xaxis.set_tick_params(labelsize=14)
    ax[0,0].yaxis.set_tick_params(labelsize=14)
    ax[0,0].legend(prop={'size':15}, loc="best", facecolor='white', framealpha=0.5)

    #--------------------------------------------------------------------------#
    # <(F/F_p)^3>
    #--------------------------------------------------------------------------#

    label = r'$\frac{\langle (F/F_p)^3 \rangle_{sim}-\langle (F/F_p)^3 \rangle_{real}}{\sqrt{\sigma^2_{sim}+\sigma^2_{real}}}$'
    # plots
    errs     = averaged_fluxes_cube_rms[1:]     / np.sqrt(n_grb_real)
    errs_sim = averaged_fluxes_cube_rms_sim[1:] / np.sqrt(n_grb_sim)
    ax[0,1].plot(test_times[1:]**(1/3), (averaged_fluxes_cube_sim[1:]-averaged_fluxes_cube[1:])/np.sqrt(errs_sim**2+errs**2), c='b', label=label)
    # sigma
    ax[0,1].axhline(y=+sigma,  xmin=test_times[0]**(1/3), xmax=test_times[-1]**(1/3), c='k', ls='--', label=r'$1.96\sigma$')
    ax[0,1].axhline(y=-sigma,  xmin=test_times[0]**(1/3), xmax=test_times[-1]**(1/3), c='k', ls='--')
    ax[0,1].axhline(y=+sigma3, xmin=test_times[0]**(1/3), xmax=test_times[-1]**(1/3), c='r', ls='--', label=r'$3\sigma$')
    ax[0,1].axhline(y=-sigma3, xmin=test_times[0]**(1/3), xmax=test_times[-1]**(1/3), c='r', ls='--')
    # set labels
    ax[0,1].set_xlabel(r'$(\mathrm{time}\ [s])^{1/3}$',         size=18) 
    # set limits
    ax[0,1].set_ylim(-5,5)
    # other
    ax[0,1].grid(True, which="major", lw=1.0, ls="-")
    ax[0,1].grid(True, which="minor", lw=0.3, ls="-")
    ax[0,1].xaxis.set_tick_params(labelsize=14)
    ax[0,1].yaxis.set_tick_params(labelsize=14)
    ax[0,1].legend(prop={'size':15}, loc="best", facecolor='white', framealpha=0.5)

    #--------------------------------------------------------------------------#
    # AUTOCORRELATION
    #--------------------------------------------------------------------------#

    label = r'$\frac{\langle ACF \rangle_{sim}-\langle ACF \rangle_{real}}{\sqrt{\sigma^2_{sim}+\sigma^2_{real}}}$'
    # plots
    errs     = acf_rms[1:]     / np.sqrt(n_grb_real)
    errs_sim = acf_rms_sim[1:] / np.sqrt(n_grb_sim)
    ax[1,0].plot((steps[1:]*bin_time)**(1/3),(acf_sim[1:]-acf[1:])/np.sqrt(errs_sim**2+errs**2), c='b', label=label)
    # sigma
    ax[1,0].axhline(y=+sigma,  xmin=(steps[0]*bin_time)**(1/3), xmax=(steps[-1]*bin_time)**(1/3), c='k', ls='--', label=r'$1.96\sigma$')
    ax[1,0].axhline(y=-sigma,  xmin=(steps[0]*bin_time)**(1/3), xmax=(steps[-1]*bin_time)**(1/3), c='k', ls='--')
    ax[1,0].axhline(y=+sigma3, xmin=(steps[0]*bin_time)**(1/3), xmax=(steps[-1]*bin_time)**(1/3), c='r', ls='--', label=r'$3\sigma$')
    ax[1,0].axhline(y=-sigma3, xmin=(steps[0]*bin_time)**(1/3), xmax=(steps[-1]*bin_time)**(1/3), c='r', ls='--')
    # set labels
    ax[1,0].set_xlabel(r'$(\mathrm{timelag}\ [s])^{1/3}$', size=18)
    # set limits
    ax[1,0].set_ylim(-8,8)
    # other
    ax[1,0].grid(True, which="major", lw=1.0, ls="-")
    ax[1,0].grid(True, which="minor", lw=0.3, ls="-")
    ax[1,0].xaxis.set_tick_params(labelsize=14)
    ax[1,0].yaxis.set_tick_params(labelsize=14)
    ax[1,0].legend(prop={'size':15}, loc="best", facecolor='white', framealpha=0.5)

    #--------------------------------------------------------------------------#
    # 
    #--------------------------------------------------------------------------#

    # plots
    ax[1,1].plot([0,1],[0,1], c='k')
    ax[1,1].plot([1,0],[0,1], c='k')

    #--------------------------------------------------------------------------#
    #--------------------------------------------------------------------------#

    if(save_fig):
        plt.savefig(name_fig)

    plt.show()

################################################################################

def runMEPSA(mepsa_path, ex_pattern_file_path, grb_file_path, nbins, grb_name, sn_level=5):
    """
    Run the MEPSA code on an input GRB and gives the number of peaks significative above 
    a chosen S/N level.
    Input:
    - mepsa_path: path to the compiled shared mepsa library (mepsa.so). Must be an absolute path
    - ex_pattern_file_path: path to the excess pattern file necessary to run MEPSA. 
    - grb_file_path: path to the file containing the light-curve information for the GRB. To work
                     properly, MEPSA needs a 3 column file in which the column are times, bkg
                     subtracted counts and errors on the counts.
    - nbins: Maximum rebin factor to be used in MEPSA
    - grb_name: name of the GRB. It is also used to name the output file that MEPSA saves
    - sn_level: minimum S/N level that the peaks must have to be considered significative
    Output:
    - signif_peaks: number of significative peaks
    """
    mepsa_lib = ctypes.CDLL(mepsa_path)
    peak_find = mepsa_lib.main
    peak_find.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    peak_find.restype  = ctypes.c_int

    grb_file_path = grb_file_path.encode('ascii')
    exp_file = ex_pattern_file_path.encode('ascii')
    reb      = str(nbins).encode('ascii')
    out_file = (grb_name + '.dat').encode('ascii')
    argv     = (ctypes.c_char_p * 5) (b'pyMEPSA', grb_file_path, exp_file, reb, out_file)
    peak_find(len(argv), argv)

    with open(out_file, 'r') as result_file:
        result_file.readline()
        signif_peaks=0  # number of significative peaks
        for line in result_file:
            res_tmp = line.split()
            if float(res_tmp[7]) >= sn_level:
                signif_peaks  += 1

    return signif_peaks

################################################################################

def readMEPSAres(mepsa_out_file_list, maximum_reb_factor = np.inf, sn_level = 5):
    """
    Reads a list of output files generated by MEPSA and gets the number of significative peaks for
    each GRB in the list.
    Input:
    - mepsa_out_file_list: list of paths to the output MEPSA files to analyze
    - maximum_reb_factor: maximum rebin factor that can be accepted. Discards all the peaks that
                          are visible only if we rebin more than the maximum rebin factor. 
                          This parameters can be used to ensure consistency while analyzing MEPSA
                          outputs that where produced at different times and with different maximum
                          rebins. 
    - sn_level: minimum S/N level that the peaks must have to be considered significative
    Output:
    - all_signif_peaks: list containing the number of significative peaks for each GRB
    """
    all_signif_peaks = []

    for mepsa_out_file in mepsa_out_file_list:
        with open(mepsa_out_file, 'r') as mepsa_out:
            signif_peaks = 0
            mepsa_out.readline()
            for line in mepsa_out:
                res_tmp = line.split()
                if res_tmp[0] !='#' and float(res_tmp[1]) <= maximum_reb_factor and float(res_tmp[7]) >= sn_level:
                    signif_peaks  += 1
            all_signif_peaks.append(signif_peaks)

    return all_signif_peaks

################################################################################

def generate_GRBs(N_grb,                                            # number of simulated GRBs to produce
                  mu, mu0, alpha, delta1, delta2, tau_min, tau_max, # 7 parameters
                  instrument, bin_time, eff_area, bg_level,         # instrument parameters
                  t90_threshold, sn_threshold, t_f, filter=True,    # constraint parameters
                  export_files=False, export_path='None',           # other parameters
                  n_cut=2000, with_bg=False, seed=None,             # other parameters
                  test_pulse_distr=False                            # other parameters
                  ):
    """
    This function generates a list of GRBs using the avalanche by Stern+96. As
    input it takes the 7 parameters needed for the avalance model, and the 
    parameters of the instrument considered. As output it returns a list, where 
    each element of the list is an RGB object, that has passed the cosntraints 
    selection (see "apply_constraints()" function). 
    Input:
    - N_grb: total number of simulated GRBs to produce in output;
    ### 7 parameters
    - mu:
    - mu0:
    - alpha:
    - delta1:
    - delta2:
    - tau_min:
    - tau_max:
    ### instrument parameters
    - res:
    - eff_area:
    - bg_level;
    ### constraint parameters
    - t90_threshold:
    - sn_threshold: 
    - t_f:
    - filter:
    ### other parameters
    - export_files: if True, every GRB that passed the constraint selection is
                    exorted into an external file;
    - export_path
    - n_cut:
    - with_bg:
    - seed: random seed;
    - test_pulse_distr: if True, it appends to each GRB object also the info
                        on the number of significative pulses inside that GRB,
                        and we also compute the time distances between all the
                        pulses in that GRB;
    Output:
    -grb_list_sim: list containing N_grb GRB objects, each lc satisfying the
                imposed constraints;

    """

    def export_lc(LC, idx, instrument, path='../simulations/'):
        """
        Export the simulated light curves in a file with these columns: 
            times, counts, err_counts, T90.
        Input:
        - LC: object that contains the light curve;
        - idx: number of the light curve;
        - instrument: string with the name of the instrument;
        - path: path where to store the results of the simulations;
        """
        outfile  = path+instrument+'/'+'lc'+str(idx)+'.txt'
        savefile = open(outfile, 'w', encoding='utf-8')
        times    = LC._times
        lc       = LC._plot_lc
        err_lc   = LC._err_lc
        T90      = LC._t90
        for i in range(len(times)):
            savefile.write('{0} {1} {2} {3}\n'.format(times[i], lc[i], err_lc[i], T90))
        savefile.close()

    def count_significative_pulses(LC, verbose=False):
        """
        Count the number of significative pulses in a simulated LC. A pulse is significative
        if its peak rate is bigger than 50*(FWHM)**-0.6 counts/64 ms, where FWHM is the 
        FWHM of the peak.
        Input:
        - LC: object that contains the light curve;
        Output:
        - n_of_sig_pulses: number of significative pulses
        - n_of_total_pulses: number of total pulses generated for the LC
        """
        pulses_param_list = lc._lc_params
        ampl              = lc._ampl
        eff_area          = lc._eff_area

        n_of_sig_pulses      = 0
        significative_pulses = []
        n_of_total_pulses    = len(pulses_param_list)

        for pulse in pulses_param_list:
            # Reads parameters of the pulse and generates it
            norm    = pulse['norm']
            t_delay = pulse['t_delay']
            tau     = pulse['tau']
            tau_r   = pulse['tau_r']

            pulse_curve = lc.norris_pulse(norm, t_delay, tau, tau_r) * ampl * eff_area

            # Find peak rate
            peak_rate = np.max(pulse_curve)

            # Evaluate the FWHM of the pulse (analytical evaluation)
            t_1 = t_delay - tau_r *np.sqrt(np.log(2))
            t_2 = t_delay + tau *np.log(2)
            peak_fwhm = t_2 - t_1

            if verbose:
                print('----')
                print('Delay time: ', t_delay, 's')
                print('Pulse peak rate: ', peak_rate, 'counts/64 ms')
                print('Pulse FWHM: ', peak_fwhm, 's' )
            
            # Evaluate the minimum peak rate for the pulse to be significative (CG formula) 
            # and check if the peak rate of the pulse is above the minimum  
            minimum_peak_rate = 50 * peak_fwhm**(-0.6)
            if peak_rate >= minimum_peak_rate:
                n_of_sig_pulses += 1
                significative_pulses.append(pulse)
        
        if verbose:
            print('-------------------------------------')
            print('Number of generated pulses: ', len(pulses_param_list))
            print('Number of significative pulses: ', n_of_sig_pulses)
            print('-------------------------------------')

        return n_of_sig_pulses, n_of_total_pulses, significative_pulses
    
    def getPulsesTimeDistance(pulses):
        delay_times = np.sort(np.array([pulse['t_delay'] for pulse in pulses]))
        time_distances = np.diff(delay_times)
        return time_distances


    # check that the parameters are in the correct range
    assert delta1<0
    assert delta2>=0
    assert np.abs(delta1)>np.abs(delta2)
    assert tau_min>0
    assert tau_max>0
    assert tau_max>tau_min

    cnt=0
    grb_list_sim         = []
    pulse_time_distances = []
    while (cnt<N_grb):
        lc = LC(### 7 parameters
                mu=mu,
                mu0=mu0,
                alpha=alpha,
                delta1=delta1,
                delta2=delta2,
                tau_min=tau_min, 
                tau_max=tau_max,
                ### instrument parameters:
                res=bin_time,
                eff_area=eff_area,
                bg_level=bg_level,
                ### other parameters:
                n_cut=n_cut,
                with_bg=with_bg)
        lc.generate_avalanche(seed=None)
        if lc.check==0:
            # check that we have generated a lc with non-zero values; otherwise,
            # skip it and continue in the generation process
            del(lc)
            continue

        if test_pulse_distr:
            # count how many pulses are signficative enough to be detected by MEPSA according to CG's formula
            n_of_sig_pulses, n_of_total_pulses, sig_pulses = count_significative_pulses(lc, verbose=False)
        else: 
            n_of_sig_pulses = None

        # convert the lc generated from the avalance into a GRB object
        grb = GRB('lc_candidate.txt', 
                  lc._times, 
                  lc._plot_lc, 
                  lc._err_lc, 
                  lc._t90,
                  export_path+instrument+'/'+'lc_candidate.txt',  
                  n_of_sig_pulses)
        # we use a temporary list that contains only _one_ lc, then we
        # check if that GRB satisfies the constraints imposed, ad if that is
        # the case, we append it to the final list of GRBs
        if grb.t90<t90_threshold: 
            # preliminary check to ensure that the savgol will not fail due
            # to short GRBs, for which often this filter fails. The reason
            # is that the `window_length` of savgol filter must be greater than 
            # `polyorder`, but for short GRBs the computed `window_length` is
            # very small.
            del(lc)
            continue
        grb_list_sim_temp = [ grb ]
        grb_list_sim_temp = apply_constraints(grb_list=grb_list_sim_temp, 
                                              bin_time=bin_time, 
                                              t90_threshold=t90_threshold, 
                                              sn_threshold=sn_threshold, 
                                              t_f=t_f,
                                              filter=filter,
                                              verbose=False)
        # save the GRB into the final list _only_ if it passed the
        # constraints selection
        if (len(grb_list_sim_temp)==1):
            if export_files:
                export_lc(LC=lc, 
                          idx=cnt, 
                          instrument=instrument,
                          path=export_path)
                grb.name = 'lc'+str(cnt)+'.txt'
                #grb.data_file_path = export_path+instrument+'/'+'lc'+str(cnt)+'.txt'
            
            if test_pulse_distr:
                #get all the time distances between the generated peaks and save them to a file. 
                pulse_time_distances.extend(getPulsesTimeDistance(sig_pulses))
                np.savetxt('time_distances.txt',np.array(pulse_time_distances))
            ###############################################################################

            grb_list_sim.append(grb)
            cnt+=1
        del(lc)

    return grb_list_sim

################################################################################



################################################################################



################################################################################
