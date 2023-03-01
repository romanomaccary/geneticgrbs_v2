import ctypes
from datetime import datetime

path='/home/lorenzo/git/lc_pulse_avalanche/pyMEPSA/mepsa.so'
#path='/mnt/c/Users/Lisa/Desktop/pyMEPSA_test/mepsa.so'

start_program = datetime.now()

# A. Create library
C_library = ctypes.CDLL(path)

# B. Specify function signatures
peak_find = C_library.main
peak_find.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)

# C. Invoke function
test_grb = b'./test/input_lc.dat'
exp_file = b'./excess_pattern_MEPSA_v0.dat'
reb      = b'16'
grb_name = 'grb12345'
out_file = (grb_name + '.dat').encode('ascii')
argv     = (ctypes.c_char_p * 5) (b'pyMepsa', test_grb, exp_file, reb, out_file)
#print(out_file)

start_peak_find = datetime.now()
peak_find(len(argv), argv)
end_time = datetime.now()

print("Time to load the libraries: ", (start_peak_find - start_program)  )
print("Time to execute MEPSA:      ", (end_time        - start_peak_find))
