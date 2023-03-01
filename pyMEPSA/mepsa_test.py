
import ctypes
from datetime import datetime

start_program = datetime.now()

# A. Create library
C_library = ctypes.CDLL("/mnt/c/Users/Lisa/Desktop/pyMEPSA_test/mepsa.so")

# B. Specify function signatures
peak_find = C_library.main
peak_find.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)

# C. Invoke function
test_grb = b'test/input_lc.dat'
exp_file = b'excess_pattern_MEPSA_v0.dat'
reb = b'16'
argv = (ctypes.c_char_p * 4) (b'pyMepsa', test_grb, exp_file, reb)

start_peak_find = datetime.now()
peak_find(len(argv), argv)
end_time = datetime.now()

print("Time to load the library: ", (start_peak_find - start_program))
print("Time to execute MEPSA: ", (end_time - start_peak_find))
