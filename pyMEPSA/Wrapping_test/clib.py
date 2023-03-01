
import ctypes

# A. Create library
C_library = ctypes.CDLL("/mnt/c/Users/Lisa/Desktop/test_wrapping_C/Clib.so")

# B. Specify function signatures
hello_fxn = C_library.say_hello
hello_fxn.argtypes = [ctypes.c_int]

# C. Invoke function
num_repeats = 5
hello_fxn(num_repeats)
