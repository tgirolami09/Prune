import ctypes
clib = ctypes.CDLL("./zipmanager.so")
clib.allocate_zips.argtypes = (ctypes.c_int,)
clib.open_zip.argtypes = ctypes.c_char_p, ctypes.c_int
clib.add_data.argtypes = ctypes.c_int, ctypes.c_int, ctypes.c_char_p
clib.read_data.argtypes = ctypes.c_int, ctypes.c_char_p, ctypes.c_int
clib.get_nb_files.argtypes = (ctypes.c_int,)
clib.get_nb_files.restype = ctypes.c_int
allocate = clib.allocate_zips
open_zip = clib.open_zip
add_data = clib.add_data
read_data= clib.read_data
get_entries = clib.get_nb_files
clear_zip = clib.clear_zip