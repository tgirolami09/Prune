from ctypes import *
from math import *
c_int_p = POINTER(c_int)

ZIP_CREATE = 0x0001
ZIP_EXCL = 0x0002
ZIP_FL_OVERWRITE = 0x0004
ZIP_CM_DEFLATE = 8
ZIP_CM_BZIP2 = 12
libzip = CDLL("/usr/lib/x86_64-linux-gnu/libzip.so")

libzip.zip_open.argtypes = (c_char_p, c_int, c_int_p)
libzip.zip_open.restype = c_void_p

libzip.zip_source_buffer.argtypes = c_void_p, c_char_p, c_int, c_int
libzip.zip_source_buffer.restype = c_void_p

libzip.zip_file_add.argtypes = c_void_p, c_char_p, c_void_p, c_int

libzip.zip_source_free.argtypes = (c_void_p,)

libzip.zip_get_num_entries.restype = c_int
libzip.zip_get_num_entries.argtypes = (c_void_p, c_int)

libzip.zip_close.argtypes = (c_void_p,)

libzip.zip_set_file_compression.argtypes = (c_void_p, c_int, c_int, c_int)

libzip.zip_name_locate.argtypes = (c_void_p, c_char_p, c_int)
libzip.zip_name_locate.restype = c_int

libzip.zip_fopen_index.argtypes = (c_void_p, c_int, c_int)
libzip.zip_fopen_index.restype = c_void_p

libzip.zip_fopen.argtypes = (c_void_p, c_char_p, c_int)
libzip.zip_fopen.restype = c_void_p

libzip.zip_fread.argtypes = (c_void_p, c_char_p, c_int)
libzip.zip_fread.restype = c_int

libzip.zip_fclose.argtypes = (c_void_p,)

for name in dir(libzip):
    if not name.startswith("_"):
        globals()[name] = getattr(libzip, name)

def id_to_name(id):
    idS = id
    span = ord('z')-ord('a')+ord('Z')-ord('A')+2
    b = []
    while id:
        b.append(id%span)
        if(b[-1] >= 26):
            b[-1] += ord('a')-26
        else:
            b[-1] += ord('A')
        id //= span
    b = bytes(b)
    #print(idS, b.decode(), end=" ")
    return bytes(b)

error = c_int()
if __name__ == "__main__":
    z = zip_open(b"archive.zip", ZIP_CREATE, byref(error))
    source = zip_source_buffer(z, b"bonjour", 7, 0)
    zip_file_add(z, b"n.txt", source, ZIP_FL_OVERWRITE)
    zip_close(z)