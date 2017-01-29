import struct
import numpy as np
with open("feature.bin", mode='rb') as file: # b is important -> binary
	fileContent = file.read()
	x = struct.unpack("f" * ((len(fileContent)) // 4), fileContent)
	print(len(x))
	x_ = np.asarray(x)
	x_res = np.resize(x_,(25000,4096))
	