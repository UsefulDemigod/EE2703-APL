import sys
import cmath
import math
import numpy as np

PI = np.pi


try:
    len(sys.argv) == 2 and ".netlist" in sys.argv

except Exception:
    print("Please enter the file")
    exit()
File = sys.argv[1]

f = open(File, "r")
list = []
for line in f.readlines():
    list.append(line)

print(list)
