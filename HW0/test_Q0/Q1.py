import sys
import numpy as np

f = open(sys.argv[1], 'r')
A = []
for line in f.readlines():
    A.append( [ int(x) for x in line.split(',') ] )
f.close()
f = open(sys.argv[2], 'r')
B = []
for line in f.readlines():
    B.append( [ int(x) for x in line.split(',') ] )
f.close()

# A = np.loadtxt(sys.argv[1], delimiter=',', dtype='long')
# B = np.loadtxt(sys.argv[2], delimiter=',', dtype='long')
AB = np.dot(A, B)

AB = AB.ravel()
AB.sort()

f = open("ans_one.txt", "w")
for i in range(AB.size):
	print(AB[i], file = f, end = "\n")
f.close()