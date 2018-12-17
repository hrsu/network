import numpy as np

data = 'bitcoin.txt'
data1 = 'facebook.txt'
data2 = 'wiki.txt'

t = np.loadtxt(data)
t1 = np.loadtxt(data1)
t2 = np.loadtxt(data2)

print(data, ": ", np.max(t) + 1)
print(data1, ": ", np.max(t1) + 1)
print(data2, ": ", np.max(t2) + 1)
