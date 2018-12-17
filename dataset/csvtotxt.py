import csv
import numpy as np

file = 'bitcoin.csv'

f = open(file, 'r')
line = f.readline()
txt_ls = []
while line:
    txt_l = []
    ls = line.split(',')
    txt_l.append(ls[0])
    txt_l.append(ls[1])
    txt_ls.append(txt_l)
    line = f.readline()
f.close()

f1 = open('bitcoin.txt', 'w')
for k in txt_ls:
    s = ''
    s += k[0] + ' ' + k[1] + '\n'
    f1.write(s)

