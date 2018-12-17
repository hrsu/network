#propagation.py：基于网络链路权重的自然传播

import numpy as np
import random
import math

'''
covers：受传播影响的节点，受影响的矩阵为1，不受影响为0
lambda1：传播系数乘以链路权重
times：时间间隔
nodes：阻塞节点的数组，阻塞为1，非阻塞为0
n：候选人人数
'''
def propagation(covers, lambda1, times, edges, nodes, n, cover):
    tmpnodes = nodes
    tmpedges = edges
    tmpcover = cover

    #propagation process
    for i in range(1, times):
        tmp = np.zeros([n, 1])
        for j in range(0, n):
            if tmpnodes[j] == 1:
                for k in range(0, n):
                    if tmpnodes[k] == 0:
                        #产生一个0到1的随机数并保留四位小数
                        dice = round(random.random(), 4)
                        if dice <= tmpedges[j][k]:
                            tmp[k] = 1
                            break

        for m in range(0, n):
            if tmp[m] == 1:
                tmpcover = tmpcover + 1
                tmpnodes[m] = 1

        covers[i] = covers[i] + tmpcover

        #math.exp(x) 即e^x
        #tmpedges = tmpedges * math.exp(-lambda1)
        # 衰变功能
        tmpedges = np.dot(tmpedges , math.exp(-lambda1))
    return covers, tmpnodes, tmpedges

if __name__ == '__main__':
    covers = [0,1,0,1,0]
    lambda1 = 3
    times = 3
    edges = [[0, 1, 0, 0, 1],
              [1, 0, 1, 1, 1],
              [0, 1, 0, 1, 0],
              [0, 1, 1, 0, 1],
              [1, 1, 0, 1, 0]]
    nodes = [1,2,3,4,5]
    n = 3
    cover = 3
    covers, tmpnodes, tmpedges = propagation(covers, lambda1, times, edges, nodes, n, cover)
    print(covers)
    print(tmpnodes)
    print(tmpedges)