'''
根据传出链接权重的总和选择n个候选
'''
import numpy as np
def ChooseNodesWeightsum(nint, n, edges):
    tmp = np.zeros([n,1],int)
    nodes = np.zeros([n,1],int)
    for i in range(n):
        for j in range(n):
            num=int(edges[i][j])
            tmp[i] = tmp[i] + num

    for i in range(nint):
        posi = 0
        maxd = -1
        for j in range(n):
            if tmp[j]>maxd:
                maxd = tmp[j]
                posi = j
        tmp[posi] = -1
        nodes [posi] = 1
    return nodes
nodes,tmp=ChooseNodesWeightsum(3,3,[[5 ,12, 3] ,[6,27,9] , [4,9,3] ])