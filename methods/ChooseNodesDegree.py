'''
选择n最高度数节点作为候选
'''
import numpy as np
def ChooseNodesDegree(ninit, n, edges):
    '''

    :param ninit:
    :param n:
    :param edges:
    :return:
    '''
    tmp = np.zeros([n,1],int)
    nodes = np.zeros([n,1],int)
    for i in range(0,n):
        for j in range(1,n):
            if edges(i,j)>0:
                tmp [i] = tmp[i]+1
    for i in range(0,ninit):
        posi = 0
        maxd = -1
        for j in range(1,n):
            if tmp(j)> maxd:
                maxd = tmp[j]
                posi = j
        tmp[posi] = -1
        nodes[posi] = 1
    return nodes

