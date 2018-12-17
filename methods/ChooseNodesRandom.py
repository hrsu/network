'''
随机选择n个节点的基线
'''
import numpy as np
import math
import random
def ChooseNodesRandom(ninit, n):
    '''

    :param ninit:
    :param n: 候选人人数
    :return:
    '''
    nodes = np.zeros([n,1],int)
    tmp = []
    while len(tmp) < ninit:
        dice = math.floor(random.random()*n + 1)
        if np.in1d(dice,tmp)==False:
            tmp.append(dice)

    tmp_max=max(tmp)
    for i in range(0,len(tmp)):
        if tmp_max==tmp[i]:
            nodes[i]=1
    return nodes

print(ChooseNodesRandom(5,5))
