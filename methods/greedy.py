'''
贪心算法
功能：
对于n个节点，一次贪心算法阻塞nint个节点
阻塞顺序：已经被影响的节点>没有被影响的节点但可能影响的范围最大的节点>其他节点
'''
import numpy as np
def Greedy(nint, n ,edges, alreadyInfectedNodes,blockedNodes):
    '''

    :param nint:
    :param n:
    :param edges:
    :param alreadyInfectedNodes: 已经被传播的节点
    :param blockedNodes: 已经被阻止处理的节点
    :return:
    '''
    tmp = np.zeros([n,1],int)
    nodes = np.zeros([n,1],int)

    #tmp,与当前节点相连的有多少节点没有被影响
    for i in range(n):
        #删除动态方案中所有先前被阻止的节点
        if blockedNodes[i]:
            tmp[i] = -1
            nodes[i] = 1
            continue
        for j in range(n):
            if alreadyInfectedNodes[j] ==0:
                num = int(edges[i][j])
                tmp[i] = tmp[i] + num
    for i in range(nint):
        posi = 0
        maxd = -1
        for j in range(n):
            if tmp[j] > maxd:
                maxd = tmp[j]
                posi = j
        tmp[posi] = -1
        nodes[posi] = 1
    return nodes

if __name__ == '__main__':
    '''
    对于n个节点，一次贪心算法阻塞nint个节点
    阻塞顺序：已经被影响的节点>没有被影响的节点但可能影响的范围最大的节点>其他节点
    '''
    n = 5
    ninit = 1  #阻塞个数
    edges1 = [[0, 1, 0, 0, 1],
              [1, 0, 1, 1, 1],
              [0, 1, 0, 1, 0],
              [0, 1, 1, 0, 1],
              [1, 1, 0, 1, 0]]
    alreadyInfectedNodes = [0, 1, 0, 0, 0]      #已经被影响的节点
    blockedNodes = [0, 0, 0, 0, 0]      #阻塞前节点的阻塞状态
    print(Greedy(ninit, n, edges1, alreadyInfectedNodes, blockedNodes))

