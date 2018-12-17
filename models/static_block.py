

import numpy as np
import math
from methods import greedy

def caculate_rate(lst):
    count1=0
    count0=0
    if type(lst)==list:
        count0=lst.count(0)
        count1=lst.count(1)
    else:
        count1 = np.sum(lst == 1)
        count0 = np.sum(lst == 0)
    if count1==0:
        return 0
    else:
        return count1/(count1+count0)
'''
covers：受传播影响的节点，受影响的矩阵为1，不受影响为0
lambda1：传播系数乘以链路权重
times：时间间隔
n：候选人人数
block_start：开始阻塞的时刻
nodes：阻塞节点的数组，阻塞为1，非阻塞为0
block_duration：阻塞blocked_nodes的时隙长度
'''
def list2string(lst):
    str=''
    for line in lst:
        line_str=''
        for each in line:
            line_str = line_str+'{},'.format(each)
        str = str + '{}\n'.format(line_str[:-1])
    return str+'\n'

def propagation_with_dynamic_blockage (covers, times, edges, nodes, n, cover, block_start, block_amount,filename):
    result = []
    line_result=[]
    tmpnodes = nodes
    tmpedges = edges
    tmpcover = cover
    block_nodes = np.zeros((n,1),dtype=int)    #创建一个n*1的数组，type为int
    block_amount = math.floor(block_amount / 2)    #向下取整

    #计算相连的没被影响的节点最多的block_amount个节点
    #比如tmp=[2,2,1,3,1]，block_amount=2
    #block_nodes，是根据tmp和block_amount返回一个列表，此处为[1,0,0,1,0]
    #初始化阻塞情况
    block_nodes = greedy.Greedy(block_amount, n, tmpedges, tmpnodes, block_nodes)
    '''
    对于n个节点，一次贪心算法阻塞block_amount个节点
    阻塞顺序：已经被影响的节点>没有被影响的节点但可能影响的范围最大的节点>其他节点
    tmpnodes是已经被影响的节点
    block_nodes是阻塞前节点的阻塞状态
    '''
    divisor = 1
    # propagation process
    #传播过程
    for i in range(block_start-1, times):
        if (i - block_start) % divisor  == 0:
            block_amount = math.floor(block_amount / 2)       #向下取整
            block_nodes = greedy.Greedy(block_amount, n, tmpedges, tmpnodes, block_nodes)
            print('tmpnodes rate:')
            tmpnodes_rate=caculate_rate(tmpnodes)
            print(tmpnodes_rate)
            print('block_nodes rate:')
            block_nodes_rate=caculate_rate(block_nodes.transpose())
            print(block_nodes_rate)
            line_result.append(round(tmpnodes_rate,4))
            line_result.append(round(block_nodes_rate,4))
            line_result.append(covers[i])
            result.append(line_result)
            line_result=[]
        #之后是每一轮之后的感染情况：会根据传播时期情况产生新的感染者
        #在此时期被阻塞就不会被感染

        # decay function
        # math.exp(x) 即e^x
        #tmpedges = tmpedges * math.exp(-lambda1)
        print("covers:")
        print(covers)
        #tmpedges = np.dot(tmpedges , math.exp(-lambda1))
        #print(block_nodes)
    str = list2string(line_result)
    f=open(filename,'a')
    f.write(str+'\n\n\n\nabc666\n')
    return covers


if __name__ == '__main__':
    covers = [0,1,0,0,0]   #受传播影响的节点矩阵
    lambda1 = 3
    times = 3
    edges = np.array([[0, 1, 0, 0, 1],
              [1, 0, 1, 1, 1],
              [0, 1, 0, 1, 0],
              [0, 1, 1, 0, 1],
              [1, 1, 0, 1, 0]])
    nodes = [1,0,0,1,0]    #被影响为1，为被影响为0
    n = 5     #总的节点个数
    cover = 1     #受传播影响的节点个数
    block_start = 2      #开始阻塞的时刻
    block_amount = 3    #每轮阻塞的个数
    block_duration = 1    #产生阻塞的时期，
    covers = propagation_with_dynamic_blockage(covers, times, edges, nodes, n, cover, block_start, block_amount)
    print(covers)

