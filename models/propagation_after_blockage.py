#propagation_after_blockage.py：某些节点被阻塞一段时间后的传播

import numpy as np
import random



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
block_nodes：阻塞节点的数组，阻塞为1，非阻塞为0
block_duration：阻塞blocked_nodes的时隙长度
'''
def propagation_after_blockage (covers, times, edges, nodes, n, cover, block_start, block_nodes, block_duration,filename):
    tmpnodes = nodes
    tmpedges = edges
    tmpcover = cover


    # propagation process
    for i in range(block_start-1, times):
        f = open(filename, 'a')
        str = ''
        tmp = np.zeros((n,1),dtype=int)    #创建一个n*1的数组，type为int
        # block period
        if i - block_start <-1000000000:
            for j in range(0, n):
                # judgement for blockage
                if block_nodes[j] == 1:
                    continue

                if tmpnodes[j] == 1:
                    for k in range(0, n):
                        # judgement for blockage
                        if block_nodes[k] == 1:
                            continue

                        if tmpnodes[k] == 0:
                            #取一个0到1的随机数，保留小数位4位
                            dice = round(random.random(), 4)
                            if dice <= tmpedges[j][k]:
                                tmp[k] = 1
                                break
        else:
            # non - blockage period
            for j in range(0, n):
                if tmpnodes[j] == 1:
                    for k in range(0, n):
                        if tmpnodes[k] == 0:
                            # 取一个0到1的随机数，保留小数位4位
                            dice = round(random.random(), 4)
                            if dice <= tmpedges[j][k]:
                                tmp[k] = 1
                                break

        for m in range(0, n):
            if tmp[m] == 1:
                tmpcover = tmpcover + 1
                tmpnodes[m] = 1

        covers[i] = covers[i] + tmpcover

        # decay function
        # math.exp(x) 即e^x
        #tmpedges = tmpedges * math.exp(-lambda1)
        print(covers)

        print('tmpnodes rate:')
        tmonodes_rate=caculate_rate(tmpnodes)
        print(tmonodes_rate)

        print('block_nodes rate:')
        block_nodes_rate=caculate_rate(block_nodes)
        print(block_nodes_rate)
        str = str+'{},{},{}\n'.format(tmonodes_rate,block_nodes_rate,covers[i])
        f.write(str)
    return covers

if __name__ == '__main__':
    covers = [1,2,3,4,5]
    lambda1 = 3
    times = 5
    edges = [[1.1,1.2,1.3,1.4,1.5],[2.1,2.2,2.3,2.4,2.5],[3.1,3.2,3.3,3.4,3.5]]
    nodes = [1,2,3,4,5]
    n = 3
    cover = 2
    block_start = 2
    block_nodes = [1,2,4,5]
    block_duration = 5
    covers = propagation_after_blockage (covers, lambda1, times, edges, nodes, n, cover, block_start, block_nodes, block_duration)
    print(covers)