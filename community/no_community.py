import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from models import propagation_with_dynamic_blockage as pro
import os
'''
不存在社区划分，存在感染和阻塞
'''
#先调用clean内的kmeans算法划分社区
#再分别使用每个社区调用动态规划算法

'''
划分为5个社区
'''

#调用动态规划算法
def getpro(Submatrix,filename):
    G = nx.Graph()
    #edges = np.array([[0, 1, 1, 1, 1, 1, 0, 0],[0, 0, 1, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 1, 0, 0, 0, 1, 1],[0, 0, 0, 0, 0, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 0]])
    for i in range(len(Submatrix)):
        for j in range(len(Submatrix)):
            G.add_edge(i, j)

    # nx.draw(G)
    # plt.show()

    lens = len(Submatrix)

    covers = [0]*lens  # 初始节点的评级情况
    times = lens  # 阻塞结束时刻，必须小于等于lambdal
    nodes = [0]*(lens-1) # 节点的感染情况
    nodes.append(1)
    n = lens  # 总的节点个数
    cover = 3  # 一次感染的评级情况
    block_start = 0  # 开始阻塞的时刻
    block_amount = lens//2-3  # 贪心算法每次阻塞的节点个数，每次阻塞后阻塞的个数除以2，如果最后为0则一直为1
    block_duration =lens//2+3   #阻塞时期，处于此时期的节点被阻塞后就不会被感染，超过此时期后，被阻塞也可能会被感染

    pro.propagation_with_dynamic_blockage(covers, times, Submatrix, nodes, n, cover, block_start,
                                                   block_amount, block_duration,filename)


def get_dataG(filen, node_num):
    f = open(filen, 'r')
    links = []
    datas = []
    while 1:
        line = f.readline()
        if not line:
            break
        ts = line[:-1].split(' ')
        t = (int(ts[0]), int(ts[1]))
        data = [int(ts[0]), int(ts[1])]
        links.append(t)
        datas.append(data)
    G = nx.Graph(links)

    #node_num = max(max(links)) + 1
    #node_num = 4039
    #bitcoin.txt: 6006
    #facebook.txt: 4039
    #wiki.txt: 8298

    data = np.zeros([node_num, node_num], np.int32)
    # 根据图进行设置邻接矩阵
    for s in range(0, len(datas)):
        #print(s)
        data[datas[s][0], datas[s][1]] = 1
        data[datas[s][1], datas[s][0]] = 1

    return data, G

#调用clean内的kmeans算法划分社区
#from data.clean import *
#Division, G1, data1 = Divs(1)

#调用Louvain和LFM算法划分社区
#更改数据集
node_num = {'facebook.txt': 4039, 'bitcoin.txt': 6006, 'wiki.txt': 8298}
d = 'facebook.txt'
file = os.path.abspath('.') + '\\data\\' + d
data1, G1 = get_dataG(file, node_num[d])
#print(Division)

datas = np.array(data1)
getpro(Submatrix=datas,filename=d)


'''
只需要更改这一行
d = 'facebook.txt'   #更改数据集
'''
