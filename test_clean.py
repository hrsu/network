import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from models import propagation_with_dynamic_blockage as pro

#先调用clean内的kmeans算法划分社区
#再分别使用每个社区调用动态规划算法

'''
划分为5个社区：
22
28
25
18
22
'''

#调用动态规划算法
def getpro(Submatrix):
    G = nx.Graph()
    #edges = np.array([[0, 1, 1, 1, 1, 1, 0, 0],[0, 0, 1, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 1, 0, 0, 0, 1, 1],[0, 0, 0, 0, 0, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 0]])
    for i in range(len(Submatrix)):
        for j in range(len(Submatrix)):
            G.add_edge(i, j)

    # nx.draw(G)
    # plt.show()

    covers = [5, 0, 0, 0, 0, 0, 0, 0]  # 初始节点的评级情况
    lambda1 = 8  # 一般和n同值
    times = 8  # 阻塞结束时刻，必须小于等于lambdal
    nodes = [1, 0, 0, 0, 0, 0, 0, 1]  # 节点的感染情况
    n = 8  # 总的节点个数
    cover = 1  # 一次感染的评级情况
    block_start = 0  # 开始阻塞的时刻
    block_amount = 5  # 贪心算法每次阻塞的节点个数，每次阻塞后阻塞的个数除以2，如果最后为0则一直为1
    block_duration = 100  # 根据此值决定是否运行贪心算法
    # 从block_start-1到times轮，每轮i根据block_duration情况产生结果
    # 当i小于等于block_duration时，每轮运行贪心算法，之后随机影响新的节点
    # 当i大于block_duration时，不运行贪心算法，之后随机影响新的节点

    covers = pro.propagation_with_dynamic_blockage(covers, lambda1, times, edges, nodes, n, cover, block_start,
                                                   block_amount, block_duration)
    print(covers)

#调用clean内的kmeans算法划分社区
from data.clean import *
Division, G1, data1 = Divs(5)
print(Division)
#print(G1._adj)
#print(data1)

#datas = np.array(data1)

getpro(Submatrix=Division[0])

'''
for Sub in Division:
    Submatrix = datas[Sub]  # 先取出想要的行数据
    Submatrix = Submatrix[:, Sub]  # 再取出要求的列数据
    print(len(Submatrix))
    #getpro(Submatrix)
'''

'''
edges = np.array(
[
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0]
])
'''
