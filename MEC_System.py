#!/usr/bin/env python
# coding=utf-8
"""
Python(3.9) Pytorch(2.0.0+cu118) CUDA(11.8)
唐宁昆(广西大学 605282008@qq.com)于2022.12.27创建。
此文件包含MEC系统中的移动设备、边缘服务器、云服务器，以及它们的基本功能，同时规定了此环境下通信、计算、卸载的规则。
与GYM等环境不同的是，本系统理论上不采取时隙机制，因此没有step函数，一个任务从开始到结束为一步。需要引用DRL算法文件并在此MEC系统的主函数中训练。
"""
import copy  # 深复制
import csv  # 用于任务数据保存至CSV文件
import math
import os  # 文件路径相关操作
import random  # 随机数
import matplotlib.pyplot as plt  # 绘图
import networkx as nx  # 图数据结构
import numpy as np  # 矩阵处理
import torch  # 深度学习
from collections import defaultdict  # 字典

import MAC_A2C as a2c  # A2C算法和用到的神经网络
import data_processing as dp  # 用户数据处理

#####################################
#                                   #
#              系统参数               #
#                                   #
#####################################
show_task_log = False  # 是否显示任务去向日志，Debug时启用
# collect_state_data = False  # 收集一次训练的状态，取其均值和标准差，用于后面训练的标准化。若为True则这次训练是收集数据而用
device = torch.device('cpu')  # 训练设备指定为'cuda' 或 'cpu'
a2c.device = device
# torch.autograd.set_detect_anomaly(True)  # Pytorch无法反向传播时启用（会降低训练速度）
time_step = 10  # 时间步长度（ms）
run_time = 1000 * 60 * 3  # 每回合运行总时长（ms）
episode_num = 1000  # 训练需要的回合数
noise = 1.5 * 10 ** (-8)  # 噪声功率（W）(车联网边缘计算环境下基于深度强化学习的分布式服务卸载方法)
C_fiber = 1000  # 节点或云服务器用光纤通信的发送速率（Mbit/s）（此速度并非取决于光纤好坏，而在于设备处理和发送的快慢）
lr_policy = 1e-4  # 策略网络学习率
lr_value = 1e-4  # 价值网络学习率
reward_scale = 2  # 奖励系数（简单地放大缩小奖励，让一个回合内奖励小于1k大于500左右）

user_num = 50  # 用户数量
avg_data_size = 5  # 平均任务大小（Mbit）
sigma1 = 100  # 任务大小的随机散布程度
avg_data_size_out = 1  # 平均计算结果大小（Mbit）
sigma2 = 50  # 计算结果大小的随机散布程度
avg_rho = 0.297  # 平均计算密度（1G(10^9)周期/Mbit）(Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing Systems)
sigma3 = 0.1  # 计算密度的随机散布程度
task_prob = 0.001  # 每时间步产生任务的概率
report_period = 5000  # 用户上报位置信息的周期（ms）

node_xy_list = [(0, 0), (75, 129.9), (150, 0), (75, -129.9), (225, 129.9), (300, 0), (225, -129.9), (375, 129.9),
                (450, 0), (375, -129.9)]  # 节点坐标(节点相邻距离150m)
edge_list = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 3), (3, 5), (3, 6), (3, 7), (4, 3), (4, 7), (5, 8), (5, 6), (6, 8),
             (6, 9), (6, 10), (7, 6), (7, 10), (8, 9), (9, 10)]  # 节点互相连通性(默认全连接)
user_xy_size = 10  # 在用户位置信息记录user_xy中，每个用户记录user_xy_size个位置历史信息
r = 100  # 基站服务范围半径（m）（其实基站覆盖半径很大，100米偏小了）
node_num = 10  # 边缘节点数量
core_num = 14  # 节点CPU核心数，一个核心对应一个计算队列。(Intel Xeon E5-2680 v4)
f_node = 2.4  # 节点CPU单核频率（GHz） (Intel Xeon E5-2680 v4)
R_node = 32 * 1024 * 8  # 节点内存容量（Mbit）
P_node = 200  # 节点的无线发射功率（W）（蒙特卡洛方法用于城镇5G基站电磁暴露安全评估）
B = 20  # 节点无线带宽（MHz）(Dynamic Radio Resource Slicing for a Two-Tier Heterogeneous Wireless Network)
SYN_period = 10000  # 节点同步周期（ms）
node_dis = 150  # 相邻节点之间的距离（m）
decision_time = 10  # 决策时延(ms)
decision_penalty = 1  # 决策约束模块为不合理决策做出的惩罚
k_node = 10 ** (-28)  # 节点或云端的能耗效率系数(Multiuser Resource Control With Deep Reinforcement Learning in IoT Edge Computing)
disconnect_prob = 1e-3  # 每条光纤断连概率（每分钟随机数判断一次）
disconnect_period = 60 * 1000  # 光纤断连判定周期（ms）
repair_t_start = 10 * 1000  # 最短光纤维修时间(ms)
repair_t_end = 60 * 1000  # 最长光纤维修时间(ms)

cloud_dis = 20 * 1000  # 云服务器的地理距离（m）
f_cloud = 10  # 云服务器CPU频率（GHz）
avg_retran_delay = 50  # 平均每Mbit的核心网转发时延（ms）
sigma4 = 15  # 转发时延散布程度


#####################################
#                                   #
#              函数工具               #
#                                   #
#####################################
def distance(device1, device2):
    """计算MEC系统中两个设备的距离（m）"""
    return ((device1.x - device2.x) ** 2 + (device1.y - device2.y) ** 2) ** 0.5


def wireless_tran_time(user, node, data_size, direction):
    """无线通信时延（发送+传播时延）"""
    if len(node.linking_users) != 0:  # 如果节点服务区里有用户
        B_alloc = B / len(node.linking_users)  # 带宽平均分配（MHz）
    else:
        B_alloc = B  # 如果之前没有用户，则新增用户把带宽占满
    # short_term_fading = abs(random.gauss(0, 1))  # 短期衰落，来自信道增益公式
    short_term_fading = 1  # 若按正态分布，会导致一些任务的无线传输速度奇慢，传个几十秒都传不完，这非常不合理
    dis = distance(user, node)  # 通信距离（m）
    if dis == 0:  # 防止距离为0而报错
        dis = 1e-03
    G = 1e-03 * dis ** (-1) * short_term_fading  # 信道增益
    # 增益公式来自Energy Efficient Relay Selection and Resource Allocation in D2D-Enabled Mobile Edge Computing
    # I = 0  # 其它发射源的干扰噪声功率（W）（由于多用户时的上传速率实在太低，考虑到频分复用可以降低甚至完全消除其他用户干扰，这里忽略了I）
    C = 0  # 用户传输速率（Mbit/s）
    if direction == 'up':
        # for u in node.linking_users:  # 遍历所有连接的用户，求他们上传产生的干扰
        #     # 干扰公式来自Dynamic Radio Resource Slicing for a Two-Tier Heterogeneous Wireless Network
        #     if (u != user) and (len(u.queue_tran) != 0):  # 其他用户如果也在传输则产生干扰
        #         G_u = 1e-03 * (distance(u, node) ** (-1)) * short_term_fading
        #         I += u.P_up * G_u
        # C = B_alloc * math.log2(1 + (user.P_up * G) / (noise + I))  # 用户传输速率（Mbit/s）
        C = B_alloc * math.log2(1 + (user.P_up * G) / noise)  # 用户传输速率（Mbit/s）
    elif direction == 'down':
        # for n in user.linking_nodes:  # 遍历所有连接的节点，求他们下载产生的干扰
        #     # 干扰公式来自Dynamic Radio Resource Slicing for a Two-Tier Heterogeneous Wireless Network
        #     if (n != node) and (len(n.queue_wireless_tran) != 0):  # 其他节点如果也在传输则产生干扰
        #         G_n = 1e-03 * (distance(n, user) ** (-1)) * short_term_fading
        #         I += n.P_down * G_n
        # C = B_alloc * math.log2(1 + (node.P_down * G) / (noise + I))  # 用户传输速率（Mbit/s）
        C = B_alloc * math.log2(1 + (node.P_down * G) / noise)  # 用户传输速率（Mbit/s）
    tran_time = (data_size / C + distance(user, node) / (3 * 10 ** 8)) * 1000  # 发送+传播时延(ms)
    # 电磁波在空气为光速，此数据来自谢希仁《计算机网络》
    return round(tran_time, 2)  # 两位小数


def fiber_tran_time(distance, data_size):
    """有线通信时延（发送+传播时延）"""
    tran_time = (data_size / C_fiber + distance / (2 * 10 ** 8)) * 1000  # 发送+传播时延(ms)
    # 光纤中的传播速度为2*10^8m/s，此数据来自谢希仁《计算机网络》
    return round(tran_time, 2)  # 两位小数


def sys_init():
    """MEC系统初始化"""
    random.seed(0)  # 让环境稳定的秘诀

    # 创建节点网络与云端
    global node_graph, cloud
    node_graph = Node_Graph(node_xy_list, edge_list)  # 创建节点网络图
    cloud = Cloud()  # 创建云端
    if show_task_log:
        nx.draw(node_graph.graph_cloud, with_labels=True, font_weight='bold')  # 云与边缘节点的网络图绘图设定
        plt.show()  # 显示网络图
        nx.draw(node_graph.graph, with_labels=True, font_weight='bold')  # 节点间的网络图绘图设定
        plt.show()  # 显示网络图
    node_graph.node_list[2].if_critic = True  # 默认评论家节点为3号
    node_graph.node_list[0].nodes_SYN()  # 开始节点之间周期性同步
    node_graph.fiber_disconnect()  # 开始每分钟断光纤的可能

    # 创建用户
    for i in range(user_num):
        user_list.append(User(i + 1, random.randint(-100, 550), random.randint(-230, 230)))


def sys_reset():
    """MEC系统运行一个回合后重置"""
    # 清理对象
    drop_list.clear()
    complete_list.clear()
    user_list.clear()
    node_graph.node_list.clear()
    node_graph.graph.clear_edges()  # 清理节点网络所有边
    node_graph.graph_cloud.clear_edges()  # 清理云边网络所有边
    to_do_list.clear()
    # 系统重新初始化
    global t
    t = 0
    sys_init()


# def load_state_data():
#     """读取状态信息数据，求均值和标准差"""
#     txt = open("state_data.txt").read()
#     state_list = txt.split('\n')  # 按换行符分割文本
#
#     state_dim = len(eval(state_list[0]))  # 状态维数
#     state_data_sum = []  # 按状态的维数存状态数据，而非原来一条一条地存
#     for dim in range(state_dim):
#         state_data_sum.append([])  # 每个维度新建一个列表
#         state_mean.append([])
#         state_std.append([])
#
#     for i in range(len(state_list)):
#         state_list[i] = eval(state_list[i])  # 每条状态信息去掉引号
#         for j in range(state_dim):
#             state_data_sum[j].append(state_list[i][j])  # 每条状态按每个维度存数据
#
#     for dim in range(state_dim):
#         a = np.array(state_data_sum[dim])
#         state_mean[dim] = np.mean(a)  # 利用numpy的mean函数求均值
#         state_std[dim] = np.std(state_data_sum[dim])  # 标准差


def save_result(eval_task_log):
    """保存测试数据，用于绘图
    eval_task_log结构为：列表[任务1[类型，时延，能耗，是否完成，开销（奖励换算）]，任务2[...]...]
    """
    # 表头
    header = ['High priority', 'Critical', 'Normal', 'Low priority', 'Data size', 'Delay', 'Energy', 'If Completed',
              'Reward']
    with open('output/result_data.csv', 'w', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)  # 创建对象
        writer.writerow(header)  # 写表头
        writer.writerows(eval_task_log)  # 写入数据(一次性写入多行)


#####################################
#                                   #
#           系统各类元素定义           #
#                                   #
#####################################
class Task:
    """计算任务"""

    def __init__(self):
        self.user = None  # 是哪个用户产生的
        self.t_created = t  # 产生时刻
        self.t_end_of_queue = None  # 任务在队列中排完队的那一刻（排完队就赋值，计算或传输完毕后重置为None）
        self.delay = 0  # 处理过程耗时（ms）
        self.energy = 0  # 处理过程能耗（J）
        self.D = 0  # 卸载决策变量

        self.completed = False  # 表示是否已完成计算（不代表已经成功回传）
        self.dropped = False  # 表示任务是否已被抛弃
        self.decider = None  # 给出卸载决策的节点
        self.returner = None  # 返回计算结果时离用户最近的节点
        self.receiver = None  # 任务从云端发回边缘层时，由接收者节点接收此任务
        self.tran_time = None  # 任务传输所需时间。因为每次求得传输时间都会不同，因此需要记录而不能重复计算。传输完毕后重置为None
        self.shortest_path = []  # 任务在节点网络里传输时，此变量暂存最短传输路径，避免重复计算

        self.fst_state = None  # 记录任务决策时节点观测到的状态
        self.fst_xy_list = None  # 记录任务决策时节点记录用户的坐标历史
        self.logp_action = None  # 记录此任务的动作概率的log值
        # （每个动作对应一个log值，不能每个智能体对应一个log值，否则将报重复backward的错）
        self.penalty = 0  # 决策约束模块为不合理决策做出的惩罚
        self.reward = 0  # 奖励值

        self.data_size = random.gauss(avg_data_size, sigma1)  # 数据大小（Mbit），根据正态分布随机抽样
        while self.data_size <= 0:  # 数据大小不可能是0或负数，如果抽到就重新抽一次直到合格
            self.data_size = random.gauss(avg_data_size, sigma1)

        self.data_size_out = random.gauss(avg_data_size_out, sigma2)  # 计算结果大小（Mbit）
        while self.data_size_out <= 0:  # 数据大小不可能是0或负数，如果抽到就重新抽一次直到合格
            self.data_size_out = random.gauss(avg_data_size_out, sigma2)

        self.rho = random.gauss(avg_rho, sigma3)  # 计算密度（1Mbit需要算多少个G的CPU周期）（1G（10^9）周期/Mbit）
        while self.rho <= 0:  # 计算密度不可能是0或负数，如果抽到就重新抽一次直到合格
            self.rho = random.gauss(avg_rho, sigma3)

        self.ddl = 1000 * self.rho * self.data_size / 2 + 30000  # 能容忍的时延上限（ms）
        # 任务结果从云端下传时所需的核心网转发时延
        self.cloud_retran_delay = self.data_size * abs(random.gauss(avg_retran_delay, sigma4))

        # 任务类型，抛骰子决定
        dice = random.random()
        if dice < 0.2:
            self.type = [1, 0, 0, 0]  # 高优先级任务
        if 0.2 <= dice < 0.4:
            self.type = [0, 1, 0, 0]  # 关键任务
        if 0.4 <= dice < 0.8:
            self.type = [0, 0, 1, 0]  # 常规任务
        if 0.8 <= dice:
            self.type = [0, 0, 0, 1]  # 低优先级任务

        self.k = 0  # 卸载决策变量。D[k]=1说明此任务应该交给设备k执行。
        self.D = []  # D[0]=1表示本地计算，D[node_num+1]=1表示云计算，D[node_num+2]=1表示丢弃任务。
        for i in range(node_num + 3):  # 3种特殊情况，需要额外3个格子
            self.D.append(0)

    def compute_reward(self):
        """计算此任务的奖励值"""
        # 设定参数
        if self.type == [1, 0, 0, 0]:  # 高优先级任务
            phi1 = phi2 = phi4 = 1 / 3
            phi3 = 0
        elif self.type == [0, 1, 0, 0]:  # 关键任务
            phi2 = phi4 = 1 / 2
            phi1 = phi3 = 0
        elif self.type == [0, 0, 0, 1]:  # 低优先级任务
            phi2 = phi3 = phi4 = 1 / 3
            phi1 = 0
        else:  # 常规任务
            phi1 = phi2 = phi3 = phi4 = 1 / 4

        # self.reward = (-1) * (phi1 * self.delay / self.ddl + phi2 * int(self.dropped) +
        #                       phi3 * self.energy / (mu1 * self.rho * self.data_size) + phi4 * self.penalty)
        # 奖励中的能耗部分。如果丢包，能耗为0是无意义的，因此丢包的任务这一项奖励为0
        energy_reward = np.array([(0.5 - 30 * self.energy / (self.rho * self.data_size)) * (1 - int(self.dropped))])
        energy_reward = np.clip(energy_reward, -0.5, 0.5)  # 限定范围，避免值过大
        energy_reward = energy_reward[0]
        self.reward = reward_scale * (phi1 * (0.5 - self.delay / self.ddl) +
                                      phi2 * (0.5 - int(self.dropped)) +
                                      phi3 * 0.1 * energy_reward +
                                      # 0.1是权重，因为能耗不是那么重要
                                      phi4 * (0.5 - self.penalty))
        # if abs(self.reward) > 5:
        #     print("奖励绝对值过大")
        global Return
        Return += self.reward  # 记录单回合的回报


class User:
    """移动设备用户"""

    def __init__(self, ID, x, y):
        self.ID = ID
        self.x, self.y = x, y  # 位置坐标
        self.f = 2  # CPU频率（GHz）
        self.R_user = 6 * 1024 * 8  # 缓存容量（Mbit）（缓存区与两个队列共用）
        self.P_up = 0.2  # 发射功率（W）
        self.k_user = random.uniform(4.13, 66.16) * 10 ** (-27)  # 能耗效率系数（基于深度强化学习的多用户边缘计算任务卸载调度与资源分配算法 邝祝芳）

        self.linking_nodes = []  # 正在连接的节点列表
        self.linking_nodes_id = set()
        self.dis_list = []  # 对应记录linking_nodes中每个节点的距离
        self.cache = []  # 缓存区
        self.queue_comp = []  # 计算队列
        self.queue_tran = []  # 传输队列
        self.q_cache = 0  # 缓存区已用大小（Mbit）
        self.q_comp = 0  # 计算队列已用大小（Mbit）
        self.q_tran = 0  # 传输队列已用大小（Mbit）
        self.xy_data = []  # 记录用户坐标数据，系统查询此数据计算出用户当前坐标（数据不可能覆盖每个时隙，中间用线性填补）
        self.data_p = 0  # 上面坐标数据的指针，指向下一个要使用的坐标值
        # 结构：[第一条数据[y,x,t,f],第二条[...],...]
        self.confirm_nodes_connect()  # 更新一下linking_nodes（每次移动后调用）
        self.report_in_cycle_caller()  # 开始周期性上报

    def confirm_nodes_connect(self):
        """根据距离确认连接节点列表，以及求出最近节点"""
        ### 每次移动后调用 ###
        linking_nodes = []  # 临时的连接节点列表
        dis_list = []  # 临时的节点距离列表
        for node in node_graph.node_list:
            dis = distance(self, node)  # 当前用户与node的距离
            if dis <= r:  # 若处于节点服务范围内，记录此节点
                linking_nodes.append(node)
                dis_list.append(dis)

        # 为linking_nodes按距离排序（简单选择排序）。排序后最近节点即为列表第一个元素
        for i in range(len(dis_list)):
            min = i
            for j in range(i + 1, len(dis_list)):
                if dis_list[j] < dis_list[min]:  # 如果第j元素打破最小记录，元素j就是新的最小记录
                    min = j
            if min != i:  # 找到最小元素min后，与元素i交换
                d = dis_list.pop(min)
                n = linking_nodes.pop(min)
                dis_list.insert(min, dis_list[i])
                linking_nodes.insert(min, linking_nodes[i])
                dis_list.pop(i)
                linking_nodes.pop(i)
                dis_list.insert(i, d)
                linking_nodes.insert(i, n)
        # linking_nodes.sort(key=lambda n:distance(self,n))  # 若不自己排序也可用sort方法排序，但会重复计算距离浪费算力

        self.linking_nodes = linking_nodes  # 更新连接节点列表
        self.dis_list = dis_list  # 更新节点距离列表

        # 记录节点ID
        self.linking_nodes_id = set()
        for node in self.linking_nodes:
            self.linking_nodes_id.add(node.ID)

    def report_in_cycle(self, node):
        """周期性向连接的节点上报位置信息"""
        node.users_report_time[self.ID] = t  # 记录上报时刻

        # user_xy的记录
        if not (self.ID in node.user_xy):  # 如果历史记录没有此用户记录，就新建
            node.user_xy[self.ID] = [[self.x, self.y]]
        else:  # 如果已经有此用户记录，当前位置l_user存入首位
            node.user_xy[self.ID].insert(0, [self.x, self.y])
            if len(node.user_xy[self.ID]) > user_xy_size:  # 如果历史记录超出大小限制，删除其最旧的记录。
                node.user_xy[self.ID].pop(-1)

        # user_xy_sum的记录
        node.user_xy_sum[self.ID] = (self.x, self.y)

        # linking_users的记录
        if not (self in node.linking_users):  # 如果节点之前没有连通此用户记录，则记录此用户是连通的
            node.linking_users.append(self)

        # linking_users_sum的记录。字典{节点ID:集合{用户1ID,...},...}
        node.linking_users_sum[node.ID].add(self.ID)

    def report_in_cycle_caller(self):
        """这个函数仅用来调用report_in_cycle，为了解决周期和时延问题"""
        # if show_task_log:
        # print("时刻%d \t 用户%d周期性上报1轮" % (t, self.ID))
        for node in self.linking_nodes:
            tran_delay = wireless_tran_time(self, node, (146 * 8 / 1e6), direction='up')  # 发送+传播时延(ms)
            if tran_delay < 1:
                tran_delay = 1  # 避免延迟过低导致todolist直接跳过此代码而不执行
            to_do_list.append((t + tran_delay, "user_list[%d].report_in_cycle(node_graph.node_list[%d])"
                               % (self.ID - 1, node.ID - 1)))  # 在一定时延后，执行上报函数
        # 上报周期后，再执行一次此函数
        to_do_list.append((t + report_period, "user_list[%d].report_in_cycle_caller()" % (self.ID - 1)))

    def task_generator(self):
        """生成任务"""
        ### 每个周期调用 ###
        if random.random() < task_prob:  # 根据概率进行随机任务生成
            if show_task_log:
                print("时刻%d \t 用户%d产生任务[%d]" % (t, self.ID, t))
            task = Task()
            task.user = self
            if task.data_size + self.q_cache + self.q_comp + self.q_tran <= self.R_user:  # 如果存储容量够
                self.cache.append(task)  # 任务暂时存入缓存区
                self.q_cache += task.data_size  # 占用存储空间
                self.report_task(task)  # 上报任务初步信息
            else:
                self.drop(task, if_report=False)  # 存储不下则抛弃任务
                if show_task_log:
                    print("【丢包原因】产生任务时用户%d存储空间不足" % self.ID)
            return task

    def report_task(self, task):
        """向最近节点上报任务初步信息"""
        ### 上传初步信息+决策时延 ###
        if len(self.linking_nodes) > 0:  # 如果用户在服务区内，才上报任务
            task.decider = self.linking_nodes[0]  # 记录最近的节点为决策者节点
            # 接收者节点，先假定云端发回的任务由决策者节点接收。但由于移动性和断联的特殊情况，接收者不一定是决策者。
            task.receiver = task.decider
            delay = wireless_tran_time(self, task.decider, (146 * 8 / 1e6), direction='up') + decision_time
            task.energy += (delay - decision_time) / 1000 * self.P_up  # 发送任务初步信息的能耗（J）
            # 在一定时延后，决策者节点收集信息，为缓存区的队首任务做出决策（其实缓存区也是FIFO的队列）
            if delay < 1:
                delay = 1  # 避免延迟过低导致todolist直接跳过此代码而不执行
            to_do_list.append((t + delay, "node_graph.node_list[%d].decide(user_list[%d].cache[0])"
                               % (task.decider.ID - 1, self.ID - 1)))
        else:  # 如果用户不在任何服务区内，直接考虑本地计算
            self.put_in_queue_comp(task)

    def drop(self, task, if_report):
        """任务抛弃。将任务移出队列的工作并不由此函数负责"""
        if show_task_log:
            print("时刻%d \t 用户%d丢弃任务[%d]" % (t, self.ID, task.t_created))
        task.delay = t - task.t_created  # 记录处理耗时
        task.energy = 0  # 丢包则能耗无意义。因为计算公式是针对计算完毕的情况的，计算到一半的任务的能耗是无法求得的
        task.dropped = True
        task.compute_reward()  # 计算奖励
        drop_list.append(task)  # 记录丢弃任务
        if if_eval:  # 如果是测试阶段，记录每个任务的完成信息用于绘图
            eval_task_log.append(
                task.type + [task.data_size, task.delay / 1000, 0, 0, task.reward])  # 丢弃任务的能耗记为0
        if if_report and task.decider is not None:  # 若任务还没让节点知道就被抛弃，则不需要上报抛弃信息。否则需要上报并让节点更新参数
            # 若没有决策者也不需要上报
            task.decider.put_in_update_cache(task)  # 节点根据算法更新参数。未考虑丢弃信息上报的时延

    def put_in_queue_comp(self, task):
        """进入用户计算队列"""
        self.cache.remove(task)  # 任务退出缓存区
        self.q_cache -= task.data_size  # 释放存储空间
        if t - task.t_created > task.ddl:
            self.drop(task, if_report=True)  # 如果已经超时则丢弃
            if show_task_log:
                print("【丢包原因】任务[%d]在进入用户%d计算队列时已经超时" % (task.t_created, self.ID))
        else:
            self.queue_comp.append(task)  # 压入计算队列。这里不用检查存储容量，因为是内部文件移动而非新产生的文件
            self.q_comp += task.data_size  # 占用存储空间
            if show_task_log:
                print("时刻%d \t 任务[%d]进入用户%d的计算队列" % (t, task.t_created, self.ID))

    def compute(self):
        """用户单核处理计算队列"""
        ### 每个时间步执行 ###
        # 监测队列中是否有超时任务
        for task_i in self.queue_comp:
            if t - task_i.t_created > task_i.ddl:
                self.drop(task_i, if_report=True)  # 如果已经超时则丢弃
                if show_task_log:
                    print("【丢包原因】任务[%d]在用户%d计算队列内被发现超时" % (task_i.t_created, self.ID))
                self.queue_comp.remove(task_i)  # 退出计算队列
                self.q_comp -= task_i.data_size  # 释放存储空间

        # 判断队首任务是否已经完成计算
        if len(self.queue_comp) > 0:  # 如果计算队列为空则不需要以下处理
            task = self.queue_comp[0]  # 取队首任务
            if task.t_end_of_queue is None:
                task.t_end_of_queue = t  # 记录任务排完队的时刻
                if show_task_log:
                    print("时刻%d \t 任务[%d]在用户%d的计算队列内开始计算" % (t, task.t_created, self.ID))
            comp_time = (task.data_size * task.rho) / self.f * 1000  # 计算时延（ms）
            if abs(t - (task.t_end_of_queue + comp_time)) < time_step:  # 如果完成时间与当前时间相差小于时间步则视为计算完毕
                task.completed = True
                task.energy += self.k_user * task.data_size * task.rho * (self.f * 10 ** 9) ** 2  # 计算能耗（J）
                # (中间两个变量的单位相互约掉了，所以不用换算成标准单位)（此公式导致无法求得计算一半而丢弃的任务能耗）
                self.queue_comp.remove(task)  # 退出计算队列
                self.q_comp -= task.data_size  # 释放存储空间
                task.delay = t - task.t_created  # 记录处理耗时
                task.t_end_of_queue = None  # 重置
                task.compute_reward()  # 计算奖励
                complete_list.append(task)  # 完成记录
                if if_eval:  # 如果是测试阶段，记录每个任务的完成信息用于绘图
                    eval_task_log.append(
                        task.type + [task.data_size, task.delay / 1000, task.energy, 1, task.reward])
                if show_task_log:
                    print("时刻%d \t 用户%d完成任务[%d]" % (t, self.ID, task.t_created))
                if task.decider is not None:  # 如果任务没有经过节点决策，则不需要下边的更新
                    # 向决策者发送收到结果确认的时延
                    delay = wireless_tran_time(self, task.decider, (146 * 8 / 1e6), direction='up')
                    task.energy += delay / 1000 * self.P_up  # 上报能耗
                    task.decider.put_in_update_cache(task)  # 节点根据RL算法更新参数。理论上上报有时延，这里未考虑

    def put_in_queue_tran(self, task):
        """进入用户传输队列"""
        self.cache.remove(task)  # 任务退出缓存区
        self.q_cache -= task.data_size  # 释放存储空间
        if t - task.t_created > task.ddl:
            self.drop(task, if_report=True)  # 如果已经超时则丢弃
            if show_task_log:
                print("【丢包原因】任务[%d]在进入用户%d传输队列时已经超时" % (task.t_created, self.ID))
        else:  # 没超时就进入传输队列。 为什么不检查连通性？因为进入队列不代表开始传输，传输建立的时候才需要连通性
            self.queue_tran.append(task)  # 压入传输队列。这里不用检查存储容量，因为是内部文件移动而非新产生的文件
            self.q_tran += task.data_size  # 占用存储空间
            if show_task_log:
                print("时刻%d \t 任务[%d]进入用户%d的传输队列" % (t, task.t_created, self.ID))

    def transmit(self):
        """用户处理传输队列"""
        ### 每个时间步执行 ###
        # 监测队列中是否有超时任务
        for task_i in self.queue_tran:
            if t - task_i.t_created > task_i.ddl:
                self.drop(task_i, if_report=True)  # 如果已经超时则丢弃
                if show_task_log:
                    print("【丢包原因】任务[%d]在用户%d传输队列内被发现超时" % (task_i.t_created, self.ID))
                self.queue_tran.remove(task_i)  # 退出计算队列
                self.q_tran -= task_i.data_size  # 释放存储空间

        # 判断队首任务是否已经完成传输
        if len(self.queue_tran) > 0:  # 如果传输队列为空则不需要以下处理
            task = self.queue_tran[0]  # 取队首任务
            if (task.decider in task.user.linking_nodes) and \
                    (task.data_size + task.decider.q_comp + task.decider.q_tran <= task.decider.R_node):
                # 每个时隙监测用户是否离开服务区，同时监测节点存储空间是否够
                if task.t_end_of_queue is None:
                    task.t_end_of_queue = t  # 记录任务排完队的时刻
                    if show_task_log:
                        print("时刻%d \t 任务[%d]在用户%d的传输队列内开始传输" % (t, task.t_created, self.ID))
                # 求用户发送给决策者的 发送+传播时延（ms）
                if task.tran_time is None:
                    task.tran_time = wireless_tran_time(self, task.decider, task.data_size, direction='up')
                if abs(t - (task.t_end_of_queue + task.tran_time)) < time_step:
                    # 如果完成时间与当前时间相差小于时间步则视为传输完毕
                    task.energy += task.tran_time / 1000 * self.P_up  # 上传能耗(J)
                    self.queue_tran.remove(task)  # 退出传输队列
                    self.q_tran -= task.data_size  # 释放存储空间
                    task.t_end_of_queue, task.tran_time = None, None  # 重置
                    if show_task_log:
                        print("时刻%d \t 任务[%d]由用户%d发往节点%d完毕" % (t, task.t_created, self.ID, task.decider.ID))
                    if task.k == task.decider.ID:  # 传输完毕后，如果决策者决定自己计算
                        task.decider.put_in_queue_comp(task)  # 放入决策者计算队列
                    else:  # 如果决策者决定给别的节点或云计算
                        task.decider.put_in_queue_fiber_tran(task)  # 放入决策者传输队列
            else:  # 如果发送过程中用户离开服务区，或是节点空间不够，则考虑本地计算
                self.queue_tran.remove(task)  # 退出传输队列
                self.q_tran -= task.data_size  # 释放存储空间
                task.t_end_of_queue = None  # 重置
                self.cache.append(task)  # 退回缓存区
                self.q_cache += task.data_size  # 占用存储空间
                self.put_in_queue_comp(task)  # 压入计算队列

    def move(self, direction, dis):
        """用户移动位置（测试用）"""
        if direction == 0:  # 上
            self.y += dis
        elif direction == 1:  # 下
            self.y -= dis
        elif direction == 2:  # 左
            self.x -= dis
        elif direction == 4:  # 右
            self.x += dis

    def move_from_data(self):
        """从坐标数据计算得出用户当前应该在的位置"""
        t_p = self.xy_data[self.data_p][2]  # 指针指向数据的时间
        t_p_1 = self.xy_data[self.data_p - 1][2]  # 指针前一个数据的时间
        if t_p == t:  # 如果当前时间等于指针指向的数据时间，指针往后一格
            self.data_p += 1
            t_p = self.xy_data[self.data_p][2]
        if t_p == t + time_step:  # 如果下一步的时间刚好等于指针指向数据记录时间，则让用户移动到此坐标，指针后移
            self.x = self.xy_data[self.data_p][1]
            self.y = self.xy_data[self.data_p][0]
            self.data_p += 1
        elif t_p != t + time_step:  # 如果下一步时间不等于指针数据时间，
            while t_p < t + time_step:  # 如果t + time_step比p所指时间还大，说明所求时间加在p到p+1的数据之间，p应该后移了
                self.data_p += 1
                t_p = self.xy_data[self.data_p][2]
            # 经过上面指针后移，t + time_step应小于或等于p所指时间
            if t_p == t + time_step:  # 如果t + time_step刚好等于p所指时间，则让用户移动到此坐标，指针后移
                self.x = self.xy_data[self.data_p][1]
                self.y = self.xy_data[self.data_p][0]
                self.data_p += 1
                return  # 到此已不需要后续操作
            # 如果t + time_step小于p所指时间，说明所求的时间夹在p-1到p的数据之间（这应该是最多数的情况）
            x_p = self.xy_data[self.data_p][1]  # 指针指向的坐标
            y_p = self.xy_data[self.data_p][0]
            x_p_1 = self.xy_data[self.data_p - 1][1]  # 指针前一格数据的坐标
            y_p_1 = self.xy_data[self.data_p - 1][0]
            delta_x = (x_p - x_p_1) / (t_p - t_p_1) * time_step  # x应该增加多少（线性预估）
            delta_y = (y_p - y_p_1) / (t_p - t_p_1) * time_step
            # 用户移动
            self.x += delta_x
            self.y += delta_y


class Edge_server:
    """边缘服务器（节点Node）"""

    def __init__(self, ID, x, y):
        self.ID = ID
        self.x, self.y = x, y  # 位置坐标
        self.R_node = R_node  # 缓存容量（Mbit）（两种队列共用）
        self.P_down = P_node  # 节点无线发送功率（W）
        self.f = f_node  # CPU频率（GHz）
        self.k_node = k_node  # 节点能耗功率系数
        self.if_critic = False  # 是否为评论家节点

        self.user_xy = {}  # 所有用户的位置历史。字典{用户ID：[[x,y]...],...}。从左到右数据由新到旧排列
        self.user_xy_sum = {}  # 用户位置总表，各节点之间共享并汇总此表。字典{用户ID：(x,y),...} 只存最新位置
        self.linking_users = []  # 所有直连的用户
        self.linking_users_sum = {}  # 用户连接性总表，各节点之间共享并汇总此表。字典{节点ID:集合{用户1ID,...},...}
        for id in range(1, node_num + 1):  # 每个节点记录初始化一个空集
            self.linking_users_sum[id] = set()
        self.users_report_time = {}  # 用户周期性上报的时刻记录。字典{ID：上报时刻t_report}
        # 若此时刻与当前时刻t相差超过2个上报周期则记录此用户为断联状态
        self.linking_nodes = []  # 能直连的所有节点的ID
        self.subgraph_nodes = set()  # 能够直连或间接连的节点的ID，包括自身
        self.linking_cloud = None  # 若能连通云端则为True
        self.queue_comp = []  # 计算队列[队列1[...],队列2[...],...]
        for i in range(core_num):  # 每个核心创建一个队列
            self.queue_comp.append([])
        self.queue_wireless_tran = []  # 无线传输队列
        self.queue_fiber_tran = []  # 有线传输队列
        self.q_comp = 0  # 计算队列已用大小（Mbit）
        self.q_comp_sum = {}  # 计算队列使用情况总表。各节点之间共享并汇总此表。字典{节点ID:使用量,...}
        self.q_tran = 0  # 传输队列已用大小（Mbit）
        self.q_tran_sum = {}  # 传输队列使用情况总表。各节点之间共享并汇总此表。字典{节点ID:使用量,...}
        self.update_queue = []  # 任务被决策后，队列记录此任务，完成处理后按FIFO的顺序进行网络参数更新。不按顺序地更新会报错
        self.update_cache = {}  # 任务处理完成后，记录奖励和状态等信息到此字典中。
        # 字典{任务用户ID*任务产生时刻/时间步长度：初始状态，奖励，结束状态，动作概率对数}

    def confirm_nodes_connect(self, graph, graph_cloud):
        """根据网络图确认与哪些节点能连通\n
        此函数无需等待同步时延，每次网络连通变动后 就要调用
        """
        self.linking_nodes = list(graph.neighbors(self.ID))  # 确定linking_nodes
        # 确定subgraph_nodes
        for subgraph in list(nx.connected_components(graph)):
            # 此nx.connected_components返回值为：子图的组成列表[集合{节点ID，...}，{...}，...]
            if self.ID in subgraph:  # 如果此节点在这个子图内，则记录
                self.subgraph_nodes = subgraph
        # 确定linking_cloud
        if self.ID in list(graph_cloud.neighbors(node_num + 1)):
            self.linking_cloud = True
        else:
            self.linking_cloud = False

    def confirm_users_disconnect(self):
        """如果没按时收到用户上报则认为用户已经断联\n
        ###每个时隙都执行。此函数必须在节点同步之前执行###
        """
        for id, t_report in list(self.users_report_time.items()):
            if t - t_report > 1.5 * report_period:  # 若时间超过1.5个上报周期还没有上报
                self.users_report_time.pop(id)  # 则删除此用户上报记录

                # 在linking_users中删除此用户，确认断联
                for user in self.linking_users:
                    if user.ID == id:
                        self.linking_users.remove(user)
                        break

                # 在linking_users_sum中删除连接记录，确认断联
                for nodeID in self.subgraph_nodes:  # 此节点所在的整个子图都要删记录，否则一同步以后又白删了
                    node_graph.node_list[nodeID - 1].linking_users_sum[self.ID].discard(id)
                # 这样做其实也违背了理论上的同步周期说法，用户一旦断连，整个子图的节点都会瞬间知晓

    def nodes_SYN(self):
        """节点之间的定期同步。\n
        理论上说，节点需要每个同步周期去发送确认来得知自己是否与其它节点相连。也就是说有可能出现这种情况：节点A以为自己跟节点B相连，
        但实际上已经断开。要模拟这种情况比较复杂，需要一套节点“以为的”连通性，还要一套真实的连通性。因此这里做出简化，
        即用户连通同步依然需要等待周期，但节点连通性同步则不需要等。\n
        ###同步周期+通信时延###"""
        for subgraph in list(nx.connected_components(node_graph.graph)):
            # 此nx.connected_components返回值为：子图的组成列表[集合{节点ID，...}，{...}，...]
            if len(subgraph) > 1:  # 如果子图中只有一个节点，那也没有同步的必要了
                for nodeID in list(subgraph):
                    node = node_graph.node_list[nodeID - 1]
                    subgraph2 = copy.deepcopy(subgraph)  # 需要一个深复制的子图，然后把自身节点去掉。也就是遍历子图中的其它节点。
                    # 为什么要深复制？因为subgraph在上面的for作为一个遍历的对象，遍历过程中不能删除元素
                    subgraph2.remove(nodeID)  # 子图2就是其它节点集合
                    for node2ID in subgraph2:  # 节点2就是其它节点
                        node2 = node_graph.node_list[node2ID - 1]

                        # linking_users_sum的同步
                        for id in node2.linking_users_sum.keys():  # 结构：字典{节点ID:集合{用户ID,...},...}
                            node.linking_users_sum[id] = \
                                (node.linking_users_sum[id] | node2.linking_users_sum[id])  # 求并集

                        # user_xy_sum的同步
                        for userID in node2.user_xy_sum.keys():  # 结构: 字典{用户ID：(x,y),...}
                            if userID not in node.user_xy_sum:  # 如果我没有这个用户，你有,那我从你那里copy过来
                                node.user_xy_sum[userID] = node2.user_xy_sum[userID]

                        # q_comp_sum以及q_tran_sum的同步
                        node.q_comp_sum[node2ID] = node2.q_comp  # 记录其他节点的计算队列使用量
                        node.q_tran_sum[node2ID] = node2.q_tran  # 记录其他节点的传输队列使用量
        delay = 2 * fiber_tran_time(node_dis, 1518 * 8 / 1e6) + SYN_period  # 距离下一次同步的时间(ms)
        if delay < 1:
            delay = 1  # 避免延迟过低导致todolist直接跳过此代码而不执行
        to_do_list.append((t + delay, "node_graph.node_list[0].nodes_SYN()"))  # 下一次同步

    # 增加了数据标准化功能的观测函数，已弃用:
    # def observe(self, task):
    #     """针对此任务，收集观测信息，近似为状态值。"""
    #     observe_origin = [task.data_size, task.ddl,
    #                       task.user.q_comp / task.user.R_user, task.user.q_tran / task.user.R_user,
    #                       len(self.linking_users) / user_num, self.ID]  # 原始的状态信息，未经过标准化处理，记录于state_data中
    #     if collect_state_data:
    #         # 这个观测用于第一次收集状态信息时，没有以往数据来进行标准化的情况
    #         observe = [task.data_size / 100, task.ddl / (1000 * 0.3 * 100 / 2 + 30000),
    #                    task.user.q_comp / task.user.R_user, task.user.q_tran / task.user.R_user,
    #                    len(self.linking_users) / user_num, self.ID]
    #     else:  # 如果已经收集到了状态数据，则后面会根据数据均值和方差做标准化
    #         observe = observe_origin
    #
    #     linking_user_list = []  # 与所有用户连接情况矩阵，1表示连接
    #     user_xy_sum_list = []  # 所有用户坐标矩阵
    #     test = 1  # 暂时不输入所有用户坐标、用户连接性作为状态，状态空间太大了。（现在状态里没有长度为user_num的向量）
    #     for id in range(1, user_num + 1):
    #         if id in self.linking_users_sum[self.ID]:
    #             linking_user_list.append(1)
    #         else:
    #             linking_user_list.append(0)
    #         if id in self.user_xy_sum:
    #             user_xy_sum_list = user_xy_sum_list + list(self.user_xy_sum[id])
    #         else:
    #             user_xy_sum_list = user_xy_sum_list + [650, 460]  # 如果没有用户坐标，就记一个很大的坐标值表示不适合卸载
    #
    #     linking_nodes_list = []  # 与所有节点连接情况矩阵，1表示连接（包括间接连通）
    #     q_comp_list = []  # 所有节点计算队列长度（百分比）矩阵
    #     q_tran_list = []  # 所有节点传输队列长度（百分比）矩阵
    #     user_opposite_xy = []  # 用户相对所有节点的相对坐标
    #     for id in range(1, node_num + 1):
    #         user_opposite_xy.append(node_graph.node_list[id - 1].x - task.user.x)
    #         user_opposite_xy.append(node_graph.node_list[id - 1].y - task.user.y)
    #         if id != self.ID:
    #             q_comp_list.append(self.q_comp_sum[id] / node_graph.node_list[id - 1].R_node)
    #             q_tran_list.append(self.q_tran_sum[id] / node_graph.node_list[id - 1].R_node)
    #         else:
    #             q_comp_list.append(self.q_comp / self.R_node)
    #             q_tran_list.append(self.q_tran / self.R_node)
    #         if id in self.subgraph_nodes:
    #             linking_nodes_list.append(1)
    #         else:
    #             linking_nodes_list.append(0)
    #     if self.linking_cloud:
    #         linking_nodes_list.append(1)
    #     else:
    #         linking_nodes_list.append(0)
    #
    #     # 节点的观测信息构成(两个部分，一个是用户坐标历史，一个是上边这一大堆别的。历史进入网络的LSTM层之后，再与其它状态信息拼接，进入全连接层)
    #     # observe = \
    #     #     task.type + observe + user_opposite_xy + linking_nodes_list + q_comp_list + q_tran_list
    #     observe_origin = observe_origin + user_opposite_xy + linking_nodes_list + q_comp_list + q_tran_list  # test
    #     observe = observe + user_opposite_xy + linking_nodes_list + q_comp_list + q_tran_list  # test 不含任务类型
    #
    #     user_xy_list = []  # 此任务的用户的坐标历史，从左到右由新到旧排列
    #     if task.user.ID in self.user_xy.keys():
    #         user_xy_list = self.user_xy[task.user.ID]
    #     while len(user_xy_list) < user_xy_size:  # 如果坐标历史记录不满size个，需要补0
    #         user_xy_list.append([0, 0])
    #
    #     if collect_state_data:  # 如果这次训练仅为收集数据
    #         state_data.append(str(observe_origin))  # 记录观测到的信息
    #     else:  # 如果已经收集到了状态数据
    #         for dim in range(len(observe)):  # 执行标准化
    #             observe[dim] = (observe[dim] - state_mean[dim]) / (state_std[dim] + 1e-8)
    #
    #     return user_xy_list, observe

    def observe(self, task):
        """针对此任务，收集观测信息，近似为状态值。
        实验证明观测数据做标准化（(数据-均值)/标准差）处理会让结果变差，归一化（除以一个常数使得数据小于1）会让结果变好。
        完全不处理的数据是不能训练的，不同维度的量级差距太大。"""
        observe = [task.data_size / 100, task.ddl / (1000 * 0.3 * 100 / 2 + 30000),
                   task.user.q_comp / task.user.R_user, task.user.q_tran / task.user.R_user,
                   len(self.linking_users) / 50, self.ID / node_num]

        linking_user_list = []  # 与所有用户连接情况矩阵，1表示连接
        user_xy_sum_list = []  # 所有用户坐标矩阵
        test = 1  # 暂时不输入所有用户坐标、用户连接性作为状态，状态空间太大了。（现在状态里没有长度为user_num的向量）
        for id in range(1, user_num + 1):
            if id in self.linking_users_sum[self.ID]:
                linking_user_list.append(1)
            else:
                linking_user_list.append(0)
            if id in self.user_xy_sum:
                user_xy_sum_list = user_xy_sum_list + list(self.user_xy_sum[id])
            else:
                user_xy_sum_list = user_xy_sum_list + [650, 460]  # 如果没有用户坐标，就记一个很大的坐标值表示不适合卸载

        linking_nodes_list = []  # 与所有节点连接情况矩阵，1表示连接（包括间接连通）
        q_comp_list = []  # 所有节点计算队列长度（百分比）矩阵
        q_tran_list = []  # 所有节点传输队列长度（百分比）矩阵
        user_opposite_xy = []  # 用户相对所有节点的相对坐标，再除以服务区宽高
        for id in range(1, node_num + 1):
            user_opposite_xy.append((node_graph.node_list[id - 1].x - task.user.x) / 650)
            user_opposite_xy.append((node_graph.node_list[id - 1].y - task.user.y) / 460)
            if id != self.ID:
                q_comp_list.append(self.q_comp_sum[id] / node_graph.node_list[id - 1].R_node)
                q_tran_list.append(self.q_tran_sum[id] / node_graph.node_list[id - 1].R_node)
            else:
                q_comp_list.append(self.q_comp / self.R_node)
                q_tran_list.append(self.q_tran / self.R_node)
            if id in self.subgraph_nodes:
                linking_nodes_list.append(1)
            else:
                linking_nodes_list.append(0)
        if self.linking_cloud:
            linking_nodes_list.append(1)
        else:
            linking_nodes_list.append(0)

        # 节点的观测信息构成(两个部分，一个是用户坐标历史，一个是上边这一大堆别的。历史进入网络的LSTM层之后，再与其它状态信息拼接，进入全连接层)
        observe = task.type + observe + user_opposite_xy + linking_nodes_list + q_comp_list + q_tran_list

        user_xy_list = []  # 此任务的用户的坐标历史，从左到右由新到旧排列
        if task.user.ID in self.user_xy.keys():
            user_xy_list = self.user_xy[task.user.ID]
        while len(user_xy_list) < user_xy_size:  # 如果坐标历史记录不满size个，需要补0
            user_xy_list.append([0, 0])

        return user_xy_list, observe

    def decide(self, task):
        """为任务task做卸载决策。收集状态信息->输入策略网络->选取概率最大的动作->执行动作"""
        if show_task_log:
            print("时刻%d \t 节点%d为用户%d的任务[%d]做出决策" % (t, self.ID, task.user.ID, task.t_created))

        ### 观测收集 ###

        user_xy_list, state = self.observe(task)
        task.fst_state = state  # 记录初始状态
        task.fst_xy_list = user_xy_list
        test = 1  # 状态暂时不含用户坐标

        ### 将观测值近似为状态，输入策略网络，得到动作 ###

        state_tensor = torch.as_tensor(state).float()  # 列表变成tensor。as_tensor是一种类似tensor的类型，为优化性能使用
        user_xy_tensor = torch.as_tensor(user_xy_list).float()
        state_tensor = state_tensor.to(device)  # tensor转移到GPU或CPU
        user_xy_tensor = user_xy_tensor.to(device)

        # 状态输入策略网络，输出动作对应的整数k，和概率分布的对数
        action, logp_action = central_controller.policy(state_tensor, f"agent_{self.ID - 1}")
        task.logp_action = logp_action  # 记录策略网络输出的概率分布的对数值
        self.update_queue.append(task)  # 记录这个待更新的任务

        ### 执行动作 ###

        # k为0表示本地计算，取1到n表示边缘计算，取n+1为云计算，取n+2表示丢包
        # task.k = random.randint(0, node_num + 2)  # 全随机卸载
        # task.k = self.ID  # 全决策者卸载
        # task.k = node_num + 1  # 全云计算
        # task.k = 0  # 全本地计算
        # task.k = node_num + 2  # 全丢包
        task.k = action.data.item()  # 记录决策结果。网络输出的是tensor，需要转换为int
        self.decision_constraint(task)  # 动作输入决策约束模块
        task.D[task.k] = 1  # 卸载决策变量

        return_delay = wireless_tran_time(task.user, self, (146 * 8 / 1e6), direction='down')  # 决策信息返回时延
        # task.energy += return_delay / 1000 * self.P_down  # 决策结果回传能耗（J） （只考虑用户能耗）
        if return_delay < 1:
            return_delay = 1  # 避免延迟过低导致todolist直接跳过此代码而不执行
        user_id = task.user.ID

        if task.user in self.linking_users:  # 如果决策结果出来以后，用户仍然在原来的节点范围内，决策结果才能返回给用户
            if task.D[node_num + 2] == 1:  # 丢
                to_do_list.append((t + return_delay, "user_list[%d].drop(user_list[%d].cache[0],if_report=True)"
                                   % (user_id - 1, user_id - 1)))
                if show_task_log:
                    to_do_list.append((t + return_delay,
                                       "print(\"【丢包原因】节点%d主动决定丢弃任务[%d]\")" % (self.ID, task.t_created)))
                to_do_list.append((t + return_delay, "user_list[%d].cache.pop(0)" % (user_id - 1)))  # 任务退出缓存区
                to_do_list.append((t + return_delay, "user_list[%d].q_cache -= user_list[%d].cache[0].data_size"
                                   % (user_id - 1, user_id - 1)))  # 释放存储空间
            elif task.D[0] == 1:  # 本地算
                to_do_list.append((t + return_delay, "user_list[%d].put_in_queue_comp(user_list[%d].cache[0])"
                                   % (user_id - 1, user_id - 1)))
            elif 1 <= task.k <= node_num + 1:  # 卸载或上云
                ###决策结果返回时延###
                to_do_list.append((t + return_delay, "user_list[%d].put_in_queue_tran(user_list[%d].cache[0])"
                                   % (user_id - 1, user_id - 1)))  # 压入传输队列，发给决策者
        else:
            task.user.put_in_queue_comp(task)  # 如果用户没来得及收到决策结果就离开了，则本地计算

    def put_in_update_cache(self, task):
        """一个任务处理结束后，决策者节点将此任务信息存入缓存。回合结束后再进行网络参数更新。
        为什么不立刻更新网络参数？若采用一任务一更新的做法，一个任务forward之后，
        还没来得及backward就可能被另一个任务提前backward，从而释放了计算图，改变了网络参数，
        因此之前的导数无法正确求出"""
        next_user_xy_list, next_state = self.observe(task)  # 再次观测状态
        # 任务信息存入更新缓存
        self.update_cache[int(task.user.ID * task.t_created / time_step)] = \
            [task.fst_state, task.reward, next_state, task.logp_action]

    def update(self):
        """任务信息转换为tensor，喂给神经网络，更新网络参数"""
        if len(self.update_queue) == 0:  # 如果更新队列为空则下边的更新操作没有必要
            return

        # 按照forward的顺序，从任务缓存中取出信息，排成列表
        s_list, r_list, ns_list, logp_action_list = [], [], [], []
        for task in self.update_queue:
            id = int(task.user.ID * task.t_created / time_step)  # 任务标识
            s_list.append(self.update_cache[id][0])  # 初始状态
            r_list.append(self.update_cache[id][1])  # 奖励
            ns_list.append(self.update_cache[id][2])  # 完成状态
            logp_action_list.append(self.update_cache[id][3])  # 动作概率对数

        # 将状态、奖励等列表转换为tensor格式
        # h = task.fst_xy_list  # 用户坐标历史
        bs = torch.tensor(np.asarray(s_list)).float()
        bs = bs.to(device)
        br = torch.tensor(np.asarray(r_list)).float()
        br = br.to(device)
        # nh = torch.tensor(np.asarray(next_user_xy_list)).float()  # 完成任务后的坐标历史
        # nh = nh.to(device)
        bns = torch.tensor(np.asarray(ns_list)).float()
        bns = bns.to(device)
        blogp_action = torch.stack(logp_action_list)
        blogp_action = blogp_action.to(device)

        # 寻找Critic节点
        critic_id = 3  # 默认为3
        for id in node_graph.critic_id:  # 遍历所有评论家节点，看哪个是能连通的
            if id in self.subgraph_nodes:
                critic_id = id
                break

        # 训练价值网络
        value_loss = central_controller.compute_value_loss(bs, br, bns, f"agent_{critic_id - 1}")  # 计算价值损失
        value_optimizer[critic_id - 1].zero_grad()  # 梯度清零，避免梯度累加
        value_loss.backward()  # 计算梯度
        value_optimizer[critic_id - 1].step()  # 优化器根据梯度更新网络参数
        central_controller.update_target_value(f"agent_{critic_id - 1}")  # 将价值网络参数按比例tau赋给目标价值网络

        # 训练策略网络
        policy_loss = central_controller.compute_policy_loss(bs, br, bns, blogp_action, f"agent_{critic_id - 1}")
        policy_optimizer[self.ID - 1].zero_grad()
        policy_loss.backward()
        policy_optimizer[self.ID - 1].step()

        log["value_loss"].append(value_loss.item())  # 记录损失
        log["policy_loss"].append(policy_loss.item())

        # 每个任务更新一次的Update方案（失败）：
        # while len(self.update_queue) > 0:
        #     task = self.update_queue[0]  # 从队首任务开始
        #     task_id = int(task.user.ID * task.t_created / time_step)  # 靠这个id唯一确定任务
        #
        #     if task_id in self.update_cache.keys():  # 如果缓存里有队首任务信息，则更新此任务
        #         s, r, ns, logp_action = self.update_cache[task_id]  # 从缓存中取出任务信息
        #
        #         # 将状态、奖励等列表转换为tensor格式
        #         # h = task.fst_xy_list  # 用户坐标历史
        #         r = torch.tensor(np.asarray(r)).float()  # 奖励
        #         r = r.to(device)
        #         # nh = torch.tensor(np.asarray(next_user_xy_list)).float()  # 完成任务后的坐标历史
        #         # nh = nh.to(device)
        #         ns = torch.tensor(ns).float()  # 完成任务后的状态
        #         ns = ns.to(device)
        #
        #         # 寻找Critic节点
        #         critic_id = 3  # 默认为3
        #         for id in node_graph.critic_id:  # 遍历所有评论家节点，看哪个是能连通的
        #             if id in self.subgraph_nodes:
        #                 critic_id = id
        #                 break
        #
        #         # 训练价值网络
        #         value_loss = \
        #         central_controller.compute_value_loss(s, r, ns, state_data_f"agent_{critic_id - 1}")  # 计算价值损失
        #         value_optimizer[critic_id - 1].zero_grad()  # 梯度清零，避免梯度累加
        #         value_loss.backward()  # 计算梯度
        #         value_optimizer[critic_id - 1].step()  # 优化器根据梯度更新网络参数
        #         central_controller.update_target_value(state_data_f"agent_{critic_id - 1}")  # 将价值网络参数按比例tau赋给目标价值网络
        #
        #         # 训练策略网络
        #         policy_loss = \
        #         central_controller.compute_policy_loss(s, r, ns, logp_action, state_data_f"agent_{self.ID - 1}")
        #         policy_optimizer[self.ID - 1].zero_grad()
        #         policy_loss.backward()
        #         policy_optimizer[self.ID - 1].step()
        #
        #         log["value_loss"].append(value_loss.item())  # 记录损失
        #         log["policy_loss"].append(policy_loss.item())
        #
        #         self.update_queue.remove(task)  # 队首任务移出更新队列
        #         del self.update_cache[task_id]  # 从缓存中删除此任务记录
        #
        #     else:  # 如果缓存里没有有队首任务信息，则跳出循环
        #         break

    def decision_constraint(self, task):
        """决策约束模块，将极不合理的节点决策改正，并产生决策惩罚"""
        if (task.k == 0) and (task.user.q_comp >= 0.9 * task.user.R_user):  # 如果本地计算队列太长，则不适合本地计算
            task.penalty = decision_penalty  # 给予惩罚
            task.k = task.decider.ID  # 纠正为卸载计算
            return None
        else:
            node = None  # 待检测节点
            if 1 <= task.k <= node_num:  # 卸载到节点
                node = node_graph.node_list[task.k - 1]  # 检测卸载目标节点
            elif task.k == node_num + 1:  # 卸载到云
                node = task.decider  # 检测决策者节点
            if (node is not None) and \
                    ((node.q_comp + node.q_tran >= 0.95 * node.R_node) or (task.user.q_tran >= 0.9 * task.user.R_user)):
                # 如果检测到节点可用空间不多，或本地传输队列过长，则不适合卸载计算
                task.penalty = decision_penalty  # 给予惩罚
                task.k = 0  # 纠正为本地计算
                return None
            if (task.k == node_num + 2) and (
                    task.user.q_comp + task.user.q_tran + task.user.q_cache < 0.9 * task.user.R_user):
                # 如果用户空间足够则不适合丢包
                task.penalty = decision_penalty  # 给予惩罚
                task.k = 0  # 纠正为本地计算

    def put_in_queue_comp(self, task):
        """进入节点计算队列"""
        # 不用检查连通性，因为那是传输队列的事情
        if t - task.t_created > task.ddl:
            task.user.drop(task, if_report=True)  # 如果已经超时则丢弃
            if show_task_log:
                print("【丢包原因】任务[%d]在进入节点%d计算队列时已经超时" % (task.t_created, self.ID))
        else:
            if task.data_size + self.q_comp + self.q_tran <= self.R_node:  # 如果节点的存储容量够
                # 搜索长度（任务大小）最短的一条计算队列
                shortest_q = None
                shortest_length = 99999
                for q in range(len(self.queue_comp)):  # 遍历每条计算队列
                    q_length = 0
                    for task_i in self.queue_comp[q]:  # 遍历队列的每个任务
                        q_length += task_i.data_size
                    if q_length < shortest_length:  # 如果是最小记录则记住他的下标
                        shortest_q = q
                        shortest_length = q_length
                self.queue_comp[shortest_q].append(task)  # 压入最短计算队列
                self.q_comp += task.data_size  # 占用存储空间
                if show_task_log:
                    print("时刻%d \t 任务[%d]进入节点%d的计算队列" % (t, task.t_created, self.ID))
            else:
                # 如果节点存不下则考虑发回本地计算
                if (task.user in self.linking_users) and \
                        (task.data_size + task.user.q_comp + task.user.q_tran + task.user.q_cache <= task.user.R_user):
                    # 如果能直连用户且用户有空间，就回传，否则丢包
                    self.put_in_queue_wireless_tran(task)
                else:
                    task.user.drop(task, if_report=True)
                    if show_task_log:
                        print("【丢包原因】因节点%d空间不足，任务[%d]无法进入其计算队列，且无法退回用户" % (self.ID, task.t_created))

    def compute(self):
        """节点多核处理计算队列"""
        ### 每个时间步执行 ###
        for queue in self.queue_comp:
            # 监测队列中是否有超时任务
            for task_i in queue:
                if t - task_i.t_created > task_i.ddl:
                    task_i.user.drop(task_i, if_report=True)  # 如果已经超时则丢弃
                    if show_task_log:
                        print("【丢包原因】任务[%d]在节点%d计算队列内被发现超时" % (task_i.t_created, self.ID))
                    queue.remove(task_i)  # 退出计算队列
                    self.q_comp -= task_i.data_size  # 释放存储空间

            # 判断队首任务是否已经完成计算
            if len(queue) > 0:  # 如果计算队列为空则不需要以下处理
                task = queue[0]  # 取队首任务
                if task.t_end_of_queue is None:
                    task.t_end_of_queue = t  # 记录任务排完队的时刻
                    if show_task_log:
                        print("时刻%d \t 任务[%d]在节点%d的计算队列内开始计算" % (t, task.t_created, self.ID))
                comp_time = (task.data_size * task.rho) / self.f * 1000  # 计算时延（ms）
                if abs(t - (task.t_end_of_queue + comp_time)) < time_step:  # 如果完成时间与当前时间相差小于时间步则视为计算完毕
                    # 计算能耗（J）
                    # task.energy += self.k_node * task.data_size * task.rho * (self.state_data_f * 10 ** 9) ** 2
                    queue.remove(task)  # 退出计算队列
                    self.q_comp -= task.data_size  # 释放存储空间
                    task.t_end_of_queue = None  # 重置
                    task.completed = True
                    if show_task_log:
                        print("时刻%d \t 节点%d完成1个任务[%d]" % (t, self.ID, task.t_created))
                    if task.user in self.linking_users:  # 计算完毕后检查用户是否在服务区
                        task.returner = self  # 记录返回者节点
                        self.put_in_queue_wireless_tran(task)  # 结果回传
                    elif (len(task.user.linking_nodes) != 0) and (
                            len(task.user.linking_nodes_id & self.subgraph_nodes) != 0):
                        # 如果用户已经离开决策者，但仍在别的节点的服务区内，则考虑跨区发回
                        # 用户可用的节点ID集 与 当前节点的子图节点ID集，求交集，选第一个ID为返回者节点
                        returner_id = list(task.user.linking_nodes_id & self.subgraph_nodes)[0]
                        task.returner = node_graph.node_list[returner_id - 1]  # 记录返回者节点
                        self.put_in_queue_fiber_tran(task)  # 结果回传
                    else:  # 用户不在任何服务区，丢包
                        self.q_comp -= task.data_size  # 释放存储空间
                        task.user.drop(task, if_report=True)
                        if show_task_log:
                            print("【丢包原因】任务[%d]在节点%d计算完毕后未发现用户处于可通信的服务区" % (task.t_created, self.ID))

    def put_in_queue_fiber_tran(self, task):
        """任务进入有线传输队列。如果是结果回传，调用前应指明task.returner。不考虑任务包从边缘层退回用户的传输情况"""
        # 为什么不检查连通性？因为进入队列不代表开始传输
        if t - task.t_created > task.ddl:
            task.user.drop(task, if_report=True)  # 如果已经超时则丢弃
            if show_task_log:
                print("【丢包原因】任务[%d]在进入节点%d有线传输队列时已经超时" % (task.t_created, self.ID))
        else:
            if task.data_size_out + self.q_comp + self.q_tran > self.R_node:
                # 如果节点的存储容量不够，不能发往任何地方也不能自己计算
                task.user.drop(task, if_report=True)  # 丢包
                if show_task_log:
                    print("【丢包原因】因节点%d空间不足，任务[%d]无法进入其有线传输队列" % (self.ID, task.t_created))
            else:
                if task.completed:  # 如果传的是计算结果
                    if self != task.returner:  # 如果本节点不是返回路径的终点
                        self.queue_fiber_tran.append(task)  # 压入有线传输队列
                        self.q_tran += task.data_size_out  # 占用存储空间
                    else:
                        self.put_in_queue_wireless_tran(task)  # 进行无线回传
                else:  # 如果传的是任务包
                    if task.k != self.ID:  # 如果本节点不是卸载目标
                        self.queue_fiber_tran.append(task)  # 压入有线传输队列
                        self.q_tran += task.data_size  # 占用存储空间
                    else:
                        self.put_in_queue_comp(task)  # 压入计算队列

    def put_in_queue_wireless_tran(self, task):
        """任务或是计算结果进入无线传输队列"""
        # 为什么不检查连通性？因为进入队列不代表开始传输
        if t - task.t_created > task.ddl:
            task.user.drop(task, if_report=True)  # 如果已经超时则丢弃
            if show_task_log:
                print("【丢包原因】任务[%d]在进入节点%d无线传输队列时已经超时" % (task.t_created, self.ID))
        else:
            if task.completed is True:
                data = task.data_size_out  # 要传的数据大小
            else:
                data = task.data_size
            if data + self.q_comp + self.q_tran <= self.R_node:  # 如果节点的存储容量够
                self.queue_wireless_tran.append(task)  # 压入传输队列
                self.q_tran += data  # 占用存储空间
                if show_task_log:
                    print("时刻%d \t 任务[%d]进入节点%d的传输队列" % (t, task.t_created, self.ID))
            else:  # 如果节点存不下就丢包
                task.user.drop(task, if_report=True)
                if show_task_log:
                    print("【丢包原因】任务[%d]的回传者节点%d没有足够空间进行回传" % (task.t_created, self.ID))

    def wireless_tran(self):
        """节点处理无线传输队列"""
        ### 每个时间步执行 ###
        # 监测队列中是否有超时任务
        for task_i in self.queue_wireless_tran:
            if t - task_i.t_created > task_i.ddl:
                task_i.user.drop(task_i, if_report=True)  # 如果已经超时则丢弃
                if show_task_log:
                    print("【丢包原因】任务[%d]在节点%d无线传输队列内被发现超时" % (task_i.t_created, self.ID))
                self.queue_wireless_tran.remove(task_i)  # 退出计算队列
                self.q_tran -= task_i.data_size  # 释放存储空间

        # 判断队首任务是否已经完成传输
        if len(self.queue_wireless_tran) > 0:  # 如果传输队列为空则不需要以下处理
            task = self.queue_wireless_tran[0]  # 取队首任务
            if task.completed:  # 如果回传的是计算结果
                data = task.data_size_out
            else:  # 如果回传的是任务包
                data = task.data_size

            if (self in task.user.linking_nodes) and \
                    (data + task.user.q_comp + task.user.q_tran + task.user.q_cache <= task.user.R_user):
                # 每个时隙监测用户是否离开服务区，同时监测用户存储空间是否够

                # 记录任务排完队的时刻
                if task.t_end_of_queue is None:
                    task.t_end_of_queue = t
                    if show_task_log:
                        print("时刻%d \t 任务[%d]在节点%d的无线传输队列内开始传输" % (t, task.t_created, self.ID))
                # 求用户发送给决策者的 发送+传播时延（ms）
                if task.tran_time is None:
                    task.tran_time = wireless_tran_time(task.user, self, data, direction='down')
                if abs(t - (task.t_end_of_queue + task.tran_time)) < time_step:
                    # 如果完成时间与当前时间相差小于时间步则视为传输完毕
                    # task.energy += task.tran_time / 1000 * self.P_down  # 任务回传能耗（J）
                    self.queue_wireless_tran.remove(task)  # 退出传输队列
                    self.q_tran -= data  # 释放存储空间
                    task.t_end_of_queue, task.tran_time = None, None  # 重置
                    if task.completed:  # 传输完毕后，如果用户收到结果
                        task.delay = t - task.t_created  # 记录处理耗时
                        task.compute_reward()  # 计算奖励
                        complete_list.append(task)  # 完成记录
                        if if_eval:  # 如果是测试阶段，记录每个任务的完成信息用于绘图
                            eval_task_log.append(task.type + [task.data_size, task.delay / 1000, task.energy, 1,
                                                              task.reward])
                        if show_task_log:
                            print("时刻%d \t 用户%d收到节点%d发回任务[%d]的计算结果" % (t, task.user.ID, self.ID, task.t_created))
                        task.decider.put_in_update_cache(task)  # 节点根据算法更新参数
                    else:  # 如果收到的是任务包
                        if show_task_log:
                            print("时刻%d \t 用户%d收到节点%d退回的任务[%d]" % (t, task.user.ID, self.ID, task.t_created))
                        task.user.cache.append(task)  # 任务暂时存入缓存区
                        task.user.q_cache += task.data_size  # 占用存储空间
                        task.user.put_in_queue_comp(task)  # 压入本地计算队列

            else:  # 如果发送过程中用户离开服务区，或是用户空间不够，则丢包
                self.queue_wireless_tran.remove(task)  # 退出传输队列
                self.q_tran -= data  # 释放存储空间
                task.user.drop(task, if_report=True)
                if show_task_log:
                    print("【丢包原因】节点%d向用户%d回传任务[%d]时用户存储空间不足或用户离开服务区"
                          % (self.ID, task.user.ID, task.t_created))

    def fiber_tran(self):
        """节点处理有线传输队列"""
        ### 每个时间步执行 ###
        for task_i in self.queue_fiber_tran:
            if t - task_i.t_created > task_i.ddl:
                task_i.user.drop(task_i, if_report=True)  # 如果已经超时则丢弃
                if show_task_log:
                    print("【丢包原因】任务[%d]在节点%d有线传输队列内被发现超时" % (task_i.t_created, self.ID))
                self.queue_fiber_tran.remove(task_i)  # 退出计算队列
                self.q_tran -= task_i.data_size  # 释放存储空间

        # 判断队首任务是否已经完成传输
        if len(self.queue_fiber_tran) > 0:  # 如果传输队列为空则不需要以下处理
            task = self.queue_fiber_tran[0]  # 取队首任务
            # 传输有3种可能，1是上云，2是发给卸载目标节点，3是计算结果发给返回者节点
            if task.completed:  # 如果是结果回传
                data = task.data_size_out
                final_target_id = task.returner.ID  # 目标确定为返回值节点
            else:  # 如果发任务包，可能是往云发，也可能是往别的节点发
                data = task.data_size
                final_target_id = task.k  # 目标确定为卸载目标或云
            # 记录最短路径
            if len(task.shortest_path) == 0:
                if (final_target_id == node_num + 1) and task.completed is False:  # 如果上云
                    task.shortest_path = [self.ID, node_num + 1]
                else:  # 上面排除了上云的情况，只剩下两种同层传输的情况
                    try:  # 根据最短路径寻找下一个转发目标
                        task.shortest_path = nx.shortest_path(node_graph.graph, source=self.ID, target=final_target_id)
                    except:  # 如果与目标节点没有连通路径，则考虑从中间的3或6节点转发
                        if final_target_id != 3 and final_target_id < 6:  # 小于6的从节点3转发
                            task.shortest_path = [self.ID, 3, final_target_id]
                        elif final_target_id != 6 and final_target_id > 5:  # 大于5的从节点6转发
                            task.shortest_path = [self.ID, 6, final_target_id]
                        elif final_target_id == 3:  # 如果目标节点为3，说明3走不通，只能考虑走6
                            task.shortest_path = [self.ID, 6, 3]
                        elif final_target_id == 6:
                            task.shortest_path = [self.ID, 3, 6]
            target_id = 0  # 下一站节点ID
            for i in range(len(task.shortest_path)):
                if task.shortest_path[i] == self.ID:  # 如果在路径中找到了自己ID，则下一站ID就是列表的下一位元素
                    target_id = task.shortest_path[i + 1]
            if (final_target_id == node_num + 1) and task.completed is False:  # 如果上云
                target = cloud  # 下一站设备
                if_linking = self.linking_cloud  # 是否与云连接
                if_enough_R = True  # 云不考虑存储空间
                dis = cloud_dis  # 传输距离（m）
                retran_delay = task.cloud_retran_delay  # 核心网转发时延(ms)
            else:  # 如果不上云
                target = node_graph.node_list[target_id - 1]  # 下一站设备
                if_linking = (target_id in self.linking_nodes)  # 是否与下一站节点连接
                if_enough_R = (data + target.q_comp + target.q_tran <= target.R_node)  # 是否下一站节点空间足够
                dis = node_dis  # 传输距离（m）
                retran_delay = 0  # 边缘层不考虑核心网转发时延

            if if_linking and if_enough_R:
                # 每个时隙监测与下一站的连通性，同时监测下一站存储空间是否够

                # 记录任务排完队的时刻
                if task.t_end_of_queue is None:
                    task.t_end_of_queue = t
                    if show_task_log:
                        print("时刻%d \t 任务[%d]在节点%d的有线传输队列内开始传输" % (t, task.t_created, self.ID))
                # 求转发的 发送+传播时延（ms）
                if task.tran_time is None:
                    task.tran_time = fiber_tran_time(dis, data)

                if abs(t - (task.t_end_of_queue + task.tran_time + retran_delay)) < time_step:
                    # 如果完成时间与当前时间相差小于时间步则视为传输完毕
                    self.queue_fiber_tran.remove(task)  # 退出传输队列
                    self.q_tran -= data  # 释放存储空间
                    task.t_end_of_queue, task.tran_time, task.shortest_path = None, None, []  # 重置
                    if (final_target_id == node_num + 1) and task.completed is False:  # 如果上云
                        if show_task_log:
                            print("时刻%d \t 节点%d将待处理任务[%d]发往云服务器完毕" % (t, self.ID, task.t_created))
                        cloud.put_in_comp_list(task)  # 压入云计算列表
                    else:
                        if show_task_log:
                            print("时刻%d \t 节点%d将任务[%d]发往节点%d完毕" % (t, self.ID, task.t_created, target_id))
                        target.put_in_queue_fiber_tran(task)  # 下一站节点继续转发

            else:  # 如果发送过程中线路中断，或是下一站空间不够，则考虑自己计算
                self.queue_fiber_tran.remove(task)  # 退出传输队列
                self.q_tran -= data  # 释放存储空间
                task.t_end_of_queue, task.tran_time, task.shortest_path = None, None, []  # 重置
                self.put_in_queue_comp(task)  # 压入计算队列


class Cloud:
    """云服务器"""
    x = 150
    y = cloud_dis
    ID = node_num + 1
    f = f_cloud  # CPU频率（GHz）

    comp_list = []  # 任务进入列表等待计算，完毕后退出
    tran_list = []  # 传输列表，存储正在往下层传输的任务

    def put_in_comp_list(self, task):
        """任务进入计算列表"""
        if t - task.t_created > task.ddl:
            task.user.drop(task, if_report=True)  # 如果已经超时则丢弃
            if show_task_log:
                print("【丢包原因】任务[%d]在进入云端计算列表时已经超时" % task.t_created)
        else:
            self.comp_list.append(task)  # 压入计算列表
            if show_task_log:
                print("时刻%d \t 任务[%d]在云端开始计算" % (t, task.t_created))

    def compute(self):
        """云端处理计算列表"""
        ### 每个时间步执行 ###
        for task in self.comp_list:
            if t - task.t_created > task.ddl:  # 超时检测
                task.user.drop(task, if_report=True)  # 如果已经超时则丢弃
                if show_task_log:
                    print("【丢包原因】任务[%d]在云端计算列表内被发现超时" % task.t_created)
                self.comp_list.remove(task)  # 退出计算列表
                continue
            if task.t_end_of_queue is None:
                task.t_end_of_queue = t  # 记录任务开始计算的时刻
            comp_time = (task.data_size * task.rho) / self.f * 1000  # 计算时延（ms）
            if abs(t - (task.t_end_of_queue + comp_time)) < time_step:  # 如果完成时间与当前时间相差小于时间步则视为计算完毕
                # task.energy += k_node * task.data_size * task.rho * (self.state_data_f * 10 ** 9) ** 2  # 计算能耗（J）
                self.comp_list.remove(task)  # 退出计算列表
                task.t_end_of_queue = None  # 重置
                task.completed = True
                if show_task_log:
                    print("时刻%d \t 云端完成1个任务[%d]" % (t, task.t_created))
                self.put_in_tran_list(task)  # 进行回传

    def put_in_tran_list(self, task):
        """任务进入传输列表"""
        if t - task.t_created > task.ddl:
            task.user.drop(task, if_report=True)  # 如果已经超时则丢弃
            if show_task_log:
                print("【丢包原因】任务[%d]在进入云端传输列表时已经超时" % task.t_created)
        else:
            self.tran_list.append(task)  # 压入传输列表
            # print("时刻%d \t 任务[%d]进入云端传输队列" % (t, task.t_created))

    def transmit(self):
        """云端处理传输列表（发往decider节点）"""
        ### 每个时间步执行 ###
        for task in self.tran_list:
            # 超时检测
            if t - task.t_created > task.ddl:
                task.user.drop(task, if_report=True)  # 如果已经超时则丢弃
                if show_task_log:
                    print("【丢包原因】任务[%d]在云端传输列表内被发现超时" % task.t_created)
                self.tran_list.remove(task)  # 退出计算队列
                continue

            if task.receiver.linking_cloud and (
                    task.data_size_out + task.receiver.q_comp + task.receiver.q_tran < task.receiver.R_node):
                # 每个时隙监测接收者连接性以及其存储空间是否够用

                if task.t_end_of_queue is None:
                    task.t_end_of_queue = t  # 记录任务排完队的时刻
                    if show_task_log:
                        print("时刻%d \t 任务[%d]在云端传输列表内开始传输" % (t, task.t_created))
                tran_time = fiber_tran_time(cloud_dis, task.data_size_out)  # 发送+传播时延（ms）

                if abs(t - (task.t_end_of_queue + tran_time + task.cloud_retran_delay)) < time_step:
                    # 如果完成时间与当前时间相差小于时间步则视为传输完毕
                    self.tran_list.remove(task)  # 退出传输列表
                    task.t_end_of_queue = None  # 重置
                    if task.user in task.receiver.linking_users:  # 发给节点完毕后检查用户是否在服务区
                        if show_task_log:
                            print("时刻%d \t 云服务器将任务[%d]计算结果发往节点%d完毕" % (t, task.t_created, task.receiver.ID))
                        task.returner = task.receiver  # 正式记录记录返回者节点
                        task.receiver.put_in_queue_wireless_tran(task)  # 结果回传
                    elif (len(task.user.linking_nodes) != 0) and \
                            (len(task.user.linking_nodes_id & task.receiver.subgraph_nodes) != 0):
                        # 如果用户已经离开接收者，但仍在别的节点的服务区内，则考虑跨区发回
                        # 用户直连的节点ID集 与 当前节点的子图节点ID集，求交集，选最近的节点为返回者节点
                        if show_task_log:
                            print("时刻%d \t 云服务器将任务[%d]计算结果发往节点%d完毕" % (t, task.t_created, task.receiver.ID))
                        # 可以用来回传的节点ID
                        available_nodes_id = list(task.user.linking_nodes_id & task.receiver.subgraph_nodes)
                        # 搜索离用户最近的节点
                        available_nodes_id.sort(key=lambda id: distance(task.user, node_graph.node_list[id - 1]))
                        returner_id = available_nodes_id[0]  # 取最近节点作为返回者
                        task.returner = node_graph.node_list[returner_id - 1]  # 记录返回者节点
                        task.receiver.put_in_queue_fiber_tran(task)  # 结果回传
                    else:  # 用户不在任何服务区，丢包
                        task.user.drop(task, if_report=True)
                        if show_task_log:
                            print("【丢包原因】任务[%d]在节点%d计算完毕后未发现用户处于可通信的服务区" % (task.t_created, self.ID))
            else:  # 若接收者断联或其存储空间不够用，考虑换一个接收者。但云端不知道边缘层情况，因此选择决策者的邻近节点，然后重试发回任务结果
                receiver_id = task.decider.ID + 1  # 决策者相邻节点做接收
                if receiver_id > node_num:  # 如果超过最大节点数，就-1取相邻
                    receiver_id = task.decider.ID - 1
                task.receiver = node_graph.node_list[receiver_id - 1]  # 记录接收者节点，下一个时间步会检查此节点能否做接收


class Node_Graph:
    """节点网络总图 \n
    node_xy_list: [(x1,y1),(x2,y2)...]\n
    edge_list: [(node1,node2),(node1,node3)...]\n
    """
    graph = nx.Graph()  # 节点之间的网络
    graph_cloud = nx.Graph()  # 节点与云服务器的网络
    node_list = []  # 节点列表。列表下标+1 = 节点ID
    critic_id = [3]  # 评论家节点ID.默认为3

    def __init__(self, node_xy_list, edge_list):
        # 创建节点
        nodeID = 1
        for node_xy in node_xy_list:
            self.node_list.append(Edge_server(nodeID, node_xy[0], node_xy[1]))
            self.graph.add_node(nodeID)  # 用ID号做图节点
            nodeID = nodeID + 1
        self.graph_cloud = copy.deepcopy(self.graph)  # 深复制
        self.graph_cloud.add_node(node_num + 1)  # graph_cloud的节点只比graph多了一个云端（ID为node_num+1）

        # 连接节点之间的线路
        self.graph.add_edges_from(edge_list)
        for node_id in range(1, node_num + 1):
            self.graph_cloud.add_edge(node_id, node_num + 1)  # 每个节点都和云服务器连一条线
        for node in self.node_list:
            node.confirm_nodes_connect(self.graph, self.graph_cloud)  # 每个节点确认自己的邻接节点

    def fiber_disconnect(self):
        """光纤每分钟有断开的可能，需要随机时间修复"""
        if_node_disconnect, if_cloud_disconnect = False, False
        for edge in list(self.graph.edges):  # 边edge结构：元组（顶点1，顶点2）
            if random.random() < disconnect_prob:
                self.graph.remove_edge(edge[0], edge[1])  # 节点之间光纤断连
                if_node_disconnect = True
                repair_time = random.randint(repair_t_start, repair_t_end)  # 断连后的维修时间(ms)
                to_do_list.append((t + repair_time, f"node_graph.graph.add_edge({edge[0]}, {edge[1]})"))  # 一段时间后修复
                # 显示修复后的网络图
                if show_task_log:
                    to_do_list.append(
                        (t + repair_time, "nx.draw(node_graph.graph, with_labels=True, font_weight='bold')"))
                    to_do_list.append((t + repair_time, "plt.show()"))

        for edge in list(self.graph_cloud.edges):
            if random.random() < disconnect_prob:
                self.graph_cloud.remove_edge(edge[0], edge[1])  # 节点与云端光纤断连
                if_cloud_disconnect = True
                repair_time = random.randint(3 * 60 * 1000, 30 * 60 * 1000)  # 断连后的维修时间(ms)
                # 一段时间后修复
                to_do_list.append((t + repair_time, f"node_graph.graph_cloud.add_edge({edge[0]}, {edge[1]})"))
                # 显示修复后的网络图
                if show_task_log:
                    to_do_list.append(
                        (t + repair_time, "nx.draw(node_graph.graph_cloud, with_labels=True, font_weight='bold')"))
                    to_do_list.append((t + repair_time, "plt.show()"))

        if if_node_disconnect:  # 如改动了节点之间的网络边集
            self.choose_critic()  # 选择新的评论家节点
            if show_task_log:
                nx.draw(node_graph.graph, with_labels=True, font_weight='bold')  # 网络图绘图设定
                plt.show()  # 显示节点网络图
            for node in node_graph.node_list:
                node.confirm_nodes_connect(node_graph.graph, node_graph.graph_cloud)  # 节点确认连接性
        if if_cloud_disconnect:  # 如改动了节点与云之间的网络边集
            if show_task_log:
                nx.draw(node_graph.graph_cloud, with_labels=True, font_weight='bold')  # 网络图绘图设定
                plt.show()  # 显示云边线路网络图
            for node in node_graph.node_list:
                node.confirm_nodes_connect(node_graph.graph, node_graph.graph_cloud)  # 节点确认连接性

        to_do_list.append((t + disconnect_period, "node_graph.fiber_disconnect()"))  # 下一次断连判定

    def choose_critic(self):
        """根据网络图，为每个连通子图选择一个评论家节点\n
        此函数无需等待同步时延，每次网络连通变动后 就要调用
        """
        self.critic_id = []
        for subgraph in list(nx.connected_components(self.graph)):
            # 搜索子图中节点是否存在评论家，然后选择第一个评论家作为延续，其余节点全部剥夺评论家身份
            critic_founded = False
            for id in subgraph:
                if (critic_founded is False) and node_graph.node_list[id - 1].if_critic:  # 如果之前没找到，选用第一个找到的
                    critic_founded = True
                    self.critic_id.append(id)  # 记录评论家ID
                    continue
                if critic_founded:
                    node_graph.node_list[id - 1].if_critic = False  # 如果找到了Critic则后面的节点通通置if_critic为False
            # 如果搜遍了这个子图还是没有评论家，就随机挑一个来当
            if not critic_founded:
                id = random.choice(list(subgraph))
                node_graph.node_list[id - 1].if_critic = True
                self.critic_id.append(id)  # 记录评论家ID


#####################################
#                                   #
#              全局变量               #
#                                   #
#####################################
t = 0  # 时间变量（ms）。主函数循环中应以1ms为周期
# 理论上没有时隙，但实际实现还是用了。但并非传统的一时间步一更新神经网络的做法，此外毫秒级别的时间步已经足够精确了。
drop_list = []  # 丢弃任务记录
complete_list = []  # 计算完毕记录
user_list = []  # 用户列表。列表下标+1 = 用户ID
node_graph: Node_Graph  # 节点网络图
cloud: Cloud  # 云端
to_do_list = []  # 存储系统将来要执行的代码。有些函数不是每个时隙都要执行的，因此先把这些要做的事存起来，需要的时候再做。
# 列表[元组(执行时间,"执行代码")]（按执行时间排序）
if_eval = False  # 是否开始测试（决定任务信息是否记录在下面的列表）
eval_task_log = []  # 测试时的任务数据收集。 列表[任务1[类型，大小(Mbit)，时延(s)，能耗(J)，是否完成，开销（奖励换算）]，任务2[...]...]

# state_data = []  # 状态数据统计。收集一次训练的状态，取其均值和标准差，用于后面状态的标准化。列表[[状态1],[状态2]...]
# state_data_f = None  # 状态数据文件的文件句柄
# state_mean = []  # 状态单个维度的均值
# state_std = []  # 状态单个维度的标准差


#####################################
#                                   #
#               主函数               #
#                                   #
#####################################
if __name__ == '__main__':
    user_info = dp.data_process(dp.load_data(user_num))  # 读取用户数据
    sys_init()  # 系统初始化
    # A2C算法初始化
    central_controller, policy_optimizer, value_optimizer = \
        a2c.init(num_agents=node_num, num_states=5 * node_num + 11, num_actions=node_num + 3,
                 num_episode=episode_num, lr_policy=lr_policy, lr_value=lr_value, h_size=user_xy_size)

    central_controller.train()  # 模型进入训练模式
    return_lst = []  # 回报值记录
    log = defaultdict(list)  # 字典记录“价值损失”和“策略损失”

    # if collect_state_data:
    #     state_data_f = open("state_data.txt", "w")  # 收集状态信息
    # else:
    #     load_state_data()  # 如果已经有状态信息，则读取，计算均值和标准差，用于标准化状态值

    # 开始训练（一回合更新一次参数。理论上一个任务更新一次，但这样代码修改成本高）
    print("\n开始训练！\n")
    for episode in range(1, episode_num + 1):  # 训练episode_num个回合
        if show_task_log:
            print(f"\n回合{episode}开始")
        Return = 0  # 回报置零
        done = False  # 完成信号（与RL更新算法无关。系统产生任务时间截止到runtime，然后停止生成任务直到所有任务处理完毕，done信号才发出）

        # 单回合主循环
        while not done:
            # 节点每周期的同步
            for node in node_graph.node_list:
                node.confirm_users_disconnect()  # 确认用户是否断联

            # 处理to_do_list。 这个列表存储系统将来要执行的代码。有些函数不是每个时隙都要执行的，因此先把这些要做的事存起来，需要的时候再做。
            to_do_list.sort(key=lambda job: job[0])  # 按照执行时刻排序，先做的工作放前面
            for job in to_do_list:  # to_do_list结构:[元组(执行时间,"执行代码")]（按执行时间排序）
                if abs(job[0] - t) < time_step:  # 如果任务执行时间已到
                    eval(job[1])  # 执行此代码
                elif job[0] > t:  # 如果执行时间尚早，说明后面的列表没必要在遍历了
                    # 删除已经执行过的工作（避免列表过长）
                    index = to_do_list.index(job)
                    for i in range(index):
                        to_do_list.pop(0)  # 删除表头元素
                    break

            # 产生任务
            if t <= run_time:
                for user in user_list:
                    user.task_generator()
            # 用户处理队列
            for user in user_list:
                user.compute()
                user.transmit()
            # 节点处理队列
            for node in node_graph.node_list:
                node.compute()
                node.wireless_tran()
                node.fiber_tran()
            # 云端处理任务
            cloud.compute()
            cloud.transmit()
            # 用户移动 test
            for user in user_list:
                user.move(random.randint(0, 4), 0.1 * random.random())  # 随机移动
                user.confirm_nodes_connect()  # 检测连接性

            # 检查是否所有任务都处理完了
            if t > run_time:
                unfinish = False
                for node in node_graph.node_list:  # 遍历每个节点，如果更新列表里的任务都已经处理完毕则认为没有任务要处理了
                    if unfinish:
                        break
                    for task in node.update_queue:
                        id = int(task.user.ID * task.t_created / time_step)  # 任务标识
                        if id in node.update_cache.keys():  # 如果缓存信息里有这个任务说明已经完成处理
                            pass
                        else:
                            unfinish = True
                            break
                    if (node == node_graph.node_list[-1]) and (not unfinish):
                        done = True

            t += time_step  # 时间往后走一步

        # 到此，回合结束，开始打印信息，训练模型

        for node in node_graph.node_list:
            node.update()  # 更新网络参数

        return_lst.append(Return)  # 记录回报
        if episode % 100 == 0:  # 每100个回合数据后绘制一次回报走势图
            x_axis = np.arange(len(return_lst))  # 为x轴选择取值（0至回报列表长度的整数，不含终止值）
            plt.plot(x_axis, return_lst)
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.savefig("回报走势.png", bbox_inches="tight")
            plt.close()

        if episode % 100 == 0:  # 每100个回合数据后保存一次策略网络参数至外部文件
            agent2policynet = {}
            for agent, policynet in central_controller.agent2policy.items():
                agent2policynet[agent] = policynet.state_dict()
            torch.save(agent2policynet, os.path.join("output", "model.pt"))

        if episode % 20 == 0:  # 每20回合打印一次信息
            avg_value_loss = np.mean(log["value_loss"][-20:])
            avg_policy_loss = np.mean(log["policy_loss"][-20:])
            avg_reward = np.mean(return_lst[-20:])
            print(
                f"回合 {episode}, 回报 {avg_reward:.2f}, 价值损失 {avg_value_loss:.4f}, 策略损失 {avg_policy_loss:.4f}")

        if show_task_log:
            print("\n系统运行时间 %ds" % (t / 1000))
            print("完成任务数量 %d" % len(complete_list))
            print("丢弃任务数量 %d" % len(drop_list))
            print(f"本回合的回报 {Return}")
            print(f"回合{episode}结束\n")

        sys_reset()  # 系统重置

    ### 训练完毕，开始测试 ###

    print("\n训练结束，开始测试~\n")
    if_eval = True
    central_controller.eval()  # 将模型设定为测试模式
    return_lst = []  # 回报值记录
    for episode in range(1, 11):  # 测试10个回合
        print(f"\n回合{episode}开始")
        Return = 0  # 回报置零
        done = False  # 完成信号（与RL更新算法无关。系统产生任务时间截止到runtime，然后停止生成任务直到所有任务处理完毕，done信号才发出）

        # 单回合主循环
        while not done:
            # 节点每周期的同步
            for node in node_graph.node_list:
                node.confirm_users_disconnect()  # 确认用户是否断联

            # 处理to_do_list。 这个列表存储系统将来要执行的代码。有些函数不是每个时隙都要执行的，因此先把这些要做的事存起来，需要的时候再做。
            to_do_list.sort(key=lambda job: job[0])  # 按照执行时刻排序，先做的工作放前面
            for job in to_do_list:  # to_do_list结构:[元组(执行时间,"执行代码")]（按执行时间排序）
                if abs(job[0] - t) < time_step:  # 如果任务执行时间已到
                    eval(job[1])  # 执行此代码
                elif job[0] > t:  # 如果执行时间尚早，说明后面的列表没必要在遍历了
                    # 删除已经执行过的工作（避免列表过长）
                    index = to_do_list.index(job)
                    for i in range(index):
                        to_do_list.pop(0)  # 删除表头元素
                    break

            # 产生任务
            if t <= run_time:
                for user in user_list:
                    user.task_generator()
            # 用户处理队列
            for user in user_list:
                user.compute()
                user.transmit()
            # 节点处理队列
            for node in node_graph.node_list:
                node.compute()
                node.wireless_tran()
                node.fiber_tran()
            # 云端处理任务
            cloud.compute()
            cloud.transmit()
            # 用户移动 test
            for user in user_list:
                user.move(random.randint(0, 4), 0.1 * random.random())  # 随机移动
                user.confirm_nodes_connect()  # 检测连接性

            # 检查是否所有任务都处理完了
            if t > run_time:
                unfinish = False
                for node in node_graph.node_list:  # 遍历每个节点，如果更新列表里的任务都已经处理完毕则认为没有任务要处理了
                    if unfinish:
                        break
                    for task in node.update_queue:
                        id = int(task.user.ID * task.t_created / time_step)  # 任务标识
                        if id in node.update_cache.keys():  # 如果缓存信息里有这个任务说明已经完成处理
                            pass
                        else:
                            unfinish = True
                            break
                    if (node == node_graph.node_list[-1]) and (not unfinish):
                        done = True

            t += time_step  # 时间往后走一步

        # 到此，回合结束，开始打印信息

        return_lst.append(Return)  # 记录回报
        print("\n系统运行时间%ds" % (t / 1000))
        print("完成任务数量%d" % len(complete_list))
        print("丢弃任务数量%d" % len(drop_list))
        print(f"本回合的回报值为{Return}")
        print(f"回合{episode}结束\n")
        sys_reset()  # 系统重置

    # if collect_state_data:
    #     state_data_f.writelines('\n'.join(state_data))
    #     state_data_f.close()

    ### 测试完毕 ###

    save_result(eval_task_log)  # 保存测试数据，用于绘图
