#!/usr/bin/env python

# 导入必要的库，创建cartpole游戏环境
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

env = gym.make("CartPole-v1", render_mode="rgb_array")
BATCH_SIZE = 32  # 批处理大小
LR = 0.01  # 学习率
EPSILON = 0.9  # 随机选取的概率，如果概率小于这个随机数，就采取greedy的行为
GAMMA = 0.9  # 折扣因子
TARGET_REPLACE_ITER = 100  # 目标网络更新间隔
MEMORY_CAPACITY = 2000  # 经验回放缓冲区容量

env = env.unwrapped
N_ACTIONS = env.action_space.n  # 小车动作空间
# print(N_ACTIONS)
N_STATES = env.observation_space.shape[0]  # 实验观测空间
# print(N_STATES)
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


# 网络结构初始化
class Net(nn.Module):
    #网络组成：输入层4个输入（位置，速度，角度，角速度） 隐藏层50个神经元   输出层两个输出（向左/向右）
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)  # 输入层到隐藏层
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化
        self.out = nn.Linear(50, N_ACTIONS)  # 隐藏层到输出层
        self.out.weight.data.normal_(0, 0.1)  # 权重初始化

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU激活函数
        actions_value = self.out(x)  # 输出每个动作的Q值
        return actions_value


# DQN算法
class DQN(object):
    def __init__(self):
        # 创建两个网络，评估网络和目标网络
        # DQN是Q-Leaarning的一种方法，但是有两个神经网络，一个是eval_net一个是target_net
        # 两个神经网络相同，参数不同，是不是把eval_net的参数转化成target_net的参数，产生延迟的效果
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 学习步数计数器
        self.memory_counter = 0  # 记忆库中位值的计数器
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库
        # 记忆库初始化为全0，存储两个state的数值加上一个a(action)和一个r(reward)的数值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()  # 优化器和损失函数，损失函数是均方误差

    # 接收环境中的观测值，并采取动作
    def choose_action(self, x):
        # x-->(array([ 0.02041975, -0.02956359, -0.04300894, -0.01443677], dtype=float32), {})

        # x[0]就是观测值
        #将numpy数组转换为PyTorch张量
        #使用unsqueeze（0）添加批次维度，因为神经网络期望输入是批次格式
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # print(x)
        # x = torch.unsqueeze(torch.FloatTensor(x[0]),0)
        # print(x)
        if np.random.uniform() < EPSILON:
            # 随机值得到的数有百分之九十的可能性<0.9,所以该if成立的几率是90%
            # 90%的情况下采取actions_value高的作为最终动作

            # 向前传播，获取本次step所有动作的Q值
            actions_value = self.eval_net.forward(x)
            # 选择Q值最大的动作，转化为numpy
            # torch.max（张量，维度）返回一个包含两个元素的元组，如：result=([0.7],[1])，[0]最大值，[1]该维度最大值的索引
            action = torch.max(actions_value, 1)[1].data.numpy()
            #如果动作是离散动作，ENV_A_SHAPE = 0 ， 返回Q最大值的动作
            #如果动作是连续动作，ENV_A_SHAPE = 动作数组的形状  返回Q最大值的连续环境期望的动作格式（数组）
            #连续环境是多维控制，标量只能控制一个维度，数组可以控制多个维度
            action = action[0] if ENV_A_SHAPE == 0 else action.resape(ENV_A_SHAPE)  # return the argmax index
        else:
            # 其他10%采取随机选取动作
            #此处N_ACTIONS=2，也就是随机返回0或1,0左推小车，1右推小车
            action = np.random.randint(0, N_ACTIONS)
            #依旧处理连续环境动作格式
            #action是标量，需要转换为数组才能作为连续环境期望的动作格式输出
            action = action if ENV_A_SHAPE == 0 else np.array(action).reshape(ENV_A_SHAPE)

        # print(action)
        return action

        # 记忆库，存储之前的记忆，学习之前的记忆库里的东西

    def store_transition(self, s, a, r, s_):
        #s是初始状态数组，s_是最终状态数组，利用hstack把三部分拼接起来
        transition = np.hstack((s, [a, r], s_))
        # print(transition)
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        #[index，：]给某一行赋值
        self.memory[index, :] = transition
        self.memory_counter += 1
        if (self.memory_counter < 2000):
            print(self.memory_counter)

    def learn(self):
        # 目标网络参数更新,每隔TARGET_REPLACE_ITE（100）更新一下
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            #tarnet和eval都是net类，有torch中的api，load_state_dick是把状态字典加载，state_dict()是提取状态字典
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # targetNet是时不时更新一下，evalNet是每一步都更新

        # 抽取记忆库中的批数据，2000里面随机抽取32组数据，每组数据有10个参数
        #这里的sample_index形状是（32，）
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        #把32组数据存入b_memory32*10的张量，形状（32，10）
        b_memory = self.memory[sample_index, :]
        # 打包记忆，分开保存进b_s，b_a，b_r，b_s
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]) #前四列初始状态
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)) #第五列动作
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]) #第六列奖励
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]) #后四列最终状态

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        #执行的实际是b_a,  gather(1, b_a) 沿着维度1(列方向)根据b_a的索引选择值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (32, 1)是实际执行动作的Q值一阶张量
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach阻止梯度传播
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)#计算目标值和评估值的损失

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()  # 误差反向传播
        self.optimizer.step()

    def save_model(self, e):
        torch.save(self.eval_net.state_dict(), "cartpole_dqn.pt")
        print("save model of ", e)


dqn = DQN()

print('\nCollection experience...')
path = "CartPole_model.pt"
for i_episode in range(800):
    s = env.reset()  # 得到环境的反馈，现在的状态
    s = s[0]
    ep_r = 0
    save_flag = False
    time = 0
    while True:
        time = time + 1
        # env.render() #环境渲染，可以看到屏幕上的环境
        a = dqn.choose_action(s)  # 根据dqn来接受现在的状态，得到一个行为
        s_, r, done, info, _ = env.step(a)  # 根据环境的行为，给出一个反馈

        # 修改 reward, 使 DQN 快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)  # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态

        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if (i_episode % 10 == 0) and (save_flag == False) and (dqn.memory_counter > MEMORY_CAPACITY):
            dqn.save_model(str(i_episode))
            save_flag = True
        if done or (time > 5000):
            break

        s = s_  # 现在的状态赋值到下一个状态上去