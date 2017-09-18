import matplotlib.pyplot as plt
import numpy as np

from agent import Agent, Agent_moment_CDC2017, new_Agent, new_Agent_moment_CDC2017, new_Agent_L2, \
    new_Agent_moment_CDC2017_L2, new_Agent_harnessing_L2, new_Agent_harnessing_L2_quantize,new_Agent_harnessing_L2_x_quantize
from make_communication import Communication
from problem import Lasso_problem, New_Lasso_problem, New_Ridge_problem


class iteration_L1(object):
    def __init__(self, n, m, step, lamb, R, pattern, iterate):
        """
        :param n: int
        :param m: int
        :param lamb: float
        :return: float,float,float
        """
        self.n = n
        self.m = m
        self.step = step
        self.lamb = lamb
        self.R = R
        self.pattern = pattern
        self.iterate = iterate

        self.main()

    def optimal(self):  # L1
        """
        :return:  float, float
        """
        self.p = [np.random.randn(self.m) for i in range(self.n)]
        # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
        self.p_num = np.array(self.p)
        # np.reshape(p)
        prob = Lasso_problem(self.n, self.m, self.p_num, self.lamb, self.R)
        prob.solve()
        x_opt = np.array(prob.x.value)  # 最適解
        x_opt = np.reshape(x_opt, (-1,))  # reshape
        f_opt = prob.send_f_opt()
        return x_opt, f_opt

    def main(self):
        self.x_opt, self.f_opt = self.optimal()
        print('最適解計算')
        self.P, self.P_history = self.make_communication_graph()
        print('通信グラフ作成')
        f_error_history = [[] for i in range(self.pattern)]
        for agent in range(self.pattern):
            f_error_history[agent] = self.iteration(agent)
        print('計算終了')
        print('finish')

        self.make_graph(f_error_history)

    def make_graph(self, f_error):
        label = ['DSM', 'Proposed']
        line = ['-', '-.']
        for i in range(self.pattern):
            # stepsize = '_s(k)=' + str(self.step[i]) + '/k+10'
            stepsize = ' c=' + str(self.step[i])
            plt.plot(f_error[i], label=label[i % 2] + stepsize, linestyle=line[i % 2], linewidth=1)
        plt.legend()
        plt.yscale('log')
        plt.xlabel('iteration $k$', fontsize=10)
        plt.ylabel('$max_{i}$ $f(x_i(k))-f^*$', fontsize=10)
        plt.show()

    def make_communication_graph(self):  # 通信グラフを作成＆保存
        weight_graph = Communication(self.n, 4, 0.3)
        weight_graph.make_connected_WS_graph()
        P = weight_graph.P
        P_history = []
        for k in range(self.iterate):  # 通信グラフを作成＆保存
            weight_graph.make_connected_WS_graph()
            P_history.append(weight_graph.P)
        return P, P_history

    def make_agent(self, pattern):  # L1専用
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(Agent(self.n, self.m, self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    Agent_moment_CDC2017(self.n, self.m, self.p[i], s, self.lamb, name=i, weight=None, R=self.R))

        return Agents

    def iteration(self, pattern):
        Agents = self.make_agent(pattern)
        f_error_history = []
        for k in range(self.iterate):
            # グラフの時間変化
            for i in range(self.n):
                Agents[i].weight = self.P_history[k][i]

            for i in range(self.n):
                for j in range(self.n):
                    x_i, name = Agents[i].send(None, None)
                    Agents[j].receive(x_i, name, None)

            for i in range(self.n):
                Agents[i].update(k)

            # x_ave = 0
            # for i in range(n):
            #     x_ave += 1.0/n * Agents[i].x_i
            f_value = []
            for i in range(self.n):
                x_i = Agents[i].x_i
                estimate_value = self.optimal_value(x_i)
                f_value.append(estimate_value)

            # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
            f_error_history.append(np.max(f_value) - self.f_opt)

        return f_error_history

    def optimal_value(self, x_i):  # L1専用
        """\
        :param x_i: float
        :param p:float
        :param n:int
        :param m:int
        :param lamb:float
        :return:float
        """

        c = np.ones(self.n)
        d = np.reshape(c, (self.n, -1))
        e = np.kron(x_i, d)
        tmp = e - self.p_num
        # p_all = np.reshape(self.p, (-1,))
        # c = np.ones(self.n)
        # d = np.reshape(c, (self.n, -1))
        # A = np.kron(d, np.identity(self.m))
        # tmp = np.dot(A, x_i) - p_all
        L1 = self.lamb * self.n * np.linalg.norm(x_i, 1)
        f_opt = 1 / 2 * (np.linalg.norm(tmp, ord='fro')) ** 2 + L1
        return f_opt


class new_iteration_L1(iteration_L1):
    def optimal(self):  # L1
        """
        :return:  float, float
        """
        self.p = [np.random.randn(self.m) for i in range(self.n)]
        self.A = np.array([np.identity(self.m) for i in range(self.n)])
        self.A += 0.1 * np.array([np.random.randn(self.m, self.m) for i in range(self.n)])
        # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
        self.p_num = np.array(self.p)
        # self.A_num = np.array(self.A)
        # np.reshape(p)
        prob = New_Lasso_problem(self.n, self.m, self.p_num, self.lamb, self.R, self.A)
        prob.solve()
        x_opt = np.array(prob.x.value)  # 最適解
        x_opt = np.reshape(x_opt, (-1,))  # reshape
        f_opt = prob.send_f_opt()
        return x_opt, f_opt

    def optimal_value(self, x_i):  # L1専用
        """\
        :param x_i: float
        :param p:float
        :param n:int
        :param m:int
        :param lamb:float
        :return:float
        """
        p = np.reshape(self.p_num, -1)
        # c = np.ones(self.n)
        # d = np.reshape(c, (self.n, -1))
        # e = np.kron(x_i,d)
        # tmp = e-self.p_num
        A_tmp = np.reshape(self.A, (-1, self.m))

        # p_all = np.reshape(self.p, (-1,))
        # c = np.ones(self.n)
        # d = np.reshape(c, (self.n, -1))
        # A = np.kron(d, np.identity(self.m))
        tmp = np.dot(A_tmp, np.array(x_i)) - p
        L1 = self.lamb * self.n * np.linalg.norm(x_i, 1)
        f_opt = 1 / 2 * (np.linalg.norm(tmp, 2)) ** 2 + L1
        return f_opt

    def make_agent(self, pattern):  # L1専用
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_moment_CDC2017(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None,
                                             R=self.R))

        return Agents


class new_iteration_L2(new_iteration_L1):
    def optimal(self):  # L1
        """
        :return:  float, float
        """
        self.p = [np.random.randn(self.m) for i in range(self.n)]
        self.A = np.array([np.identity(self.m) for i in range(self.n)])
        self.A += 0.1 * np.array([np.random.randn(self.m, self.m) for i in range(self.n)])
        # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
        self.p_num = np.array(self.p)
        # self.A_num = np.array(self.A)
        # np.reshape(p)
        prob = New_Ridge_problem(self.n, self.m, self.p_num, self.lamb, self.R, self.A)
        prob.solve()
        x_opt = np.array(prob.x.value)  # 最適解
        x_opt = np.reshape(x_opt, (-1,))  # reshape
        f_opt = prob.send_f_opt()
        return x_opt, f_opt

    def make_agent(self, pattern):
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent_L2(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_moment_CDC2017_L2(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None,
                                                R=self.R))

        return Agents

    def optimal_value(self, x_i):
        """\
        :param x_i: float
        :param p:float
        :param n:int
        :param m:int
        :param lamb:float
        :return:float
        """
        p = np.reshape(self.p_num, -1)
        # c = np.ones(self.n)
        # d = np.reshape(c, (self.n, -1))
        # e = np.kron(x_i,d)
        # tmp = e-self.p_num
        A_tmp = np.reshape(self.A, (-1, self.m))

        # p_all = np.reshape(self.p, (-1,))
        # c = np.ones(self.n)
        # d = np.reshape(c, (self.n, -1))
        # A = np.kron(d, np.identity(self.m))
        tmp = np.dot(A_tmp, np.array(x_i)) - p
        L2 = self.lamb * self.n * (np.linalg.norm(x_i, 2)) ** 2
        f_opt = 1 / 2 * (np.linalg.norm(tmp, 2)) ** 2 + L2
        return f_opt


class new_iteration_L2_harnessing(new_iteration_L2):
    def make_agent(self, pattern):
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent_L2(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_harnessing_L2(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None,
                                            R=self.R))
        return Agents

    def make_communication_graph(self):  # 通信グラフを作成＆保存
        weight_graph = Communication(self.n, 4, 0.3)
        weight_graph.make_connected_WS_graph()
        P = weight_graph.P
        P_history = []
        for k in range(self.iterate):  # 通信グラフを作成＆保存
            weight_graph.make_connected_WS_graph()
            P_history.append(weight_graph.P)
        return P, P_history


class new_iteration_L2_harnessing_quantize(new_iteration_L2_harnessing):
    def make_agent(self, pattern):
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent_harnessing_L2(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_harnessing_L2_quantize(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i,
                                                     weight=None,
                                                     R=self.R))
        return Agents

    def iteration(self, pattern):
        Agents = self.make_agent(pattern)
        f_error_history = []
        for k in range(self.iterate):
            # print(k)
            # グラフの時間変化
            for i in range(self.n):
                Agents[i].weight = self.P_history[k][i]

            for i in range(self.n):
                for j in range(self.n):
                    x_i, name = Agents[i].send(j,k)
                    Agents[j].receive(x_i, name, k)

            for i in range(self.n):
                Agents[i].update(k)

            # x_ave = 0
            # for i in range(n):
            #     x_ave += 1.0/n * Agents[i].x_i
            f_value = []
            for i in range(self.n):
                x_i = Agents[i].x_i
                estimate_value = self.optimal_value(x_i)
                f_value.append(estimate_value)

            # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
            f_error_history.append(np.max(f_value) - self.f_opt)

        return f_error_history

    def make_communication_graph(self):  # 通信グラフを作成＆保存
        weight_graph = Communication(self.n, 4, 0.3)
        weight_graph.make_connected_WS_graph()
        P = weight_graph.P
        P_history = []
        for k in range(self.iterate):  # 通信グラフを作成＆保存
            # weight_graph.make_connected_WS_graph()
            P_history.append(weight_graph.P)
        return P, P_history

class new_iteration_L2_harnessing_x_quantize(new_iteration_L2_harnessing_quantize):
    def make_agent(self, pattern):
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent_harnessing_L2(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_harnessing_L2_x_quantize(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i,
                                                       weight=None,
                                                       R=self.R))
        return Agents

if __name__ == '__main__':
    n = 50
    m = 50
    lamb = 0.1
    R = 10
    np.random.seed(0)  # ランダム値固定
    pattern = 10
    test = 2000
    step = [0.25, 0.25, 0.5, 0.5, 1., 1., 2., 2.]
    # step = [0.25, 0.5, 0.25, 0.7, 0.25, 0.8, 0.25, 0.9, 0.25, 0.95]
    # step = np.array([[0.1 *(j+1) for i in range(2)] for j in range(10)])
    step = np.reshape(step, -1)
    tmp = new_iteration_L1_paper(n, m, step, lamb, R, pattern, test)
    # tmp = new_iteration_L1_paper2(n, m, step, lamb, R, pattern, test)
