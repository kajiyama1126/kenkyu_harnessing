import numpy as np


class Agent(object):
    def __init__(self, n, m, p, step, lamb, name, weight=None, R=100000):
        self.n = n
        self.m = m
        self.p = p
        self.step = step
        self.lamb = lamb
        self.name = name
        self.weight = weight
        self.R = R
        self.x_i = np.random.rand(self.m)
        self.x_i = self.project(self.x_i)
        self.x = np.zeros([self.n, self.m])

    def subgrad(self):
        grad = self.x_i - self.p
        subgrad_l1 = self.lamb * np.sign(self.x_i)
        subgrad = grad + subgrad_l1
        return subgrad

    def send(self, j, k):
        return self.x_i, self.name

    def receive(self, x_j, name, k):
        self.x[name] = x_j

    def s(self, k):
        # return self.step/ (k + 1.0)
        return self.step / (k + 10)

    def project(self, x):
        if np.linalg.norm(x) <= self.R:
            return x
        else:
            y = (self.R / np.linalg.norm(x)) * x
            return y

    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x)
        self.x_i = self.x_i - self.s(k) * self.subgrad()
        self.x_i = self.project(self.x_i)


class Agent_L2(Agent):
    def subgrad(self):
        grad = self.x_i - self.p
        grad_l2 = 2 * self.lamb * self.x_i
        return grad + grad_l2


class new_Agent(Agent):
    def __init__(self, n, m, A, p, step, lamb, name, weight=None, R=100000):
        super(new_Agent, self).__init__(n, m, p, step, lamb, name, weight=weight, R=R)
        self.A = A

    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.p))
        subgrad_l1 = self.lamb * np.sign(self.x_i)
        subgrad = grad + subgrad_l1
        return subgrad


class new_Agent_L2(new_Agent):
    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.p))
        grad_l2 = 2 * self.lamb * self.x_i
        subgrad = grad + grad_l2
        return subgrad


class new_Agent_harnessing_L2(new_Agent_L2):
    def __init__(self, n, m, A, p, s, lamb, name, weight=None, R=1000000):
        self.A = A
        super(new_Agent_harnessing_L2, self).__init__(n, m, A, p, s, lamb, name, weight, R=R)
        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])
        self.eta = s

    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.p))
        grad_l2 = 2 * self.lamb * self.x_i
        subgrad = grad + grad_l2
        return subgrad

    def send(self, j, k):
        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name, k):
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i
        grad_bf = self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.eta *self.v_i
        self.v_i = np.dot(self.weight, self.v) +  (self.subgrad() - grad_bf)
#
class new_Agent_harnessing_L2_x_code(new_Agent_harnessing_L2):
    def __init__(self, n, m, A, p, s, lamb, name, weight=None, R=1000000):
        super(new_Agent_harnessing_L2_x_code, self).__init__(n, m, A, p, s, lamb, name, weight, R=R)
        self.Encoder = Encoder(self.n, self.m, self.x_i)
        self.Decoder = Decoder(self.n, self.m, self.x_i)
        # self.z_ij= np.zeros_like(self.x,int)
        self.xi_ij = np.zeros_like(self.x)
        self.zeta_ij = np.zeros_like(self.x)
        # self.code = Code_memory()
        # self.eta = 0.01


    def send(self, j, k):
        z = self.Encoder.x_encode(self.x_i, j, k)
        zeta = self.Encoder.v_encode(self.v_i, j, k)
        # z = self.x_i
        # zeta = self.v_i
        self.xi_ij[j] = self.Encoder.xi_update(z, j, k)
        # self.xi_ij[j] = self.x_i
        self.zeta_ij[j] = self.Encoder.zeta_update(zeta, j, k)
        # self.zeta_ij[j] = zeta
        return (z, zeta), self.name


    def receive(self, x_j, name, k):
        self.x[name] = self.Decoder.x_decode(x_j[0], name, k)
        self.v[name] = self.Decoder.v_decode(x_j[1], name, k)
        # self.v[name] = x_j[1]

    def update(self, k):
        h = 1
        self.x[self.name] = self.x_i
        self.xi_ij[self.name] = self.x_i
        self.v[self.name] = self.v_i
        self.zeta_ij[self.name] = self.v_i
        one = np.array([[1] for i in range(self.n)])
        # w = self.x - np.kron(one, self.x_i)
        w = np.array(self.x - self.xi_ij)
        # print(w)
        # if self.name == 0:
        #     print(w)
        # y = self.v - np.kron(one, self.v_i)
        y = np.array(self.v - self.zeta_ij)

        grad_bf = self.subgrad()
        self.x_i = self.x_i + h*(np.dot(self.weight, w) - self.eta * self.v_i)
        self.v_i = self.v_i + h*(np.dot(self.weight, y)) + self.subgrad() - grad_bf

        self.Encoder.G_update(k)
        self.Encoder.H_update(k)
        self.Decoder.G_update(k)
        self.Decoder.H_update(k)
class new_Agent_harnessing_L2_quantize(new_Agent_harnessing_L2):
    def __init__(self, n, m, A, p, s, lamb, name, weight=None, R=1000000):
        super(new_Agent_harnessing_L2_quantize, self).__init__(n, m, A, p, s, lamb, name, weight, R=R)
        self.Encoder = Encoder(self.n, self.m, self.x_i)
        self.Decoder = Decoder(self.n, self.m, self.x_i)
        # self.z_ij= np.zeros_like(self.x,int)
        self.xi_ij = np.zeros_like(self.x)
        self.zeta_ij = np.zeros_like(self.x)
        # self.code = Code_memory()

    def send(self, j, k):
        z = self.Encoder.x_encode(self.x_i, j, k)
        zeta = self.Encoder.v_encode(self.v_i, j, k)
        self.xi_ij[j] = self.Encoder.xi_update(z, j, k)
        self.zeta_ij[j] = self.Encoder.zeta_update(zeta, j, k)
        return (z, zeta), self.name

    def receive(self, x_j, name, k):
        self.x[name] = self.Decoder.x_decode(x_j[0], name, k)
        self.v[name] = self.Decoder.v_decode(x_j[1], name, k)

    def update(self, k):
        h = 1.0
        self.x[self.name] = self.x_i
        self.xi_ij[self.name] = self.x_i
        self.v[self.name] = self.v_i
        self.zeta_ij[self.name] = self.v_i
        one = np.array([[1] for i in range(self.n)])
        # w = self.x - np.kron(one, self.x_i)
        w = np.array(self.x -self.xi_ij)
        # print(w)
        # if self.name == 0:
        #     print(w)
        # y = self.v - np.kron(one, self.v_i)
        y = np.array(self.v - self.zeta_ij)

        grad_bf = self.subgrad()
        self.x_i = self.x_i + h*(np.dot(self.weight, w) - self.eta * self.v_i)
        self.v_i = self.v_i + h*(np.dot(self.weight, y)) + self.subgrad() - grad_bf

        self.Encoder.G_update(k)
        self.Encoder.H_update(k)
        self.Decoder.G_update(k)
        self.Decoder.H_update(k)

class Coder(object):
    def __init__(self):
        self.eta = 0.97
        self.eta2 = 0.97
        self.G = 1.0
        self.H = 1.0

    def G_update(self,k):
        self.G = self.G * self.eta

    def H_update(self,k):
        self.H = self.H * self.eta2
    #
    # def G_update(self,k):
    #     self.G = 10/(k+10)
    #
    # def H_update(self,k):
    #     self.H = 10 / (k + 10)
    #
    def quantize(self, x_i):
        tmp = np.around(x_i)
        # print(max(tmp))
        return tmp

class Encoder(Coder):
    def __init__(self, n, m, x_i):
        super(Encoder,self).__init__()
        self.xi = np.zeros([n, m])
        self.zeta = np.zeros([n, m])

    def x_encode(self, x_i, j, k):
        tmp = 1.0 / (self.G) * (x_i - self.xi[j])
        return self.quantize(tmp)

    def v_encode(self, v_i, j, k):
        tmp = 1.0 / (self.H) * (v_i - self.zeta[j])
        return self.quantize(tmp)

    def xi_update(self, z, j, k):
        self.xi[j] = (self.G ) * z + self.xi[j]
        return self.xi[j]

    def zeta_update(self, z, j, k):
        self.zeta[j] = (self.H) * z + self.zeta[j]
        return self.zeta[j]

class Decoder(Coder):
    def __init__(self, n, m, x_i):
        super(Decoder, self).__init__()
        self.x_Q = np.zeros([n, m])
        self.v_Q = np.zeros([n, m])

    def x_decode(self, x, name, k):
        self.x_Q[name] = (self.G) * x + self.x_Q[name]
        return self.x_Q[name]

    def v_decode(self, x, name, k):
        self.v_Q[name] = (self.H) * x + self.v_Q[name]
        return self.v_Q[name]


class new_Agent_moment_CDC2017(new_Agent):
    def __init__(self, n, m, A, p, step, lamb, name, weight=None, R=100000):
        super(new_Agent_moment_CDC2017, self).__init__(n, m, A, p, step, lamb, name, weight, R)
        self.gamma = 0.9
        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])

    def send(self, j, k):
        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name, k):
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma * np.dot(self.weight, self.v) + self.s(k) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i
        self.x_i = self.project(self.x_i)


class new_Agent_moment_CDC2017_L2(new_Agent_moment_CDC2017):
    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.p))
        grad_l2 = 2 * self.lamb * self.x_i
        subgrad = grad + grad_l2
        return subgrad


class Agent_moment_CDC2017(Agent):
    def __init__(self, n, m, p, step, lamb, name, weight=None, R=100000):
        super(Agent_moment_CDC2017, self).__init__(n, m, p, step, lamb, name, weight, R)
        self.gamma = 0.9

        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])

    def send(self, j, k):
        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name, k):
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma * np.dot(self.weight, self.v) + self.s(k) * (0.1) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i
        self.x_i = self.project(self.x_i)
