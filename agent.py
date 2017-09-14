import numpy as np


class Agent(object):
    def __init__(self, n, m, p,step, lamb, name, weight=None,R = 100000):
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
        subgrad_l1 = self.lamb*np.sign(self.x_i)
        subgrad = grad + subgrad_l1
        return subgrad

    def send(self, j, k):
        return self.x_i, self.name

    def receive(self, x_j, name, k):
        self.x[name] = x_j

    def s(self, k):
        # return self.step/ (k + 1.0)
        return self.step/(k+10)
    def project(self,x):
        if np.linalg.norm(x) <= self.R:
            return x
        else:
            y = (self.R/np.linalg.norm(x)) * x
            return y


    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x)
        self.x_i = self.x_i- self.s(k) * self.subgrad()
        self.x_i = self.project(self.x_i)

class Agent_L2(Agent):
    def subgrad(self):
        grad = self.x_i-self.p
        grad_l2 = 2*self.lamb * self.x_i
        return grad+grad_l2

class new_Agent(Agent):
    def __init__(self, n, m, A, p, step, lamb, name, weight=None, R=100000):
        super(new_Agent,self).__init__(n,m,p,step,lamb,name,weight=weight,R=R)
        self.A = A

    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i) - self.p))
        subgrad_l1 = self.lamb*np.sign(self.x_i)
        subgrad = grad + subgrad_l1
        return subgrad

class new_Agent_L2(new_Agent):
    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i) - self.p))
        grad_l2 = 2 * self.lamb * self.x_i
        subgrad = grad + grad_l2
        return subgrad


class new_Agent_harnessing_L2(new_Agent_L2):
    def __init__(self, n, m,A, p,s, lamb, name, weight=None,R=1000000):
        self.A = A
        super(new_Agent_harnessing_L2, self).__init__(n, m, A,p, s,lamb, name, weight,R=R)
        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])
        self.eta = 0.01

    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i) - self.p))
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
        self.x_i = np.dot(self.weight, self.x) - self.v_i
        self.v_i = np.dot(self.weight, self.v) + self.eta*( self.subgrad() -grad_bf)

class new_Agent_harnessing_L2_code(new_Agent_harnessing_L2):
    def __init__(self, n, m,A, p,s, lamb, name, weight=None,R=1000000):
        super(new_Agent_harnessing_L2_code, self).__init__(n, m, A,p, s,lamb, name, weight,R=R)
        self.Encoder = Encoder(self.n,self.m,self.x_i)
        self.Decoder = Decoder(self.n,self.m,self.x_i)
        # self.z_ij= np.zeros_like(self.x,int)
        self.xi_ij= np.zeros_like(self.x)
        # self.code = Code_memory()

    def send(self, j, k):
        z = self.Encoder.encode(self.x_i,j,k)
        # self.v_i
        self.xi_ij[j] = self.Encoder.xi_update(z,j,k)
        return (z,self.v_i),self.name


    def receive(self, x_j, name, k):
        self.v[name] = x_j[1]
        self.x[name] = self.Decoder.decode(x_j[0],name,k)

    def update(self, k):
        self.x[self.name] = self.x_i
        one = np.array([[1] for i in range(self.n)])
        w = self.x-np.kron(one,self.x_i)
        self.v[self.name] = self.v_i
        grad_bf = self.subgrad()
        self.x_i = self.x_i + np.dot(self.weight, w) - self.eta*self.v_i
        self.v_i = np.dot(self.weight, self.v) +  self.subgrad() -grad_bf

class Encoder(object):
    def __init__(self,n,m,x_i):
        self.z_ij = np.zeros([n,m])
        self.eta = 0.99
        self.G = 10
        self.xi = np.zeros([n, m])

    def encode(self,x_i,j,k):
        tmp = 1.0/(self.G * self.eta**float(k)) * (x_i-self.xi[j])
        return self.quantize(tmp)

    def xi_update(self,z,j,k):
        self.xi[j] = (self.G * self.eta**float(k))*z + self.xi[j]
        return self.xi[j]

    def quantize(self,x_i):
        return np.around(x_i)
        # x = np.zeros_like(x_i,int)
        # for j in range(len(x_i)):
        #     x[j] = self.quant(x_i[j])
        #
        # return x

    def quant(self,x):
        for i in range(10000):
            if abs(x) <= (i + 1):
                if x >= 0:
                    return i
                else:
                    return -i
            if i == 9999:
                print('error')
                return

class Decoder(object):
    def __init__(self,n,m,x_i):
        self.z_ij = np.zeros([n,m],int)
        self.eta = 0.99
        self.G = 10
        self.x_Q = np.zeros([n, m])

    def decode(self,x,name,k):
        self.x_Q[name] = (self.G * self.eta**k) * x + self.x_Q[name]
        return self.x_Q[name]


# class Code_memory(object):
#     def __init__(self,n,m,x_i):
#         self.n = n
#         self.m = m
#         self.eta = 0.99
#         self.G = 100
#         self.x_i_bf = np.zeros([self.n, self.m],int)
#         self.x_j_bf = np.zeros([self.n, self.m],int)
#
#     def decode(self,x_i,name,k):
#         tmp_x_j = (x_i - self.x_j_bf[name]) / (self.G * self.eta ** k)
#
#     def encoder(self, x_j, name, k):
#         tmp_x_j = (x_j - self.x_j_bf[name]) / (self.G * self.eta ** k)
#         for i in range(self.m):
#             tmp_x_j[i] = self.decode(tmp_x_j[i])
#
#         self.x_i_bf[name] = tmp_x_j
#         return tmp_x_j
#
#     def decoder(self,x_j,name,k):
#         tmp_x_j = (x_j-self.x_j_bf[name])/(self.G * self.eta ** k)
#         for i in range(self.m):
#             tmp_x_j[i] = self.decode(tmp_x_j[i])
#
#         self.x_j_bf[name] = tmp_x_j
#         return tmp_x_j

    # def decode(self,x):
    #     for i in range(1000):
    #         if abs(x)<= (i+1):
    #             if x>= 0 :
    #                 return i
    #             else:
    #                 return -i
    #         if i == 99:
    #             print('error')
    #             break



class new_Agent_moment_CDC2017(new_Agent):
    def __init__(self, n, m,A, p,step, lamb, name, weight=None,R = 100000):
        super(new_Agent_moment_CDC2017, self).__init__(n, m,A, p,step, lamb, name, weight,R)
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
        grad = np.dot(A_to,(np.dot(self.A,self.x_i) - self.p))
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