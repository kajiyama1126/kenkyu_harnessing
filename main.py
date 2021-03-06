import numpy as np

from iteration import new_iteration_L2_harnessing_quantize,new_iteration_L2_harnessing_x_quantize

if __name__ == '__main__':
    n = 20
    m = 20
    lamb = 0.0
    R = 400
    np.random.seed(0)  # ランダム値固定
    pattern = 4
    test = 1000
    # step = [0.25, 0.25, 0.5, 0.5, 1., 1., 2., 2.]
    # step = [2.0,0.2,5.0,0.5,10.0,1.0,20.0,2.0]
    # step = [0.5, 0.5, 0.5, 0.7, 0.5, 0.9, 0.5, 0.99]
    step = [0.05,0.05,0.01,0.01]
    print(n,m,lamb,R,test)
    if pattern != len(step):
        print('error')
        pass
    else:
        # step = np.array([[0.1 *(j+1) for i in range(2)] for j in range(10)])
        step = np.reshape(step, -1)
        # tmp = new_iteration_Dist(n, m, step, lamb, R, pattern, test)
        # tmp = new_iteration_L1_paper2(n, m, step, lamb, R, pattern, test)
        # tmp = new_iteration_L2_harnessing(n, m, step, lamb, R, pattern, test)
        tmp = new_iteration_L2_harnessing_quantize(n, m, step, lamb, R, pattern, test)

    print('finish2')