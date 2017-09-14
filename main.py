import numpy as np

from iteration import new_iteration_L2_harnessing_code,new_iteration_L2_harnessing_x_code

if __name__ == '__main__':
    n = 50
    m = 20
    lamb = 0.1
    R = 400
    np.random.seed(0)  # ランダム値固定
    pattern = 2
    test = 5000
    # step = [0.25, 0.25, 0.5, 0.5, 1., 1., 2., 2.]
    # step = [2.0,0.2,5.0,0.5,10.0,1.0,20.0,2.0]
    # step = [0.5, 0.5, 0.5, 0.7, 0.5, 0.9, 0.5, 0.99]
    step = [1.0,0.05]
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
        tmp = new_iteration_L2_harnessing_x_code(n, m, step, lamb, R, pattern, test)

    print('finish2')