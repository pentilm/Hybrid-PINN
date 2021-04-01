# Hybird PINN for Poisson's equation
# Author: Zhiwei Fang
# Email: zhiweifang1987@gmail.com

import tensorflow as tf
import numpy as np
import time
import dmsh
import matplotlib.pyplot as plt
from Utilities_epinn import adjacency_mat, poly_nk, poly_bnd_2D, sparse_block, convert_sparse_matrix_to_sparse_tensor, plot_mesh
from Geometry_Data import *
import sympy as sp
from pyDOE import lhs
np.random.seed(1234)
tf.set_random_seed(1234)

class ePINN:
    # Initialize the class
    def __init__(self, **kwargs):
        self.lb, self.ub = kwargs['lbub']
        self.P, self.T = kwargs['PT']
        self.amat = kwargs['amat']
        self.deg_hard, self.deg_soft = kwargs['deg']
        self.layers = kwargs['layers']
        self.u_exa = kwargs['u_exa']
        self.w_A_hard, self.w_A_soft = kwargs['w_A']
        self.rhs_exa = kwargs['rhs_exa']

        self.Np = P.shape[0]
        xa, ya = self.lb
        xb, yb = self.ub
        ver = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])
        self.inp = np.logical_not(poly_bnd_2D(ver, P))  # marker for interior points
        self.Ninp = np.sum(self.inp)
        self.in_P = self.P[self.inp, :]
        self.A = self.xavier_init([self.Ninp, self.Np])
        self.weights, self.biases = self.initialize_NN(layers)

        # sparse tensor method
        self.amat_tensor = convert_sparse_matrix_to_sparse_tensor(sparse_block(self.amat, list(self.inp), [True]*self.Np))
        # full tensor method
        # self.amat_tensor = tf.constant(self.amat.todense()[list(self.inp),:], dtype=tf.float32)

        # tf Placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.x_in_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_in_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.rhs_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.x_bnd_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_bnd_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_bnd_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.x_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # tf Graphs
        self.u_bnd_pred = self.net_u(self.x_bnd_tf, self.y_bnd_tf)
        self.u_pred = self.net_u(self.x_pred_tf, self.y_pred_tf)
        self.f_pred = self.net_f(self.x_tf, self.y_tf)

        # generate polynomials
        tmp = []
        for i in self.deg_hard:
            tmp.append(poly_nk(2, i))
        self.poly_hard_list = np.vstack(tmp)
        tmp = []
        for i in self.deg_soft:
            tmp.append(poly_nk(2, i))
        self.poly_soft_list = np.vstack(tmp)

        self.poly_fcn = lambda x, y, a, p: (x-p[0])**a[0]*(y-p[1])**a[1]
        # generate solutions
        self.dpoly_fcn = lambda x, y, a, p: (-a[0]*(a[0]-1)*(x-p[0])**(a[0]-2)*(y-p[1])**a[1] if a[0]>=2 else 0.)\
                                         + (-a[1]*(a[1]-1)*(x-p[0])**a[0]*(y-p[1])**(a[1]-2) if a[1]>=2 else 0.)


        # make loss
        MSE_A = lambda x: tf.reduce_sum(tf.square(x))
        self.loss_A_hard = 0
        self.loss_A_soft = 0
        for k in self.poly_hard_list:
            p, dp = [], []
            for pts in self.in_P:
                p.append(self.poly_fcn(self.P[:, 0:1], self.P[:, 1:2], k, pts).reshape((-1, 1)))
                dp.append(self.dpoly_fcn(pts[0], pts[1], k, pts))
            Ap = self.net_A(tf.constant(np.hstack(p), dtype=tf.float32))
            self.loss_A_hard += MSE_A(tf.diag_part(Ap) - tf.constant(dp, dtype=tf.float32))
        for k in self.poly_soft_list:
            p, dp = [], []
            for pts in self.in_P:
                p.append(self.poly_fcn(self.P[:, 0:1], self.P[:, 1:2], k, pts).reshape((-1, 1)))
                dp.append(self.dpoly_fcn(pts[0], pts[1], k, pts))
            Ap = self.net_A(tf.constant(np.hstack(p), dtype=tf.float32))
            self.loss_A_soft += MSE_A(tf.diag_part(Ap) - tf.constant(dp, dtype=tf.float32))
        self.loss_A = self.w_A_hard*self.loss_A_hard+self.w_A_soft*self.loss_A_soft

        # loss for PDE
        MSE_PDE = lambda x: tf.reduce_mean(tf.square(x))
        self.loss_PDE = MSE_PDE(self.f_pred-self.rhs_tf) \
                        + MSE_PDE(self.u_bnd_pred-self.u_bnd_tf)

        # Optimizers
        self.optimizer_BFGS_A = tf.contrib.opt.ScipyOptimizerInterface(self.loss_A,
                                                                       var_list = [self.A],
                                                                       method = 'L-BFGS-B',
                                                                       options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam_A = tf.train.AdamOptimizer()
        self.train_op_Adam_A = self.optimizer_Adam_A.minimize(self.loss_A, var_list=[self.A])

        self.optimizer_BFGS_PDE = tf.contrib.opt.ScipyOptimizerInterface(self.loss_PDE,
                                                                       var_list = self.weights + self.biases,
                                                                       method = 'L-BFGS-B',
                                                                       options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam_PDE = tf.train.AdamOptimizer()
        self.train_op_Adam_PDE = self.optimizer_Adam_PDE.minimize(self.loss_PDE, var_list=self.weights + self.biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_A(self, u):
        # sparse tensor
        y0 = self.amat_tensor.__mul__(self.A)
        y1 = tf.sparse_tensor_dense_matmul(y0, u)

        # full tensor
        # y0 = self.amat_tensor*self.A
        # y1 = tf.matmul(y0, u)

        return y1

    def net_u(self, x, y):
        X = tf.concat([x, y], 1)

        u = self.neural_net(X, self.weights, self.biases)

        return u

    def net_f(self, x, y):
        u = self.net_u(x, y)
        f = self.net_A(u)

        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train_A(self, nIter):
        in_p = self.P[self.inp, :]
        tf_dict = {self.x_tf: self.P[:, 0:1], self.y_tf: self.P[:, 1:2],
                   self.x_in_tf: in_p[:, 0:1], self.y_in_tf: in_p[:, 1:2]}
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam_A, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_A, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer_BFGS_A.minimize(self.sess,
                                       feed_dict = tf_dict,
                                       fetches = [self.loss_A],
                                       loss_callback = self.callback)
        # test A
        # fcn = lambda x: tf.reduce_sum(tf.abs(x))
        # print(self.sess.run([fcn(self.net_A(tf.ones((self.Np,1)))),
        #                      fcn(self.net_A(self.x_tf)),
        #                      fcn(self.net_A(self.y_tf)),
        #                      fcn(self.net_A(self.x_tf**2)+2),
        #                      fcn(self.net_A(self.y_tf**2)+2),
        #                      fcn(self.net_A(self.x_tf*self.y_tf))], tf_dict))


    def train_PDE(self, nIter):
        bnd_p = self.P[np.logical_not(self.inp), :]
        tf_dict = {self.x_tf: self.P[:, 0:1], self.y_tf: self.P[:, 1:2],
                   self.x_in_tf: self.in_P[:, 0:1], self.y_in_tf: self.in_P[:, 1:2],
                   self.rhs_tf: self.rhs_exa(self.in_P[:, 0:1], self.in_P[:, 1:2]),
                   self.x_bnd_tf: bnd_p[:, 0:1], self.y_bnd_tf: bnd_p[:, 1:2],
                   self.u_bnd_tf: self.u_exa(bnd_p[:, 0:1], bnd_p[:, 1:2])}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam_PDE, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_PDE, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer_BFGS_PDE.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss_PDE],
                                loss_callback = self.callback)


    def predict(self, xy):
        tf_dict = {self.x_pred_tf: xy[:, 0:1], self.y_pred_tf: xy[:, 1:2]}
        u = self.sess.run(self.u_pred, tf_dict)
        return u

if __name__ == "__main__":
    # Domain bounds
    xa, xb, ya, yb = 0., 1., 0., 1.
    lb = np.array([xa, ya])
    ub = np.array([xb, yb])
    # geometry
    edge_size = 0.1
    geo = dmsh.Rectangle(xa, xb, ya, yb)
    P, T = dmsh.generate(geo, edge_size=edge_size)
    # plot_mesh(P,T,bnd=np.array(np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])))
    # adjacency matrix
    amat = adjacency_mat(T)
    # degree of polynomials to train the finite difference operator
    deg_hard = [0, 1, 2]    # hard constraints
    deg_soft = [3]    # soft constraints
    w_A_hard = 10
    w_A_soft = 1

    layers = [2, 100, 100, 100, 100, 1]
    x_var, y_var = sp.symbols('x, y')

    u_exa_sp = sp.exp(x_var**2)*sp.sin(y_var)
    rhs_exa_sp = -sp.diff(u_exa_sp, x_var, 2) - sp.diff(u_exa_sp, y_var, 2)
    u_exa = sp.lambdify([x_var, y_var], u_exa_sp)
    rhs_exa = sp.lambdify([x_var, y_var], rhs_exa_sp)

    kwargs = {'lbub': [lb, ub], 'PT':[P, T], 'amat': amat, 'deg': [deg_hard, deg_soft],
              'layers': layers, 'u_exa': u_exa, 'w_A': [w_A_hard, w_A_soft],
              'rhs_exa': rhs_exa}
    model = ePINN(**kwargs)

    start_time = time.time()
    model.train_A(5000)
    model.train_PDE(5000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    # xy = lb + (ub - lb)*lhs(2, 100000)
    xy = P
    u_ref = u_exa(xy[:, 0:1], xy[:, 1:2])
    u_ref = u_ref.reshape((-1, 1))
    u_pred = model.predict(xy)
    u_pred = u_pred.reshape((-1, 1))
    # e_u = np.linalg.norm(u_pred - u_ref, 2) / np.linalg.norm(u_ref, 2)
    e_u = np.linalg.norm(u_pred - u_ref, 2)/P.shape[0]
    print('error of u: %.6e' % e_u)

    fig = plt.figure(0)
    plt.scatter(xy[:, 0], xy[:, 1], c=u_pred.flatten(), cmap='jet')
    plt.show()





