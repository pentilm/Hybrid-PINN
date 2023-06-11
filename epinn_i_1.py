# Hybird PINN for Inverse Problem
# Author: Zhiwei Fang
# Email: zhiweifang1987@gmail.com
import tensorflow as tf
import numpy as np
import time
import dmsh
import matplotlib.pyplot as plt
from Utilities_epinn import adjacency_mat, poly_nk, poly_bnd_2D, sparse_block, convert_sparse_matrix_to_sparse_tensor, plot_mesh
from scipy import sparse
from scipy.optimize import least_squares
import sympy as sp
from pyDOE import lhs
np.random.seed(1234)
tf.set_random_seed(1234)
class operator:
    def __init__(self, **kwargs):
        self.P, self.T = kwargs['PT']
        self.amat = kwargs['amat']
        self.deg_hard, self.deg_soft = kwargs['deg']
        self.w_A_hard, self.w_A_soft = kwargs['w_A']

        self.Np = P.shape[0]
        ver = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])
        self.inp = np.logical_not(poly_bnd_2D(ver, P))  # marker for interior points
        self.Ninp = np.sum(self.inp)
        self.in_P = self.P[self.inp, :]
        self.amat_sub = sparse_block(self.amat, list(self.inp), [True]*self.Np)
        self.N_var = self.amat_sub.row.shape[0]
        self.p_data_hard = []
        self.dp_data_hard = []
        self.p_data_soft =[]
        self.dp_data_soft = []
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

        self.dpoly_fcn = lambda x, y, a, p: (-a[0]*(a[0]-1)*(x-p[0])**(a[0]-2)*(y-p[1])**a[1] if a[0] >= 2 else 0.) \
                                            + (-a[1]*(a[1]-1)*(x-p[0])**a[0]*(y-p[1])**(a[1]-2) if a[1] >= 2 else 0.)

        for k in self.poly_hard_list:
            p, dp = [], []
            for pts in self.in_P:
                p.append(self.poly_fcn(self.P[:, 0:1], self.P[:, 1:2], k, pts).reshape((-1, 1)))
                dp.append(self.dpoly_fcn(pts[0], pts[1], k, pts))
            self.p_data_hard.append(np.hstack(p))
            self.dp_data_hard.append(np.array(dp))
        self.n1 = len(self.p_data_hard)

        for k in self.poly_soft_list:
            p, dp = [], []
            for pts in self.in_P:
                p.append(self.poly_fcn(self.P[:, 0:1], self.P[:, 1:2], k, pts).reshape((-1, 1)))
                dp.append(self.dpoly_fcn(pts[0], pts[1], k, pts))
            self.p_data_soft.append(np.hstack(p))
            self.dp_data_soft.append(np.array(dp))
        self.n2 = len(self.p_data_soft)

        self.make_der_sparsity()

    def loss_fcn(self, x):
        A = sparse.coo_matrix((x, (self.amat_sub.row, self.amat_sub.col)), shape=(self.Ninp, self.Np))
        p = []
        for i in range(self.n1):
            p.append(self.w_A_hard*((A.dot(self.p_data_hard[i])).diagonal()-self.dp_data_hard[i]))
        for i in range(self.n2):
            p.append(self.w_A_soft*((A.dot(self.p_data_soft[i])).diagonal()-self.dp_data_soft[i]))
        return np.concatenate(p)

    def make_der_sparsity(self):
        tmp0 = self.amat_sub.todok()
        dict0 = {k: v for (k, v) in zip(tmp0.keys(), list(range(len(tmp0.keys()))))}
        sub_block = sparse.dok_matrix((self.Ninp, self.N_var), dtype=int)
        for (i, j) in dict0.keys():
            sub_block[i, dict0[(i, j)]]= 1
        tmp = []
        for i in range(self.n1+self.n2):
            tmp.append(sub_block)
        self.fcn_der = sparse.vstack(tmp)

    def solve(self):
        x0 = np.ones(self.N_var)
        res = least_squares(self.loss_fcn, x0, jac_sparsity=self.fcn_der, verbose=2)
        self.x = res.x
        self.A = sparse.coo_matrix((res.x, (self.amat_sub.row, self.amat_sub.col)), shape=(self.Ninp, self.Np))

    def test(self):
        A = sparse.coo_matrix((self.x, (self.amat_sub.row, self.amat_sub.col)), shape=(self.Ninp, self.Np))
        val = []
        fcn = lambda x: np.sum(np.abs(x))
        val.append(fcn(A.dot(np.ones((self.Np,1)))))
        val.append(fcn(A.dot(self.P[:, 0:1])))
        val.append(fcn(A.dot(self.P[:, 1:2])))
        val.append(fcn(A.dot(self.P[:, 0:1]**2)+2))
        val.append(fcn(A.dot(self.P[:, 1:2]**2)+2))
        val.append(fcn(A.dot(self.P[:, 0:1]*self.P[:, 1:2])))
        print(val)



class ePINN:
    def __init__(self, **kwargs):
        self.lb, self.ub = kwargs['lbub']
        self.P, self.T = kwargs['PT']
        self.layers = kwargs['layers']
        self.u_exa = kwargs['u_exa']
        self.ux_exa = kwargs['ux_exa']
        self.uy_exa = kwargs['uy_exa']
        self.rhs_exa = kwargs['rhs_exa']
        self.A = kwargs['A']

        self.Np = P.shape[0]
        xa, ya = self.lb
        xb, yb = self.ub
        ver = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])
        self.inp = np.logical_not(poly_bnd_2D(ver, P))  # marker for interior points
        self.Ninp = np.sum(self.inp)
        self.in_P = self.P[self.inp, :]
        self.weights, self.biases = self.initialize_NN(layers)
        self.lambda_0 = tf.Variable(np.random.rand(1), dtype=tf.float32)

        self.A_tensor = convert_sparse_matrix_to_sparse_tensor(self.A)

        # tf Placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.x_in_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_in_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.rhs_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.x_bnd_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_bnd_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_bnd_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ux_bnd_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.uy_bnd_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.x_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # tf Graphs
        self.u_bnd_pred, self.ux_bnd_pred, self.uy_bnd_pred = self.net_u(self.x_bnd_tf, self.y_bnd_tf)
        self.u_pred, _, _ = self.net_u(self.x_pred_tf, self.y_pred_tf)
        self.f1_pred = self.lambda_0*self.net_f(self.x_tf, self.y_tf)
        self.f2_pred, _, _ = self.net_u(self.x_in_tf, self.y_in_tf)

        # loss for PDE
        MSE = lambda x: tf.reduce_mean(tf.square(x))
        self.loss = MSE(self.f1_pred + self.f2_pred - self.rhs_tf) \
                    + MSE(self.u_bnd_pred-self.u_bnd_tf) \
                    + MSE(self.ux_bnd_pred-self.ux_bnd_tf) \
                    + MSE(self.uy_bnd_pred-self.uy_bnd_tf)

        # Optimizers

        self.optimizer_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                   method = 'L-BFGS-B',
                                                                   options = {'maxiter': 50000,
                                                                       'maxfun': 50000,
                                                                       'maxcor': 50,
                                                                       'maxls': 50,
                                                                       'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

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
        return tf.sparse_tensor_dense_matmul(self.A_tensor, u)

    def net_u(self, x, y):
        X = tf.concat([x, y], 1)
        u = self.neural_net(X, self.weights, self.biases)
        ux = tf.gradients(u, x)[0]
        uy = tf.gradients(u, y)[0]
        return u, ux, uy

    def net_f(self, x, y):
        u, _, _ = self.net_u(x, y)
        f = self.net_A(u)
        return f

    def callback(self, loss, lambda_0):
        print('Loss: %.3e, Lambda_0: %.3e' % (loss, lambda_0))

    def train(self, nIter):
        bnd_p = self.P[np.logical_not(self.inp), :]
        tf_dict = {self.x_tf: self.P[:, 0:1], self.y_tf: self.P[:, 1:2],
                   self.x_in_tf: self.in_P[:, 0:1], self.y_in_tf: self.in_P[:, 1:2],
                   self.rhs_tf: self.rhs_exa(self.in_P[:, 0:1], self.in_P[:, 1:2]),
                   self.x_bnd_tf: bnd_p[:, 0:1], self.y_bnd_tf: bnd_p[:, 1:2],
                   self.u_bnd_tf: self.u_exa(bnd_p[:, 0:1], bnd_p[:, 1:2]),
                   self.ux_bnd_tf: self.ux_exa(bnd_p[:, 0:1], bnd_p[:, 1:2]),
                   self.uy_bnd_tf: self.uy_exa(bnd_p[:, 0:1], bnd_p[:, 1:2])}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_0_value = self.sess.run(self.lambda_0)
                print('It: %d, Lambda_0: %.3e Loss: %.3e, Time: %.2f' %
                      (it, lambda_0_value, loss_value, elapsed))
                start_time = time.time()

        self.optimizer_BFGS.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda_0],
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
    # edge_size = 0.1
    # geo = dmsh.Rectangle(xa, xb, ya, yb)
    # P, T = dmsh.generate(geo, edge_size=edge_size)
    import meshzoo
    P, T = meshzoo.rectangle(
        xmin=xa, xmax=xb,
        ymin=ya, ymax=yb,
        nx=21, ny=21,
        variant="zigzag")
    # plot_mesh(P,T,bnd=np.array(np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])))
    # adjacency matrix
    amat = adjacency_mat(T)
    # degree of polynomials to train the finite difference operator
    deg_hard = [0, 1, 2]    # hard constraints
    deg_soft = [3]    # soft constraints
    w_A_hard = 10
    w_A_soft = 1

    layers = [2, 100, 100, 100, 1]
    x_var, y_var = sp.symbols('x, y')

    k0 = 2.
    u_exa_sp = sp.exp(x_var**2)*sp.sin(y_var)
    ux_exa_sp = sp.diff(u_exa_sp, x_var)
    uy_exa_sp = sp.diff(u_exa_sp, y_var)
    rhs_exa_sp = -k0*(sp.diff(u_exa_sp, x_var, 2) + sp.diff(u_exa_sp, y_var, 2)) + u_exa_sp
    u_exa = sp.lambdify([x_var, y_var], u_exa_sp)
    ux_exa = sp.lambdify([x_var, y_var], ux_exa_sp)
    uy_exa = sp.lambdify([x_var, y_var], uy_exa_sp)
    rhs_exa = sp.lambdify([x_var, y_var], rhs_exa_sp)

    kwargs_A = {'PT':[P, T], 'amat': amat, 'deg': [deg_hard, deg_soft], 'w_A': [w_A_hard, w_A_soft]}

    op_A=operator(**kwargs_A)
    op_A.solve()
    op_A.test()

    kwargs_PDE = {'lbub': [lb, ub], 'PT':[P, T], 'layers': layers, 'u_exa': u_exa,
                  'ux_exa': ux_exa, 'uy_exa': uy_exa, 'rhs_exa': rhs_exa, 'A': op_A.A}
    model = ePINN(**kwargs_PDE)

    start_time = time.time()
    model.train(5000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    xy = lb + (ub - lb)*lhs(2, 100000)
    # xy = P
    u_ref = u_exa(xy[:, 0:1], xy[:, 1:2])
    u_ref = u_ref.reshape((-1, 1))
    u_pred = model.predict(xy)
    u_pred = u_pred.reshape((-1, 1))
    # e_u = np.linalg.norm(u_pred - u_ref, 2) / np.linalg.norm(u_ref, 2)
    e_u = np.linalg.norm(u_pred - u_ref, 2)/xy.shape[0]
    print('error of u: %.6e' % e_u)
    np.savetxt('xyz', xy)
    np.savetxt('u', u_pred)

    fig = plt.figure(0)
    plt.scatter(xy[:, 0], xy[:, 1], c=u_pred.flatten(), cmap='jet')
    plt.show()