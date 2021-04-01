import tensorflow as tf
import numpy as np
import time
import meshzoo
import matplotlib.pyplot as plt
from Utilities_epinn import adjacency_mat, poly_nk, convert_sparse_matrix_to_sparse_tensor, plot_mesh_3D, fibonacci_sphere
import sympy as sp
from scipy import sparse
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
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
        self.N_var = self.amat.row.shape[0]
        self.p_data_hard = []
        self.dp_data_hard = []
        self.p_data_soft =[]
        self.dp_data_soft = []
        # generate polynomials
        tmp = []
        for i in self.deg_hard:
            tmp.append(poly_nk(3, i))
        self.poly_hard_list = np.vstack(tmp)
        tmp = []
        for i in self.deg_soft:
            tmp.append(poly_nk(3, i))
        self.poly_soft_list = np.vstack(tmp)

        for k in self.poly_hard_list:
            poly_fcn, dpoly_fcn = self.f_Lap_Bet(k)
            self.p_data_hard.append(poly_fcn(self.P[:, 0:1], self.P[:, 1:2], self.P[:, 2:3])+self.P[:,0:1]*0.)
            self.dp_data_hard.append(dpoly_fcn(self.P[:, 0:1], self.P[:, 1:2] , self.P[:, 2:3])+self.P[:,0:1]*0.)
        self.n1 = len(self.p_data_hard)

        for k in self.poly_soft_list:
            poly_fcn, dpoly_fcn = self.f_Lap_Bet(k)
            self.p_data_soft.append(poly_fcn(self.P[:, 0:1], self.P[:, 1:2] , self.P[:, 2:3])+self.P[:,0:1]*0.)
            self.dp_data_soft.append(dpoly_fcn(self.P[:, 0:1], self.P[:, 1:2] , self.P[:, 2:3])+self.P[:,0:1]*0.)
        self.n2 = len(self.p_data_soft)

        self.make_der_sparsity()

    def f_Lap_Bet(self, a):
        xv, yv, zv = sp.symbols('x, y, z')
        fcn_sp = (xv**a[0])*(yv**a[1])*(zv**a[2])/(sp.sqrt(xv**2+yv**2+zv**2)**(a[0]+a[1]+a[2]))
        res_sp = (1-xv**2)*sp.diff(fcn_sp, xv, 2) + (1-yv**2)*sp.diff(fcn_sp, yv, 2)\
                 + (1-zv**2)*sp.diff(fcn_sp, zv, 2) \
                 - 2*xv*yv*sp.diff(fcn_sp, xv, yv) - 2*yv*zv*sp.diff(fcn_sp, yv, zv) \
                 - 2*xv*zv*sp.diff(fcn_sp, xv, zv) \
                 - 4*xv*sp.diff(fcn_sp, xv) - 4*yv*sp.diff(fcn_sp, yv) - 4*zv*sp.diff(fcn_sp, zv)\
                 - fcn_sp
        return sp.lambdify([xv, yv, zv], fcn_sp), sp.lambdify([xv, yv ,zv], sp.simplify(res_sp))

    def loss_fcn(self, x):
        A = sparse.coo_matrix((x, (self.amat.row, self.amat.col)), shape=(self.Np, self.Np))
        p = []
        for i in range(self.n1):
            p.append(self.w_A_hard*((A.dot(self.p_data_hard[i]))-self.dp_data_hard[i]))
        for i in range(self.n2):
            p.append(self.w_A_soft*((A.dot(self.p_data_soft[i]))-self.dp_data_soft[i]))
        return np.concatenate(p).flatten()

    def make_der_sparsity(self):
        tmp0 = self.amat.todok()
        dict0 = {k: v for (k, v) in zip(tmp0.keys(), list(range(len(tmp0.keys()))))}
        sub_block = sparse.dok_matrix((self.Np, self.N_var), dtype=int)
        for (i, j) in dict0.keys():
            sub_block[i, dict0[(i, j)]] = 1
        tmp = []
        for i in range(self.n1+self.n2):
            tmp.append(sub_block)
        self.fcn_der = sparse.vstack(tmp)

    def solve(self):
        x0 = np.ones(self.N_var)*100.
        res = least_squares(self.loss_fcn, x0, jac_sparsity=self.fcn_der, verbose=2)
        self.x = res.x
        self.A = sparse.coo_matrix((res.x, (self.amat.row, self.amat.col)), shape=(self.Np, self.Np))

    def test(self):
        A = self.A
        val = []
        fcn = lambda x: np.sum(np.abs(x))
        for k in self.poly_hard_list:
            poly_fcn, dpoly_fcn = self.f_Lap_Bet(k)
            val.append(fcn(A.dot(poly_fcn(self.P[:, 0:1], self.P[:, 1:2], self.P[:, 2:3])+self.P[:,0:1]*0.)-dpoly_fcn(self.P[:, 0:1], self.P[:, 1:2], self.P[:, 2:3])+self.P[:,0:1]*0.))
        print(val)

class ePINN:
    # Initialize the class
    def __init__(self, **kwargs):
        self.lb, self.ub = kwargs['lbub']
        self.P, self.T = kwargs['PT']
        self.layers = kwargs['layers']
        self.u_exa = kwargs['u_exa']
        self.rhs_exa = kwargs['rhs_exa']
        self.A = kwargs['A']

        self.Np = P.shape[0]
        self.weights, self.biases = self.initialize_NN(layers)

        self.A_tensor = convert_sparse_matrix_to_sparse_tensor(self.A)

        # tf Placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.rhs_tf = tf.placeholder(tf.float32, shape=[None, 1])


        self.x_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.z_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # tf Graphs
        self.u_pred = self.net_u(self.x_pred_tf, self.y_pred_tf, self.z_pred_tf)
        self.f_pred = self.net_f(self.x_tf, self.y_tf, self.z_tf)

        # loss for PDE
        MSE = lambda x: tf.reduce_mean(tf.square(x))
        self.loss = MSE(self.f_pred - self.rhs_tf)

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
            H = tf.sin(np.pi*tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_A(self, u):
        return tf.sparse_tensor_dense_matmul(self.A_tensor, u)

    def net_u(self, x, y, z):
        X = tf.concat([x, y, z], 1)
        u = self.neural_net(X, self.weights, self.biases)
        return u

    def net_f(self, x, y, z):
        u = self.net_u(x, y, z)
        f = self.net_A(u)
        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):
        tf_dict = {self.x_tf: self.P[:, 0:1], self.y_tf: self.P[:, 1:2],
                   self.z_tf: self.P[:, 2:3],
                   self.rhs_tf: self.rhs_exa(self.P[:, 0:1], self.P[:, 1:2], self.P[:, 2:3])}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer_BFGS.minimize(self.sess,
                                     feed_dict = tf_dict,
                                     fetches = [self.loss],
                                     loss_callback = self.callback)


    def predict(self, xyz):
        tf_dict = {self.x_pred_tf: xyz[:, 0:1], self.y_pred_tf: xyz[:, 1:2],
                   self.z_pred_tf: xyz[:, 2:3]}
        u = self.sess.run(self.u_pred, tf_dict)
        return u

if __name__ == "__main__":
    # Domain bounds
    r = 1.
    xa, xb, ya, yb, za, zb = -r, r, -r, r, -r, r
    lb = np.array([xa, ya, za])
    ub = np.array([xb, yb, zb])
    # geometry
    P, T = meshzoo.icosa_sphere(40)
    # plot_mesh_3D(P, T)
    # adjacency matrix
    amat = adjacency_mat(T)
    # degree of polynomials to train the finite difference operator
    deg_hard = [0, 1]    # hard constraints
    deg_soft = [2]    # soft constraints
    w_A_hard = 20
    w_A_soft = 1

    layers = [3, 100, 100, 100, 1]
    x_var, y_var, z_var = sp.symbols('x, y, z')

    u_exa = lambda x, y, z: x*np.sin(y) + z
    rhs_exa = lambda x, y, z: -(3*z + 4*x*y*np.cos(y) + x*(4 - y**2)*np.sin(y))
    # u_exa = lambda x, y, z: x*y*z
    # rhs_exa = lambda x, y, z: -12*x*y*z
    kwargs_A = {'PT':[P, T], 'amat': amat, 'deg': [deg_hard, deg_soft],
                'w_A': [w_A_hard, w_A_soft]}

    op_A=operator(**kwargs_A)
    start_time = time.time()
    op_A.solve()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    op_A.test()

    kwargs_PDE = {'lbub': [lb, ub], 'PT':[P, T], 'layers': layers, 'u_exa': u_exa,
                  'rhs_exa': rhs_exa, 'A':op_A.A}
    model = ePINN(**kwargs_PDE)

    start_time = time.time()
    model.train(5000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    xyz, _ = fibonacci_sphere(samples=10000)
    # xyz = P
    u_ref = u_exa(xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3])
    u_ref = u_ref.reshape((-1, 1))
    u_pred = model.predict(xyz)
    u_pred = u_pred.reshape((-1, 1))
    # e_u = np.linalg.norm(u_pred - u_ref, 2) / np.linalg.norm(u_ref, 2)
    e_u = np.linalg.norm(u_pred - u_ref, 2)/np.sqrt(xyz.shape[0])
    np.savetxt('xyz', xyz)
    np.savetxt('u', u_pred)

    print('error of u: %.6e' % e_u)

    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=u_pred.flatten(), cmap='jet')
    ax.set_aspect(1.)
    plt.show()