from scipy import sparse
import numpy as np
from itertools import permutations, product
from scipy.special import comb
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

def vet_polygon(n, r=1):
# return an n*2 array of a vertexes of polygon
    vetx_arr = []
    for i in range(n):
        vetx_arr.append([r*np.cos(2*np.pi/n*i), r*np.sin(2*np.pi/n*i)])
    return vetx_arr	# return point class and array

def point_inside_polygon(x,y,poly):
    n = len(poly)
    inside =False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

def fibonacci_sphere(samples=1, randomize=True, R=1.):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    nor = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([R*x,R*y,R*z])
        nor.append([x,y,z])

    return np.array(points), np.array(nor)

def cart2sph(xyz):
    # lb = np.array([0, -np.pi])
    # ub = np.array([np.pi, np.pi])
    # theta_phi = lb + (ub - lb) * np.random.rand(100000, 2)
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def get_E(T):
    Ne = T.shape[0]
    res = []
    for i in range(Ne):
        for k in permutations(T[i, :], 2):
            res.append(k)
    return np.unique(np.sort(np.array(res), 1), axis=0)

def get_mesh_size(P, var, choice='T'):
    if choice == 'T':
        E = get_E(var)
    else:
        E = var
    res = []
    for e in E:
        res.append(np.linalg.norm(P[e[0]] - P[e[1]], 2))
    return np.array(res)

def adjacency_mat(T):
    # create adjacency matrix given simplex T
    # return a sparse matrix
    Np = np.max(T)+1
    Ne = T.shape[0]
    res = sparse.dok_matrix((Np, Np))

    for i in range(Ne):
        for k in permutations(T[i, :], 2):
            res[k[0], k[1]] = 1
    for i in range(Np):
        res[i, i] = 1

    # y = res.tocoo()
    # y.data = y.data.astype(np.float32)
    y = res
    return y


def sparse_block(smat, r, c):
    if type(r[0])==np.bool_:
        r0, c0 = np.where(r)[0], np.where(c)[0]
    else:
        r0, c0 = np.array(r), np.array(c)
    tmp = smat.todok()[r0, :]
    return (tmp[:, c0]).tocoo()


def convert_sparse_matrix_to_sparse_tensor(X, dtype=np.float32):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data.astype(dtype), coo.shape)

def poly_nk(n, k):
    ans = []
    def deep(n, k,st=""):
        if n != 1:
            for a in range(k+1):
                t = st + ","+str(a)
                deep(n-1,k-a,t)
        else:
            t = st + ","+str(k)
            ans.append(t[1:])
    deep(n, k)
    for i in range(len(ans)):
        ans[i] = ans[i].split(",")
    return np.array(ans).astype(np.int64)

def num_poly_nk(n, k):
    return comb(n + k - 1, k, True)

def cum_poly_nk(n, k):
    return comb(n + k, k, True)

def tri_area_2D(ver):
    # ver = [[[x1,y1],[x2,y2],[x3,y3]],...]
    edges = ver[:, 0:-1, :] - ver[:, 1:, :]
    return 0.5*np.abs(np.linalg.det(edges))

def poly_bnd_2D(bnd, pts, eps=1e-16):
    Nv = bnd.shape[0]
    Np, dim = pts.shape
    pts_tmp = np.array([pts]).reshape((Np, 1, dim))
    ans = np.zeros((Np, 1))
    for i in range(Nv):
        i1 = (i+1) % Nv
        tmp = np.vstack((bnd[i, :], bnd[i1, :]))
        bnd_line = np.tile(tmp, (Np, 1, 1))
        tris_tmp = np.concatenate((bnd_line, pts_tmp), axis=1)
        area_val = tri_area_2D(tris_tmp).reshape((-1, 1))
        val = np.where(area_val < eps, True, False)
        ans = np.logical_or(val, ans)
    return ans.flatten()

def plot_mesh(P, T, bnd=None, with_labels = False, eps=1e-16):
    amat = adjacency_mat(T).tocoo()
    dict = {k: v for (k, v) in zip(list(zip(amat.row, amat.col)), list(amat.data))}
    G = nx.Graph()
    G.add_nodes_from(nx.path_graph(P.shape[0]))
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if dict.get((i, j)) is not None:
                G.add_edge(i, j)

    if bnd is None:
        c_list = 'blue'
    else:
        bnd_P = poly_bnd_2D(bnd, P, eps=eps)
        c_list = np.where(bnd_P, 'red', 'blue')
    nx.draw(G, P, node_size=5, node_color=c_list, with_labels = with_labels)

    plt.show()

def tet_vol_3D(ver):
    # ver = [[[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], ...]
    edges = ver[:, 1:, :] - ver[:, 0:1, :]
    return np.abs(np.linalg.det(edges))/6

def poly_bnd_3D(bnd_P, bnd_T, pts, eps=1e-16):
    Nv = bnd_T.shape[0]
    Np, dim = pts.shape
    pts_tmp = np.array([pts]).reshape((Np, 1, dim))
    ans = np.zeros((Np, 1))
    for i in range(Nv):
        t1, t2, t3 = bnd_T[i, :]
        tmp = np.vstack((bnd_P[t1, :], bnd_P[t2, :], bnd_P[t3, :]))
        bnd_surf = np.tile(tmp, (Np, 1, 1))
        tets_tmp = np.concatenate((bnd_surf, pts_tmp), axis=1)
        vol_val = tet_vol_3D(tets_tmp).reshape((-1, 1))
        val = np.where(vol_val < eps, True, False)
        ans = np.logical_or(val, ans)
    return ans.flatten()

def plot_mesh_3D(P, var, bnd=None, choice='T', linewidth=0.5, marker_size=3, eps=1e-16):
    if choice == 'T':
        E = get_E(var)
    else:
        E = var
    lines = []
    for i in range(E.shape[0]):
        lines.append([tuple(P[E[i, 0], :]), tuple(P[E[i, 1], :])])
    lc = Line3DCollection(lines, colors='black', linewidth=linewidth)
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.add_collection3d(lc)

    if bnd is None:
        c_list = 'blue'
    else:
        bnd_P = poly_bnd_3D(bnd[0], bnd[1], P, eps=eps)
        c_list = np.where(bnd_P, 'red', 'blue')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=marker_size, c=c_list)

    xa, ya, za = np.min(P, axis=0)
    xb, yb, zb = np.max(P, axis=0)
    ax.set_xlim(xa, xb)
    ax.set_ylim(ya, yb)
    ax.set_zlim(za, zb)

    plt.show()

def poly_bnd_3D_convex(pts, eps=1e-16):
    hull = ConvexHull(pts)
    bnd_P, bnd_T = [hull.points, hull.simplices]
    Nv = bnd_T.shape[0]
    Np, dim = pts.shape
    pts_tmp = np.array([pts]).reshape((Np, 1, dim))
    ans = np.zeros((Np, 1))
    for i in range(Nv):
        t1, t2, t3 = bnd_T[i, :]
        tmp = np.vstack((bnd_P[t1, :], bnd_P[t2, :], bnd_P[t3, :]))
        bnd_surf = np.tile(tmp, (Np, 1, 1))
        tets_tmp = np.concatenate((bnd_surf, pts_tmp), axis=1)
        vol_val = tet_vol_3D(tets_tmp).reshape((-1, 1))
        val = np.where(vol_val < eps, True, False)
        ans = np.logical_or(val, ans)
    return ans.flatten()

def plot_mesh_3D_convex(P, var, choice='T', linewidth=0.5, marker_size=3, eps=1e-16):
    hull = ConvexHull(P)
    bnd = [hull.points, hull.simplices]
    if choice == 'T':
        E = get_E(var)
    else:
        E = var
    lines = []
    for i in range(E.shape[0]):
        lines.append([tuple(P[E[i, 0], :]), tuple(P[E[i, 1], :])])
    lc = Line3DCollection(lines, colors='black', linewidth=linewidth)
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.add_collection3d(lc)

    if bnd is None:
        c_list = 'blue'
    else:
        bnd_P = poly_bnd_3D(bnd[0], bnd[1], P, eps=eps)
        c_list = np.where(bnd_P, 'red', 'blue')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=marker_size, c=c_list)

    xa, ya, za = np.min(P, axis=0)
    xb, yb, zb = np.max(P, axis=0)
    ax.set_xlim(xa, xb)
    ax.set_ylim(ya, yb)
    ax.set_zlim(za, zb)
    ax.set_aspect(1)


    plt.show()

class Degree_nk:

    def __init__(self, dim):
        self.dim = dim

        self.L = 0

        self.last_degrees = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        dim = self.dim

        if self.L == 0:
            degrees = np.array([np.zeros(dim, dtype=int)])
        else:

            degrees = []

            mask0 = np.ones(len(self.last_degrees[0]), dtype=bool)
            if self.L > 1:
                mask1 = np.ones(len(self.last_degrees[1]), dtype=bool)

            for i in range(dim):
                idx0 = self.last_degrees[0][mask0][:, i]

                if self.L > 1:
                    idx1 = self.last_degrees[1][mask1][:, i]
                    yy = idx1 + 1 > 0

                deg = self.last_degrees[0][mask0]
                deg[:, i] += 1
                degrees.append(deg)
                # mask is True for all entries where the first `i` degrees are 0
                mask0 &= self.last_degrees[0][:, i] == 0
                if self.L > 1:
                    mask1 &= self.last_degrees[1][:, i] == 0

            degrees = np.concatenate(degrees)


        self.last_degrees[1] = self.last_degrees[0]
        self.last_degrees[0] = degrees
        self.L += 1

        return degrees