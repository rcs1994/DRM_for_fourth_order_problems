import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
import g_tr as gt


N = 6000
dataname = '5000pts'



domain_data_x = uniform.rvs(size=N)
domain_data_y = uniform.rvs(size=N)

domain_data = np.array([domain_data_x,domain_data_y]).T
print(domain_data.shape)
#print("domain_data\n", domain_data)


Nb = 2000

def generate_random_bdry(Nb):
    '''
    Generate random boundary points.
    '''
    bdry_col = uniform.rvs(size=Nb*2).reshape([Nb,2])
    for i in range(Nb):
        randind = np.random.randint(0,2)
        if bdry_col[i,randind] <= 0.5:
            bdry_col[i,randind] = 0.0
        else:
            bdry_col[i,randind] = 1.0

    return bdry_col


def compute_normals(bdry_col, eps=1e-8):
    """
    Given bdry_col of shape (Nb,2) with points on the edges of [0,1]^2,
    returns two arrays n1,n2 of shape (Nb,1) giving the outward unit normals.

    Assumes:
      - if x ≈ 0 → normal = (-1, 0)
      - if x ≈ 1 → normal = ( 1, 0)
      - if y ≈ 0 → normal = ( 0,-1)
      - if y ≈ 1 → normal = ( 0, 1)
    """
    x = bdry_col[:, 0]
    y = bdry_col[:, 1]

    n1 = np.zeros_like(x)
    n2 = np.zeros_like(y)

    # left edge x=0
    mask = np.isclose(x, 0.0, atol=eps)
    n1[mask] = -1.0;  n2[mask] =  0.0

    # right edge x=1
    mask = np.isclose(x, 1.0, atol=eps)
    n1[mask] =  1.0;  n2[mask] =  0.0

    # bottom edge y=0
    mask = np.isclose(y, 0.0, atol=eps)
    n1[mask] =  0.0;  n2[mask] = -1.0

    # top edge y=1
    mask = np.isclose(y, 1.0, atol=eps)
    n1[mask] =  0.0;  n2[mask] =  1.0

    # reshape to column vectors
    return n1.reshape(-1,1), n2.reshape(-1,1)

bdry_col = generate_random_bdry(Nb)
n1_np, n2_np = compute_normals(bdry_col)
normal_vec = np.hstack([n1_np, n2_np])
#normal_vec = np.array([n1_np,n2_np]).reshape(Nb,2)

print(normal_vec.shape)
print(bdry_col.shape)
# print("normal_vec\n", normal_vec) 
# print("bdry_col\n",bdry_col)




if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)
    pkl.dump(normal_vec,pfile)


ygt,fgt = gt.data_gen_interior(domain_data)
dirichlet_data, Neumann_data = gt.data_gen_bdry(bdry_col,normal_vec)

# print("ygt\n",ygt)
#print("fgt[50:100]=\n",fgt[50:100])
# print("dirichlet_data\n",dirichlet_data)
# print("Neumann_data\n",Neumann_data)




with open("dataset/gt_on_{}".format(dataname),'wb') as pfile:
    pkl.dump(ygt,pfile)
    pkl.dump(fgt,pfile)
    pkl.dump(dirichlet_data,pfile)
    pkl.dump(Neumann_data,pfile)

   