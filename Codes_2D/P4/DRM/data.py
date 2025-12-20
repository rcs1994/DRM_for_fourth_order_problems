import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
import g_tr as gt


N = 8000
dataname = '5000pts'



domain_data_x = uniform.rvs(size=N)
domain_data_y = uniform.rvs(size=N)

domain_data = np.array([domain_data_x,domain_data_y]).T
print(domain_data.shape)


Nb = 3000

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

print(bdry_col.shape,normal_vec.shape)
# print("n1_np\n",n1_np)
# print("n2_np\n",n2_np)
# print("bdry_col\n",bdry_col)
# print("normal_vec\n",normal_vec)





if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)
    pkl.dump(normal_vec,pfile)


ygt,fgt = gt.data_gen_interior(domain_data)
g_1, g_2 = gt.data_gen_bdry(bdry_col,normal_vec)



with open("dataset/gt_on_{}".format(dataname),'wb') as pfile:
    pkl.dump(ygt,pfile)
    pkl.dump(fgt,pfile)
    pkl.dump(g_1,pfile)
    pkl.dump(g_2,pfile)

   