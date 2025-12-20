import numpy as np
from sympy import symbols, diff, sin, pi, lambdify

# define symbols
x1, x2 = symbols('x1 x2')

# exact solution
#y_sym = x1**2*x2**2*(1-x1)**2*(1-x2)**2
y_sym = (1/(2*pi**2))*sin(pi*x1)*sin(pi*x2) 
#y_sym = x1*x2*(1-x1)*(1-x2)


# source term f = -Δy
laplacian_y = diff(y_sym, x1, 2) + diff(y_sym, x2, 2)
bilaplacian_y = diff(laplacian_y, x1, 2) + diff(laplacian_y, x2, 2)

f_sym = bilaplacian_y 

#print("bilaplacian_y",bilaplacian_y)

# # second derivatives for Neumann–type data
# y_x_sym = diff(y_sym, x1, 2)
# #y_xy_sym = diff(y_sym, x1, x2)
# y_y_sym = diff(y_sym, x2, 2)

# lambdify all  
ldy     = lambdify((x1, x2), y_sym,     'numpy')
ldf     = lambdify((x1, x2), f_sym,     'numpy')
# ldy_x  = lambdify((x1, x2), y_x_sym,  'numpy')
# #ldy_xy  = lambdify((x1, x2), y_xy_sym,  'numpy')
# ldy_y  = lambdify((x1, x2), y_y_sym,  'numpy')
ldy_laplacian_y = lambdify((x1, x2), laplacian_y, 'numpy')


def from_seq_to_array(items):
    out = []
    for item in items:
        out.append(np.array(item).reshape(-1, 1))
    if len(out) == 1:
        return out[0]
    return out


def data_gen_interior(collocations):
    y_gt = [ldy(x, y) for x, y in collocations]
    f_gt = [ldf(x, y) for x, y in collocations]
    return from_seq_to_array([y_gt, f_gt])


def data_gen_bdry(collocations, normal_vec):
    """
    collocations:   array-like of shape (Nb,2) with boundary pts (x,y)
    normal_vec:     array-like of shape (Nb,2) with outward normals (n1,n2)
    returns:        [u(x,y), (D^2u·n)·n] each of shape (Nb,1)
    """
    ybdry_vals       = []
    neumann_vals = []

    for (x, y), (n1, n2) in zip(collocations, normal_vec):
        # function value
        ybdry_vals.append(ldy(x, y))
        neumann_vals.append(ldy_laplacian_y(x, y))



    # pack into column‐vectors and return both
    return from_seq_to_array([ybdry_vals, neumann_vals])
