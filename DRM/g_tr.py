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

# second derivatives for Neumann–type data
y_xx_sym = diff(y_sym, x1, 2)
y_xy_sym = diff(y_sym, x1, x2)
y_yy_sym = diff(y_sym, x2, 2)

# lambdify all
ldy     = lambdify((x1, x2), y_sym,     'numpy')
ldf     = lambdify((x1, x2), f_sym,     'numpy')
ldy_xx  = lambdify((x1, x2), y_xx_sym,  'numpy')
ldy_xy  = lambdify((x1, x2), y_xy_sym,  'numpy')
ldy_yy  = lambdify((x1, x2), y_yy_sym,  'numpy')


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

        # second derivatives
        u_xx = ldy_xx(x, y)
        u_xy = ldy_xy(x, y)   # = u_yx
        u_yy = ldy_yy(x, y)

        # [D^2u·n]·n = u_xx n1^2 + 2 u_xy n1 n2 + u_yy n2^2
        neumann_vals.append(
            u_xx*(n1**2) 
          + 2*u_xy*(n1*n2) 
          + u_yy*(n2**2)
        )

    # pack into column‐vectors and return both
    return from_seq_to_array([ybdry_vals, neumann_vals])
