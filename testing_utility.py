import numpy as np
import utility as ut
import define_SDP_cg as dsdp_cg


def testing_eigen(A, n):
    H = ds.Build_H_XXZ_full(N-2, dim=2, Delta=1)
    print('H\n', H)
    eigvals, eigvecs = ut.eigen(H, chi)
    print('eigvals =\n', eigvals)
    print('eigvecs =\n', eigvecs)
    print('eigvecs.shape =\n', eigvecs.shape)
    return eigvals, eigvecs


def testing_func_test(X):
    f = ut.func_test(X)
    return f

np.random.seed(99)
X = np.random.randint(10, size=(12))
X = X.reshape((3,4))
#f, shape = testing_func_test(X)
#print('shape of input matrix X: ', shape)
#print('trace = ', f)


def testing_create_basis_matrix(shape, index_tuple):
    return ut.create_basis_matrix(shape, index_tuple)

#E = testing_create_basis_matrix((3,4), (2,1))
#print('E with (2,1) =\n', E)

def testing_calc_derivative_test(func, index_tuple, *args):
    deriv = ut.calc_derivative_test(func, index_tuple, *args)
    return deriv

f = ut.func_test
# index (i,j)
index_tuple = (2,1)

#deriv = testing_calc_derivative_test(f, index_tuple, X)
#print('df/dx_(21) = ', deriv)





def testing_get_derivative_V(N, dim=2, chi=3):
    L, R = ut.get_derivative_V(N, dim=dim, chi=chi)
    return L, R
N = 4

L, R = testing_get_derivative_V(N, dim=2, chi=3)
print('dV_L = ', L)
print('dV_R = ', R)
print('dV_L.shape ', L[0].shape)
#L = L.flatten()

# get cg map
W = dsdp_cg.generate_linear_map(dim=2, chi=3)
V_L_list, V_R_list = dsdp_cg.get_V_matrices(N, W, dim=2, chi=3)
print('V_L_list', V_L_list)
print('V_R_list', V_R_list)
print('V_L_list[0].shape', V_L_list[0].shape)
# L is a matrix with a loooot of zeros and some non zero elements
'''
for i in range(0, len(L)):
    print(L[i])
'''





#
