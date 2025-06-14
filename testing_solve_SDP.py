import numpy as np
import cvxpy as cp

import utility as ut
import solve_SDP as ssdp
import solve_SDP_cg as ssdp_cg
import define_SDP as dsdp
import define_SDP_cg as dsdp_cg
import define_system as ds


def testing_rhos_cp_variable(N, dim=2):
    rhos = ssdp.rhos_cp_variable(N, dim=dim)
    return rhos

# create an example of rhos
rhos = testing_rhos_cp_variable(4)
#print(rhos)
#print(rhos[0].shape[0])



def testing_Minimize_trace_rho_times_h(ham, rhos, N=4):
    prob = ssdp.Minimize_trace_rho_times_h(ham, rhos, N=N)
    return prob
'''
#ham = ds.Build_H_TFIM_Interaction_Term()
ham = ds.Build_H_XXZ(Delta=1)
#print(ham)

rhos = testing_rhos_cp_variable(4)
prob = testing_Minimize_trace_rho_times_h(ham, rhos)
#ham = ds.Build_H_TFIM_Interaction_Term()
prob1, prob2 = ssdp.compare_solutions_no_cg(4, 6, ham, dim=2)
'''

'''
print('N = 5')
rhos = testing_rhos_cp_variable(5)
prob = testing_Minimize_trace_rho_times_h(ham, rhos, N=3)
'''




#print('rho2 = ', rhos[0].value)
#rho2 = rhos[0].value
#rho3 = rhos[1].value
#rho4 = rhos[2].value

#x_vec = dsdp.Build_x_vector_cp(4)
#print('x_vec', x_vec)
#print('x_vec.shape', x_vec.shape)

#x_vec = dsdp_cg.Build_x_vector_cp_cg(4)
#print('x_vec', x_vec)
#print('x_vec.shape', x_vec.shape)

'''
testing coarse graining
'''

'''
rho_cg_L, rho_cg_R, omega = dsdp.coarse_grain_map(rho3, rho4, dim=2)

ptr_R = ut.ptrace(omega, (4, 2), 2)
ptr_L = ut.ptrace(omega, (2, 4), 1)
print('rho_cg_L \n', rho_cg_L)
print('rho_cg_R \n', rho_cg_R)
print('omega \n', omega)
print('ptr_L \n', ptr_L)
print('ptr_R \n', ptr_R)
'''


#def testing_Minimize_trace_rho_times_h_cg(ham, rhos, W, N=4, dim=2, chi=3):
def testing_Minimize_trace_rho_times_h_cg(ham, W, N=4, dim=2, chi=3, output=True):
    #prob, rho, constraints = ssdp.Minimize_trace_rho_times_h_cg(ham, rhos, W, N=N, dim=dim, chi=chi)
    prob, rho, constraints = ssdp.Minimize_trace_rho_times_h_cg(ham, W, N=N, dim=dim, chi=chi, output=output)
    return prob, rho, constraints



for s in range(1,2):
    N = 5
    dim = 2
    chi = 4
    seed = s
    Delta = 1
    output = True
    a = -1
    b = 1
    #print('N =', N)
    #print('dim =', dim)
    #print('chi =', chi)
    #rhos = testing_rhos_cp_variable(N)
    #ham = ds.Build_H_TFIM_Interaction_Term()
    ham = ds.Build_H_XXZ(Delta=Delta)
    W = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=seed, a=a, b=b)
    #prob, rho, constraints = testing_Minimize_trace_rho_times_h_cg(ham, rhos, W, N=N, dim=dim, chi=chi)
    prob, rho, constraints = testing_Minimize_trace_rho_times_h_cg(ham, W, N=N, dim=dim, chi=chi, output=output)
    #print('prob.value =', prob.value)
    '''
    norm = ssdp.calc_norm_cg_rho(rho, W, dim=dim)
    if norm > 1e-4:
        print('s =', s)
        print('norm of corase grained rho:', norm)
    '''
#print('rho3 matrix', rho)
#print(rho.shape)
N=4
rhos = testing_rhos_cp_variable(N)
ham = ds.Build_H_XXZ(Delta=1)
prob = testing_Minimize_trace_rho_times_h(ham, rhos, N=N)



'''
print('dual variable to constraint  1\n', constraints[0].dual_value)
print('dual variable to constraint  2\n', constraints[1].dual_value)
print('dual variable to constraint  3\n', constraints[2].dual_value)
print('dual variable to constraint  4\n', constraints[3].dual_value)
print('dual variable to constraint  5\n', constraints[4].dual_value)
print(constraints[4].dual_value.shape)
'''

def testing_get_vectors_and_matrices(N, ham):
    c_vec, x_vec, b_vec, A_mat, B_mat = ssdp.get_vectors_and_matrices(N, ham)
    return c_vec, x_vec, b_vec, A_mat, B_mat


#c_vec, x_vec, b_vec, A_mat, B_mat = testing_get_vectors_and_matrices(4, ham)

'''
print('len(c_vec) ', len(c_vec))
print(c_vec)
print('x_vec.shape', x_vec.shape)
print(x_vec[0])
print('bvec.shape', b_vec.shape)
print(b_vec)
print('A_mat', len(A_mat))
#print(A_mat)
print('B_mat.shape', B_mat.shape)
print(B_mat)
print('-------------')
print(c_vec@x_vec)

#s = 0
#for i in range(0, (x_vec.shape[0])):
#    s += x_vec[i]*A_mat[i]
#print(s)
'''


'''
# get the x_vector from the rho list for testing
x_vector = dsdp.Build_x_vector2(rhos)
#print('x_vector = ', x_vector)
#print(len(x_vector))
#print('len A_mat', len(A_mat))
#print(A_mat[0])
def check_constraint_x_times_A(x_vector, A_mat):
    # sum x_i * A_i >= 0
    s = 0
    for i in range(0, len(x_vector)):
        #print('i = ', i)
        s += x_vector[i]*A_mat[i]
    return s
s = check_constraint_x_times_A(x_vector, A_mat)
print('s', s)


print('s.shape', s.shape)


def check_constraint_B_times_x(B_mat, x_vec):
    res = B_mat@x_vec
    return res

res = check_constraint_B_times_x(B_mat, x_vector)
print('B itmes  = ', res)

def check_c_times_x(c_vec, x_vec):
    c_times_x = c_vec @ x_vec
    return c_times_x

c_times_x = check_c_times_x(c_vec, x_vector)
print('c_times_x = ', c_times_x)


Bx = B_mat @ x_vec
#print(Bx)
#print(Bx.shape)
'''

def testing_Minimize_cvec_timec_xvec(N, ham):
    prob = ssdp.Minimize_cvec_timec_xvec(N, ham)
    return prob

#ham = ds.Build_H_TFIM_Interaction_Term()
ham = ds.Build_H_XXZ(Delta=2)
#print('ham = ', ham)

#prob = testing_Minimize_cvec_timec_xvec(4, ham)

'''
WITH CG
'''
def testing_get_vectors_and_matrices_cg(N, ham, W, dim=2, chi=3):
    c_vec, x_vec, b_vec, A_mat, B_mat, basis_vec = ssdp_cg.get_vectors_and_matrices_cg(N, ham, W, dim=dim, chi=chi)
    return c_vec, x_vec, b_vec, A_mat, B_mat, basis_vec
'''
N=4
W = dsdp_cg.generate_linear_map(dim=2, chi=3, seed=0)
c_vec, x_vec, b_vec, A_mat, B_mat, basis_vec = testing_get_vectors_and_matrices_cg(N, ham, W, dim=2, chi=3)


print('len(c_vec) ', len(c_vec))
print('c_vec', c_vec)
print('x_vec.shape', x_vec.shape)
print('x_vec[0]', x_vec[0])
print('bvec.shape', b_vec.shape)
print('b_vec', b_vec)
print('basis_vec', basis_vec[0])
print('len A_mat', len(A_mat))
print(A_mat)
print('B_mat.shape', B_mat.shape)
print('B_mat', B_mat)
print('-------------')
print('c_vec@x_vec', c_vec@x_vec)
'''

def testing_Minimize_cvec_timec_xvec_cg(N, ham, W, dim=2, chi=3):
    prob, optimal_variable, optimal_dual_variable, constraints = ssdp_cg.Minimize_cvec_timec_xvec_cg(N, ham, W, dim=dim, chi=chi)
    return prob, constraints

'''
#ham = ds.Build_H_TFIM_Interaction_Term()
ham = ds.Build_H_XXZ(Delta=2)
W = dsdp_cg.generate_linear_map(dim=2, chi=1, seed=5)
prob, constraints = testing_Minimize_cvec_timec_xvec_cg(4, ham, W, dim=2, chi=1)
'''

'''
print('constraints', constraints)
print('constraints[0]', constraints[0].dual_value)
print('constraints[1]', constraints[1].dual_value)
'''

def testing_gradient_descent(N, W, ham, dim=2, chi=3):
    res, history, g_tol = ssdp_cg.gradient_descent(N, W, ham, dim=dim, chi=chi)
    return res, history, g_tol

'''
N = 5
dim = 2
chi = 2
Delta = 1

for i in range(0, 1):
    seed = i
    #H = ds.Build_H_XXZ_full(N-2, dim=2, Delta=1)
    #print('H\n', H)
    #eigvals, eigvecs = ut.eigen(H, chi)
    #get some pertubation
    #epsilon = 0.1
    a = -1
    b = 1
    W = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=seed, a=a, b=b)
    #W0 = (eigvecs +epsilon*W).T
    #print(eigvecs)
    #print(eigvecs.T)
    #W = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=seed)
    #ham = ds.Build_H_TFIM_Interaction_Term()
    ham = ds.Build_H_XXZ(Delta=Delta)
    prob, rho, __ = testing_Minimize_trace_rho_times_h_cg(ham, W, N=N, dim=dim, chi=chi, output=False)
    print('prob.value =', prob.value)
    norm = ssdp.calc_norm_cg_rho(rho, W, dim=dim)
    print('\n\nnorm', norm)
    res, history, _ = testing_gradient_descent(N, W, ham, dim=dim, chi=chi)
    #W_after = res.x
    #norm_after = ssdp.calc_norm_cg_rho(rho, W_after, dim=dim)
    #print('res.fun =', res.fun)
    #prob1, prob2 = ssdp.compare_solutions_no_cg(4, 5, ham, dim=2)
    #e = ssdp.compare_normalized_solutions(4, 5, ham, dim=2)
    e, _, __ = ssdp.compare_normalized_solutions(4, 5, prob.value, -res.fun, ham, dim=2)

    #convert the history data into normalized solution, in oder to plot it
    history_normalized = ssdp.normalize_history(4, 5, prob.value, history, ham, dim=2, output=False)
    print(history_normalized)
    print('---------------------------------------------------------------------')

'''


#
