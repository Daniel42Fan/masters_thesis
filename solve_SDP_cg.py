import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

import utility as ut
import define_SDP as dsdp
import define_SDP_cg as dsdp_cg
import define_system as ds


def get_vectors_and_matrices_cg(N, ham, W, dim=2, chi=3):
    x_vec_cp = dsdp_cg.Build_x_vector_cp_cg(N, dim=dim, chi=chi)
    basis_vec = dsdp_cg.get_basis_vector_cg(N, dim=dim, chi=chi)
    A_mat = dsdp_cg.Build_A_matrices_cg(N, basis_vec, dim=dim, chi=chi)
    #W = dsdp_cg.generate_linear_map(dim=dim, chi=chi, seed=0)
    B_mat = dsdp_cg.Build_B_matrix_cg(N, W, dim=dim, chi=chi)
    b_vec = dsdp_cg.Build_b_vector_cg(N, dim=dim, chi=chi)
    c_vec = dsdp_cg.Build_c_vector_cg(N, ham, basis_vec, dim=dim, chi=chi)
    return c_vec, x_vec_cp, b_vec, A_mat, B_mat, basis_vec

'''
added output variable, to also hide print statements, if wanted
'''
def Minimize_cvec_timec_xvec_cg(N, ham, W, dim=2, chi=3, output=True):
    # get all neede components
    c_vec, x_vec_cp, b_vec, A_mat, B_mat, basis_vec = get_vectors_and_matrices_cg(N, ham, W, dim=dim, chi=chi)

    # define constraints
    constraints = []

    # sum x_i * A_i >= 0
    s = 0
    #print('x_vec_cp.shape', x_vec_cp.shape)
    #print('len A_mat', len(A_mat))
    for i in range(0, (x_vec_cp.shape[0])):
        #print('i = ', i)
        s += x_vec_cp[i]*A_mat[i]
        #print('x_vec_cp[{}] ='.format(i), x_vec_cp[i])
        #print('A_mat[{}] ='.format(i), A_mat[i])
        #print('A_mat[{}].shape ='.format(i), A_mat[i].shape)
    #print('s.shape', s.shape)
    # >> is the matrix inequality
    constraints.append(s >> 0)

    # cast B times x into right shape to match with b
    constraints.append((B_mat@x_vec_cp).T[0] == b_vec)

    # form objective
    obj = cp.Minimize(c_vec@x_vec_cp)

    # form and solve problem
    prob = cp.Problem(obj, constraints)
    #prob.solve(solver=cp.SCS)
    prob.solve(solver=cp.MOSEK)
    #prob.solve(verbose=True)
    #prob.solve()
    if output==True:
        print("\n\nRESULTS:\n")
        print("WITH coarse graining\n")
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("for cg map\n", W)
        print("-------------------------------------------------------")
        #print("unpack results", prob.unpack_results)
    #print("optimal dual variable to constraint xA >= 0", constraints[0].dual_value)
    #print("optimal dual variable to constraint Bx = b", constraints[1].dual_value)
    optimal_dual_variable = constraints[1].dual_value
    #print("optimal var", x_vec_cp.value)
    optimal_variable = x_vec_cp.value
    #rho2 = optimal_variable[0:16]
    #print(rho2.reshape(4,4))
    #print('x_vec_cp.value.shape', x_vec_cp.value.shape)
    #print((c_vec@x_vec_cp).value)

    return prob, optimal_variable, optimal_dual_variable, constraints

'''
already done int define_SDP_cg
'''
def calc_SDP_gradient(N, W, dim=2, chi=3):
    '''
    want to check, if the the calculated dB is right, by checking it with the
    SDP gradient formula.
    1. Use the SDP gradient forumla
    2. Solve the human readable SDP problem and use the dual variables to check
       the solution above.
    '''
    # get the calculated derivative of B
    dB = dsdp_cg.Build_B_matrix_cg(N, W, dim=dim, chi=chi, gradient=True)
    return 0





'''
add  gradient descent
'''
def gradient_descent(N, W, ham, g_tol=1e-6, dim=2, chi=3, _2D=False):
    '''
    compute energy and gradient of the energy (using the SDP gradient formula)
    with respect to the coarse graining map W
    '''
    # get hamiltonian
    ham = ham
    # flatten W
    W0 = W.flatten()
    '''
    save the energy AFTER each iteration with the callback function
    also include the intitial energy
    '''
    #history = []
    #energy, grad = dsdp_cg.get_energy_and_derivative_of_energy(W0, N, ham, dim=dim, chi=chi)
    #history.append(energy)
    #print('initial energy', energy)

    # callback funciton to get intermediate results.
    # used an global list instead to get the results
    '''
    def callbackF(W):

        # get energy from the global variable
        energy = dsdp_cg.global_energy
        history.append(energy)
        print('global energy from callback', energy)
    '''
    '''
    apply gradient descent
    Use scipy.optimize and its minimize BFGS algorithm
    '''
    res = 0
    if _2D == False:
        if chi==1:
            # jac = True, indicates that gradient is also as 2nd return variable
            res = minimize(dsdp_cg.get_energy_and_derivative_of_energy, W0, method='BFGS',
                        args=(N, ham, dim, chi), jac=True, tol=10**(-6))#, callback=callbackF)#,
                        #options={'disp': True}) # for getting immediate results
        else:
            res = minimize(dsdp_cg.get_energy_and_derivative_of_energy, W0, method='BFGS',
                        args=(N, ham, dim, chi), jac=True, options={'gtol': g_tol}) #range  e-2 e-8
                        # get reason why solver stops!
        history = dsdp_cg.global_energy_list
        print('intermediate result from global list', history)
        # RESET the global list, otherwise the old data would still remain
        dsdp_cg.global_energy_list = []

    else:
        # do 2D minimization here
        pass
    # print some results
    print('result res\n', res)
    print('result res.x = ', res.x)
    return res, history, g_tol
