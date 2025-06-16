from scipy.optimize import minimize
import define_SDP_cg as dsdp_cg


def gradient_descent(N, W, ham, g_tol=1e-8, dim=2, chi=3):
    W0 = W.flatten()
    res = minimize(
        dsdp_cg.get_energy_and_derivative_of_energy,
        W0,
        method="BFGS",
        args=(N, ham, dim, chi),
        jac=True,
        options={"gtol": g_tol},
    )

    history = dsdp_cg.global_energy_list
    dsdp_cg.global_energy_list = []
    return res, history, g_tol


import cvxpy as cp
import define_SDP_cg as dsdp_cg


def Minimize_trace_rho_times_h_cg(ham, W, N=4, dim=2, chi=3):
    rho_N_minus_1 = get_rho_cp_variable(N - 1, dim=dim)
    constraints = []
    constraints.append(cp.trace(rho_N_minus_1) == 1)

    rho_shape_0 = rho_N_minus_1.shape[0]
    ptr_R = cp.partial_trace(rho_N_minus_1, [int(rho_shape_0 / 2), 2], axis=1)
    ptr_L = cp.partial_trace(rho_N_minus_1, [2, int(rho_shape_0 / 2)], axis=0)
    # LTI constraint
    constraints.append(ptr_L == ptr_R)
    # apply coarse graining
    rho_cg_L, rho_cg_R = dsdp_cg.coarse_grain_map_N(
        rho_N_minus_1, W, N=N, dim=dim, chi=chi
    )

    omega = cp.Variable((chi * dim**2, chi * dim**2), PSD=True)
    ptr_R = cp.partial_trace(omega, (chi * dim, dim), 1)
    ptr_L = cp.partial_trace(omega, (dim, chi * dim), 0)

    constraints.append(ptr_R == rho_cg_R)
    constraints.append(ptr_L == rho_cg_L)
    # form objective
    obj = cp.Minimize(
        cp.trace(rho_N_minus_1 @ sparse.kron(ham, sparse.identity(dim ** (N - 3))))
    )
    # form and solve problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK)

    rho_N_minus_1 = rho_N_minus_1.value
    return prob, rho_N_minus_1, constraints
