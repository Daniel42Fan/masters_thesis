import numpy as np
import re
import pickle as pickle
import scipy.linalg as sLA
from numpy import linalg as nLA

import define_system as ds
import solve_SDP as ssdp
import calc_energies as ce
import define_SDP_cg as dsdp_cg


"""
gets pickled data from a certain path
"""


def get_pickled_data(path, data_name):
    data = 0
    with open(path + data_name + ".pickle", "rb") as f:
        data = pickle.load(f)
    return data


"""
define a path, where the data are stored
"""


def get_data_path(model, *args, normalize=False, map=""):
    path = ""
    if normalize == True:
        path = "data_normalized/" + model + map
    else:
        if model == "random_Hamiltonian/":
            for arg in args:
                s = arg
                # print('seed for Hamiltonian =', s)
                model_ = model + "seed" + str(s) + "/"
                path = "data/" + model_ + map
                # print('path =', path)
        else:
            path = "data/" + model + map
    return path


def get_data_path2(Delta, N, chi, *args, model="XXZ/", map="", normalize=False):
    """
    go to the path, where the energies with different settings are stored
    """
    path_1 = get_data_path(model, *args, normalize=normalize, map=map)
    print("path_1 = ", path_1)
    # path_1 = 'data/XXZ/pertubation/'
    """
    go into file with different settings
    """
    path_2 = "Delta{}_N{}_chi{}/".format(Delta, N, chi)
    if model == "random_Hamiltonian/":
        path_2 = "N{}_chi{}/".format(N, chi)
    else:
        pass
    """
    REWRITE IT IF THIS IS NEEDED SOMEDAY. path1 already uses args, need a way
    to not get the same data
    # if there is something additional in the path string, get it from args
    if len(args)==1:
        for arg in args:
            path_2 = path_2+arg
    else:
        pass
    """
    path = path_1 + path_2
    print("Current path:\n", path)
    return path


"""
arguments:
*args:  expect 1 optional argument, that is added to the path_2 string
models: which kind of Hamiltonian is considered
        default: XXZ
maps:   what kind of map was applied to the coarse graining.
        e.g. random_maps
        default: empty string (= data with random maps from earlier runs)
"""


def get_energies(Delta, N, chi, *args, model="XXZ/", map="", normalize=False):
    """
    go to the path, where the energies with different settings are stored
    """
    # path = get_data_path2(Delta, N, chi, args[0], model=model, map=map, normalize=normalize)
    """
    if args is not empty use this
    TO DO: implement for different cases, if args is empty or not
    """
    path_1 = ""
    # path_1 = 'data/XXZ/pertubation/'
    if not args:
        path_1 = get_data_path(model, normalize=normalize, map=map)
    else:
        path_1 = get_data_path(model, args[0], normalize=normalize, map=map)
    """
    go into file with different settings
    """
    path_2 = "Delta{}_N{}_chi{}/".format(Delta, N, chi)

    """
    # outdated with gtol stuff, need do adjust if this is needed
    # if there is something additional in the path string, get it from args
    if len(args)==1:
        for arg in args:
            path_2 = path_2+str(arg)

    else:
        pass
    """
    path = path_1 + path_2
    print("Current path:\n", path)
    """
    get data name
    """
    data_1 = "energies_before_gd"
    data_2 = "energies_after_gd"

    e_before_gd = get_pickled_data(path, data_1)
    e_after_gd = get_pickled_data(path, data_2)
    return e_before_gd, e_after_gd


"""
for random Hamiltonians
get the normalized energies and compute them back into absolute values
"""


def normalized_energies_to_absolute_values(
    Delta, N, chi, ham, *args, dim=2, model="XXZ/", map="", normalize=True
):
    prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(
        N - 1, N, ham, dim=dim, output=False
    )
    energy_N_minus_1 = prob_N_minus_1.value
    energy_N = prob_N.value
    print("Energy for N = {}:".format(N - 1), energy_N_minus_1)
    print("Energy for N = {}:".format(N), energy_N)
    shift = energy_N_minus_1
    norm_factor = 1 / (energy_N - shift)

    # get the normalized energies
    e_before_gd_n, e_after_gd_n = get_energies(
        Delta, N, chi, *args, model=model, map=map, normalize=normalize
    )
    print("Normalized energies before gradient descent:\n", e_before_gd_n)
    print("Normalized energies after gradient descent:\n", e_after_gd_n)
    # convert the normalized energies back
    e_before_gd = [(e / norm_factor) + shift for e in e_before_gd_n]
    e_after_gd = [(e / norm_factor) + shift for e in e_after_gd_n]
    return e_before_gd, e_after_gd


"""
after conversion of the normalized energies, save them into the right folder!
"""


def save_converted_energies(
    Delta, N, chi, ham, *args, dim=2, model="XXZ/", map="", normalize=True
):
    # convert data
    e_before_gd, e_after_gd = normalized_energies_to_absolute_values(
        Delta, N, chi, ham, *args, dim=dim, model=model, map=map, normalize=normalize
    )
    print("e_before_gd =\n", e_before_gd)
    print("e_after_gd =\n", e_after_gd)

    # create path and save converted data
    path = get_data_path(model, normalize=False, map=map)
    path_2 = "Delta{}_N{}_chi{}/".format(Delta, N, chi)
    path = path + path_2
    ce.create_path(path)
    ce.save_as_pickle(path, "energies_before_gd", e_before_gd)
    ce.save_as_pickle(path, "energies_after_gd", e_after_gd)
    return 0


def get_res_history(Delta, N, chi, *args, model="XXZ/", map="", normalize=False):
    path = get_data_path2(
        Delta, N, chi, *args, model=model, map=map, normalize=normalize
    )
    data_name = "res_history"
    h = get_pickled_data(path, data_name)
    return h


def get_history(Delta, N, chi, *args, model="XXZ/", map="", normalize=False):
    path = get_data_path2(
        Delta, N, chi, *args, model=model, map=map, normalize=normalize
    )
    data_name = "history"
    history = get_pickled_data(path, data_name)
    return history


def get_error_txt(Delta, N, chi, *args, model="XXZ/", map="", normalize=False):
    path = get_data_path2(
        Delta, N, chi, *args, model=model, map=map, normalize=normalize
    )
    data_name = "error"
    data = 0
    with open(path + data_name + ".txt", "r") as f:
        data = f.read()
    return data


def extract_error_seeds(text):
    # get the text file
    data = text
    # Use regular expression to find all numbers after "Error with seed"
    numbers = re.findall(r"Error with seed (\d+)", data)
    # Convert the numbers from strings to integers
    numbers = [int(num) for num in numbers]
    return numbers


def get_optimized_cg_map(
    Delta, N, chi, *args, dim=2, model="XXZ/", map="", normalize=False
):
    print(args)
    print(*args)
    path = get_data_path2(
        Delta, N, chi, *args, model=model, map=map, normalize=normalize
    )
    data_name = "optimized_cg_map"
    W = get_pickled_data(path, data_name)
    k = N - 2
    for i in range(len(W)):
        W[i] = W[i].reshape((chi, int(dim**k)))
    return W


"""
investigate the optimized maps where the associated energy
after gradient descent is larger than it should be
Calculate its eigenvalues
(optional: calc. the QR decomposition look at R)
"""


def investigate_optimized_maps(
    Delta, N, chi, dim=2, *args, model="XXZ/", map="", normalize=False
):
    # get the optimized maps
    W = get_optimized_cg_map(
        Delta, N, chi, dim=dim, *args, model=model, map=map, normalize=normalize
    )
    # get the energies after gradient descent
    e_before_gd, e_after_gd = get_energies(
        Delta, N, chi, model=model, map=map, normalize=normalize
    )
    print("energy after gd =\n", e_after_gd)
    # sort energy from lowest to highest
    idx = np.argsort(e_after_gd)
    # flip indices to sort from largest to lowest
    idx = np.flip(idx)
    # print('idx =', idx)
    sorted_energy = np.array(e_after_gd)[idx]
    # print('sorted_energy =\n', sorted_energy)
    sorted_maps = np.array(W)[idx]
    # print('sorted_maps =\n', sorted_maps)
    for i in range(len(sorted_energy)):
        # print(sorted_maps[i])
        # Q, R = qr_decomposition(sorted_maps[i])
        # print('R after QR decomp =\n', R)
        U, S, Vh = svd_decomposition(sorted_maps[i])
        print("S after SVD decomposition =\n", S)
        # eigvals, _ = nLA.eig(R)
        # print('eigvals of R =',  eigvals)
        # eigenValues, eigenVectors = nLA.eig(sorted_maps[i])
        # print('Eigenvalues of map {}\n'.format(i), eigenValues)

    return 0


"""
calculates the subspace angle between two matrices A and B
"""


def calc_subspace_angles(A, B):
    angles = np.rad2deg(sLA.subspace_angles(A, B))
    return angles


"""
TO DO: RUN MORE INSTANCES FOR DIFFERENT SETTINGS
plot heatmap (log if neccessary)
display maximum entry of the a matrix
"""


def get_subspace_angles(W):
    a = np.zeros((len(W), len(W)))
    for i in range(0, len(W)):
        for j in range(i + 1, len(W)):
            angles = calc_subspace_angles(W[i].T, W[j].T)
            a[i][j] = angles[0]
    # print('angles =\n', a)
    return a


"""
calculate SVD decomposition
"""


def svd_decomposition(mat):
    U, S, Vh = nLA.svd(mat)
    return U, S, Vh


"""
get the random and optimized maps W_rand, W_opt
do QR decomposition on W = QR, where only the first chi columns of Q are
considered giving Q'
calculate the eigenvalues of H' = Q'_dagger H Q' for both W_rand and W_opt
and compare
"""


def qr_decomposition(mat):
    q, r = sLA.qr(mat, mode="economic")
    # q, r = sLA.qr(mat)
    # print('q =', q)
    # print('r =', r)
    return q, r


def calc_q_dagger_H_q(Delta, N, chi, ham, model, map, *args, dim=2):
    Delta = Delta
    N = N
    chi = chi
    W = get_optimized_cg_map(Delta, N, chi, *args, model=model, map=map)
    # print('W[0].shape = ', W[0].shape)
    # print('len(W)', len(W))
    # get the seeds of the maps, that produces errors
    errors = get_error_txt(Delta, N, chi, model=model, map=map)
    error_seeds = extract_error_seeds(errors)
    # get the initial maps and exclude the maps from the list of initial maps that had errors
    W0 = [
        dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=seed)
        for seed in range(100)
        if seed not in error_seeds
    ]
    # print('len(W0)', len(W0))
    H = ham
    # H = ds.Build_H_XXZ_full(N-2, Delta=Delta)
    # store the Qs into a list, after QR decomposition
    eigval = []
    spectrum = []
    eigval0 = []
    spectrum0 = []

    eigvals_H = sLA.eigvalsh(H)
    print(eigvals_H)
    print("H.shape =", H.shape)
    for map in W:
        # print('------------------------------')
        # print('shape of Ws', map.shape)
        q, r = qr_decomposition(map.T)
        # print('q.shape =', q.shape)
        # print('r.shape =', r.shape)
        # print(np.linalg.norm(r.T@q.T-map))
        # print('------------------------------')
        H_bar = q.T @ H @ q
        # print('H_bar.shape = ', H_bar.shape)
        eigvals = sLA.eigvalsh(H_bar)
        # print('eigvals =', eigvals)
        # print('eigvals.shape =', eigvals.shape)
        spectrum.append(eigvals)
        max_eigval = max(eigvals)
        min_eigval = min(eigvals)
        # print(max_eigval)
        eigval.append([min_eigval, max_eigval])
        # print('eigenvalues H =', eigvals_H)
        # print('eigenvalues of q.T x H x q =', eigvals)
    for map in W0:
        q0, r0 = qr_decomposition(map.T)
        H_bar = q0.T @ H @ q0
        eigvals0 = sLA.eigvalsh(H_bar)
        spectrum0.append(eigvals0)
        max_eigval0 = max(eigvals0)
        min_eigval0 = min(eigvals0)
        eigval0.append([min_eigval0, max_eigval0])
    # print('spectrum = ', spectrum)
    # print('len(spectrum) = ', len(spectrum))

    return eigval, eigvals_H, spectrum, eigval0, spectrum0


def spectrum_of_H(N=5, Delta=1):
    H = ds.Build_H_XXZ_full(N - 2, Delta=Delta)
    eigval_H = []
    eigvals_H = sLA.eigvalsh(H)
    max_eigval_H = max(eigvals_H)
    min_eigval_H = min(eigvals_H)
    eigval_H.append([min_eigval_H, max_eigval_H])
    return eigval_H
