"""
plot routines for the plots used in the thesis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle as pickle

import define_system as ds
import calc_energies as ce
import analyze_data as ad
import solve_SDP as ssdp

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica"
    }
)

TINY_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc("font", size=TINY_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

"""
create a template for plots
"""


def create_plot_template():
    fig, ax = plt.subplots(figsize=((10, 6)))
    # fig, ax = plt.subplots()
    # plt.rcParams['text.usetex'] = True
    return fig, ax


"""
for plotting random Hamiltonians
"""


def plot_energies(model, map, *args, normalize=True, xscale="log", yscale="log"):
    markers = [".", "1", "v", "p", "*", "x", "+", "s"]
    model = model
    map = map
    seed = args[0]
    e_before_gd_list = []
    e_after_gd_list = []
    chi_list = []
    figs_list = []
    axes_list = []
    for Delta in range(1, 2):
        for N in range(5, 7, 2):
            # calc the energies for no cg
            """
            TO DO: need a better way to extract Hamiltonian!!!!!
            """
            prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(
                N - 1, N, args[1], dim=2, output=False
            )
            # prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(N-1, N, args[0], dim=2, output=True)
            e_N_minus_1 = prob_N_minus_1.value
            e_N = prob_N.value
            de = e_N - e_N_minus_1
            # try:
            for chi in range(2, 8):
                # get data
                e_before_gd, e_after_gd = ad.get_energies(
                    Delta, N, chi, model=model, map=map, normalize=normalize
                )
                if normalize == True:
                    e_after_gd = 1 - np.asarray(e_after_gd)
                    e_before_gd = 1 - np.asarray(e_before_gd)
                else:
                    e_after_gd = e_N - np.asarray(e_after_gd)
                    e_before_gd = e_N - np.asarray(e_before_gd)
                    # e_after_gd = (e_N+1e-06) - np.asarray(e_after_gd)
                    # e_before_gd = (e_N+1e-06) - np.asarray(e_before_gd)
                e_before_gd_list.append(e_before_gd)
                e_after_gd_list.append(e_after_gd)
                chi_list.append(chi)
            # print('Plotting energies for Delta = {}, N = {}, chi = {}'.format(Delta, N, chi))
            print("Plotting energies for Delta = {}, N = {}".format(Delta, N))
            fig, ax = create_plot_template()
            for i in range(len(e_after_gd_list)):
                # print('before gd\n', e_before_gd_list[i])
                # print('after gd\n', e_after_gd_list[i])
                ax.plot(
                    e_before_gd_list[i],
                    e_after_gd_list[i],
                    marker=markers[i],
                    linestyle="None",
                    label=r"$\chi = {}$".format(chi_list[i]),
                )
            plt.legend(loc="upper left")
            """
            TEST THE SIGNIFICANT NUMBERS AND FORMAT
            """
            plt.title(r"$\Delta E = {0:.2e}$".format(de), loc="right")
            # extract seed out of args
            # if random Hamiltonian: get seed name
            # plt.title('seed = {}'.format(args[0]), loc='right')
            path = ce.create_path("plots/energies/" + model + map)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            # plt.xscale(xscale)
            # plt.yscale(yscale)

            """
            CHANGE TITLE NAME ACCORDINGLY!!!
            """
            if normalize == True:
                ax.set_title(
                    "Normalized Energies of a Random Hamiltonian\nN = {}, Delta = {}".format(
                        N, Delta
                    ),
                    fontsize=BIGGER_SIZE,
                )
                ax.set_xlabel("Accuracy Before Optimization")
                ax.set_ylabel("Accuracy After Optimization")
                # plt.savefig(path+'normalized_energies_Delta{}_N{}.pdf'.format(Delta, N))
            else:
                # ax.set_title('Energies for the XXZ Hamiltonian $\Delta = {}$\nN = {}'.format(Delta, N))
                # ax.set_title('Ground State Energy of a Random Hamiltonian (seed {})\nn = {}'.format(args[0], N))
                ax.set_title(
                    "Ground State Energy of a Random Hamiltonian (seed {})\nn = {}".format(
                        seed, N
                    ),
                    fontsize=BIGGER_SIZE,
                )
                # ax.set_xlabel('Energy Before Gradient Descent')
                # ax.set_ylabel('Energy After Gradient Descent')
                ax.set_xlabel("Accuracy Before the Gradient Search")
                ax.set_ylabel("Accuracy After the Gradient Search")
                # plt.savefig(path+'energies_Delta{}_N{}.pdf'.format(Delta, N))
                plt.savefig(path + "seed{}_energies_N{}.pdf".format(seed, N))
            print("Plot saved successfully")
            figs_list.append(fig)
            axes_list.append(ax)
            plt.close("all")
            # plt.show()
            # except:
            #    print('No data found')
    # return 0
    return figs_list, axes_list


"""
#seeds = [1,2,4,5,7,8]
seeds = [3,6,9]
for s in seeds:
#for s in range(1,11):
    #plot_energies('random_Hamiltonian/seed{}/'.format(s), 'random_maps/')
    ham = ds.Build_random_H(s)
    scale = 'log'
    figs, axes = plot_energies('random_Hamiltonian/seed{}/'.format(s), 'random_maps/', s, ham,
                                normalize=False, xscale=scale, yscale=scale)
"""

"""
for plotting the XXZ model
"""


def plot_energies2(model, map, *args, normalize=True, xscale="log", yscale="log"):
    chi_start = 2
    markers = [".", "1", "v", "p", "*", "x", "+", "s"]
    # Get the list of standard colors
    standard_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if chi_start == 1:
        markers = ["d", ".", "1", "v", "p", "*", "x", "+", "s"]
        # cut out the first color
        standard_colors = [standard_colors[-1]] + standard_colors
    model = model
    map = map
    e_before_gd_list = []
    e_after_gd_list = []
    chi_list = []
    figs_list = []
    axes_list = []
    for Delta in range(0, 1):
        ham = ds.Build_H_XXZ(Delta=Delta)
        for N in range(5, 7, 2):
            # calc the energies for no cg
            prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(
                N - 1, N, ham, dim=2, output=True
            )
            e_N_minus_1 = prob_N_minus_1.value
            e_N = prob_N.value
            de = e_N - e_N_minus_1
            # try:
            for chi in range(chi_start, 8):
                # get data
                e_before_gd, e_after_gd = ad.get_energies(
                    Delta, N, chi, model=model, map=map, normalize=normalize
                )
                if normalize == True:
                    e_after_gd = 1 - np.asarray(e_after_gd)
                    e_before_gd = 1 - np.asarray(e_before_gd)
                else:
                    e_after_gd = np.asarray(e_after_gd)
                    e_before_gd = np.asarray(e_before_gd)
                    if xscale == "log":
                        e_after_gd = e_N - np.asarray(e_after_gd)
                        e_before_gd = e_N - np.asarray(e_before_gd)
                    # e_after_gd = (e_N+1e-06) - np.asarray(e_after_gd)
                    # e_before_gd = (e_N+1e-06) - np.asarray(e_before_gd)
                e_before_gd_list.append(e_before_gd)
                e_after_gd_list.append(e_after_gd)
                chi_list.append(chi)
            # print('Plotting energies for Delta = {}, N = {}, chi = {}'.format(Delta, N, chi))
            print("Plotting energies for Delta = {}, N = {}".format(Delta, N))
            fig, ax = create_plot_template()
            for i in range(len(e_after_gd_list)):
                # print('before gd\n', e_before_gd_list[i])
                # print('after gd\n', e_after_gd_list[i])
                # for log axes, consider only positive values
                if xscale == "log":
                    print(e_before_gd_list[i])
                    print(e_after_gd_list[i])
                    pos_e_after = []
                    pos_e_before = []
                    for e_before, e_after in zip(
                        e_before_gd_list[i], e_after_gd_list[i]
                    ):
                        if e_after >= 0:
                            pos_e_before.append(e_before)
                            pos_e_after.append(e_after)
                    ax.plot(
                        pos_e_before,
                        pos_e_after,
                        marker=markers[i],
                        color=standard_colors[i],
                        linestyle="None",
                        label=r"$\chi = {}$".format(chi_list[i]),
                    )
                    print(pos_e_after)
                else:
                    ax.plot(
                        e_before_gd_list[i],
                        e_after_gd_list[i],
                        marker=markers[i],
                        color=standard_colors[i],
                        linestyle="None",
                        label=r"$\chi = {}$".format(chi_list[i]),
                    )
            if xscale == "log":
                plt.legend(loc="best")
            else:
                plt.legend(loc="lower center")
            # clear list for next loop
            e_before_gd_list.clear()
            e_after_gd_list.clear()
            chi_list.clear()
            """
            TEST THE SIGNIFICANT NUMBERS AND FORMAT
            """
            plt.title(r"$\Delta E = {0:.2e}$".format(de), loc="right", y=0.99)
            path = ce.create_path("plots/energies/" + model + map)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            # plt.xscale(xscale)
            # plt.yscale(yscale)
            x_left, x_right = plt.xlim()
            y_lower, y_upper = plt.ylim()
            # distance between left and right corners
            x_dis = x_right - x_left
            y_dis = y_upper - y_lower
            # add vertical and horizontal lines with energies E_4 and E_5
            ax.hlines(
                y=e_N, xmin=-1, xmax=0, colors="k", linestyle="--", linewidth=1
            )  # , label='E_{}'.format(N))
            ax.text(
                x_left - 0.018 * (x_dis),
                e_N + 0.02 * (y_dis),
                r"$E_{{ \mathrm{{ {} }} }}({})$".format("LTI", N),
                color="k",
            )
            ax.hlines(
                y=e_N_minus_1,
                xmin=-1,
                xmax=0,
                colors="darkred",
                linestyle="--",
                linewidth=1,
            )  # , label='E_{}'.format(N))
            ax.text(
                x_left + 0.05 * (x_dis),
                e_N_minus_1 + 0.02 * (y_dis),
                r"$E_{{ \mathrm{{ {} }} }}({})$".format("LTI", N - 1),
                color="darkred",
            )
            ax.vlines(
                e_N_minus_1,
                ymin=-1,
                ymax=1,
                colors="darkgreen",
                linestyles="--",
                linewidth=1,
            )  # , label='E_{}'.format(N-1))
            ax.text(
                x_left - 0.018 * (x_dis),
                y_lower
                - 0.01
                * (y_dis),  # ax.text(e_N_minus_1+0.01*(x_dis), y_lower-0.01*(y_dis),
                r"$E_{{ \mathrm{{ {} }} }}({})$".format("LTI", N - 1),
                color="darkgreen",
            )
            ax.vlines(
                e_N, ymin=-1, ymax=1, colors="purple", linestyles="--", linewidth=1
            )  # , label='E_{}'.format(N))
            ax.text(
                e_N - 0.07 * (x_dis),
                y_lower - 0.01 * (y_dis),
                r"$E_{{ \mathrm{{ {} }} }}({})$".format("LTI", N),
                color="purple",
            )
            # plt.xlim(x_left-0.02*x_dis, x_right-0.02*x_dis)
            # plt.ylim(y_lower-0.03*y_dis, y_upper+0.03*y_dis)

            """
            CHANGE TITLE NAME ACCORDINGLY!!!
            """
            if normalize == True:
                ax.set_title(
                    "Normalized Energies for a Random Hamiltonian\nN = {}, Delta = {}".format(
                        N, Delta
                    )
                )
                ax.set_xlabel(
                    "Accuracy Before Optimizing $E_{\text{LTI}}(n,W)$ over $W$"
                )
                ax.set_ylabel(
                    "Accuracy After Optimizing $E_{\text{LTI}}(n,W)$ over $W$"
                )
                plt.savefig(
                    path + "normalized_energies_Delta{}_N{}.pdf".format(Delta, N)
                )
            else:
                ax.set_title(
                    "Initial and Final Energy Value for XX Model\nn = {}".format(N),
                    fontsize=BIGGER_SIZE,
                )
                # ax.set_title('Initial and Final Energy Value for the Heisenberg Model \nn = {}'.format(N),
                #                fontsize=BIGGER_SIZE)
                # ax.set_title('Initial and Final Energy Value for the XXZ Model ($\Delta = {}$) \nn = {}'.format(Delta, N),
                #                fontsize=BIGGER_SIZE)
                # ax.set_title('Energies for a Random Hamiltonian\nN = {}'.format(N))
                ax.set_xlabel("Ground State Energy Before the Gradient Search")
                ax.set_ylabel("Ground State Energy After the Gradient Search")
                # ax.set_xlabel('Accuracy Before Gradient Descent')
                # ax.set_ylabel('Accuracy After Gradient Descent')
                if xscale == "log":
                    ax.set_xlabel("Accuracy Before the Gradient Search")
                    ax.set_ylabel("Accuracy After the Gradient Search")
                if chi_start == 1:
                    if xscale == "log":
                        plt.savefig(
                            path
                            + "log_energies_all_chis_Delta{}_N{}.pdf".format(Delta, N)
                        )
                    else:
                        plt.savefig(
                            path + "energies_all_chis_Delta{}_N{}.pdf".format(Delta, N)
                        )
                else:
                    if xscale == "log":
                        plt.savefig(
                            path + "log_energies_Delta{}_N{}.pdf".format(Delta, N)
                        )
                    else:
                        plt.savefig(path + "energies_Delta{}_N{}.pdf".format(Delta, N))
                    pass
            print("Plot saved successfully")
            figs_list.append(fig)
            axes_list.append(ax)
            plt.close("all")
            # plt.show()
            # except:
            #    print('No data found')
    # return 0
    return figs_list, axes_list


figs, axes = plot_energies2(
    "XXZ/", "random_maps/", normalize=False, xscale="log", yscale="log"
)


"""
plot the intermediate results from the gradient descent minimization
"""


def plot_grad_descent_intermediate_results():
    model = "XXZ/"
    map = "random_maps/"
    markers = ["o", "s", "^", "D", "v", "p", ">", "<", "*", "H"]
    for Delta in range(0, 1):
        ham = ds.Build_H_XXZ(Delta=Delta)
        for N in range(5, 7, 2):
            prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(
                N - 1, N, ham, dim=2, output=True
            )
            e_N_minus_1 = prob_N_minus_1.value
            e_N = prob_N.value
            for chi in range(2, 3):
                # try:
                # get result
                history = ad.get_history(
                    Delta, N, chi, model=model, map=map, normalize=False
                )
                print(
                    "Plotting intermediate energy values of gradient search for Delta = {}, N = {}, chi = {}".format(
                        Delta, N, chi
                    )
                )
                fig, ax = create_plot_template()
                # ax.set_title('Intermediate Results of the Gradient Descent Using the XXZ Model\nn = {}, $\Delta = {}$, $\chi = {}$'.format(N, Delta, chi),
                #                fontsize=BIGGER_SIZE)
                ax.set_title(
                    "Intermediate Energy Values of the Gradient Search for the XX Model\nn = {}, $\chi = {}$".format(
                        N, chi
                    ),
                    fontsize=BIGGER_SIZE,
                )
                # plot first 10 maps
                for i in range(10):
                    # ax.plot(-1*np.asarray(history[i]), marker='x', linestyle='None')#, label='energies')
                    # print('markers[{}]'.format(i), markers[i])
                    ax.plot(
                        -1 * np.asarray(history[i]),
                        marker=markers[i],
                        label="Map {}".format(i + 1),
                    )
                # only integer ticks on x-axis
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                x_left, x_right = plt.xlim()
                y_lower, y_upper = plt.ylim()
                # distance between left and right corners
                x_dis = x_right - x_left
                y_dis = y_upper - y_lower
                ax.hlines(
                    y=e_N, xmin=-10, xmax=1000, colors="k", linestyle="--", linewidth=1
                )  # , label='E_{}'.format(N))
                ax.text(
                    x_left + 0.05 * (x_dis),
                    e_N + 0.01 * (y_dis),
                    r"$E_{{ \mathrm{{ {} }} }}({})$".format("LTI", N),
                    fontsize="x-large",
                    color="k",
                )
                ax.hlines(
                    y=e_N_minus_1,
                    xmin=-10,
                    xmax=1000,
                    colors="darkred",
                    linestyle="--",
                    linewidth=1,
                )  # , label='E_{}'.format(N))
                ax.text(
                    x_left + 0.05 * (x_dis),
                    e_N_minus_1 + 0.01 * (y_dis),
                    r"$E_{{ \mathrm{{ {} }} }}({})$".format("LTI", N - 1),
                    fontsize="x-large",
                    color="darkred",
                )
                ax.set_xlabel("Iteration Step")
                ax.set_ylabel("Ground State Energy")
                plt.xlim(-1, 42)
                plt.legend(loc="center right")
                path = ce.create_path("plots/history/" + model + map)
                plt.savefig(
                    path + "history_Delta{}_N{}_chi{}.pdf".format(Delta, N, chi)
                )
                print("plot saved successfully")
                # plt.close("all")
                # plt.show()
                # except:
                #    print('No data found')
    return 0


# plot_grad_descent_intermediate_results()


def boxplots(model, map, *args, normalize=True, xscale="log", yscale="log"):
    markers = [".", "1", "v", "p", "*", "x", "+", "s"]
    standard_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model = model
    map = map
    seed = 0
    e_before_gd = 0
    e_after_gd = 0
    e_N_minus_1 = 0
    e_N = 0
    de = 0
    e_before_gd_list = []
    e_after_gd_list = []
    chi_list = []
    figs_list = []
    axes_list = []
    for Delta in range(1, 2):
        for N in range(5, 7, 2):
            # calc the energies for no cg
            # initilialize
            prob_N_minus_1 = 0
            prob_N = 0
            if model == "XXZ/":
                # for XXZ model
                ham = ds.Build_H_XXZ(Delta=Delta)
                prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(
                    N - 1, N, ham, dim=2, output=True
                )
                e_N_minus_1 = prob_N_minus_1.value
                e_N = prob_N.value
                de = e_N - e_N_minus_1
            # if there are even more models to consider,
            # then this must be changed accordingly
            else:
                """
                # since the data are in one plot, this is not necessary anymore
                # for random Hamiltonians
                seed = args[0]
                prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(N-1, N, args[1], dim=2, output=False)
                """
                pass

            # try:
            if model == "random_Hamiltonian/":
                e_before_seed_list = []
                e_after_seed_list = []
                for s in args[0]:
                    ham = ds.Build_random_H(s)
                    prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(
                        N - 1, N, ham, dim=2, output=False
                    )
                    e_N = prob_N.value
                    for chi in range(2, 8):
                        # get data
                        e_before_gd, e_after_gd = ad.get_energies(
                            Delta, N, chi, s, model=model, map=map, normalize=normalize
                        )
                        if normalize == True:
                            """
                            needs to be adapted if one wants to use this ever again
                            """
                            e_after_gd = 1 - np.asarray(e_after_gd)
                            e_before_gd = 1 - np.asarray(e_before_gd)
                        else:
                            e_after_gd = e_N - np.asarray(e_after_gd)
                            e_before_gd = e_N - np.asarray(e_before_gd)
                            # sort list
                            e_after_gd = sorted(e_after_gd)
                            e_before_gd = sorted(e_before_gd)
                            # extract the top 10 energies with best accuracy
                            e_after_gd = e_after_gd[:10]
                            e_before_gd = e_before_gd[:10]
                        chi_list.append(chi)
                        e_before_seed_list.append(e_before_gd)
                        e_after_seed_list.append(e_after_gd)
                    e_before_gd_list.append(e_before_seed_list)
                    e_after_gd_list.append(e_after_seed_list)
                    # make list empty again for next iteration
                    e_before_seed_list = []
                    e_after_seed_list = []
            else:
                for chi in range(2, 8):
                    # get data
                    e_before_gd, e_after_gd = ad.get_energies(
                        Delta, N, chi, model=model, map=map, normalize=normalize
                    )
                    if normalize == True:
                        e_after_gd = 1 - np.asarray(e_after_gd)
                        e_before_gd = 1 - np.asarray(e_before_gd)
                    else:
                        e_after_gd = e_N - np.asarray(e_after_gd)
                        e_before_gd = e_N - np.asarray(e_before_gd)
                    e_before_gd_list.append(e_before_gd)
                    e_after_gd_list.append(e_after_gd)
                    chi_list.append(chi)
            # print('Plotting energies for Delta = {}, N = {}, chi = {}'.format(Delta, N, chi))
            print("Plotting energies for Delta = {}, N = {}".format(Delta, N))
            fig, ax = plt.subplots(figsize=(10, 6))
            chi_label = ["$\chi = $ {}".format(i) for i in range(2, 8)]
            l = [i for i in range(2, 8)]

            p = list(range(2, 8))
            positions = [p for _ in range(len(e_after_gd_list))]
            # Create a boxplot
            # ax.boxplot(e_after_gd_list, labels=chi_label)
            for i, (data_set, pos) in enumerate(zip(e_after_gd_list, positions)):
                # Create boxplot for each data set with specified position and color
                # dont show outlier points
                print("data_set =\n", data_set)
                bp = ax.boxplot(data_set, 0, "", positions=pos)

                # Set color for the box, whiskers, maximum, and minimum
                for key in bp.keys():
                    if isinstance(bp[key], list):
                        for component in bp[key]:
                            # if key in ['boxes', 'whiskers', 'caps', 'medians']:
                            if key in ["boxes", "whiskers", "caps"]:
                                component.set(color=standard_colors[i], linewidth=1)
                            elif key in ["medians"]:
                                component.set(color="black", linewidth=1)

            # Plot invisible scatter points with labels
            for i, s in zip(range(len(positions)), args[0]):
                ax.scatter(
                    [],
                    [],
                    c=standard_colors[i],
                    label="Random Hamiltonian seed {}".format(s),
                )
            plt.legend()
            ax.set_xticks(p)
            ax.set_xticklabels(chi_label)

            # plt.title(r'$\Delta E = {0:.2e}$'.format(de), loc='right')
            # extract seed out of args
            # if random Hamiltonian: get seed name
            # plt.title('seed = {}'.format(args[0]), loc='right')
            path = ce.create_path("plots/energies/" + model + map)
            ax.set_yscale(yscale)
            # ax.set_title('Accuracy of the Ground State Energy of a Random Hamiltonian (seed {})\nn = {}'.format(seed, N),
            #            fontsize=BIGGER_SIZE)
            ax.set_title(
                "Accuracy of the Ground State Energy of Random Hamiltonians\nn = {}".format(
                    N
                ),
                fontsize=BIGGER_SIZE,
            )
            ax.set_xlabel("Coarse-Graining Dimension $\chi$")
            ax.set_ylabel("Accuracy After Optimization")
            # y_lower, y_upper = plt.ylim()
            # plt.ylim(10e-42, y_upper)
            # plt.savefig(path+'energies_Delta{}_N{}.pdf'.format(Delta, N))
            # plt.savefig(path+'boxplots_N{}.pdf'.format(N))
            # print('Plot saved successfully')
            figs_list.append(fig)
            axes_list.append(ax)
            # plt.close("all")
            plt.show()
            # except:
            #    print('No data found')
    return figs_list, axes_list


"""
seeds = [3,6,9]
for s in seeds:
    #plot_energies('random_Hamiltonian/seed{}/'.format(s), 'random_maps/')
    ham = ds.Build_random_H(s)
    xscale='linear'
    yscale = 'log'
    figs, axes = boxplots('random_Hamiltonian/seed{}/'.format(s), 'random_maps/', s, ham,
                                normalize=False, xscale=xscale, yscale=yscale)
"""
"""
seeds = [3,6,9]
xscale='linear'
yscale = 'log'
figs, axes = boxplots('random_Hamiltonian/', 'random_maps/', seeds,
                            normalize=False, xscale=xscale, yscale=yscale)
"""
# figs, axes = boxplots('XXZ/', 'random_maps/', normalize=False, xscale='linear', yscale='log')
#


def plot_subspace_energies(model, map, *args):
    model = model
    map = map
    for Delta in range(1, 2):
        for N in range(5, 7, 2):
            ham = ds.get_Hamiltonian_full(model, N, args)
            fig, ax = create_plot_template()
            for chi in range(2, 3):
                # try:
                # get data
                eigvals, eigvals_H, spectrum, eigval0, spectrum0 = ad.calc_q_dagger_H_q(
                    Delta, N, chi, ham, model, map, *args
                )
                # Extract first and second entries separately
                entries = []
                entries0 = []
                # print('len(spectrum[0]) = ', len(spectrum[0]))
                for i in range(len(spectrum[0])):
                    e = [entry[i] for entry in spectrum]
                    entries.append(e)
                for i in range(len(spectrum0[0])):
                    e = [entry[i] for entry in spectrum0]
                    entries0.append(e)
                print(
                    "Plotting energies for Delta = {}, N = {}, chi = {}".format(
                        Delta, N, chi
                    )
                )
                # ax.set_title('Spectrum of the {}-body XX Model Hamiltonian and the Coarsed-Grained Hamiltonian\n$ \chi = {}$'.format(N-2, chi),
                #            fontsize=MEDIUM_SIZE)
                ax.set_title(
                    "Spectrum of the {}-body Heisenberg Model Hamiltonian and the Coarsed-Grained Hamiltonian\n$ \chi = {}$".format(
                        N - 2, chi
                    ),
                    fontsize=MEDIUM_SIZE,
                )
                labels = [
                    "Ground State of the Optimized Coarsed-Grained Hamiltonian",
                    "1. Excited State of the Optimized Coarsed-Grained Hamiltonian",
                    "2. Excited State of the Optimized Coarsed-Grained Hamiltonian",
                    "3. Excited State of the Optimized Coarsed-Grained Hamiltonian",
                    "4. Excited State of the Optimized Coarsed-Grained Hamiltonian",
                    "5. Excited State of the Optimized Coarsed-Grained Hamiltonian",
                    "6. Excited State of the Optimized Coarsed-Grained Hamiltonian",
                    "7. Excited State of the Optimized Coarsed-Grained Hamiltonian",
                ]
                labels0 = [
                    "Ground State of the Coarsed-Grained Hamiltonian",
                    "1. Excited State of the Coarsed-Grained Hamiltonian",
                    "2. Excited State of the Coarsed-Grained Hamiltonian",
                    "3. Excited State of the Coarsed-Grained Hamiltonian",
                    "4. Excited State of the Coarsed-Grained Hamiltonian",
                    "5. Excited State of the Coarsed-Grained Hamiltonian",
                    "6. Excited State of the Coarsed-Grained Hamiltonian",
                    "7. Excited State of the Coarsed-Grained Hamiltonian",
                ]
                for i in range(len(spectrum[0])):
                    # ax.plot(entries[i], marker='x', linestyle='None', label=labels[i])
                    # ax.plot(entries0[i], marker='*', linestyle='None', label=labels0[i])
                    ax.plot(entries[i], marker="x", linestyle="dotted", label=labels[i])
                    ax.plot(entries0[i], marker="*", label=labels0[i])
                ax.hlines(
                    y=eigvals_H,
                    xmin=0,
                    xmax=len(eigvals),
                    colors="k",
                    linestyle="--",
                    label="Spectrum of the XX Hamiltonian",
                )
                # ax.hlines(y=eigvals_H, xmin=0, xmax=len(eigvals), colors='k', linestyle='--', label='Spectrum of the Heisenberg Hamiltonian')
                ax.set_xlabel("Random Coarse-Graining Maps $W$")
                ax.set_ylabel("Energy")
                # ax.set_ylabel(r'$E_{{ \mathrm{{ {} }} }}({},W)$'.format('LTI', N-2))
                # r'$E_{{ \mathrm{{ {} }} }}({})$'.format('LTI', N)
                plt.legend(fontsize=10)
                y_lower, y_upper = plt.ylim()
                plt.ylim(y_lower, 1.8 * y_upper)
                # plt.legend([a], ['min eigval','max eigval'])
                path = ce.create_path("plots/subspace_energies/" + model + map)
                plt.savefig(
                    path
                    + "subspace_energies_Delta{}_N{}_chi{}.pdf".format(Delta, N, chi)
                )
                print("saved")
                plt.close("all")
                # plt.show()
                # except:
                #    print('No data found')
    return 0


"""
seeds  = [3,6,9]
ham = ds.Build_random_H(3)
ham_full = ds.Build_H_full(N-2, ham)
"""
"""
model = 'XXZ/'
Delta = 1
plot_subspace_energies(model, 'random_maps/', Delta)
"""
# model = 'random_Hamiltonian/'
# plot_subspace_energies(model, 'random_maps/', 3)
