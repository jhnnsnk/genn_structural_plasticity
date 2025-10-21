import os
import numpy as np
import pickle
from pygenn import GeNNModel
import topographic_map_model as topomodel
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helpers


def plot_network_sketch(net_dict, sim_dict, data_source="precomputed"):

    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0], projection="3d", computed_zorder=False)
    ax_plot_network_sketch(ax, net_dict, sim_dict, data_source)

    plt.savefig(os.path.join(sim_dict["dir"], "topomap_network_sketch.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(sim_dict["dir"], "topomap_network_sketch.png"),
                bbox_inches="tight")


def ax_plot_network_sketch(ax, net_dict, sim_dict, data_source="precomputed"):
    """
    Parameters
    ----------
    data_source:
        precompute: use precomputed data arrays
        generate: use simple model to generate new data
        load: load from directory
    """
    xpos = net_dict["x_positions"]
    ypos = net_dict["y_positions"]
    # identify one neuron approximately in the centre and get connectivity data
    tgt_ind = int(net_dict["num_neurons"] / 2 -
                  xpos[-1] / 2 - 1)
    tgt_ind += 1

    if data_source == "precomputed":
        data_ff = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        data_lat = np. array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    else:
        if data_source == "generate":
            pre_inds_ff, post_inds_ff, pre_inds_lat, post_inds_lat = \
                get_connectivity_from_simple_model(net_dict, sim_dict)

        elif data_source == "load":
            pre_inds_ff, post_inds_ff, pre_inds_lat, post_inds_lat = \
                load_connectivity_data(sim_dict["dir"])

        data_ff, data_lat = get_data_tgt_ind(
            pre_inds_ff, post_inds_ff, pre_inds_lat, post_inds_lat,
            xpos, ypos, tgt_ind)

    # define colors
    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    # in-degrees
    color_ff = "darkgreen"
    cmap_ff = {0: "white",
               1: "mediumaquamarine",
               2: "seagreen",
               3: "darkgreen",
               4: "darkgreen"}
    color_lat = "saddlebrown"
    cmap_lat = {0: "white",
                1: "sandybrown",
                2: "chocolate",
                3: "chocolate",
                4: "chocolate"}

    # target layer
    for x in np.arange(net_dict["grid_num_x"]):
        for y in np.arange(net_dict["grid_num_y"]):
            X, Y = np.meshgrid([x-0.5, x+0.5], [y-0.5, y+0.5])
            ax.plot_surface(X, Y, np.array([[0]]),
                            cstride=1, rstride=1,
                            shade=False, edgecolors="lightgrey",
                            color=cmap_lat[data_lat[x, y]])

    # lateral connections
    for x in np.arange(net_dict["grid_num_x"]):
        for y in np.arange(net_dict["grid_num_y"]):
            if data_lat[x, y] > 0:
                ax.plot(xs=[xpos[tgt_ind], x], ys=[ypos[tgt_ind], y], zs=[0, 0],
                        color=color_lat)

    # feed-forward connections
    for x in np.arange(net_dict["grid_num_x"]):
        for y in np.arange(net_dict["grid_num_y"]):
            if data_ff[x, y] > 0:
                ax.plot(xs=[xpos[tgt_ind], x], ys=[ypos[tgt_ind], y], zs=[0, 1],
                        color=color_ff)

    # source layer
    for x in np.arange(net_dict["grid_num_x"]):
        for y in np.arange(net_dict["grid_num_y"]):
            X, Y = np.meshgrid([x-0.5, x+0.5], [y-0.5, y+0.5])
            ax.plot_surface(X, Y, np.array([[1]]),
                            cstride=1, rstride=1,
                            shade=False, edgecolors="lightgrey",
                            color=cmap_ff[data_ff[x, y]], zorder=10)

    # preferred location
    ax.plot(xs=xpos[tgt_ind], ys=ypos[tgt_ind],
            zs=1, color="k", marker="x", linestyle="", zorder=20)

    ticks = np.arange(0, net_dict["grid_num_x"]+1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xlabel("x-position")
    ax.set_ylabel("y-position")

    ax.set_xlim(-0.9, net_dict["grid_num_x"]-0.1)
    ax.set_ylim(-0.9, net_dict["grid_num_y"]-0.1)

    # hide
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_zticks([])
    ax.grid(False)

    # text
    ax.text2D(0., 0.79, "Source layer\n(Correlated input)",
              transform=ax.transAxes, ha="left", zorder=100)
    ax.text2D(0., 0.35, "Target layer\n(Neurons)",
              transform=ax.transAxes, ha="left", zorder=100)
    ax.text2D(0.75, 0.5, "Feed-forward\nconnections",
              transform=ax.transAxes, ha="center", color=color_ff, zorder=100)
    ax.text2D(0.75, 0.33, "Lateral\nconnections",
              transform=ax.transAxes, ha="center", color=color_lat, zorder=100)
    ax.text2D(0.5, 0.77, "Ideal\nlocation",
              transform=ax.transAxes, ha="center", zorder=100)

    ax.view_init(elev=20, azim=-50)
    # plt.axis("off")
    return


def get_connectivity_from_simple_model(net_dict, sim_dict):
    # set up GeNN model
    model = GeNNModel(model_name="NetworkSketch")

    # neuron groups
    ng_src = topomodel.add_source_layer_neuron_population_correlated_input(
        model, net_dict)
    ng_tgt = topomodel.add_target_layer_neuron_population(
        model, net_dict)

    # synapse groups
    sg_ff = model.add_synapse_population(
        pop_name="FeedForwardInitConnectivity",
        matrix_type="SPARSE",
        source=ng_src, target=ng_tgt,
        weight_update_init=net_dict["weight_update_init_STDP"],
        postsynaptic_init=topomodel.get_postsynaptic_init(ng_tgt, net_dict),
        connectivity_init=net_dict["connectivity_initialiser_ff_GaussWoReplace"])

    sg_lat = model.add_synapse_population(
        pop_name="LateralInitConnectivity",
        matrix_type="SPARSE",
        source=ng_src, target=ng_tgt,
        weight_update_init=net_dict["weight_update_init_STDP"],
        postsynaptic_init=topomodel.get_postsynaptic_init(ng_tgt, net_dict),
        connectivity_init=net_dict["connectivity_initialiser_lat_GaussWoReplace"])

    # initialise simulation
    num_timesteps = topomodel.initialise_simulation(model, sim_dict)

    # build and load
    model.build()
    model.load(num_recording_timesteps=num_timesteps)

    # get connectivity
    pre_inds_ff, post_inds_ff = topomodel.get_connectivity(sg_ff)
    pre_inds_lat, post_inds_lat = topomodel.get_connectivity(sg_lat)

    return pre_inds_ff, post_inds_ff, pre_inds_lat, post_inds_lat


def load_connectivity_data(dir):

    # load connection data of selected connection type at initial time step
    with open(os.path.join(dir, "data_pre_inds.pkl"),  'rb') as f:
        data_pre_inds = pickle.load(f)
        pre_inds_ff = data_pre_inds["ff"][0]
        pre_inds_lat = data_pre_inds["lat"][0]
    with open(os.path.join(dir, "data_post_inds.pkl"),  'rb') as f:
        data_post_inds = pickle.load(f)
        post_inds_ff = data_post_inds["ff"][0]
        post_inds_lat = data_post_inds["lat"][0]

    return pre_inds_ff, post_inds_ff, pre_inds_lat, post_inds_lat


def get_data_tgt_ind(pre_inds_ff, post_inds_ff, pre_inds_lat, post_inds_lat,
                     xpos, ypos, tgt_ind):

    def get_data(pre_inds, post_inds, tgt_ind):
        data = np.zeros((xpos[-1] + 1, ypos[-1] + 1), dtype=int)
        num_scr = 0
        for i, post_ind in enumerate(post_inds):
            if post_ind == tgt_ind:
                pre_ind = pre_inds[i]
                data[xpos[pre_ind], ypos[pre_ind]] += 1
                num_scr += 1
        return data

    data_ff = get_data(pre_inds_ff, post_inds_ff, tgt_ind)
    data_lat = get_data(pre_inds_lat, post_inds_lat, tgt_ind)

    print("data_ff:", repr(data_ff), "sum:", np.sum(data_ff))
    print("data_lat:", repr(data_lat), "sum", np.sum(data_lat))

    # just for checking
    if 0:
        plot_sources_matrix(
            f"test_figure", pre_inds_ff, post_inds_ff, xpos, ypos, tgt_ind)

    return data_ff, data_lat


def ax_plot_correlated_input_sketch(ax, net_dict, shift=(2, 3)):

    grid_num_x_default = net_dict["grid_num_x_default"]
    grid_num_y_default = grid_num_x_default
    corr_base_rate = net_dict["corr_base_rate"]
    corr_peak_rate = net_dict["corr_peak_rate"]
    corr_std = net_dict["corr_std"]

    xpre = 0
    ypre = 0
    # Gauss profile
    stim_rates = np.ones(
        (grid_num_x_default, grid_num_y_default)) * corr_base_rate
    for xpost in np.arange(grid_num_x_default):
        for ypost in np.arange(grid_num_y_default):
            dx, dy = helpers.get_dx_dy_pbc(
                xpre, xpost, ypre, ypost, grid_num_x_default, grid_num_y_default)
            dd = dx * dx + dy * dy

            gauss_rate = corr_peak_rate * \
                np.exp(-dd / (2. * corr_std * corr_std))
            stim_rates[xpost, ypost] += gauss_rate

    # roll matrix
    matrix = np.roll(stim_rates, shift=shift, axis=(0, 1))

    im = ax.imshow(matrix.T, extent=(
        -0.5, grid_num_x_default-0.5, grid_num_y_default-0.5, -0.5),
        cmap=matplotlib.cm.plasma, vmin=corr_base_rate, vmax=corr_base_rate+corr_peak_rate)

    ticks = np.arange(0, grid_num_x_default+1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # ax.set_title("Correlated input")
    ax.set_xlabel("x-position")
    ax.set_ylabel("y-position")

    # color bar
    cax = ax.inset_axes(
        # left, bottom, width, height
        [1.05, 0, 0.06, 1], transform=ax.transAxes)
    cb = plt.colorbar(im, cax=cax, label="Rate (spikes/s)")
    cb.set_ticks([corr_base_rate, 50, 100, 150])
    # cb.locator = MaxNLocator(nbins=4)
    cb.update_ticks()

    # center marker
    ax.plot(shift[0], shift[1], color="k", marker="x", linestyle="")
    ax.text(shift[0]+0.5, shift[1], "$t_i$",
            color="k", verticalalignment="top")

    return


def plot_sources_matrix(
        fname,
        pre_inds, post_inds, xpos, ypos, tgt_ind=None, cbar=True):
    """Assume that xpos and ypos are the same for the pre- and postsynaptic
    populations.
    """
    # if tgt_ind is not specified, one approximately at the centre is picked
    if tgt_ind == None:
        pop_size = len(xpos)
        if pop_size % 2 == 0:
            tgt_ind = int(pop_size / 2 - xpos[-1] / 2 - 1)
        else:
            tgt_ind = int(pop_size / 2)

    # get connection data
    data = np.zeros((xpos[-1] + 1, ypos[-1] + 1))
    num_scr = 0
    for i, post_ind in enumerate(post_inds):
        if post_ind == tgt_ind:
            pre_ind = pre_inds[i]
            data[xpos[pre_ind], ypos[pre_ind]] += 1
            num_scr += 1
    print(num_scr)

    data = data.T

    # define color map
    color_map = {0: np.array([255, 255, 255]),  # white
                 1: np.array([119, 183, 125]),
                 2: np.array([84, 158, 179]),
                 3: np.array([85, 104, 184]),
                 4: np.array([155, 98, 167])}

    # make a 3d numpy array that has a color channel dimension
    data_3d = np.ndarray(shape=(data.shape[0], data.shape[1], 3), dtype=int)

    # set no sources to nan
    for i in np.arange(np.shape(data)[0]):
        for j in np.arange(np.shape(data)[1]):
            data_3d[i][j] = color_map[data[i][j]]

    fig, ax = plt.subplots(figsize=(5, 5))

    # plot all positions of presynaptic neurons
    ax.plot(xpos, ypos, "o", color="k", label="source population")

    # plot sources
    im = ax.imshow(data_3d, aspect='auto')

    # plot target neuron
    ms_tgt = 2 * matplotlib.rcParams["lines.markersize"]
    ax.plot(xpos[tgt_ind], ypos[tgt_ind], "o", color="red",
            markersize=ms_tgt, label=f"target (id={tgt_ind})")

    # axes limits with PBC
    ax.set_xlim(0, np.max(xpos)+0.999)
    ax.set_ylim(0, np.max(ypos)+0.999)

    ax.set_xlabel("x-position")
    ax.set_ylabel("y-position")

    legend_elements = []
    max_indegree = np.max(data)
    for i in np.arange(1, max_indegree + 1):
        legend_elements.append(
            matplotlib.lines.Line2D(
                [], [], color=color_map[i]/255, marker="s", linestyle='None', markersize=ms_tgt, label=int(i)))
    legend_elements.append(matplotlib.lines.Line2D(
        [], [], color="red", marker="o", linestyle='None', markersize=ms_tgt, label="Target\n" + f"(ID={tgt_ind})"))

    ax.legend(handles=legend_elements,
              title="In-degree", bbox_to_anchor=(1, 1))

    plt.savefig(f"{fname}.pdf", bbox_inches="tight")
