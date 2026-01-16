import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
import helpers
import network_sketch as ns

pd = helpers.plot_dict

plt.rc('font', size=pd["SMALL_SIZE"])          # controls default text sizes
plt.rc('axes', titlesize=pd["BIGGER_SIZE"])    # fontsize of the axes title
plt.rc('axes', labelsize=pd["MEDIUM_SIZE"])    # fontsize of the x and y labels
plt.rc('xtick', labelsize=pd["SMALL_SIZE"])    # fontsize of the tick labels
plt.rc('ytick', labelsize=pd["SMALL_SIZE"])    # fontsize of the tick labels
plt.rc('legend', fontsize=pd["SMALL_SIZE"])    # legend fontsize
plt.rc('figure', titlesize=pd["BIGGER_SIZE"])  # fontsize of the figure title

################################################################################
# 1. Methods figure
################################################################################


def network_sketch_init_final(net_dict, sim_dict, data_source="load"):
    print("Plotting network sketch, and initial vs. final")

    fig = plt.figure(figsize=(pd["figure_width"], 6))
    gs = gridspec.GridSpec(6, 5)
    gs.update(left=0.01, right=0.98, wspace=2,
              hspace=0.5, top=1.05, bottom=0.08)

    # A: network sketch
    ax = plt.subplot(gs[:4, :3], projection="3d", computed_zorder=False)
    ns.ax_plot_network_sketch(ax, net_dict, sim_dict, data_source)

    # B: correlated input illustration
    ax = plt.subplot(gs[-2:, :3])
    helpers.add_label(ax, "B Correlated input")
    ns.ax_plot_correlated_input_sketch(ax, net_dict)

    # label for A with non 3D axes
    helpers.add_label(ax, "A Initial connections", offset=(0, 2.1))

    # C: feed-forward connections and weights
    ax = gs_plot_mean_conns_and_weights_vs_distance(
        gs[1:3, 3:], net_dict, sim_dict, conntype="ff")
    helpers.add_label(ax, "C Feed-forward connections", offset=(-0.3, 0.7))

    # D: lateral connections and weights
    ax = gs_plot_mean_conns_and_weights_vs_distance(
        gs[4:, 3:], net_dict, sim_dict, conntype="lat")
    helpers.add_label(ax, "D Lateral connections", offset=(-0.3, 0.7))

    plt.savefig(os.path.join(sim_dict["dir"],
                "topomap_network_sketch_init_final.png"))
    plt.savefig(os.path.join(sim_dict["dir"],
                             "topomap_network_sketch_init_final.pdf"))
    return


def gs_plot_mean_conns_and_weights_vs_distance(gspec, net_dict, sim_dict, conntype="ff"):

    grid_num_x = net_dict["grid_num_x"]
    grid_num_y = net_dict["grid_num_y"]
    scale = net_dict["scale"]

    t_init, t_final, all_num_conns, all_mean_conns, all_mean_weights = \
        get_data_mean_conns_and_weights_vs_distance(
            sim_dict["dir"], net_dict, conntype)

    max_conns = np.max((all_mean_conns[t_init], all_mean_conns[t_final]))
    max_weights = np.max((all_mean_weights[t_init], all_mean_weights[t_final]))

    # sub-gridspec
    gssub = gspec.subgridspec(2, 2, wspace=0.3)

    for t, time in enumerate([t_init, t_final]):
        for i, data, vmin, vmax, label in zip(
                [0, 1],
                [all_mean_conns[time], all_mean_weights[time]],
                [0, 0],
                [max_conns, max_weights],
                ["Conn. probs.",
                 "Weights"]):
            ax = plt.subplot(gssub[2*t + i])
            masked_data = np.ma.masked_where(all_num_conns[time] == 0., data)

            # for checking:
            # if i == 1:
            #    masked_data[4:10, 5] = 2

            data_mT = masked_data.transpose()

            im = ax.imshow(data_mT,
                           origin="lower", vmin=vmin, vmax=vmax,
                           cmap=matplotlib.cm.viridis)

            xticks = np.arange(1, grid_num_x, step=4*scale)
            yticks = np.arange(1, grid_num_y, step=4*scale)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([int(xtick - grid_num_x/2 + 1)
                               for xtick in xticks])
            ax.set_yticklabels([int(ytick - grid_num_y/2 + 1)
                               for ytick in yticks])

            if t == 0 and i == 0:
                ax.set_ylabel(
                    r"$\mathbf{Initial}$" + "\ny-displacem.")
                ax_label = ax
            elif t == 1 and i == 0:
                ax.set_ylabel(
                    r"$\mathbf{Final}$" + "\ny-displacem.")

            if t == 1:
                ax.set_xlabel("x-displacem.")

            if t == 0:
                ax.set_xticklabels([])

                # color bar
                cax = ax.inset_axes(
                    # left, bottom, width, height
                    [0, 1.3, 1, 0.1], transform=ax.transAxes)
                cb = plt.colorbar(im, cax=cax, label=label,
                                  orientation="horizontal")
                cb.locator = MaxNLocator(nbins=3)
                cb.update_ticks()
                cax.xaxis.set_label_position("top")

            if i == 1:
                ax.set_yticklabels([])
    return ax_label


def get_data_mean_conns_and_weights_vs_distance(dir, net_dict, conntype="ff"):
    grid_num_x = net_dict["grid_num_x"]
    grid_num_y = net_dict["grid_num_y"]

    # use only grid_num_x throughout
    assert (grid_num_x == grid_num_y)
    assert (grid_num_x % 2 == 0)

    num_neurons = grid_num_x * grid_num_x

    # load connection data of selected connection type
    with open(os.path.join(dir, "data_pre_inds.pkl"),  'rb') as f:
        data_pre_inds = pickle.load(f)[conntype]
    with open(os.path.join(dir, "data_post_inds.pkl"),  'rb') as f:
        data_post_inds = pickle.load(f)[conntype]
    with open(os.path.join(dir, "data_weights.pkl"),  'rb') as f:
        data_weights = pickle.load(f)[conntype]

    # initial and final data
    all_num_conns = {}
    all_mean_conns = {}
    all_mean_weights = {}
    t_init = np.min(list(data_pre_inds.keys()))
    t_final = np.max(list(data_pre_inds.keys()))

    # gather and combine data
    for time in [t_init, t_final]:
        mean_weights = np.zeros((grid_num_x, grid_num_x))
        num_conns = np.zeros_like(mean_weights, dtype=int)
        for i in np.arange(len(data_weights[time])):
            pre_id = data_pre_inds[time][i]
            post_id = data_post_inds[time][i]

            dx, dy = helpers.get_dx_dy_pbc_from_ids(
                pre_id, post_id, grid_num_x, grid_num_x)

            # -1 because of how PBC is implemented for getting dx and dy
            xidx = dx + int(grid_num_x / 2 - 1)
            yidx = dy + int(grid_num_x / 2 - 1)

            mean_weights[xidx, yidx] += data_weights[time][i]
            num_conns[xidx, yidx] += 1

        # normalization of weights by number of connections
        for xidx in np.arange(grid_num_x):
            for yidx in np.arange(grid_num_x):
                if num_conns[xidx, yidx] > 0:
                    mean_weights[xidx, yidx] /= num_conns[xidx, yidx]

        # mean number of connections is existing number / maximal possible number
        # note: this does not account for multapses
        mean_conns = num_conns / num_neurons

        all_num_conns[time] = num_conns
        all_mean_conns[time] = mean_conns
        all_mean_weights[time] = mean_weights

    return t_init, t_final, all_num_conns, all_mean_conns, all_mean_weights


################################################################################
# 2. Topomap formation (analysis + plotting)
################################################################################

def analyze_connectivity_evolution(dir, net_dict, conntype="ff"):
    print(f"Analyzing connectivity evolution for {conntype}.")
    with open(os.path.join(dir, "data_pre_inds.pkl"),  'rb') as f:
        data_pre_inds = pickle.load(f)[conntype]
    times = list(data_pre_inds.keys())

    with open(os.path.join(dir, "data_post_inds.pkl"),  'rb') as f:
        data_post_inds = pickle.load(f)[conntype]

    with open(os.path.join(dir, "data_weights.pkl"),  'rb') as f:
        data_weights = pickle.load(f)[conntype]

    def sets_to_distances(sets):
        distances = np.empty(len(sets))
        for i, pre_post in enumerate(sets):
            dx, dy = helpers.get_dx_dy_pbc_from_ids(
                pre_post[0], pre_post[1], net_dict["grid_num_x"], net_dict["grid_num_y"])
            distances[i] = np.sqrt(dx**2 + dy**2)
        return distances

    # changed connections: eliminated and formed
    changed_conns = {
        "times_ms": times,
        "eliminated": [],
        "formed": []}
    for i in np.arange(len(times)-1):
        pairs_prev = set(
            zip(data_pre_inds[times[i]], data_post_inds[times[i]]))
        pairs_curr = set(
            zip(data_pre_inds[times[i+1]], data_post_inds[times[i+1]]))

        eliminated = pairs_prev - pairs_curr
        if len(eliminated) > 0:
            changed_conns["eliminated"].append(
                # time stamp, list of distances at which synapses have been eliminated
                [times[i], sets_to_distances(eliminated)])
        formed = pairs_curr - pairs_prev
        if len(formed) > 0:
            changed_conns["formed"].append(
                # time stamp, list of distances at which synapses have been formed
                [times[i], sets_to_distances(formed)])

    with open(os.path.join(dir, f"connectivity_evolution_{conntype}.pkl"),  'wb') as f:
        pickle.dump(changed_conns, f)

    ############################################################################

    # connection probabilities and weights
    if net_dict["grid_num_x"] != net_dict["grid_num_y"]:
        raise Exception
    grid_num_x = net_dict["grid_num_x"]

    # start after first rewiring
    all_mean_conns = np.empty((len(times)-1, grid_num_x))
    all_mean_weights = np.empty((len(times)-1, grid_num_x))

    # gather and combine data
    for t, time in enumerate(times[1:]):
        num_conns = np.zeros((grid_num_x), dtype=int)
        mean_weights = np.zeros((grid_num_x))
        for i in np.arange(len(data_pre_inds[time])):
            pre_id = data_pre_inds[time][i]
            post_id = data_post_inds[time][i]

            # only interested in dx = 0
            if np.mod(post_id - pre_id, grid_num_x) != 0:
                continue

            dx, dy = helpers.get_dx_dy_pbc_from_ids(
                pre_id, post_id, grid_num_x, grid_num_x)
            if dx != 0:
                raise Exception

            # -1 because of how PBC is implemented for getting dx and dy
            # xidx = dx + int(grid_num_x / 2 - 1)
            yidx = dy + int(grid_num_x / 2 - 1)

            mean_weights[yidx] += data_weights[time][i]
            num_conns[yidx] += 1

        # normalization of weights by number of connections
        for yidx in np.arange(grid_num_x):
            if num_conns[yidx] > 0:
                mean_weights[yidx] /= num_conns[yidx]

        # mean number of connections is existing number / maximal possible number
        # note: this does not account for multapses
        # (which are not used in the model anyways)
        all_mean_conns[t] = num_conns / net_dict["num_neurons"]
        all_mean_weights[t] = mean_weights

    with open(os.path.join(dir, f"connectivity_evolution_mean_conns_{conntype}.pkl"),  'wb') as f:
        pickle.dump(all_mean_conns, f)
    with open(os.path.join(dir, f"connectivity_evolution_mean_weights_{conntype}.pkl"),  'wb') as f:
        pickle.dump(all_mean_weights, f)
    return


def analyze_connections_per_neuron(dir, net_dict, conntype="ff"):
    print(f"Analyzing connections per neuron for {conntype}.")

    with open(os.path.join(dir, "data_pre_inds.pkl"),  'rb') as f:
        data_pre_inds = pickle.load(f)
        data_pre_inds = data_pre_inds[conntype]
    with open(os.path.join(dir, "data_post_inds.pkl"),  'rb') as f:
        data_post_inds = pickle.load(f)
        data_post_inds = data_post_inds[conntype]

    conns = {}
    conns["times"] = list(data_pre_inds.keys())  # same for pre and post

    for data_inds, label in zip([data_post_inds, data_pre_inds],
                                ["post", "pre"]):

        mean = np.ones(len(data_inds)) * np.nan
        std = np.ones(len(data_inds)) * np.nan
        for i, time in enumerate(data_inds):
            unique, counts = np.unique(data_inds[time], return_counts=True)
            mean[i] = np.mean(counts)
            std[i] = np.std(counts)

        conns[label] = {"mean": mean, "std": std}

    with open(os.path.join(dir, f"connections_per_neuron_{conntype}.pkl"),  'wb') as f:
        pickle.dump(conns, f)


def plot_connectivity_evolution(dir, net_dict, conntype="ff",
                                binwidth_heat_time_ms=200.,
                                binwidth_heat_distance=1.,
                                binwidth_hist_time_ms=200.):
    print(f"Plotting connectivity evolution for {conntype}.")

    with open(os.path.join(dir, f"connectivity_evolution_{conntype}.pkl"),  'rb') as f:
        changed_conns = pickle.load(f)
    times = changed_conns["times_ms"]

    with open(os.path.join(dir, f"connections_per_neuron_{conntype}.pkl"),  'rb') as f:
        conns = pickle.load(f)
        times_conns_per_neuron = conns["times"]
        conns_pre = conns["pre"]
        conns_post = conns["post"]

    # standard deviation auf Gaussion for formation
    if conntype == "ff":
        sigma = net_dict["std_ff"]
        title = "Feed-forward connections"
    elif conntype == "lat":
        sigma = net_dict["std_lat"]
        title = "Lateral connections"

    max_distance = 0.5 * \
        np.sqrt(net_dict["grid_num_x"]**2 + net_dict["grid_num_y"]**2)

    with open(os.path.join(dir, f"connectivity_evolution_mean_conns_{conntype}.pkl"),  'rb') as f:
        all_mean_conns = pickle.load(f)
    with open(os.path.join(dir, f"connectivity_evolution_mean_weights_{conntype}.pkl"),  'rb') as f:
        all_mean_weights = pickle.load(f)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, weight="bold")

    gs = gridspec.GridSpec(4, 1)
    gs.update(hspace=0.5, right=0.9, left=0.08, top=0.92, bottom=0.05)

    ############################################################################
    # A: Eliminations, B: Formations

    gs_elim = gs[0].subgridspec(2, 1, hspace=0.)
    gs_form = gs[1].subgridspec(2, 1, hspace=0.)

    # heat map
    for title, panel_label, change_type, gs_type in zip(
        ["Eliminations", "Formations"],
        ["A", "B"],
        ["eliminated", "formed"],
            [gs_elim, gs_form]):

        time_list = []
        distance_list = []
        for time, distances in changed_conns[change_type]:
            for d in distances:
                time_list.append(time)
                distance_list.append(d)
        time_list = np.array(time_list)
        distance_list = np.array(distance_list)

        time_bins = np.arange(
            times[1], times[-1] + binwidth_heat_time_ms, binwidth_heat_time_ms)
        distance_bins = np.arange(
            0, max_distance + binwidth_heat_distance, binwidth_heat_distance)

        heatmap, time_edges, distance_edges = np.histogram2d(
            time_list, distance_list, bins=[time_bins, distance_bins])

        ax_heat = plt.subplot(gs_type[0])
        helpers.add_label(ax_heat, panel_label)
        ax_heat.set_title(title)
        im = ax_heat.imshow(
            heatmap.T,
            interpolation="none",
            origin="lower",
            aspect="auto",
            extent=[time_edges[0], time_edges[-1],
                    distance_edges[0], distance_edges[-1]],
            # colormap that starts with white
            cmap=matplotlib.cm.ocean_r
        )

        # color bar
        cax = ax_heat.inset_axes(
            # left, bottom, width, height
            [1.005, 0, 0.01, 1], transform=ax_heat.transAxes)
        cb = plt.colorbar(im, cax=cax, label="Count")
        cb.locator = MaxNLocator(nbins=3)
        cb.update_ticks()

        # Standard deviation of Gaussian
        ax_heat.axhline(y=sigma, color=pd["color_formation"],
                        label=r"$\sigma_\mathrm{form}=$" + f"{sigma}")

        ax_heat.set_xticks([])
        ax_heat.set_ylabel("Distance")
        ax_heat.legend(loc="upper right", frameon=False)

    # histogram
    for change_type, gs_type, color in zip(["eliminated", "formed"],
                                           [gs_elim, gs_form],
                                           ["color_elimination", "color_formation"]):
        bin_counts = defaultdict(int)

        for time, distances in changed_conns[change_type]:
            bin_number = time // binwidth_hist_time_ms
            bin_start = bin_number * binwidth_hist_time_ms
            bin_counts[bin_start] += len(distances)

        bins = sorted(bin_counts.keys())
        counts = [bin_counts[b] for b in bins]
        # Elimination / formation rate
        # binwidth_hist_time is in ms
        rate = [c / (binwidth_hist_time_ms * 0.001) for c in counts]

        ax_hist = plt.subplot(gs_type[1])
        ax_hist.bar(bins, rate, width=binwidth_hist_time_ms,
                    align='edge', color=pd[color])

        ax_hist.set_xlim(times[1], times[-1])
        ax_hist.set_ylim(0, np.max(rate)*1.3)

        # change time tick labels from ms to s
        xticks = ax_hist.get_xticks()
        ax_hist.set_xticklabels(xticks * 0.001)
        ax_hist.set_xlabel("Time (s)")

        ax_hist.set_ylabel("Rate (s$^{-1}$)")

    ############################################################################
    # C: Connections per neuron (in-degree and out-degree)

    ax_degree = plt.subplot(gs[2])
    helpers.add_label(ax_degree, "C")
    ax_degree.set_title("Connections per neuron")
    for conns, label, lw, color in zip([conns_post, conns_pre],
                                       ["In-degree", "Out-degree"],
                                       [4., 1.],
                                       ["black", "firebrick"]):

        mean = conns["mean"]
        std = conns["std"]

        ax_degree.plot(times_conns_per_neuron, mean, '-',
                       linewidth=matplotlib.rcParams["lines.linewidth"]*lw,
                       color=color,
                       label=label + " (mean)")
        ax_degree.plot(times_conns_per_neuron, mean-std, '-',
                       linewidth=matplotlib.rcParams["lines.linewidth"]*lw,
                       color=color, alpha=0.5,
                       label=label + r" (mean $\pm$ std)")
        ax_degree.plot(times_conns_per_neuron, mean+std, '-',
                       linewidth=matplotlib.rcParams["lines.linewidth"]*lw,
                       color=color, alpha=0.5)

    # theoretical result of initial in-degree
    ax_degree.plot(times[0], 2*np.pi*net_dict[f"peak_prob_{conntype}"] *
                   net_dict[f"std_{conntype}"]**2, label=r"$K=2\pi p_\mathrm{form}\sigma_\mathrm{form}^2$",
                   marker="o", markersize=matplotlib.rcParams["lines.markersize"],
                   linestyle="none",
                   color="yellow",
                   markeredgecolor="black",
                   zorder=10, clip_on=False)
    ax_degree.set_ylim(bottom=0)

    ax_degree.legend(loc="lower right", ncol=3, frameon=False)
    ax_degree.set_xlim(times[1], times[-1])
    ax_degree.set_ylabel("Degree")

    # change time tick labels from ms to s
    xticks = ax_degree.get_xticks()
    ax_degree.set_xticklabels(xticks * 0.001)
    ax_degree.set_xlabel("Time (s)")

   ############################################################################
    # D: connections and weights

    gs_conns = gs[3].subgridspec(2, 1, hspace=0.)

    cmap = matplotlib.cm.get_cmap()
    cmap.set_bad(color='white')

    for i, data, label in zip([0, 1], [all_mean_conns, all_mean_weights],
                              ["Conn. probs.",
                               "Weights"]):

        ax_conns = plt.subplot(gs_conns[i])
        # mask where no connections are visible
        masked_data = np.ma.masked_where(
            all_mean_conns == 0., data)
        im_conns = ax_conns.imshow(masked_data.T,
                                   interpolation="none",
                                   origin="lower",
                                   aspect="auto",
                                   extent=[times[1], times[-1],
                                           -net_dict["grid_num_x"] // 2+1, net_dict["grid_num_x"] // 2],
                                   rasterized=True)
        if i == 0:
            helpers.add_label(ax_conns, "D")
            ax_conns.set_title("Connection probabilities and weights")
            ax_conns.set_xticks([])
            ax_conns.set_ylabel(r"y-displacement")

        if i == 1:
            # change time tick labels from ms to s
            xticks = ax_conns.get_xticks()
            ax_conns.set_xticklabels(xticks * 0.001)
            ax_conns.set_xlabel("Time (s)")

        # color bar
        cax_conns = ax_conns.inset_axes(
            # left, bottom, width, height
            [1.005, 0, 0.01, 1], transform=ax_conns.transAxes)
        cb_conns = plt.colorbar(im_conns, cax=cax_conns,
                                label=label)
        cb_conns.locator = MaxNLocator(nbins=3)
        cb_conns.update_ticks()

    plt.savefig(os.path.join(
        dir, f"topomap_connection_evolution_{conntype}.pdf"))
    plt.savefig(os.path.join(
        dir, f"topomap_connection_evolution_{conntype}.png"))
    return


################################################################################
# 3. Performance
################################################################################

def plot_performance(runs, grid_num_x_default, tsim):
    scales = []
    experiments = runs["info"]["experiments"]
    for exp in experiments:
        exp_split = exp.split("_")
        if exp_split[0] == "scale":
            scales.append(int(exp_split[1]))

    # scale to network size
    network_sizes = (grid_num_x_default * np.array(scales))**2

    time_names = {
        "py_simulate": ["Python simulation", "black"],
        "py_simulate_stimulus": ["Python stimulus", "#222255"],
        # "genn_init": ["GeNN init"],
        # "genn_sparse_init": ["GeNN sparse init"],
        "genn_neuron_update": ["GeNN neuron up.", "#666633"],
        "genn_presynaptic_update": ["GeNN presynaptic up.", "#CCEEFF"],
        "genn_postsynaptic_update": ["GeNN postsynaptic up.", "#225555"],
        "genn_connectivity_update": ["GeNN conn. up.", "#FFCCCC"],
        "genn_connectivity_update_host": ["GeNN conn. up. host", "#663333"],
        "genn_connectivity_update_remap_time": ["GeNN conn. up. remap time", "#CCDDAA"]
    }

    trunk_dir = runs["info"]["trunk_dir"]
    num_seeds = runs["info"]["num_seeds"]

    # prepare data
    data = {}
    for key in time_names.keys():
        data[key] = {}
        for scale in scales:
            data[key][scale] = {"values": []}

        for r in np.arange(len(scales) * num_seeds):
            dir = runs[r]["dir"]
            with open(os.path.join(dir, "times_s.pkl"),  'rb') as f:
                times_s = pickle.load(f)
                exp = runs[r]["experiment"]
                scale = int(exp.split("_")[1])
                data[key][scale]["values"].append(times_s[key])

        for scale in scales:
            data[key][scale]["mean"] = np.mean(data[key][scale]["values"])
            data[key][scale]["error"] = np.std(data[key][scale]["values"])

        # sort data into lists
        data[key]["means"] = [data[key][scale]["mean"] for scale in scales]
        data[key]["errors"] = [data[key][scale]["error"] for scale in scales]

    # figure
    plt.figure(figsize=(pd["figure_width"], 3))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.97, bottom=0.23,
              top=0.98, wspace=0.28, hspace=0.5)

    axlin = plt.subplot(gs[0])

    tsim_s = tsim * 0.001
    tmodel_label = r"$T_\mathrm{model}=$" + f"{int(tsim_s)} s"

    axlin.hlines(y=tsim_s,
                 xmin=np.min(network_sizes), xmax=np.max(network_sizes),
                 color="k", linestyles="dashed",
                 label=tmodel_label)

    bottom = np.zeros(len(scales))
    for key in ["genn_neuron_update",
                "genn_presynaptic_update",
                "genn_postsynaptic_update",
                "genn_connectivity_update",
                "genn_connectivity_update_host",
                "genn_connectivity_update_remap_time",
                "py_simulate_stimulus",
                "py_simulate"]:
        if not key == "py_simulate":
            top = bottom + np.array(data[key]["means"])
        label = time_names[key][0]
        color = time_names[key][1]
        if not key == "py_simulate":
            axlin.fill_between(x=network_sizes, y1=bottom, y2=top,
                               facecolor=color, edgecolor=None, label=label,
                               linewidth=matplotlib.rcParams["lines.linewidth"]*0.5)

        if key == "py_simulate":
            y = data[key]["means"]
            fmt = ''
            label = time_names[key][0]
        else:
            y = top
            fmt = 'none'
            label = ''
        axlin.errorbar(x=network_sizes, y=y, yerr=data[key]["errors"],
                       color=time_names[key][1],
                       label=label,
                       capsize=3,
                       capthick=0.5,
                       ecolor="k",
                       fmt=fmt
                       )
        if not key == "py_simulate":
            bottom = top

    axlin.set_ylabel("Time (s)")
    axlin.spines["top"].set_visible(False)
    axlin.spines["right"].set_visible(False)
    axlin.set_xticks(network_sizes)
    axlin.set_xticklabels(network_sizes, rotation=60)
    axlin.set_xlim(np.min(network_sizes), np.max(network_sizes))
    axlin.set_ylim(0, np.max(data["py_simulate"]["means"])*1.02)
    axlin.set_xlabel("Network size")
    axlin.legend(loc="upper left", frameon=False, reverse=True)

    plt.savefig(os.path.join(trunk_dir, "topomap_performance.png"))
    plt.savefig(os.path.join(trunk_dir, "topomap_performance.pdf"))
    return
