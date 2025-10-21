import os
import numpy as np
import helpers
from pygenn import GeNNModel
from pygenn import (create_var_ref, create_wu_var_ref, init_postsynaptic)


def initialize_topographic_map_model(net_dict):

    model = GeNNModel(model_name="TopographicMap")

    # source layer neuron population (different neuron model dependent on
    # whether correlated input is required)
    if net_dict["use_correlated_input"]:
        ng_src = add_source_layer_neuron_population_correlated_input(
            model, net_dict)
        stim_rates_matrix = compute_stimulus_rates_id0(
            net_dict["grid_num_x_default"],
            net_dict["corr_base_rate"],
            net_dict["corr_peak_rate"],
            net_dict["corr_std"],
            net_dict["scale"])
    else:
        ng_src = add_source_layer_neuron_population_uncorrelated_input(
            model, net_dict)
        stim_rates_matrix = None
    ng_src.spike_recording_enabled = True

    # target layer neuron population
    ng_tgt = add_target_layer_neuron_population(model, net_dict)
    ng_tgt.spike_recording_enabled = True

    # connectivity initialization type
    if net_dict["conn_init_type"] == "GaussianProfileWithoutReplacement":
        connectivity_initialiser_ff = net_dict["connectivity_initialiser_ff_GaussWoReplace"]
        connectivity_initialiser_lat = net_dict["connectivity_initialiser_lat_GaussWoReplace"]
    elif net_dict["conn_init_type"] == "GaussianProfileFixedNumberPreWithReplacement":
        connectivity_initialiser_ff = net_dict["connectivity_initialiser_ff_GaussWReplace"]
        connectivity_initialiser_lat = net_dict["connectivity_initialiser_lat_GaussWReplace"]
    elif net_dict["conn_init_type"] == "FixedProbability":
        connectivity_initialiser_ff = connectivity_initialiser_lat = \
            net_dict["connectivity_initialiser_FixedProb"]
    elif net_dict["conn_init_type"] == "OneToOne":
        connectivity_initialiser_ff = connectivity_initialiser_lat = \
            net_dict["connectivity_initialiser_OneToOne"]

    # weight update model
    if net_dict["use_STDP"]:
        weight_update_init = net_dict["weight_update_init_STDP"]
    else:
        weight_update_init = net_dict["weight_update_init_noSTDP"]

    # initial feedforward connectivity (source to target)
    sg_ff = model.add_synapse_population(
        pop_name="FeedForwardInitConnectivity",
        matrix_type="SPARSE",
        source=ng_src, target=ng_tgt,
        weight_update_init=weight_update_init,
        postsynaptic_init=get_postsynaptic_init(ng_tgt, net_dict),
        connectivity_init=connectivity_initialiser_ff)

    # initial lateral connectivity (target to target)
    sg_lat = model.add_synapse_population(
        pop_name="LateralInitConnectivity",
        matrix_type="SPARSE",
        source=ng_tgt, target=ng_tgt,
        weight_update_init=weight_update_init,
        postsynaptic_init=get_postsynaptic_init(ng_tgt, net_dict),
        connectivity_init=connectivity_initialiser_lat)

    # structural plasticity for feedforward and lateral connectivity
    splast_ff = add_connectivity_update(
        model, sg_ff, net_dict, conntype="ff")
    splast_lat = add_connectivity_update(
        model, sg_lat, net_dict, conntype="lat")

    return model, ng_src, ng_tgt, sg_ff, sg_lat, splast_ff, splast_lat, stim_rates_matrix

################################################################################
# helpers for model initialisation


def add_source_layer_neuron_population_correlated_input(mdl, ndict):
    ng = mdl.add_neuron_population(
        pop_name="SourceLayer",
        num_neurons=ndict["num_neurons"],
        neuron=ndict["neuron_model_src_corrTrue"],
        params=ndict["neuron_param_space_src_corrTrue"],
        vars=ndict["neuron_var_space_src_corrTrue"])
    return ng


def add_source_layer_neuron_population_uncorrelated_input(mdl, ndict):
    ng = mdl.add_neuron_population(
        pop_name="SourceLayer",
        num_neurons=ndict["num_neurons"],
        neuron=ndict["neuron_model_src_corrFalse"],
        params=ndict["neuron_param_space_src_corrFalse"],
        vars=ndict["neuron_var_space_src_corrFalse"])
    return ng


def add_target_layer_neuron_population(mdl, ndict):
    ng = mdl.add_neuron_population(
        pop_name="TargetLayer",
        num_neurons=ndict["num_neurons"],
        neuron=ndict["neuron_model_tgt"],
        params=ndict["neuron_param_space_tgt"],
        vars=ndict["neuron_var_space_tgt"])
    return ng


def get_postsynaptic_init(ng, ndict):
    # postsynaptic init is set here because it needs the neuron group in var_refs
    pi = init_postsynaptic(snippet=ndict["postsyn_model"],
                           params=ndict["ps_param_space"],
                           var_refs={"V": create_var_ref(ng, "V")})
    return pi


def add_connectivity_update(mdl, sg, ndict, conntype="ff"):
    """
    conntype: 'ff' (feedforward) or 'lat' (lateral)
    """
    if conntype == "ff":
        cu_name = "FeedForwardStructuralPlasticity"
        peak_prob = ndict["peak_prob_ff"]
        std = ndict["std_ff"]
    elif conntype == "lat":
        cu_name = "LateralStructuralPlasticity"
        peak_prob = ndict["peak_prob_lat"]
        std = ndict["std_lat"]

    splast = mdl.add_custom_connectivity_update(
        cu_name=cu_name,
        group_name="UpdateConnectivity",
        syn_group=sg,
        custom_conn_update_model=ndict["custom_conn_update_model"],
        params={
            "num_rewiring_attempts": ndict["num_rewiring_attempts"],
            "weight_threshold": ndict["weight_threshold"],
            "new_weight_init": ndict["new_weight_init"],
            "peak_prob": peak_prob,
            "std": std,
            "grid_num_x": ndict["grid_num_x"],
            "grid_num_y": ndict["grid_num_y"],
            "elimination_probability_dep": ndict["elimination_probability_dep"],
            "elimination_probability_pot": ndict["elimination_probability_pot"]},
        var_refs={"g": create_wu_var_ref(sg, "g")})
    splast.extra_global_params["num_rewiring_attempts_per_row"].set_init_values(
        np.zeros(ndict["num_neurons"]))
    splast.extra_global_params["bitfield"].set_init_values(
        np.zeros(ndict["bitfield_row_words"] * ndict["bitfield_num_rows"]))
    return splast

################################################################################


def initialise_simulation(mdl, sdict):
    mdl.seed = sdict['seed']
    mdl.dt = sdict['dt']

    ntimesteps = int(round(sdict['tsim'] / mdl.dt))

    mdl.timing_enabled = sdict["timing_enabled"]

    # create output directory if it does not exist yet
    if not os.path.exists(sdict["dir"]):
        os.makedirs(sdict["dir"])
    return ntimesteps


def initialise_extra_global_params_view(splast, ntimesteps, param_name):
    param_var = splast.extra_global_params[param_name]
    matrix_shape = (ntimesteps + 1, len(param_var.view))
    params = np.ones(matrix_shape, dtype=object) * np.nan

    add_to_extra_global_view(0, params, param_var)
    return params, param_var


def add_to_extra_global_view(tstep, params, param_var):
    param_var.pull_from_device()
    params[tstep, :] = param_var.view


def initialise_voltage_view(ng, ntimesteps):
    vector_shape = (ntimesteps + 1, ng.num_neurons)
    v = np.ones(vector_shape, dtype=float) * np.nan
    v_var = ng.vars["V"]

    add_to_voltage_view(0, v, v_var)
    return v, v_var


def add_to_voltage_view(tstep, v, v_var):
    v_var.pull_from_device()
    v[tstep, :] = v_var.current_view


def initialise_connectivity_view(sg, inds, ntimesteps):
    matrix_shape = (ntimesteps + 1,
                    sg.src.num_neurons * sg.trg.num_neurons)  # flattened
    c = np.ones(matrix_shape, dtype=bool) * np.nan  # False

    c[0, inds] = True
    return c


def initialise_num_connections_view(pre_inds, ntimesteps):
    # 2nd dimension is here time stamp
    nc = np.ones((ntimesteps + 1, 2), dtype=object) * np.nan

    nc[0] = [0, len(pre_inds)]
    return nc


def get_connectivity(sg):
    sg.pull_connectivity_from_device()
    pre_inds = sg.get_sparse_pre_inds()
    post_inds = sg.get_sparse_post_inds()
    return pre_inds, post_inds


def get_weights(sg):
    var = sg.vars["g"]
    var.pull_from_device()
    weights = var.values
    return weights


def get_connectivity_multi_index(sg):
    pre_inds, post_inds = get_connectivity(sg)
    inds = np.ravel_multi_index([pre_inds, post_inds],
                                (sg.src.num_neurons, sg.trg.num_neurons))
    return pre_inds, post_inds, inds


def initialise_weight_view(sg, inds, ntimesteps):
    matrix_shape = (ntimesteps + 1,
                    sg.src.num_neurons * sg.trg.num_neurons)  # flattened
    w = np.ones(matrix_shape, dtype=float) * np.nan
    w_var = sg.vars["g"]

    w_var.pull_from_device()
    w[0, inds] = w_var.values
    return w, w_var


def set_stimulus_rates(ng, grid_num_x_default, stim_rates_matrix):
    var = ng.vars["stim_rates"]
    var.pull_from_device()
    var.values = roll_stimulus_id(
        grid_num_x_default, stim_rates_matrix, randomize=True)
    var.push_to_device()


def compute_stimulus_rates_id0(
        grid_num_x_default, corr_base_rate, corr_peak_rate, corr_std, scale):
    grid_num_y_default = grid_num_x_default

    # assume stimulus id being first neuron id (top left corner of grid)
    xpre = 0
    ypre = 0
    # Gauss profile in first block
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

    # extend to full network size
    stim_rates_matrix = np.tile(stim_rates, reps=(scale, scale))

    return stim_rates_matrix


def roll_stimulus_id(grid_num_x_default, stim_rates_matrix,
                     randomize=True,
                     row_fix=0, col_fix=0):

    # stimulus location (row, col) in first block
    if randomize:
        row, col = np.random.randint(
            low=0, high=grid_num_x_default - 1, size=2)
    else:
        row = row_fix
        col = col_fix

    # roll matrix
    matrix_roll = np.roll(
        stim_rates_matrix, shift=(row, col), axis=(0, 1))

    # flatten
    stim_rates = matrix_roll.flatten()
    return stim_rates
