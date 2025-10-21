import os
from pygenn import (init_sparse_connectivity, init_weight_update, init_var)

from custom_models_and_snippets import *

################################################################################
# network parameters
################################################################################

net_dict = {
    ############################################################################
    # network scale must be a square number
    "scale": 1,

    ############################################################################
    # spatial grid

    # number of neurons in x-direction (same number in y direction)
    "grid_num_x_default": 16,

    ############################################################################
    # neuron model of source layer

    # whether to use correlated input
    "use_correlated_input": True,

    # neuron model and parameters for correlated input
    "neuron_model_src_corrTrue": PoissonSpatiallyCorrelatedInput,
    # change location after each interval
    "neuron_param_space_src_corrTrue": {
        "stim_interval": 20.,  # ms
    },

    # variables of source neuron model
    "neuron_var_space_src_corrTrue": {
        "timeStepToSpike": 0.,
        "stim_rates": 0.  # 1/s,
    },

    # further parameters for calculating the stimulus rates
    "corr_base_rate": 5.,  # 1/s
    "corr_peak_rate": 152.8,  # 1/s
    "corr_std": 2.,  # in units of grid spacing

    # neuron model and parameters for non-correlated input
    "neuron_model_src_corrFalse": "Poisson",
    "neuron_param_space_src_corrFalse": {
        "rate": 20.  # 1/s
    },

    # variables of source neuron model
    "neuron_var_space_src_corrFalse": {
        "timeStepToSpike": 0.,
    },

    ############################################################################
    # neuron model of target layer

    "neuron_model_tgt": "LIF",
    # parameters of target neuron model
    "neuron_param_space_tgt": {
        "C": 20.,  # nF
        "TauM": 20.,  # ms
        "Vrest": -70.,  # mV
        "Vreset": -70.,  # mV
        # Bogdan2018 code uses -50 mV, but -54 mV is in paper and Bamford2010
        "Vthresh": -54.,  # mV
        "Ioffset": 0.,
        "TauRefrac": 5.,  # ms (Bogdan uses 5 ms, Bamford none)
    },

    # variables of target neuron model
    "neuron_var_space_tgt": {
        "V": init_var("Normal", {"mean": -60., "sd": 5.}),  # arbitrarily set
        "RefracTime": 0.,  # ms (counted downwards)
    },

    ############################################################################
    # postsynaptic model

    # (Bogdan2018 uses PyNN's IF_cond_exp model)
    "postsyn_model": "ExpCond",
    # parameters of postsynaptic model (E is here set for excitatory input)
    "ps_param_space": {
        "tau": 5.,  # ms, decay time constant
        "E": 0.  # mV, reversal potential
    },

    ############################################################################
    # connectivity initialisation

    # choose connectivity initialization type, options are:
    "conn_init_type": "GaussianProfileWithoutReplacement",
    # "conn_init_type": "GaussianProfileFixedNumberPreWithReplacement",
    # "conn_init_type": "FixedProbability",
    # "conn_init_type": "OneToOne",
    # "conn_init_type": "AllToAll",
    # "conn_init_type": "Unconnected",


    # Spatial Gaussian profile,see derived params
    # initial in-degree for FixedNumberPre initialiser only
    "Gauss_init_indegree": 16,
    # peak connection probability and standard deviation for feedforward and
    # lateral connections
    "peak_prob_ff": 0.16,
    "std_ff": 2.5,
    "peak_prob_lat": 1.,
    "std_lat": 1.,

    # Pairwise Bernoulli initialiser
    "connectivity_initialiser_FixedProb": init_sparse_connectivity(
        snippet="FixedProbability",
        params={"prob": 0.1}
    ),

    # One-to-one initialiser
    "connectivity_initialiser_OneToOne": init_sparse_connectivity(
        snippet="OneToOne",
        params={}
    ),

    # All-to-all initialiser
    "connectivity_initialiser_AllToAll": init_sparse_connectivity(
        snippet="FixedProbability",
        params={"prob": 1.}
    ),

    # Unconnected initialiser
    "connectivity_initialiser_Unconnected": init_sparse_connectivity(
        snippet="FixedProbability",
        params={"prob": 0.0001}
    ),

    ############################################################################
    # weight update model

    # whether to use STDP
    "use_STDP": True,

    # maximum conductance
    "g_max": 0.2,

    # initial conductance
    # set to maximum as described in Bamford2010
    "g_init": 0.2,

    # for weight update init see derived params

    # STDP
    "tauPlus": 20.,  # ms
    "tauMinus": 64.,  # ms
    "aPlus": 0.1 * 0.2,  # Bogdan2018 use 0.1, but SpiNNaker scales with max weight
    "B": 1.2,  # to be converted into A_-

    ############################################################################
    # structural plasticity

    # name of custom connectivity update model
    "custom_conn_update_model": BamfordStructuralPlasticity,
    # update interval
    # (rewiring frequency of 10 kHz  = 10 attempts per time step of 1 ms)
    "update_interval": 1.,  # ms
    # number of rewiring attempts per update per block
    "num_rewiring_attempts_block": 10,
    # weight threshold (0.5 * g_max)
    "weight_threshold": 0.1,
    # initial weight of newly created synapse (g_max)
    "new_weight_init": 0.2,
    # Here, we use a much higher elimination probability (factor 50) than
    # Bamford because no competition imposed through maximum in-degree
    # elimination probability if weight below threshold
    "elimination_probability_dep": 0.0245 * 50,
    # elimination probability if weight above threshold
    "elimination_probability_pot": 0.000136 * 50,
}

################################################################################
# derived network parameters
################################################################################


def set_derived_network_params(net_dict):
    ############################################################################

    # number of neurons in x-and y-direction
    grid_num_x = net_dict["grid_num_x_default"] * net_dict["scale"]
    grid_num_y = grid_num_x

    # total number of neurons per population
    num_neurons = grid_num_x * grid_num_y

    net_dict.update({
        ########################################################################
        # spatial grid

        # number of neurons in x-direction
        "grid_num_x": grid_num_x,

        # number of neurons in y-direction
        "grid_num_y": grid_num_y,

        # total number of neurons per population
        "num_neurons": num_neurons,

        # x-positions: [0, 1, 2, ..., 0, 1, 2, ..., 0, 1, 2, ...]
        "x_positions": np.tile(np.arange(grid_num_x), grid_num_y).astype(int),

        # y-positions: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
        "y_positions": np.repeat(np.arange(grid_num_y), grid_num_x).astype(int),

        ########################################################################
        # connectivity initialisation

        # Spatial Gaussian profile initialiser without replacement
        "connectivity_initialiser_ff_GaussWoReplace": init_sparse_connectivity(
            snippet=GaussianProfileWithoutReplacement,
            params={"peak_prob": net_dict["peak_prob_ff"],
                    "std": net_dict["std_ff"],
                    "grid_num_x": grid_num_x,
                    "grid_num_y": grid_num_y
                    }
        ),
        "connectivity_initialiser_lat_GaussWoReplace": init_sparse_connectivity(
            snippet=GaussianProfileWithoutReplacement,
            params={"peak_prob": net_dict["peak_prob_lat"],
                    "std": net_dict["std_lat"],
                    "grid_num_x": grid_num_x,
                    "grid_num_y": grid_num_y
                    }
        ),

        # Spatial Gaussian profile initialiser with replacement
        "connectivity_initialiser_ff_GaussWReplace": init_sparse_connectivity(
            snippet=GaussianProfileFixedNumberPreWithReplacement,
            params={"in_degree": net_dict["Gauss_init_indegree"],
                    "peak_prob": net_dict["peak_prob_ff"],
                    "std": net_dict["std_ff"],
                    "grid_num_x": grid_num_x,
                    "grid_num_y": grid_num_y
                    }
        ),
        "connectivity_initialiser_lat_GaussWReplace": init_sparse_connectivity(
            snippet=GaussianProfileFixedNumberPreWithReplacement,
            params={"in_degree": net_dict["Gauss_init_indegree"],
                    "peak_prob": net_dict["peak_prob_lat"],
                    "std": net_dict["std_lat"],
                    "grid_num_x": grid_num_x,
                    "grid_num_y": grid_num_y
                    }
        ),

        ########################################################################
        # weight update model

        # custom STDP with all-to-all spike pairing
        "weight_update_init_STDP": init_weight_update(
            snippet=STDPAllToAllSpikePairing,
            params={
                "tauPlus": net_dict["tauPlus"],
                "tauMinus": net_dict["tauMinus"],
                "aPlus":  net_dict["aPlus"],
                # A_ = B * A_ + * tau_ + / tau_ - = 1.2 * 0.1 *  20. / 64.
                "aMinus": net_dict["B"] * net_dict["aPlus"]
                * net_dict["tauPlus"] / net_dict["tauMinus"],
                "wMin": 0.,
                "wMax": net_dict["g_max"]},
            vars={"g": net_dict["g_init"]},
            pre_vars={"preTrace": 0.},
            post_vars={"postTrace": 0.}
        ),

        # no STDP
        "weight_update_init_noSTDP": init_weight_update(
            snippet="StaticPulse",
            vars={"g": net_dict["g_init"]},
        ),


        ########################################################################
        # structural plasticity

        # number of rewiring attempts per update per block
        "num_rewiring_attempts":
        net_dict["num_rewiring_attempts_block"] * net_dict["scale"]**2,

        # row length of bitfield matrix (padded to word length, 32 bit)
        # (uses neuron number of postsynaptic neuron population)
        "bitfield_row_words": int((num_neurons + 31) / 32),
        # number of rows in the bitfield
        # (equals neuron number of presynaptic neuron population)
        "bitfield_num_rows": num_neurons,
    })


################################################################################
# simulation parameters
################################################################################

sim_dict = {
    # output directory
    "dir": "data",
    # GeNN random seed
    "seed": 789,
    # simulation time step in ms
    # (Bogdan2018 uses 1 ms, Bamford 0.1 ms)
    "dt": 0.1,
    # simulation time in ms
    "tsim": 60000.,
    # whether to read out the connectivity and weights directly
    # after each rewiring update
    "read_conn_after_rewiring": False,
    # whether to enable GeNN timers
    "timing_enabled": True,
}

# note on delays: Bogdan2018 use 1 time step = 1 ms, this GeNN implementation
# does not explicitly use any delay which also corresponds to 1 time step
# (here: 0.1 ms)

################################################################################


def set_params(net_dict_orig, sim_dict_orig,
               new_net_params={}, new_sim_params={}):
    new_net_dict = net_dict_orig.copy()
    new_net_dict.update(new_net_params)
    set_derived_network_params(new_net_dict)
    new_sim_dict = sim_dict_orig.copy()
    new_sim_dict.update(new_sim_params)

    # create output directory
    if not os.path.exists(new_sim_dict["dir"]):
        os.makedirs(new_sim_dict["dir"])
    return new_net_dict, new_sim_dict
