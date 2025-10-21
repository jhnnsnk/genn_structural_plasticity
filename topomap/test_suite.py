# run tests with, for example:
# python test_suite.py TestSuite.test_GaussianProfileFixedNumberPreWithReplacementIndegree

import unittest
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pygenn import GeNNModel
from pygenn import (create_var_ref, create_wu_var_ref,
                    init_postsynaptic, init_weight_update)
import topographic_map_model as topomodel
import helpers
from parameters import net_dict, sim_dict, set_params


class TestSuite(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSuite, self).__init__(*args, **kwargs)

        self.plotting = True
        self.dir = "test_output"

    def test_TestSuite(self):
        self.assertTrue(True)

    def test_GaussianProfileFixedNumberPreWithReplacementIndegree(self):
        """
        """
        # define parameters that differ from original ones
        net_dict_test, sim_dict_test = set_params(
            net_dict, sim_dict,
            new_net_params={"Gauss_init_indegree": 16},
            new_sim_params={"dir": self.dir,
                            # one time step, not simulated anyways
                            "tsim": sim_dict["dt"],
                            })

        # set up and run model
        model_name = "Test_FixedNumberPreIndegree"
        model = GeNNModel(model_name=model_name)

        # neuron population (target)
        ng = topomodel.add_target_layer_neuron_population(model, net_dict_test)

        # synapse population (no STDP)
        sg = model.add_synapse_population(
            pop_name="Synapses",
            matrix_type="SPARSE",
            source=ng, target=ng,
            weight_update_init=net_dict_test['weight_update_init_noSTDP'],
            postsynaptic_init=topomodel.get_postsynaptic_init(
                ng, net_dict_test),
            connectivity_init=net_dict_test["connectivity_initialiser_ff_GaussWReplace"])

        # initialise simulation
        num_timesteps = topomodel.initialise_simulation(model, sim_dict_test)

        # build and load
        model.build()
        model.load(num_recording_timesteps=num_timesteps)

        pre_inds, post_inds = topomodel.get_connectivity(sg)

        # count in-degree
        for post_id in np.arange(ng.num_neurons):
            in_degree = list(post_inds).count(post_id)
            assert in_degree == net_dict_test["Gauss_init_indegree"], \
                "In-degree does not equal Gauss_init_indegree"

    def test_GaussianFixedNumberPreWithReplacementSigma(self):
        """
        """
        # define parameters that differ from original ones
        net_dict_test, sim_dict_test = set_params(
            net_dict, sim_dict,
            new_net_params={"Gauss_init_indegree": 16,
                            "std_ff": 2.5},
            new_sim_params={"dir": self.dir,
                            # one time step, not simulated anyways
                            "tsim": sim_dict["dt"],
                            })

        # set up and run model
        model_name = "Test_FixedNumberPreSigma"
        model = GeNNModel(model_name=model_name)

        # neuron population (target)
        ng = topomodel.add_target_layer_neuron_population(model, net_dict_test)

        # synapse population (no STDP)
        sg = model.add_synapse_population(
            pop_name="Synapses",
            matrix_type="SPARSE",
            source=ng, target=ng,
            weight_update_init=net_dict_test['weight_update_init_noSTDP'],
            postsynaptic_init=topomodel.get_postsynaptic_init(
                ng, net_dict_test),
            connectivity_init=net_dict_test["connectivity_initialiser_ff_GaussWReplace"])

        # initialise simulation
        num_timesteps = topomodel.initialise_simulation(model, sim_dict_test)

        # build and load
        model.build()
        model.load(num_recording_timesteps=num_timesteps)

        pre_inds, post_inds = topomodel.get_connectivity(sg)

        # sort connection details for each postsynaptic neuron
        conns_post = {}
        for post_id in np.arange(net_dict_test["num_neurons"]):
            conns_post[post_id] = {"sources": []}

        # for each postsynaptic neuron, fill in presynaptic neuron ids
        # and the corresponding weights
        for i, post_id in enumerate(post_inds):
            conns_post[post_id]["sources"].append(pre_inds[i])

        # iterate over postsynaptic neurons and calculate standard deviation
        sigma = 0.
        for post_id in sorted(conns_post):
            sigma2_nom = 0.
            sigma2_denom = 0.
            for pre_id in conns_post[post_id]["sources"]:
                dx, dy = helpers.get_dx_dy_pbc_from_ids(
                    pre_id, post_id,
                    net_dict["grid_num_x_default"], net_dict["grid_num_x_default"])
                sigma2_nom += dx * dx + dy * dy
                sigma2_denom += 1.
            sigma += np.sqrt(0.5 * sigma2_nom / sigma2_denom)

        sigma /= net_dict_test["num_neurons"]
        self.assertAlmostEqual(sigma, net_dict_test["std_ff"], delta=0.1,
                               msg="Sampled sigma not equal to std_ff.")

    def test_StructuralPlasticityEliminationBelowThresholdProb1(self):
        """
        All-to-all connected neuron population; initial weights below
        threshold should trigger elimination with elimination probability _dep.
        """

        # define parameters that differ from original ones
        net_dict_test, sim_dict_test = set_params(
            net_dict, sim_dict,
            new_net_params={"grid_num_x_default": 2,
                            "g_init": 0.08,
                            "weight_threshold": 0.1,
                            "num_rewiring_attempts": 10,
                            "elimination_probability_dep": 1.,
                            },
            new_sim_params={"dir": self.dir,
                            # one time step
                            "tsim": sim_dict["dt"]
                            })

        # set up and run model
        model_name = "Test_StructuralPlasticityEliminationBelowThresholdProb1"
        model = GeNNModel(model_name=model_name)
        num_conns = _run_model_StructuralPlasticity(
            model, net_dict_test, sim_dict_test, "AllToAll", self.plotting,
            self.dir + "/" + model_name)

        # evaluate: exact number removed
        num_conns_removed = num_conns[0, 1] - num_conns[1, 1]
        self.assertEqual(num_conns_removed,
                         net_dict_test["num_rewiring_attempts"])

    def test_StructuralPlasticityEliminationBelowThresholdProb05(self):
        """
        All-to-all connected neuron population; initial weights below
        threshold should trigger elimination with elimination probability _dep.
        """
       # define parameters that differ from original ones
        net_dict_test, sim_dict_test = set_params(
            net_dict, sim_dict,
            new_net_params={"grid_num_x_default": 5,
                            "g_init": 0.08,
                            "weight_threshold": 0.1,
                            "num_rewiring_attempts": 300,
                            "elimination_probability_dep": 0.5,
                            },
            new_sim_params={"dir": self.dir,
                            # one time step
                            "tsim": sim_dict["dt"]
                            })

        # set up and run model
        model_name = "Test_StructuralPlasticityEliminationBelowThresholdProb05"
        model = GeNNModel(model_name=model_name)
        num_conns = _run_model_StructuralPlasticity(
            model, net_dict_test, sim_dict_test, "AllToAll", False,
            self.dir + "/" + model_name)

        # evaluate: 10 % error ok because of finite size
        num_conns_removed = num_conns[0, 1] - num_conns[1, 1]
        num_conns_removed_exp = net_dict_test["num_rewiring_attempts"] * \
            net_dict_test["elimination_probability_dep"]
        self.assertAlmostEqual(num_conns_removed, num_conns_removed_exp,
                               delta=0.1 * net_dict_test["num_rewiring_attempts"])

    def test_StructuralPlasticityEliminationAboveThresholdProb1(self):
        """
        All-to-all connected neuron population; initial weights above
        threshold should trigger elimination with elimination probability _pot.
        """
        net_dict_test, sim_dict_test = set_params(
            net_dict, sim_dict,
            new_net_params={"grid_num_x_default": 2,
                            "g_init": 0.12,
                            "weight_threshold": 0.1,
                            "num_rewiring_attempts": 10,
                            "elimination_probability_pot": 1.,
                            },
            new_sim_params={"dir": self.dir,
                            # one time step
                            "tsim": sim_dict["dt"]
                            })

        # set up and run model
        model_name = "Test_StructuralPlasticityEliminationAboveThresholdProb1"
        model = GeNNModel(model_name=model_name)
        num_conns = _run_model_StructuralPlasticity(
            model, net_dict_test, sim_dict_test, "AllToAll", self.plotting,
            self.dir + "/" + model_name)

        # evaluate: exact number removed
        num_conns_removed = num_conns[0, 1] - num_conns[1, 1]
        self.assertEqual(num_conns_removed,
                         net_dict_test["num_rewiring_attempts"])

    def test_StructuralPlasticityEliminationAboveThresholdProb05(self):
        """
        All-to-all connected neuron population; initial weights above
        threshold should trigger elimination with elimination probability _pot.
        """

        # define parameters that differ from original ones
        net_dict_test, sim_dict_test = set_params(
            net_dict, sim_dict,
            new_net_params={"grid_num_x_default": 5,
                            "g_init": 0.12,
                            "weight_threshold": 0.1,
                            "num_rewiring_attempts": 300,
                            "elimination_probability_dep": 0.5,
                            },
            new_sim_params={"dir": self.dir,
                            # one time step
                            "tsim": sim_dict["dt"]
                            })

        # set up and run model
        model_name = "Test_StructuralPlasticityEliminationAboveThresholdProb05"
        model = GeNNModel(model_name=model_name)
        num_conns = _run_model_StructuralPlasticity(
            model, net_dict_test, sim_dict_test, "AllToAll", False,
            self.dir + "/" + model_name)

        # evaluate: 10 % error ok because of finite size
        num_conns_removed = num_conns[0, 1] - num_conns[1, 1]
        num_conns_removed_exp = net_dict_test["num_rewiring_attempts"] * \
            net_dict_test["elimination_probability_pot"]
        self.assertAlmostEqual(num_conns_removed, num_conns_removed_exp,
                               delta=0.1 * net_dict_test["num_rewiring_attempts"])

    def test_StructuralPlasticityFormation(self):
        """
        Unconnected neuron population.
        """
       # define parameters that differ from original ones
        net_dict_test, sim_dict_test = set_params(
            net_dict, sim_dict,
            new_net_params={"grid_num_x_default": 2,
                            "grid_num_y": 2,
                            "num_rewiring_attempts": 10,
                            "peak_prob_ff": 1.,
                            "std_ff": 100},
            new_sim_params={"dir": self.dir,
                            # one time step
                            "tsim": sim_dict["dt"]
                            })

        # set up and run model
        model_name = "Test_StructuralPlasticityFormation"
        model = GeNNModel(model_name=model_name)
        num_conns = _run_model_StructuralPlasticity(
            model, net_dict_test, sim_dict_test, "Unconnected", self.plotting,
            self.dir + "/" + model_name)

        # evaluate: exact number added
        num_conns_added = num_conns[1, 1] - num_conns[0, 1]
        self.assertEqual(
            num_conns_added, net_dict_test["num_rewiring_attempts"])

    def test_ComputeStimulusRatesId0(self):
        nx = 16
        stim_rates_matrix = topomodel.compute_stimulus_rates_id0(
            grid_num_x_default=nx,
            corr_base_rate=5.,  # 1/s
            corr_peak_rate=152.8,  # 1/s
            corr_std=2.,
            scale=2)

        if 0:
            print(stim_rates_matrix)
            plt.imshow(stim_rates_matrix)
            plt.show()

        self.assertEqual(stim_rates_matrix[0, 0], 157.8)
        self.assertEqual(int(stim_rates_matrix[0, 1]), 139)
        self.assertEqual(stim_rates_matrix[16, 16], 157.8)

    def test_RollStimulusId(self):
        nx = 16
        scale = 2
        stim_rates_matrix = topomodel.compute_stimulus_rates_id0(
            grid_num_x_default=nx,
            corr_base_rate=5.,  # 1/s
            corr_peak_rate=152.8,  # 1/s
            corr_std=2.,
            scale=scale)

        stim_rates = topomodel.roll_stimulus_id(
            grid_num_x_default=nx,
            stim_rates_matrix=stim_rates_matrix,
            randomize=False, row_fix=1, col_fix=10)

        if 0:
            sheet = stim_rates.reshape(nx * scale, -1)
            print(sheet.astype(int))
            plt.imshow(sheet)
            plt.show()

        self.assertEqual(stim_rates[42], 157.8)

    def test_NeuronSynapseModel(self):
        """
        Target neuron opulation receiving only feedforward input,
        one-to-one connectivity.
        This test just works via visual inspection. Plots of voltages and spikes
        are generated.
        """
        # define parameters that differ from original ones
        net_dict_test, sim_dict_test = set_params(
            net_dict, sim_dict,
            new_net_params={"grid_num_x_default": 2,
                            "g_init": 0.18,
                            "tauPlus": 20.,  # ms
                            "tauMinus": 64.,  # ms
                            "aPlus": 0.1*0.2,
                            "B": 1.2,  # to be converted into A_-
                            # assume indegree of 16 at rate 20 Hz
                            "neuron_param_space_src_corrFalse": {"rate": 16 * 20}
                            },
            new_sim_params={"dir": self.dir,
                            "tsim": 100.
                            })

        # set up and run model
        model_name = "Test_NeuronSynapseModel"
        model = GeNNModel(model_name=model_name)
        _ = _run_model_NeuronSynapseModel(
            model, net_dict_test, sim_dict_test, "STDP",
            self.plotting, self.dir + "/" + model_name)

################################################################################
# helpers
################################################################################


def _run_model_NeuronSynapseModel(
        mdl, ndict, sdict, weightupdate="STDP", plotting=True, fname=""):
    """
    weightupdate: 'noSTDP' or 'STDP'
    """

    ng_src = topomodel.add_source_layer_neuron_population_uncorrelated_input(
        mdl, ndict)
    ng_src.spike_recording_enabled = True

    ng_tgt = topomodel.add_target_layer_neuron_population(mdl, ndict)
    ng_tgt.spike_recording_enabled = True

    # synapse population (without STDP)
    sg = mdl.add_synapse_population(
        pop_name="Synapses",
        matrix_type="SPARSE",
        source=ng_src, target=ng_tgt,
        weight_update_init=ndict["weight_update_init_" + weightupdate],
        postsynaptic_init=topomodel.get_postsynaptic_init(ng_tgt, ndict),
        connectivity_init=ndict["connectivity_initialiser_OneToOne"])

    # initialise simulation
    num_timesteps = topomodel.initialise_simulation(mdl, sdict)

    # build and load
    mdl.build()
    mdl.load(num_recording_timesteps=num_timesteps)

    # initialise views for voltages and weights
    voltages, voltage_var = topomodel.initialise_voltage_view(
        ng_tgt, num_timesteps)
    pre_inds, post_inds, inds = topomodel.get_connectivity_multi_index(sg)
    weights, weights_var = topomodel.initialise_weight_view(
        sg, inds, num_timesteps)

    while mdl.t < sdict["tsim"]:
        mdl.step_time()

        # read out views
        voltage_var.pull_from_device()
        voltages[mdl.timestep, :] = voltage_var.current_view
        _ = topomodel.get_connectivity(sg)
        weights_var.pull_from_device()
        weights[mdl.timestep, inds] = weights_var.values

    mdl.pull_recording_buffers_from_device()

    spike_times_rec_src, neuron_ids_rec_src = ng_src.spike_recording_data[0]
    # print(spike_times_rec_src, neuron_ids_rec_src)
    spike_times_rec_tgt, neuron_ids_rec_tgt = ng_tgt.spike_recording_data[0]
    # print(spike_times_rec_tgt, neuron_ids_rec_tgt)

    if plotting:
        times = np.arange(0, sdict["tsim"] + mdl.dt, mdl.dt)

        fig, axes = plt.subplots(3, sharex=True)
        times = np.arange(0., sdict["tsim"] + mdl.dt, mdl.dt)

        axes[0].plot(times, voltages)
        axes[0].set_ylabel("Mem. voltage (mV)")

        # remove np.nan, assuming one-to-one connectivity
        weights_clean = weights[~np.isnan(
            weights)].reshape(-1, ng_tgt.num_neurons)
        axes[1].plot(times, weights_clean)
        axes[1].set_ylabel("Conductance (mS)")

        axes[2].scatter(spike_times_rec_src, neuron_ids_rec_src, c="lightblue")
        axes[2].scatter(spike_times_rec_tgt, neuron_ids_rec_tgt, c="black")
        axes[2].set_ylabel("Neuron id")
        axes[2].set_xlabel("Time [ms]")
        plt.savefig(os.path.join(sdict["dir"], "Test_NeuronSynapseModel.pdf"),
                    bbox_inches="tight")

    return


def _run_model_StructuralPlasticity(
        mdl, ndict, sdict, init_conn="AllToAll", plotting=True, fname=""):
    """
    init_conn: 'AllToAll' or 'Unconnected'
    """

    # neuron population (target)
    ng = topomodel.add_target_layer_neuron_population(mdl, ndict)

    # synapse population (no STDP)
    sg = mdl.add_synapse_population(
        pop_name="Synapses",
        matrix_type="SPARSE",
        source=ng, target=ng,
        weight_update_init=ndict['weight_update_init_noSTDP'],
        postsynaptic_init=topomodel.get_postsynaptic_init(ng, ndict),
        connectivity_init=ndict["connectivity_initialiser_" + init_conn])

    if init_conn == "Unconnected":
        sg.max_connections = ng.num_neurons

    # structural plasticity for feedforward connectivity
    splast = topomodel.add_connectivity_update(
        mdl, sg, ndict, conntype="ff")

    # initialise simulation
    num_timesteps = topomodel.initialise_simulation(mdl, sdict)

    # build and load
    mdl.build()
    mdl.load(num_recording_timesteps=num_timesteps)

    # initialise view for num_rewiring_attempts_per_row
    num_rewiring_attempts_per_row, num_rewiring_attempts_per_row_var = \
        topomodel.initialise_extra_global_params_view(
            splast, num_timesteps, "num_rewiring_attempts_per_row")

    # initialise views for weights and the number of connections
    # with status at time step 0
    pre_inds, post_inds, inds = topomodel.get_connectivity_multi_index(sg)
    weights, weights_var = topomodel.initialise_weight_view(
        sg, inds, num_timesteps)
    num_conns = topomodel.initialise_num_connections_view(
        pre_inds, num_timesteps)

    # reset time, advance one time step, and update connectivity
    mdl.timestep = 0
    mdl.step_time()
    mdl.custom_update("UpdateConnectivity")

    # read out views
    pre_inds, post_inds, inds = topomodel.get_connectivity_multi_index(sg)
    num_rewiring_attempts_per_row_var.pull_from_device()
    num_rewiring_attempts_per_row[mdl.timestep] = num_rewiring_attempts_per_row_var.view
    weights_var.pull_from_device()
    weights[mdl.timestep, inds] = weights_var.values
    num_conns[mdl.timestep] = [mdl.t, len(pre_inds)]

    if plotting:
        times = np.arange(0, sdict["tsim"] + mdl.dt, mdl.dt)

        plot_time_resolved_matrix(
            fname + "_weights",
            "Weights time resolved",
            times, weights, ng.num_neurons, ng.num_neurons,
            cbar=True)

        print("num conns change (end - start): ",
              num_conns[1, 1] - num_conns[0, 1])
        print("num_rewiring_attempts_per_row: ",
              num_rewiring_attempts_per_row[1])

    return num_conns


def plot_time_resolved_matrix(
        fname, title,
        times, data, ng_pre_size, ng_post_size, cbar=False):

    pre_i, post_i = np.unravel_index(np.arange(ng_pre_size * ng_post_size),
                                     (ng_pre_size, ng_post_size))
    y_ticklabels = list(zip(pre_i, post_i))

    # scale figure size in x and y direction
    time_size = 5
    indices_size = ng_pre_size * ng_post_size / 5

    fig, ax = plt.subplots(figsize=(time_size, indices_size))
    data_T = data.T  # transpose

    cmap = matplotlib.colormaps["viridis"]
    cmap.set_bad(color="white")

    im = ax.imshow(data_T, cmap=cmap, aspect='auto')

    ax.set_title(title)

    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels(times)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    ax.set_yticks(np.arange(ng_pre_size * ng_post_size))
    ax.set_yticklabels(y_ticklabels)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Indices (pre, post)")

    if cbar:
        colorbar(ax, im, '')

    plt.savefig(f"{fname}.pdf", bbox_inches="tight")


def colorbar(
        ax,
        im,
        label,
        axis='right',
        orientation="vertical",
        size='5%',
        pad=0.05,
        nbins=5):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(axis, size=size, pad=pad)
    cb = plt.colorbar(im, cax=cax, label=label, orientation=orientation)
    cb.locator = MaxNLocator(nbins=nbins)
    cb.update_ticks()  # necessary for location
    return cax


unittest.main()
