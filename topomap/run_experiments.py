import time
import os
import random
import pickle
import numpy as np
import figures_ms
import topographic_map_model as topomodel
from parameters import net_dict, sim_dict, set_params


def run_model(net_dict, sim_dict):
    print("Simulating.")

    time_start = time.time()

    # set up model
    model, ng_src, ng_tgt, sg_ff, sg_lat, splast_ff, splast_lat, stim_rates_matrix = \
        topomodel.initialize_topographic_map_model(net_dict)

    num_timesteps = topomodel.initialise_simulation(model, sim_dict)

    model.build()
    model.load(num_recording_timesteps=num_timesteps)

    # set up data containers and fill with initial data (t = 0 ms)
    pre_inds = {"ff": {}, "lat": {}}
    post_inds = {"ff": {}, "lat": {}}
    weights = {"ff": {}, "lat": {}}

    pre_inds["ff"][0.], post_inds["ff"][0.] = \
        topomodel.get_connectivity(sg_ff)
    pre_inds["lat"][0.], post_inds["lat"][0.] = \
        topomodel.get_connectivity(sg_lat)
    weights["ff"][0.] = topomodel.get_weights(sg_ff)
    weights["lat"][0.] = topomodel.get_weights(sg_lat)

    time_prepare = time.time()

    time_cum_stimulus = 0
    time_cum_readout = 0

    # reset time and start simulation loop
    model.timestep = 0  # integer timestep
    while model.t < sim_dict["tsim"]:
        model.step_time()

        # change stimulus location
        time_start_stimulus = time.time()
        if net_dict["use_correlated_input"] and (
                model.t % net_dict["neuron_param_space_src_corrTrue"]["stim_interval"] == 0):
            topomodel.set_stimulus_rates(
                ng_src, net_dict["grid_num_x_default"], stim_rates_matrix)
            time_cum_stimulus += (time.time() - time_start_stimulus)

        # rewiring
        if (model.t % net_dict["update_interval"]) == 0:
            # print(model.t, "Rewiring.")
            model.custom_update("UpdateConnectivity")

            # read out connections and weghts directly after rewiring
            time_start_readout = time.time()
            if sim_dict["read_conn_after_rewiring"]:
                pre_inds["ff"][model.t], post_inds["ff"][model.t] = \
                    topomodel.get_connectivity(sg_ff)
                pre_inds["lat"][model.t], post_inds["lat"][model.t] = \
                    topomodel.get_connectivity(sg_lat)
                weights["ff"][model.t] = topomodel.get_weights(sg_ff)
                weights["lat"][model.t] = topomodel.get_weights(sg_lat)
            time_cum_readout += (time.time() - time_start_readout)

    time_simulate = time.time()

    # read out final connections and weights if they were not read out during
    # the simulation
    if list(weights["ff"].keys())[-1] < sim_dict["tsim"]:
        pre_inds["ff"][model.t], post_inds["ff"][model.t] = \
            topomodel.get_connectivity(sg_ff)
        pre_inds["lat"][model.t], post_inds["lat"][model.t] = \
            topomodel.get_connectivity(sg_lat)
        weights["ff"][model.t] = topomodel.get_weights(sg_ff)
        weights["lat"][model.t] = topomodel.get_weights(sg_lat)

    # write data to file
    with open(os.path.join(sdict["dir"], "data_pre_inds.pkl"),  'wb') as f:
        pickle.dump(pre_inds, f)
    with open(os.path.join(sdict["dir"], "data_post_inds.pkl"),  'wb') as f:
        pickle.dump(post_inds, f)
    with open(os.path.join(sdict["dir"], "data_weights.pkl"),  'wb') as f:
        pickle.dump(weights, f)

    time_write_data = time.time()

    # save and print measured times
    times = {
        "py_total": time_write_data - time_start,
        "py_prepare": time_prepare - time_start,
        "py_simulate": time_simulate - time_prepare,
        "py_simulate_stimulus": time_cum_stimulus,
        "py_simulate_readout": time_cum_readout,
        "py_write_data": time_write_data - time_simulate,
        # tsim is in ms
        "realtimefactor": (time_simulate - time_prepare) / (sim_dict["tsim"] * 0.001),
        "genn_init": model.init_time,
        "genn_sparse_init": model.init_sparse_time,
        "genn_neuron_update": model.neuron_update_time,
        "genn_presynaptic_update": model.presynaptic_update_time,
        "genn_postsynaptic_update": model.postsynaptic_update_time,
        "genn_connectivity_update": model.get_custom_update_time("UpdateConnectivity"),
        "genn_connectivity_update_host": model.get_custom_update_host_time("UpdateConnectivity"),
        "genn_connectivity_update_remap_time": model.get_custom_update_remap_time("UpdateConnectivity"),
    }

    with open(os.path.join(sdict["dir"], "times_s.pkl"),  'wb') as f:
        pickle.dump(times, f)

    print(
        "\nTime measurements of simulation:\n"
        + "  Python total:------------------------{:.3f} s\n".format(times["py_total"])
        + "  Python prepare:----------------------{:.3f} s\n".format(times["py_prepare"])
        + "  Python simulate:---------------------{:.3f} s\n".format(times["py_simulate"])
        + "  Python simulate: stimulus:-----------{:.3f} s\n".format(
            times["py_simulate_stimulus"])
        + "  Python simulate: readout:------------{:.3f} s\n".format(times["py_simulate_readout"])
        + "  Python write data:-------------------{:.3f} s\n".format(times["py_write_data"])
        + "  Real-time factor:--------------------{:.3f} s\n".format(times["realtimefactor"])
        + "  GeNN init:---------------------------{:.3f} s\n".format(times["genn_init"])
        + "  GeNN sparse init:--------------------{:.3f} s\n".format(times["genn_sparse_init"])
        + "  GeNN neuron update:------------------{:.3f} s\n".format(times["genn_neuron_update"])
        + "  GeNN presynaptic update:-------------{:.3f} s\n".format(
            times["genn_presynaptic_update"])
        + "  GeNN postsynaptic update:------------{:.3f} s\n".format(
            times["genn_postsynaptic_update"])
        + "  GeNN connectivity update:------------{:.3f} s\n".format(
            times["genn_connectivity_update"])
        + "  GeNN connectivity update host:-------{:.3f} s\n".format(
            times["genn_connectivity_update_host"])
        + "  GeNN connectivity update remap time:-{:.3f} s\n".format(
            times["genn_connectivity_update_remap_time"])
    )
    return


if __name__ == "__main__":

    # select parameterset
    key = 1

    # parametersets: plotting is always enabled, but set whether simulations
    # need to be rerun
    # for manuscript figures (keys 2-4), the simulated data is provided,
    # so there do_run_model=False
    p = {
        1: {"name": "test",
            "trunk_dir": "data_test",
            "experiments": ["scale_1"],
            "num_seeds": 1,
            "do_run_model": True},
        2: {"name": "figure_methods",
            "trunk_dir": "data_ms",
            "experiments": ["scale_1"],
            "num_seeds": 1,
            "do_run_model": False},
        3: {"name": "figure_topomap_formation",
            "trunk_dir": "data_ms",
            "experiments": ["scale_1_readout"],
            "num_seeds": 1,
            "do_run_model": False},
        4: {"name": "figure_performance",
            "trunk_dir": "data_ms",
            "experiments": ["scale_1", "scale_2", "scale_3", "scale_4",
                            "scale_5", "scale_6", "scale_7"],
            "num_seeds": 3,
            "do_run_model": False}
    }

    name = p[key]["name"]
    print(f"Running experiments of parameterset {key}: {name}")

    # seeds are identical for different experiments to have the same
    # initial settings
    random.seed(643)
    random_seeds = random.sample(range(1, 1000), 20)

    # base directory
    trunk_dir = p[key]["trunk_dir"]

    # list of experiments
    experiments = p[key]["experiments"]

    # number of seeds per experiment
    num_seeds = p[key]["num_seeds"]

    # whether to run the model simulation
    do_run_model = p[key]["do_run_model"]

    runs = {}
    runs["info"] = {"trunk_dir": trunk_dir,
                    "experiments": experiments,
                    "num_seeds": num_seeds}

    for e, exp in enumerate(experiments):
        seeds = random_seeds[:num_seeds]
        for s, seed in enumerate(seeds):
            r = e * num_seeds + s
            dir = os.path.join(trunk_dir, exp, str(s))
            runs[r] = {"experiment": exp, "run": s, "dir": dir, "seed": seed}
            runs[r]["new_net_params"] = {}
            runs[r]["new_sim_params"] = {"dir": dir, "seed": seed}

            exp_split = exp.split("_")
            if exp_split[0] == "scale":
                scale = int(exp_split[1])
                runs[r]["new_net_params"].update(
                    {"scale": scale})
                # disabling structural plasticity by setting the interval to a value larger than the simulation time
                if len(exp_split) >= 3:
                    if exp_split[2] == 'readout':
                        runs[r]["new_sim_params"].update(
                            {"read_conn_after_rewiring": True})

    # iterate over all runs
    for r in np.arange(len(experiments) * num_seeds):
        print(f"Run {r}")
        ndict, sdict = set_params(net_dict, sim_dict,
                                  new_net_params=runs[r]["new_net_params"],
                                  new_sim_params=runs[r]["new_sim_params"])
        if do_run_model:
            run_model(ndict, sdict)

        # methods figure
        if name == "figure_methods":
            time_start_methods = time.time()
            figures_ms.network_sketch_init_final(
                ndict, sdict, data_source="load")
            time_stop_methods = time.time()
            print("\n  Time to plot methods figure: {:.3f}\n".format(
                time_stop_methods - time_start_methods))

        # topomap formation figure (analysis)
        # (if the model is run, the analysis needs to be executed before plotting)
        if do_run_model and name == "figure_topomap_formation":
            time_start_process_conn_evo = time.time()
            figures_ms.analyze_connectivity_evolution(
                sdict["dir"], ndict, conntype="ff")
            figures_ms.analyze_connectivity_evolution(
                sdict["dir"], ndict, conntype="lat")
            figures_ms.analyze_connections_per_neuron(
                sdict["dir"], ndict, conntype="ff")
            figures_ms.analyze_connections_per_neuron(
                sdict["dir"], ndict, conntype="lat")
            time_stop_process_conn_evo = time.time()
            print("\n Time to process connectivity evolution: {:.3f}\n".format(
                time_stop_process_conn_evo - time_start_process_conn_evo))

        # topomap formation figure (plotting)
        if name == "figure_topomap_formation":
            time_start_plot_conn_evo = time.time()
            figures_ms.plot_connectivity_evolution(
                sdict["dir"], ndict, conntype="ff")
            figures_ms.plot_connectivity_evolution(
                sdict["dir"], ndict, conntype="lat")
            time_stop_plot_conn_evo = time.time()
            print("\n Time to plot connectivity evolution: {:.3f}\n".format(
                time_stop_plot_conn_evo - time_start_plot_conn_evo))

    # performance figure
    if name == "figure_performance":
        time_start_perf = time.time()
        figures_ms.plot_performance(
            runs,
            grid_num_x_default=ndict["grid_num_x_default"],
            tsim=sdict["tsim"])
        time_stop_perf = time.time()
        print("\n  Time to assess performance: {:.3f}\n".format(
            time_stop_perf - time_start_perf))
