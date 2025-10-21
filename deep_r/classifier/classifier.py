import os
import numpy as np

from argparse import ArgumentParser
from hashlib import md5
from itertools import chain
from json import dump
from pygenn.cuda_backend import DeviceSelect
from ml_genn import Connection, Population, Network
from ml_genn.callbacks import Checkpoint, VarRecorder
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import (AdaptiveLeakyIntegrateFire, LeakyIntegrate,
                             LeakyIntegrateFire, SpikeInput)
from ml_genn.serialisers import Numpy

from glob import glob
from time import perf_counter

from ml_genn.compilers.eprop_compiler import default_params

from callbacks import ConnectivityCheckpoint, CSVTrainLog, CSVTestLog
from data_loader import load_dataset


def pad_hidden_layer_argument(arg, num_hidden_layers, context, default=None):
    # If argument wasn't specified but there is a default, repeat default for each hidden layer
    if arg is None and default is not None:
        return [default] * num_hidden_layers
    elif len(arg) == 1:
        return arg * num_hidden_layers
    elif len(arg) != num_hidden_layers:
        raise RuntimeError(f"{context} either needs to be specified as a single "
                           f" value or for each {num_hidden_layers} layers")
    else:
        return arg

def inference(genn_kwargs, args, unique_suffix, network, serialiser, latest_spike_time, epoch, resume):
    print(f"Loading inference model from checkpoint {epoch}")

    # Load network state from final checkpoint
    network.load((epoch,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=int(np.ceil(latest_spike_time)),
                                 batch_size=1 if args.cpu else args.batch_size, rng_seed=args.seed,
                                 reset_vars_between_batches=False, 
                                 kernel_profiling=args.kernel_profiling, **genn_kwargs)
    model_name = (f"classifier_test_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_test_{unique_suffix}")
    compiled_net = compiler.compile(network, name=model_name)

    with compiled_net:
        # Perform warmup evaluation
        # **TODO** subset of data
        compiled_net.evaluate({input: spikes},
                              {output: labels})

        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar",
                     CSVTestLog(f"test_output_{unique_suffix}.csv", epoch, resume, output)]
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels},
                                            callbacks=callbacks)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")

        if args.kernel_profiling:
            genn_model = compiled_net.genn_model
            data = {
                "neuron_update": genn_model.neuron_update_time,
                "presynaptic_update": genn_model.presynaptic_update_time,
                "custom_update_reset": genn_model.get_custom_update_time("Reset")}
            with open(f"test_kernel_profile_{unique_suffix}.json", "w") as fp:
                dump(data, fp)

parser = ArgumentParser()
parser.add_argument("--device-id", type=int, default=0, help="CUDA device ID")
parser.add_argument("--use-mpi", action="store_true", help="Use MPI for faster training")
parser.add_argument("--mode", choices=["train", "train_validate", "validate", "test"], default="train")
parser.add_argument("--fold", type=int, default=0, help="When using 'train_validate' mode, specifies the validation fold to use")
parser.add_argument("--record-rewiring", action="store_true", help="Record number of Deep-R rewiring over time")
parser.add_argument("--cpu", action="store_true", help="Use CPU for inference")
parser.add_argument("--kernel-profiling", action="store_true", help="Output kernel profiling data")
parser.add_argument("--test-validate-all", action="store_true", help="In validate or test mode, should we test checkpoints from all epochs?")
parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--dataset", choices=["ssc", "shd", "dvs_gesture", "mnist"], required=True)
parser.add_argument("--dataset-threshold", type=int, default=None, help="Minimum number of events in timestep required to spike")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--rewire", action="store_true", help="Use Deep-R to rewire sparse networks")
parser.add_argument("--resume-epoch", type=int, default=None)
parser.add_argument("--l1-strength", type=float, default=0.01)
parser.add_argument("--hidden-size", type=int, nargs="*")
parser.add_argument("--hidden-recurrent", choices=["True", "False"], nargs="*")
parser.add_argument("--hidden-model", choices=["lif", "alif"], nargs="*")
parser.add_argument("--hidden-input-sparsity", type=float, nargs="*")
parser.add_argument("--hidden-recurrent-sparsity", type=float, nargs="*")
parser.add_argument("--row-padding-prop", type=float, default=0.0)

args = parser.parse_args()

num_hidden_layers = max(len(args.hidden_size), 
                        len(args.hidden_recurrent),
                        len(args.hidden_model))
print(f"{num_hidden_layers} hidden layers")

# Pad hidden layer arguments
args.hidden_size = pad_hidden_layer_argument(args.hidden_size, 
                                             num_hidden_layers,
                                             "Hidden layer size")
args.hidden_recurrent = pad_hidden_layer_argument(args.hidden_recurrent, 
                                                  num_hidden_layers,
                                                  "Hidden layer recurrentness")
args.hidden_model = pad_hidden_layer_argument(args.hidden_model, 
                                              num_hidden_layers,
                                              "Hidden layer neuron model")
args.hidden_input_sparsity = pad_hidden_layer_argument(args.hidden_input_sparsity, 
                                                       num_hidden_layers,
                                                       "Hidden layer input sparsity",
                                                       1.0)
args.hidden_recurrent_sparsity = pad_hidden_layer_argument(args.hidden_recurrent_sparsity, 
                                                          num_hidden_layers,
                                                          "Hidden layer recurrent sparsity",
                                                          1.0)

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items()
                         if arg not in ["mode", "cpu", "resume_epoch",
                                        "test_validate_all", "kernel_profiling"])

# Depending on whether we're operating on full or partial data, add suffix
if args.mode == "train_validate" or args.mode == "validate":
    unique_suffix += "_valid"
else:
    unique_suffix += "_full"

# When training, create parameters file containing arguments in easier-to-handle way
train = (args.mode == "train" or args.mode == "train_validate")
if train:
    with open(f"params_{unique_suffix}.json", "w") as fp:
        dump(vars(args), fp)

if train and args.use_mpi:
    from ml_genn.communicators import MPI
    communicator = MPI()
    print(f"Training on rank {communicator.rank} / {communicator.num_ranks}")

    data_start = communicator.rank
    data_step = communicator.num_ranks
else:
    communicator = None
    data_start = 0
    data_step = 1

# Load dataset
(spikes, labels, num_input, num_output,
 max_spikes, latest_spike_time) = load_dataset(args, data_start, data_step)

serialiser = Numpy("checkpoints_" + unique_suffix)
network = Network(default_params)
deep_r_conns = []
with network:
    # Add spike input population
    input = Population(SpikeInput(max_spikes=args.batch_size * max_spikes),
                       num_input)

    # Add output population
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="sum_var"),
                        num_output)

    # Loop through hidden layers
    hidden = []
    for i, (s, r, m, in_sp, rec_sp) in enumerate(zip(args.hidden_size, 
                                                 args.hidden_recurrent,
                                                 args.hidden_model,
                                                 args.hidden_input_sparsity,
                                                 args.hidden_recurrent_sparsity)):
        # Add population
        if m == "alif":
            hidden.append(Population(AdaptiveLeakyIntegrateFire(v_thresh=0.6, tau_refrac=5.0),
                                     s))
        else:
            hidden.append(Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0, tau_refrac=5.0),
                                     s))

        # If recurrent, add recurrent connections
        if r == "True":
            rec_weight = Normal(sd=1.0 / np.sqrt(s)) 
            if rec_sp == 1.0:
                Connection(hidden[-1], hidden[-1], Dense(rec_weight))
            else:
                conn = Connection(hidden[-1], hidden[-1], FixedProbability(rec_sp, rec_weight))
                if args.rewire:
                    deep_r_conns.append(conn)

        # Add connection to output layer
        Connection(hidden[-1], output, Dense(Normal(sd=1.0 / np.sqrt(hidden[-1].shape[0]))))

        # If this is first hidden layer, add input connections
        if i == 0:
            in_weight = Normal(sd=1.0 / np.sqrt(num_input))
            if in_sp == 1.0:
                Connection(input, hidden[-1], Dense(in_weight))
            else:
                conn = Connection(input, hidden[-1], FixedProbability(in_sp, in_weight, True))
                if args.rewire:
                    deep_r_conns.append(conn)
        # Otherwise, add connection to previous hidden layer
        else:
            in_weight = Normal(sd=1.0 / np.sqrt(hidden[-2].shape[0]))
            if in_sp == 1.0:
                Connection(hidden[-2], hidden[-1], Dense(in_weight))
            else:
                conn = Connection(hidden[-2], hidden[-1], FixedProbability(in_sp, in_weight))
                if args.rewire:
                    deep_r_conns.append(conn)

genn_kwargs = {"device_select_method": DeviceSelect.MANUAL,
               "manual_device_id": args.device_id}
                   
# If we're training model
if train:
    assert not args.cpu

    # If we should resume traing from a checkpoint, load checkpoint
    if args.resume_epoch is not None:
        network.load((args.resume_epoch,), serialiser)

    if args.record_rewiring:
        record_rewirings = {c: f"{c.name}_rewiring" for c in deep_r_conns}
    else:
        record_rewirings = {}

    # Create EProp compiler and compile
    compiler = EPropCompiler(example_timesteps=int(np.ceil(latest_spike_time)),
                             losses="sparse_categorical_crossentropy", rng_seed=args.seed,
                             optimiser="adam", batch_size=args.batch_size, 
                             communicator=communicator,
                             deep_r_conns=deep_r_conns,
                             deep_r_l1_strength=args.l1_strength,
                             deep_r_record_rewirings=record_rewirings,
                             kernel_profiling=args.kernel_profiling, **genn_kwargs)
    
    model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_train_{unique_suffix}")
    compiled_net = compiler.compile(network, name=model_name)
    
    # If additional row padding is required
    if args.row_padding_prop > 0.0:
        # Loop through connections and their compiled synapse groups
        for c, s in compiled_net.connection_populations.items():
            # Skip connections to output (as these are dense)
            if c.target() == output:
                continue

            # Increase row length
            s.max_connections = min(s.trg.num_neurons,
                                    int(s.max_connections * (1.0 + args.row_padding_prop)))
    with compiled_net:
        # Evaluate model on SHD
        start_time = perf_counter()
        start_epoch = 0 if args.resume_epoch is None else (args.resume_epoch + 1)
        record_data = (communicator is None or communicator.rank == 0)
        if record_data:
            callbacks = ["batch_progress_bar", Checkpoint(serialiser),
                         CSVTrainLog(f"train_output_{unique_suffix}.csv", output,
                                     args.resume_epoch is not None),
                         ConnectivityCheckpoint(serialiser)]
        else:
            callbacks = []
        metrics, callback_data  = compiled_net.train({input: spikes},
                                                     {output: labels},
                                                     num_epochs=args.num_epochs,
                                                     callbacks=callbacks, shuffle=True,
                                                     start_epoch=start_epoch)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
        
        if args.kernel_profiling:
            genn_model = compiled_net.genn_model
            data = {
                "neuron_update": genn_model.neuron_update_time,
                "presynaptic_update": genn_model.presynaptic_update_time,
                "synapse_dynamics": genn_model.synapse_dynamics_time,
                "custom_update_gradient_batch_reduce": genn_model.get_custom_update_time("GradientBatchReduce"),
                "custom_update_gradient_learn": genn_model.get_custom_update_time("GradientLearn"),
                "custom_update_reset": genn_model.get_custom_update_time("Reset"),
                "custom_update_softmax_1": genn_model.get_custom_update_time("Softmax1"),
                "custom_update_softmax_2": genn_model.get_custom_update_time("Softmax2"),
                "custom_update_softmax_2": genn_model.get_custom_update_time("Softmax2")}
            if args.rewire:
                data["custom_update_deep_r_1"] = genn_model.get_custom_update_time("DeepR1")
                data["custom_update_deep_r_2"] = genn_model.get_custom_update_time("DeepR2")
                data["custom_update_deep_r_l1"] = genn_model.get_custom_update_time("DeepRL1")
            with open(f"train_kernel_profile_{unique_suffix}.json", "w") as fp:
                dump(data, fp)

        # If we should record rewirings, store in CSV
        if len(record_rewirings) > 0:
            np.savetxt(f"rewiring_{unique_suffix}.csv", 
                       np.column_stack([callback_data[k] for k in record_rewirings.values()]),
                       header=", ".join(chain(*zip((c.name for c in record_rewirings.keys()),
                                                   (c.name + "_fail" for c in record_rewirings.keys())))),
                       fmt="%d", delimiter=",")

else:
    # Use CPU backend if desired
    if args.cpu:
        genn_kwargs["backend"]="single_threaded_cpu"

    # Loop through trained epochs
    if args.test_validate_all:
        for e in range(args.num_epochs):
             inference(genn_kwargs, args, unique_suffix, network, serialiser,
                       latest_spike_time, e, e > 0)
    else:
        inference(genn_kwargs, args, unique_suffix, network, serialiser,
                  latest_spike_time, args.num_epochs - 1, False)
