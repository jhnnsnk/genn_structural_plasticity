import numpy as np

from os import makedirs
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data, preprocess_tonic_spikes)

def load_dataset(args, data_start, data_step):
    # Do we need to load the test set?
    test = (args.mode == "test")
    
    # If dataset is MNIST
    spikes = []
    labels = []
    num_input = None
    num_output = None
    if args.dataset == "mnist":
        from mnist import download_and_parse_mnist_file

        # K-fold validation not supported/required for MNIST
        assert args.fold == 0

        # Latency encode MNIST digits
        num_input = 28 * 28
        num_output = 10
        num_validate = 10000

        # Make data directory if it doesn't exist
        # **NOTE** unlike higher-level MNIST API, download_and_parse_mnist_file doesn't do this
        makedirs("./data", exist_ok=True)

        # If we're in a mode which uses the testing data, simply load it
        if test:
            labels = download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz", target_dir="./data")
            images = download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz", target_dir="./data")
        else:
            # Get training data
            train_labels = download_and_parse_mnist_file("train-labels-idx1-ubyte.gz", target_dir="./data")
            train_images = download_and_parse_mnist_file("train-images-idx3-ubyte.gz", target_dir="./data")

            # If we're training on entire training set, take directly
            if args.mode == "train":
                labels = train_labels
                images = train_images
            # Otherwise, if we're training with validation, hold out validation
            elif args.mode == "train_validate":
                labels = train_labels[:-num_validate]
                images = train_images[:-num_validate]
            # Otherwise, if we're validating, take held out
            elif args.mode == "validate":
                labels = train_labels[-num_validate:]
                images = train_images[-num_validate:]
            else:
                assert False

            # Slice out subset required on this node
            labels = labels[data_start::data_step]
            images = images[data_start::data_step]

        # Encode images into spikes
        spikes = log_latency_encode_data(images, 20.0, 51)
    # Otherwise
    else:
        # Import bits of tonic
        from tonic.datasets import DVSGesture, SHD, SSC
        from tonic.transforms import Compose, Downsample

        # Load dataset
        if args.dataset == "shd":
            dataset = SHD(save_to='./data', train=not test)
            sensor_size = dataset.sensor_size
            
            # Speaker IDs are non-sequential so convert to sequential IDs
            _, fold = np.unique(dataset.speaker, return_inverse=True)
        elif args.dataset == "dvs_gesture":
            transform = Compose([Downsample(spatial_factor=0.25)])
            dataset = DVSGesture(save_to='./data', train=not test, transform=transform)
            sensor_size = (32, 32, 2)
            
            # For consistency, concert user IDs to unique sequential IDs starting at zero
            _, fold = np.unique(dataset.users, return_inverse=True)

            # Divide folds by 3 to reduce number of folds to 8 to obtain
            # validation sets which are roughly 10% of the size of the training set
            fold = fold // 3
        elif args.dataset == "ssc":
            if test:
                split = "test"
            elif args.mode == "validate":
                split = "valid"
            else:
                split = "train"
            dataset = SSC(save_to='./data', split=split)
            sensor_size = dataset.sensor_size
            fold = None

        # If dataset provides fold data and we're doing some sort of hold out
        if fold is not None and (args.mode == "validate" 
                                 or args.mode == "train_validate"):
            # Get indices of correct fold(s) of data
            assert args.fold <= np.amax(fold)
            if args.mode == "validate":
                indices = np.where(np.asarray(fold) == args.fold)[0]
            else:
                indices = np.where(np.asarray(fold) != args.fold)[0]
            
            # From these, select slice required for this node
            indices = indices[data_start::data_step]
        # Otherwise, just take slice of datatset required for this node
        else:
            assert args.fold == 0
            indices = np.arange(data_start, len(dataset), data_step)
        
        # Get number of input and output neurons from dataset 
        # and round up outputs to power-of-two
        num_input = int(np.prod(sensor_size))
        num_output = len(dataset.classes)

        # Preprocess dataset entries specified by indices
        for i in indices:
            events, label = dataset[i]
            spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                                  sensor_size, dt=1.0,
                                                  histogram_thresh=args.dataset_threshold))
            labels.append(label)

    # Determine max spikes and latest spike time
    max_spikes = calc_max_spikes(spikes)
    latest_spike_time = calc_latest_spike_time(spikes)
    print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")
    
    return (spikes, labels, num_input, num_output,
            max_spikes, latest_spike_time)
