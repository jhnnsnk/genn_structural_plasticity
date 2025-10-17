import csv

from ml_genn.callbacks import Callback

from time import perf_counter

class CSVTrainLog(Callback):
    def __init__(self, filename, output_pop, resume):
        # Create CSV writer
        self.file = open(filename, "a" if resume else "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")

        # Write header row if we're not resuming from an existing training run
        if not resume:
            self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])

        self.output_pop = output_pop

    def on_epoch_begin(self, epoch):
        self.start_time = perf_counter()

    def on_epoch_end(self, epoch, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()

class CSVTestLog(Callback):
    def __init__(self, filename, epoch, resume, output_pop):
        # Create CSV writer
        self.file = open(filename, "a" if resume else "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")
        if not resume:
            self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])
        self.epoch = epoch
        self.output_pop = output_pop

    def on_test_begin(self):
        self.start_time = perf_counter()

    def on_test_end(self, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([self.epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()

class ConnectivityCheckpoint(Callback):
    def __init__(self, serialiser="numpy", epoch_interval=1):
        self.serialiser = serialiser
        self.epoch_interval = epoch_interval

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def on_epoch_end(self, epoch, metrics):
        # If we should checkpoint this epoch
        if (epoch % self.epoch_interval) == 0:
            self._compiled_network.save_connectivity((epoch,), self.serialiser)
