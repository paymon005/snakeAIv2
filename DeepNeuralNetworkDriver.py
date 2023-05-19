import time
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class DnnDriver:

    def __init__(self, model_name='DNN_Model', model_dir='Models', run_dir='Snake',
                 learning_rate=1e-3, epochs=5, keep_probability=0.8, profile_run=False, observation_space_size=None):
        self._action_space_size = 3
        self._observation_space_size = observation_space_size
        self._model_name = model_name
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._keep_probability = keep_probability
        self._layers = [128, 256, 512, 256, 128]
        self._model = None
        self._training_data = None
        self._save_checkpoints = True
        self._log_dir = 'log'
        self._model_dir = model_dir
        self._run_dir = run_dir
        self._run_id = run_dir  # 'Snake_Model'
        self._profile_run = profile_run
        self._snapshot_frequency = 5
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[0], 'GPU')

    def train_model(self, model=False):
        input_shape = self._training_data[0][0].shape
        length = 0
        if len(input_shape) == 2:
            length = input_shape[0] * input_shape[1]
        elif len(input_shape) == 1:
            length = input_shape[0]
        X = np.array([i[0] for i in self._training_data]).reshape(-1, length, 1)
        y = [i[1] for i in self._training_data]
        if not model:
            model = self.neural_network_model(len(X[0]))
        if self._profile_run:
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                               python_tracer_level=1,
                                                               device_tracer_level=1)
            tf.profiler.experimental.start(self._log_dir, options=options)
        step_snapshot_interval = self._snapshot_frequency * int(len(self._training_data) / 64)
        print('Fitting the model and snapshotting every ' + str(self._snapshot_frequency) + ' epochs or every ' +
              str(step_snapshot_interval) + ' steps.')
        time.sleep(5)
        model.fit({'input': X}, {'targets': y}, n_epoch=self._epochs, snapshot_epoch=False, show_metric=True,
                  run_id=self._run_id, snapshot_step=step_snapshot_interval)
        if self._profile_run:
            tf.profiler.experimental.stop()
        return model

    def neural_network_model(self, input_length):
        tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
        network = input_data(shape=[None, input_length, 1], name='input')
        for i in range(0, len(self._layers)):
            network = fully_connected(network, self._layers[i], activation='relu')
            network = dropout(network, self._keep_probability)
        network = fully_connected(network, self._action_space_size, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=self._learning_rate,
                             loss='categorical_crossentropy', name='targets')
        if self._save_checkpoints and self._model_dir is not None:
            checkpoint_dir = os.path.join(self._model_dir, self._run_dir, self._model_name)
            best_checkpoint_dir = os.path.join(self._model_dir, self._run_dir, 'best-' + self._model_name)
            self._model = tflearn.DNN(network, tensorboard_dir=self._log_dir, checkpoint_path=checkpoint_dir,
                                      best_checkpoint_path=best_checkpoint_dir, tensorboard_verbose=3)
        else:
            self._model = tflearn.DNN(network, tensorboard_dir=self._log_dir)
        return self._model

    @staticmethod
    def create_save_dir(model_dir, run_dir):
        save_dir = os.path.join(os.getcwd(), model_dir, run_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_model(self):
        self._model.save(self._model_dir + '\\' + self._run_dir)

    def load_model(self, run_dir=None, run_name=None):
        if run_name is None or run_dir is None:
            run_name = self.get_last_run()
        else:
            self._run_dir = run_dir
        final_directory = os.path.join(os.getcwd(), self._model_dir, self._run_dir, run_name)
        if self._observation_space_size is not None:
            if isinstance(self._observation_space_size, int):
                self.neural_network_model(self._observation_space_size)
            else:
                self.neural_network_model(self._observation_space_size[0] * self._observation_space_size[1])
            print('Loading: ' + final_directory)
            self._model.load(final_directory)
        return self._model

    def plot_graphs(self):
        os.system("taskkill /IM ""tensorboard.exe"" /F")
        print(os.getcwd() + "\\venv\\Scripts\\python.exe -m tensorboard.main --logdir=" + self._log_dir)
        os.system(os.getcwd() + "\\venv\\Scripts\\python.exe -m tensorboard.main --logdir=" + self._log_dir)
        #  print("tensorboard. --logdir " + self._log_dir)
        #  os.system("tensorboard --logdir " + self._log_dir)

    def get_last_run(self):
        file_nums = []
        files = []
        folders = os.listdir(self._model_dir)
        for i in reversed(range(len(folders))):
            files = os.listdir(self._model_dir + '\\' + folders[i])
            if len(files) > 0:
                self._run_dir = folders[i]
                break
        for file in files:
            tmp = file.split('-')
            if len(tmp) == 5:
                file_nums.append(int(tmp[1].split('.')[0]))
        last_run = self._model_name + '-' + str(max(file_nums))
        return last_run

    def load_training_data(self, run_dir_to_load=None):
        if run_dir_to_load is not None:
            file_to_load = os.path.join(os.getcwd(), self._model_dir, run_dir_to_load, 'training_data.npy')
        else:
            file_to_load = os.path.join(os.getcwd(), self._model_dir, self._run_dir, 'training_data.npy')
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        self._training_data = np.load(file_to_load)
        np.load = np_load_old

    def change_run_dir(self, new_run_dir):
        self._run_dir = new_run_dir
        checkpoint_dir = os.path.join(self._model_dir, self._run_dir, self._model_name)
        best_checkpoint_dir = os.path.join(self._model_dir, self._run_dir, 'best-' + self._model_name)
        self._model.trainer.checkpoint_path = checkpoint_dir
        self._model.trainer.best_checkpoint_path = best_checkpoint_dir

    @staticmethod
    def check_version():
        print('TensorFlow version: ', tf.__version__)
        device_name = tf.test.gpu_device_name()
        if not device_name:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @learning_rate.deleter
    def learning_rate(self):
        del self._learning_rate

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        self._epochs = value

    @epochs.deleter
    def epochs(self):
        del self._epochs

    @property
    def run_dir(self):
        return self._run_dir

    @run_dir.setter
    def run_dir(self, value):
        self._run_dir = value

    @run_dir.deleter
    def run_dir(self):
        del self._run_dir

    @property
    def model_dir(self):
        return self._model_dir

    @model_dir.setter
    def model_dir(self, value):
        self._model_dir = value

    @model_dir.deleter
    def model_dir(self):
        del self._model_dir

    @property
    def run_id(self):
        return self._run_id

    @run_id.setter
    def run_id(self, value):
        self._run_id = value

    @run_id.deleter
    def run_id(self):
        del self._run_id

    @property
    def action_space_size(self):
        return self._action_space_size

    @action_space_size.setter
    def action_space_size(self, value):
        self._action_space_size = value

    @action_space_size.deleter
    def action_space_size(self):
        del self._action_space_size

    @property
    def training_data(self):
        return self._training_data

    @training_data.setter
    def training_data(self, value):
        self._training_data = value

    @training_data.deleter
    def training_data(self):
        del self._training_data

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = value

    @layers.deleter
    def layers(self):
        del self._layers

    @property
    def keep_probability(self):
        return self._keep_probability

    @keep_probability.setter
    def keep_probability(self, value):
        self._keep_probability = value

    @keep_probability.deleter
    def keep_probability(self):
        del self._keep_probability

    @property
    def observation_space_size(self):
        return self._observation_space_size

    @observation_space_size.setter
    def observation_space_size(self, value):
        self._observation_space_size = value

    @observation_space_size.deleter
    def observation_space_size(self):
        del self._observation_space_size

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @model.deleter
    def model(self):
        del self._model

    @property
    def save_checkpoints(self):
        return self._save_checkpoints

    @save_checkpoints.setter
    def save_checkpoints(self, value):
        self._save_checkpoints = value

    @save_checkpoints.deleter
    def save_checkpoints(self):
        del self._save_checkpoints
