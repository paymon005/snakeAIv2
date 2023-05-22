import time
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn
import os
import tensorflow as tf
import pickle
import MyTools
import tensorflow.python.util.deprecation as deprecation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False


class DnnDriver:

    def __init__(self, model_name='DNN_Model', model_dir='Models', log_dir='log', run_dir='Snake', learning_rate=1e-3,
                 epochs=5, keep_probability=0.8, num_of_cores=8, gpu_mem=0.4, profile_run=False):
        self.action_space_size = 3
        self.observation_space_length = None
        self.num_of_cores = num_of_cores
        self.gpu_mem = gpu_mem
        self.model_name = model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.keep_probability = keep_probability
        self.layers = [5]
        self.activations = ['relu']
        self.model = None
        self.training_data = None
        self.save_checkpoints = True
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.run_dir = run_dir
        self.run_id = run_dir  # 'Snake_Model'
        self.profile_run = profile_run
        self.snapshot_frequency = 5
        self.output_size = self.action_space_size
        tf.debugging.set_log_device_placement(True)
        self.check_version()

    def train_model(self):
        input_length = len(self.training_data[0][0])
        X = np.array([i[0] for i in self.training_data]).reshape(-1, input_length, 1)
        y = np.array([i[1] for i in self.training_data]).reshape(-1, self.output_size)
        if self.profile_run:
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                               python_tracer_level=1,
                                                               device_tracer_level=1)
            tf.profiler.experimental.start(self.log_dir, options=options)
        step_snapshot_interval = self.snapshot_frequency * int(len(self.training_data) / 64)
        print('Fitting the model and snapshotting every ' + str(self.snapshot_frequency) + ' epochs or every ' +
              str(step_snapshot_interval) + ' steps.')
        time.sleep(5)
        self.model.fit({'input': X}, {'targets': y}, n_epoch=self.epochs, snapshot_epoch=False, show_metric=True,
                       run_id=self.run_id, snapshot_step=step_snapshot_interval)
        if self.profile_run:
            tf.profiler.experimental.stop()

    def neural_network_model(self):
        tflearn.init_graph(num_cores=self.num_of_cores, gpu_memory_fraction=self.gpu_mem)
        network = input_data(shape=[None, self.observation_space_length, 1], name='input')
        for i in range(0, len(self.layers)):
            network = fully_connected(network, self.layers[i], activation=self.activations[i])
            network = dropout(network, self.keep_probability)
        network = fully_connected(network, self.output_size, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=self.learning_rate,
                             loss='mean_square', name='targets')
        if self.save_checkpoints and self.model_dir is not None:
            checkpoint_dir = os.path.join(self.model_dir, self.run_dir, self.model_name)
            best_checkpoint_dir = os.path.join(self.model_dir, self.run_dir, 'best-' + self.model_name)
            self.model = tflearn.DNN(network, tensorboard_dir=self.log_dir, checkpoint_path=checkpoint_dir,
                                     best_checkpoint_path=best_checkpoint_dir, tensorboard_verbose=3)
        else:
            self.model = tflearn.DNN(network, tensorboard_dir=self.log_dir)
        return self.model

    def save_model(self):
        self.model.save(self.model_dir + '\\' + self.run_dir)

    def load_model(self, run_name=None):
        if run_name is None:
            [model_to_load, self.run_dir] = self.get_last_run()
            final_directory = os.path.join(os.getcwd(), model_to_load)
        else:
            final_directory = MyTools.find_file_in_folder(self.model_dir, run_name)
            self.run_dir = '\\'.join(final_directory.split('\\')[1:-1])
        self.neural_network_model()
        print('Loading: ' + final_directory)
        self.model.load(final_directory)
        self.load_layers()
        return self.model

    def get_last_run(self):
        model_to_load = None
        run_dir = []
        files = []
        for (dir_path, dir_names, file_names) in os.walk(self.model_dir):
            for file in file_names:
                files.append(dir_path + '\\' + file)
        files = sorted(files, key=os.path.getctime, reverse=True)
        for file in files:
            if file.endswith('.meta'):
                model_to_load = file.split('.')[0]
                tmp = model_to_load.split('\\')
                run_dir = '\\'.join(tmp[1:-1])
                break
        if model_to_load is None:
            raise Exception('\n\nNo models to load!')
        return model_to_load, run_dir

    def plot_graphs(self):
        os.system("taskkill /IM ""tensorboard.main"" /F")
        os.system("taskkill /IM ""tensorboard.exe"" /F")
        print(os.getcwd() + "\\venv\\Scripts\\python.exe -m tensorboard.main --logdir=" + self.log_dir)
        os.system(os.getcwd() + "\\venv\\Scripts\\python.exe -m tensorboard.main --logdir=" + self.log_dir)
        #  print("tensorboard. --logdir " + self.log_dir)
        #  os.system("tensorboard --logdir " + self.log_dir)

    def load_training_data(self, filename=None):
        if filename is not None:
            file_to_load = MyTools.find_file_in_folder(self.model_dir, filename)
        else:
            file_to_load = MyTools.get_newest_file_in_folder_w_ext(self.model_dir, 'npy')
        print('Loading: ' + file_to_load)
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        self.training_data = np.load(file_to_load)
        np.load = np_load_old

    def change_run_dir(self, new_run_dir):
        self.run_dir = new_run_dir
        checkpoint_dir = os.path.join(self.model_dir, self.run_dir, self.model_name)
        best_checkpoint_dir = os.path.join(self.model_dir, self.run_dir, 'best-' + self.model_name)
        self.model.trainer.checkpoint_path = checkpoint_dir
        self.model.trainer.best_checkpoint_path = best_checkpoint_dir

    def save_layers(self):
        data = {}
        file = open(os.path.join(self.model_dir, self.run_dir, 'Model_Network.pkl'), 'wb')
        data['Layers'] = self.layers
        data['Activations'] = self.activations
        pickle.dump(data, file)
        file.close()

    def load_layers(self):
        path = os.path.join(self.model_dir, self.run_dir, 'Model_Network.pkl')
        if os.path.isfile(path):
            file = open(path, 'rb')
            data = pickle.load(file)
            self.layers = data['Layers']
            self.activations = data['Activations']
            file.close()
        else:
            print('No Model Network to load!')

    @staticmethod
    def check_version():
        print('TensorFlow version: ', tf.__version__)
        device_name = tf.test.gpu_device_name()
        if not device_name:
            print('GPU device not found')
        else:
            print('Found GPU at: {}'.format(device_name))
            # gpus = tf.config.list_physical_devices('GPU')
            # tf.config.set_visible_devices(gpus[0], 'GPU')
