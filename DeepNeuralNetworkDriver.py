import time
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn import callbacks, conv_2d, conv_1d
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
        self.game_size = None
        self.stop_accuracy = 0.95
        self.observation_space_length = None
        self.num_of_cores = num_of_cores
        self.gpu_mem = gpu_mem
        self.model_name = model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.keep_probability = keep_probability
        self.layer_types =          ['conv_2d', 'conv_2d', 'fully_connected']  # conv_2d, fully_connected, conv_1d
        self.layer_nodes =          [None, None, 256]  # number of nodes in each core layer
        self.layer_num_of_filters = [32, 64, None]  # only for conv layers
        self.layer_filter_sizes =   [8, 4, None]  # only for conv layers
        self.layer_strides =        [4, 2, None]  # only for conv layers
        self.dropouts =             [False, False, False]  # whether to add a dropout on this layer
        self.activations =          ['relu', 'relu', 'relu']  # each layers activation function
        self.output_layer_activation = 'softmax'
        self.regression_optimizer = 'adam'
        self.loss_function = 'mean_square'
        #  activations
        # linear, tanh, sigmoid, softmax, softplus, softsign, relu, relu6, leaky_relu, prelu, elu, crelu, selu
        #  optimizers
        # sgd, rmsprop, adam, momentum, adagrad, ftrl, adadelta, proxi_adagrad, nesterov
        # loss functions
        #  softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, weighted_crossentropy, mean_square, hinge_loss, roc_auc_score, weak_cross_entropy_2d
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
        self.network = None

    def train_model(self):
        self.save_layers()
        # input_length = len(self.training_data[0][0])
        if self.layer_types[0] == 'conv_2d':
            X = np.array([i[0] for i in self.training_data]).reshape(-1, self.game_size[1], self.game_size[0], 1)
        else:
            input_length = self.training_data[0][0].size
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
        early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=self.stop_accuracy)
        self.model.fit({'input': X}, {'targets': y}, n_epoch=self.epochs, snapshot_epoch=False, show_metric=True,
                       run_id=self.run_id, snapshot_step=step_snapshot_interval, callbacks=early_stopping_cb)
        if self.profile_run:
            tf.profiler.experimental.stop()

    def neural_network_model(self):
        tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
        tflearn.init_graph(num_cores=self.num_of_cores, gpu_memory_fraction=self.gpu_mem)

        if self.layer_types[0] == 'fully_connected' or self.layer_types[0] == 'conv_1d':
            self.network = input_data(shape=[None, self.observation_space_length, 1], name='input')
        else:
            self.network = input_data(shape=[None, self.game_size[1], self.game_size[0], 1], name='input')

        for i in range(0, len(self.layer_types)):
            if self.layer_types[i] == 'fully_connected':
                self.network = fully_connected(self.network,
                                               n_units=self.layer_nodes[i],
                                               activation=self.activations[i],
                                               weights_init=tnorm)
            elif self.layer_types[i] == 'conv_2d':
                self.network = conv_2d(self.network,
                                       nb_filter=self.layer_num_of_filters[i],
                                       filter_size=self.layer_filter_sizes[i],
                                       strides=self.layer_strides[i],
                                       activation=self.activations[i],
                                       weights_init=tnorm)
            elif self.layer_types[i] == 'conv_1d':
                self.network = conv_1d(self.network,
                                       nb_filter=self.layer_num_of_filters[i],
                                       filter_size=self.layer_filter_sizes[i],
                                       strides=self.layer_strides[i],
                                       activation=self.activations[i],
                                       weights_init=tnorm)
            if self.dropouts[i]:
                self.network = dropout(self.network, self.keep_probability)

        self.network = fully_connected(self.network, self.output_size, activation=self.output_layer_activation)
        self.network = regression(self.network, optimizer=self.regression_optimizer, learning_rate=self.learning_rate,
                                  loss=self.loss_function, name='targets')
        if self.save_checkpoints and self.model_dir is not None:
            checkpoint_dir = os.path.join(self.model_dir, self.run_dir, self.model_name)
            best_checkpoint_dir = os.path.join(self.model_dir, self.run_dir, 'best-' + self.model_name)
            self.model = tflearn.DNN(self.network, tensorboard_dir=self.log_dir, checkpoint_path=checkpoint_dir,
                                     best_checkpoint_path=best_checkpoint_dir, tensorboard_verbose=3)
        else:
            self.model = tflearn.DNN(self.network, tensorboard_dir=self.log_dir)
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
        self.load_layers()
        self.neural_network_model()
        print('Loading: ' + final_directory)
        self.model.load(final_directory)
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
        data['layer_nodes'] = self.layer_nodes
        data['activations'] = self.activations
        data['dropouts'] = self.dropouts
        data['output_layer_activation'] = self.output_layer_activation
        data['regression_optimizer'] = self.regression_optimizer
        data['loss_function'] = self.loss_function
        data['layer_types'] = self.layer_types
        data['layer_num_of_filters'] = self.layer_num_of_filters
        data['layer_filter_sizes'] = self.layer_filter_sizes
        data['layer_strides'] = self.layer_strides
        data['observation_space_length'] = self.observation_space_length
        pickle.dump(data, file)
        file.close()

    def load_layers(self):
        path = os.path.join(self.model_dir, self.run_dir, 'Model_Network.pkl')
        if os.path.isfile(path):
            try:
                file = open(path, 'rb')
                data = pickle.load(file)
                self.layer_nodes = data['layer_nodes']
                self.activations = data['activations']
                self.dropouts = data['dropouts']
                self.output_layer_activation = data['output_layer_activation']
                self.regression_optimizer = data['regression_optimizer']
                self.loss_function = data['loss_function']
                self.layer_types = data['layer_types']
                self.layer_num_of_filters = data['layer_num_of_filters']
                self.layer_filter_sizes = data['layer_filter_sizes']
                self.layer_strides = data['layer_strides']
                self.observation_space_length = data['observation_space_length']
                file.close()
                print('Loading: ' + path)
            except:
                print('Could not load model, trying defined model...')
        else:
            print('No Model Network to load!')
            time.sleep(1)

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


class EarlyStoppingCallback(callbacks.Callback):
    def __init__(self, val_acc_thresh):
        super().__init__()
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        if training_state.val_acc is None:
            return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration
