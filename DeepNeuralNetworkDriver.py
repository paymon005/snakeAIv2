import time
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn import callbacks, conv_2d, conv_1d, max_pool_2d
import tflearn
import os
import tensorflow as tf
import pickle
from tqdm import tqdm
import MyTools
import tensorflow.python.util.deprecation as deprecation
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from tbparse import SummaryReader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False
matplotlib.use('Agg')


class DnnDriver:

    def __init__(self, model_name='DNN_Model', model_dir='Models', log_dir='log', run_dir='Snake', learning_rate=1e-3,
                 epochs=5, num_of_cores=8, gpu_mem=0.4, profile_run=False):
        self.action_space_size = 3
        self.game_size = None
        self.stop_accuracy = 0.95
        self.observation_space_length = None
        self.num_of_cores = num_of_cores
        self.gpu_mem = gpu_mem
        self.model_name = model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        # all arrays below must be the same length (fill None at idxs where values do not apply)
        self.layer_types =           ['conv_2d', 'avg_pool_2d', 'fully_connected']  # conv_2d, fully_connected, conv_1d, max_pool_2d, avg_pool_2d
        self.layer_num_of_elements = [36, 4, 24]  # number of layer elements
        self.layer_filter_sizes =    [4, None, None]  # only for conv layers
        self.layer_strides =         [2, 2, None]  # only for conv/pool layers
        self.dropouts =              [False, False, True]  # whether to add a dropout on this layer
        self.keep_probability =      [None, None, 0.85]  # dropout keep rate
        self.activations =           ['relu', None, 'linear']  # each layers activation function
        self.output_layer_activation = 'softmax'
        self.regression_optimizer = 'adam'
        self.loss_function = 'mean_square'
        #  activations
        # linear, tanh, sigmoid, softmax, softplus, softsign, relu, relu6, leaky_relu, prelu, elu, crelu, selu
        #  optimizers
        # sgd, rmsprop, adam, momentum, adagrad, ftrl, adadelta, proxi_adagrad, nesterov
        # loss functions
        #  softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, weighted_crossentropy,
        #  mean_square, hinge_loss, roc_auc_score, weak_cross_entropy_2d
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
        if self.layer_types[0] == 'conv_2d':  # if the first layer in the model is a convolution layer, it will expect the whole game matrix
            X = np.array([i[0] for i in self.training_data]).reshape((-1, self.game_size[1], self.game_size[0], 1))
        else:
            # input_length = len(self.training_data[0][0])
            input_length = self.training_data[0][0].size  # otherwise it's just an array
            X = np.array([i[0] for i in self.training_data]).reshape((-1, input_length, 1))
        y = np.array([i[1] for i in self.training_data]).reshape(-1, self.output_size)
        if self.profile_run:
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                               python_tracer_level=1,
                                                               device_tracer_level=1)  # setup profiler so we can never use it
            tf.profiler.experimental.start(self.log_dir, options=options)
        step_snapshot_interval = self.snapshot_frequency * int(len(self.training_data) / 64)  # calculate save interval entirely wrong
        print('Fitting the model and snapshotting every ' + str(self.snapshot_frequency) + ' epochs or every ' +
              str(step_snapshot_interval) + ' steps.')  # tell user save interval rate
        time.sleep(5)
        early_stopping_cb = EarlyStoppingCallback(accuracy_threshold=self.stop_accuracy)  # define non-working stop criteria, not that it will ever matter
        self.model.fit({'input': X}, {'targets': y}, n_epoch=self.epochs, snapshot_epoch=False, show_metric=True,
                       run_id=self.run_id, snapshot_step=step_snapshot_interval, callbacks=early_stopping_cb)  # fit the model to the training data
        if self.profile_run:
            tf.profiler.experimental.stop()  # stop profiler, even though we never use that shit

    def neural_network_model(self):
        tflearn.init_graph(num_cores=self.num_of_cores, gpu_memory_fraction=self.gpu_mem)  # setup graph for usage

        if self.layer_types[0] == 'fully_connected' or self.layer_types[0] == 'conv_1d':  # again check input size, a function of first layer definition
            self.network = input_data(shape=[None, self.observation_space_length, 1], name='input')
        else:
            self.network = input_data(shape=[None, self.game_size[1], self.game_size[0], 1], name='input')
        self.add_hidden_layers()  # add all the neural networks hidden layers
        self.network = fully_connected(self.network, self.output_size, activation=self.output_layer_activation)  # add a fully connected layer at the end to give the output (the output layer)
        self.network = regression(self.network, optimizer=self.regression_optimizer, learning_rate=self.learning_rate,
                                  loss=self.loss_function, name='targets')  # add a regression method, so this shit even works at all

        if self.save_checkpoints and self.model_dir is not None:   # setup save path
            checkpoint_dir = os.path.join(self.model_dir, self.run_dir, self.model_name)
            best_checkpoint_dir = os.path.join(self.model_dir, self.run_dir, 'best-' + self.model_name)
            self.model = tflearn.DNN(self.network, tensorboard_dir=self.log_dir, checkpoint_path=checkpoint_dir,
                                     best_checkpoint_path=best_checkpoint_dir, tensorboard_verbose=3)  # initialize neural network
        else:
            self.model = tflearn.DNN(self.network, tensorboard_dir=self.log_dir)  # initialize neural network
        return self.model

    def add_hidden_layers(self):
        tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
        for i in range(0, len(self.layer_types)):
            if self.layer_types[i] == 'fully_connected':  # regular dense layer
                self.network = fully_connected(self.network,
                                               n_units=self.layer_num_of_elements[i],
                                               activation=self.activations[i],
                                               weights_init=tnorm)
            elif self.layer_types[i] == 'conv_2d':  # 2d convolution layer
                self.network = conv_2d(self.network,
                                       nb_filter=self.layer_num_of_elements[i],
                                       filter_size=self.layer_filter_sizes[i],
                                       strides=self.layer_strides[i],
                                       activation=self.activations[i],
                                       weights_init=tnorm)
            elif self.layer_types[i] == 'conv_1d':  # 1d convolution layer, basically useless
                self.network = conv_1d(self.network,
                                       nb_filter=self.layer_num_of_elements[i],
                                       filter_size=self.layer_filter_sizes[i],
                                       strides=self.layer_strides[i],
                                       activation=self.activations[i],
                                       weights_init=tnorm)
            elif self.layer_types[i] == 'max_pool_2d':  # 2d max pooling, to downsample conv2d outputs
                self.network = max_pool_2d(self.network,
                                           kernel_size=self.layer_num_of_elements[i],
                                           strides=self.layer_strides[i],
                                           )
            if self.dropouts[i]:  # add a dropout inbetween layers to add randomness to fitting
                self.network = dropout(self.network, self.keep_probability[i])

    def save_model(self):
        self.model.save(self.model_dir + '\\' + self.run_dir)

    def load_model(self, run_name=None):
        if run_name is None:  # if we didn't get a name, get the newest model in the model dir
            [model_to_load, self.run_dir] = self.get_last_run()
            final_path = os.path.join(os.getcwd(), model_to_load)
        else:  # otherwise, find the model with the given name
            final_path = MyTools.find_file_in_folder(self.model_dir, run_name)
            self.run_dir = '\\'.join(final_path.split('\\')[1:-1])
        self.load_layers()  # load the layers so we can setup a matching DNN
        self.neural_network_model()  # make the DNN with the redefined layers
        print('Loading: ' + final_path)
        self.model.load(final_path)  # load the model
        return self.model

    def get_last_run(self):
        model_to_load = None
        run_dir = []
        files = []
        for (dir_path, dir_names, file_names) in os.walk(self.model_dir):  # get all files in model dir
            for file in file_names:
                files.append(dir_path + '\\' + file)
        files = sorted(files, key=os.path.getctime, reverse=True)   # sort files so newest is first
        for file in files:
            if file.endswith('.meta'):  # if its model file, save the name
                model_to_load = file.split('.')[0]  # get rid of the extension
                tmp = model_to_load.split('\\')  # get rid of the top folder from the path
                run_dir = '\\'.join(tmp[1:-1])  # get rid of the top folder from the path
                break
        if model_to_load is None:
            raise Exception('\n\nNo models to load!')  # we couldn't find a model
        return model_to_load, run_dir

    def plot_graphs(self):
        os.system("taskkill /IM ""tensorboard.main"" /F")  # one of these works, I dunno
        os.system("taskkill /IM ""tensorboard.exe"" /F")  # one of these works, I dunno
        print(os.getcwd() + "\\venv\\Scripts\\python.exe -m tensorboard.main --logdir=" + self.log_dir +
              '--samples_per_plugin scalars=9999999999,images=9999999999')
        os.system(os.getcwd() + "\\venv\\Scripts\\python.exe -m tensorboard.main --logdir=" + self.log_dir +
                  '--samples_per_plugin scalars=9999999999,images=9999999999')  # start tensorboard without trimming data cause I have the RAM to do it
        #  print("tensorboard. --logdir " + self.log_dir)
        #  os.system("tensorboard --logdir " + self.log_dir)

    def load_training_data(self, filename=None):
        if filename is not None:
            file_to_load = MyTools.find_file_in_folder(self.model_dir, filename)  # load specific training data
        else:
            file_to_load = MyTools.get_newest_file_in_folder_w_ext(self.model_dir, 'npy')  # load newest training data
        print('Loading: ' + file_to_load)
        np_load_old = np.load  # some stack overflow np funk
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)  # some stack overflow np funk
        self.training_data = np.load(file_to_load)  # load the data
        np.load = np_load_old

    def change_run_dir(self, new_run_dir):  # change current run dir in driver for DNN
        self.run_dir = new_run_dir
        checkpoint_dir = os.path.join(self.model_dir, self.run_dir, self.model_name)
        best_checkpoint_dir = os.path.join(self.model_dir, self.run_dir, 'best-' + self.model_name)
        self.model.trainer.checkpoint_path = checkpoint_dir
        self.model.trainer.best_checkpoint_path = best_checkpoint_dir

    def save_layers(self):  # save layers so when we load the model we can make a new matching one, if its been changed
        data = {}
        file = open(os.path.join(self.model_dir, self.run_dir, 'Model_Network.pkl'), 'wb')
        data['activations'] = self.activations
        data['dropouts'] = self.dropouts
        data['output_layer_activation'] = self.output_layer_activation
        data['regression_optimizer'] = self.regression_optimizer
        data['loss_function'] = self.loss_function
        data['layer_types'] = self.layer_types
        data['layer_num_of_elements'] = self.layer_num_of_elements
        data['layer_filter_sizes'] = self.layer_filter_sizes
        data['layer_strides'] = self.layer_strides
        data['observation_space_length'] = self.observation_space_length
        data['keep_probability'] = self.keep_probability
        pickle.dump(data, file)
        file.close()

    def load_layers(self):  # load layers so we can overwrite the ones defined so the new DNN can be made, so we can process game data and feed it to the model correctly
        path = os.path.join(self.model_dir, self.run_dir, 'Model_Network.pkl')
        if os.path.isfile(path):
            try:
                file = open(path, 'rb')
                data = pickle.load(file)
                self.activations = data['activations']
                self.dropouts = data['dropouts']
                self.output_layer_activation = data['output_layer_activation']
                self.regression_optimizer = data['regression_optimizer']
                self.loss_function = data['loss_function']
                self.layer_types = data['layer_types']
                self.layer_num_of_elements = data['layer_num_of_elements']
                self.layer_filter_sizes = data['layer_filter_sizes']
                self.layer_strides = data['layer_strides']
                self.observation_space_length = data['observation_space_length']
                self.keep_probability = data['keep_probability']
                file.close()
                print('Loading: ' + path)
            except KeyError:
                raise Exception('Outdated model version, does not contain required data for network generation!')
        else:
            print('No Model Network to load!')
            time.sleep(1)

    @staticmethod
    def check_version():  # check if we can use a GPU
        print('TensorFlow version: ', tf.__version__)
        device_name = tf.test.gpu_device_name()
        if not device_name:
            print('GPU device not found')
        else:
            print('Found GPU at: {}'.format(device_name))
            # gpus = tf.config.list_physical_devices('GPU')
            # tf.config.set_visible_devices(gpus[0], 'GPU')

    def print_tensorboard_plots(self, plot_scalars=True, plot_histograms=False):  # print tensorboard plots data to PDF
        tmp = os.path.join(self.log_dir, self.run_dir)
        try:
            log_path = os.path.join(tmp, os.listdir(tmp)[0])
            print('Loading log: ' + log_path)
        except FileNotFoundError:
            print('No log data in dir: ' + tmp)  # catch the error of just not finding shit
            return
        pdf = PdfPages(os.path.join(self.model_dir, self.run_dir, "TrainingPlots.pdf"))  # make pdf to save plots
        start_time = time.time()
        reader = SummaryReader(log_path)  # launch data reader
        print(time.time() - start_time)
        if plot_scalars:
            df_scalars = reader.scalars  # get scalars out of data
            self.plot_scalars(pdf, df_scalars)  # plot them
            del df_scalars
        if plot_histograms:  # get histogram data out of data
            df_histograms = reader.histograms  # plot them, this shit doesn't work btw
            self.plot_histograms(pdf, df_histograms)
        pdf.close()

    @staticmethod
    def plot_histograms(pdf, df):  # takes 400 years, not even worth
        tags = df.tag.unique()
        for tag in tags:
            print('Plotting ' + tag + ' Histogram')
            limits = df[df['tag'] == tag].limits
            counts = df[df['tag'] == tag].counts
            steps = df[df['tag'] == tag].step
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', rasterized=True)
            for i in tqdm(range(0, len(df.step))):
                ax.bar(limits[i], counts[i], zs=steps[i], zdir='z', color='r', ec='k', alpha=0.4)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            pdf.savefig(fig)
            plt.clf()
            plt.close('all')

    @staticmethod
    def plot_scalars(pdf, df):  # EZ line plots, very cute
        tags = df.tag.unique()
        for tag in tags:
            print('Plotting ' + tag)
            values = df[df['tag'] == tag]['value']
            fig = plt.figure()
            plt.plot(values)
            plt.xlabel('Step')
            plt.grid()
            plt.title(tag)
            pdf.savefig(fig)
            plt.clf()
            plt.close('all')


class EarlyStoppingCallback(callbacks.Callback):  # stop the model fitting if its already accurate
    def __init__(self, accuracy_threshold):
        super().__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, training_state):
        if training_state.val_acc is None:
            return
        if training_state.val_acc > self.accuracy_threshold:
            raise StopIteration
