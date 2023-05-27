from datetime import datetime


class Parameters:

    def __init__(self):
        # True
        # False
        self.run_to_load = 'Snake_Model_2023_05_23___00_06_12-14410'
        self.training_data_to_load = 'FullMatrix_36x36_Req2_1000games_300goalstep.npy'
        # model loading
        self.load_model = False
        self.load_specific_model = False
        # training data loading
        self.load_training_data = True
        self.load_specific_training_data = True
        # generate training data
        self.generate_training_data = False
        # model training
        self.train_model = True
        self.recursive_training = False  # continue training the model after, using its predictions for training data
        # plotting
        self.plot_graphs = True
        # training data forcing
        self.force_training_data_length = True  # keep running until you get a certain amount of training data
        self.forced_training_data_length = 500
        self.force_score_requirement = True  # force score req, rather than using a percentile and score_check runs to define it
        self.forced_score_requirement = 2
        # other parameters
        self.score_check_runs = 100
        self.initial_games = 10
        self.accepted_percentile = 95
        self.goal_steps = 300
        self.epochs = 50
        self.LR = 0.002
        self.cores_for_games = 12
        self.cores_for_training = 12
        self.gpu_memory_fraction = 0.5
        self.kick_out_sore = -500
        self.recursive_iterations = 10
        self.recursive_score_check_runs = 200
        self.recursive_initial_games = 10000
        self.recursive_target_training_data_length = 100
        self.recursive_epochs = 10
        self.target_score_increase = 1
        self.mutation_rate = 0.01
        self.score_requirement = 9e-9
        # self.game_size = [72, 48]
        self.game_size = [36, 36]
        self.snake_speed = 45
        self.games_to_play = 5
        self.now = datetime.now()  # current date and time
        self.model_dir = 'Models'
        self.log_dir = 'log'
        self.model_name = 'Snake_Model_' + self.now.strftime("%Y_%m_%d___%H_%M_%S")
        self.run_dir = self.now.strftime("%Y_%m_%d___%H_%M_%S")
        self.run_random_game = False
        self.use_model_when_making_training_data = True
        self.include_reward_in_obs = False
        self.profile_run = False
        self.play_full_game = False
        self.first_layer_type = None
        self.print_direction_during_game = True

    def parse_inputs(self):
        if not self.load_specific_training_data:
            self.training_data_to_load = None
        else:
            self.load_training_data = True
        if not self.load_specific_model:
            self.run_to_load = None
        else:
            self.load_model = True
        if self.recursive_training:
            self.run_dir = self.run_dir + '\\Iteration_1'
        if self.force_score_requirement:
            self.score_requirement = self.forced_score_requirement
        if self.load_training_data:
            self.generate_training_data = False
        if self.generate_training_data or self.train_model or self.load_model:
            self.run_random_game = False
        if not self.generate_training_data and not self.train_model:
            self.play_full_game = True
        if not self.generate_training_data and not self.load_training_data and self.train_model:
            raise Exception('This shit wont run. No training data for model')
