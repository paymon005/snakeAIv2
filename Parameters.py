from datetime import datetime


class Parameters:

    def __init__(self):
        # True
        # False
        self.run_to_load = 'Snake_Model-18330'
        self.training_data_to_load = 'Req2_1000games_100goalSteps_6ob.npy'
        self.load_model = False
        self.load_specific_model = False
        self.load_training_data = False
        self.load_specific_training_data = False
        self.generate_training_data = True
        self.train_model = True
        self.recursive_training = False
        self.plot_graphs = False
        self.profile_run = False
        self.use_target_training_data_length = True
        self.force_score_requirement = False
        self.include_reward_in_obs = False
        self.forced_score_requirement = 3
        self.target_training_data_length = 1000
        self.score_check_runs = 1000
        self.accepted_percentile = 98
        self.initial_games = 1000
        self.goal_steps = 300
        self.cores_for_games = 16
        self.cores_for_training = 12
        self.gpu_memory_fraction = 0.5
        self.epochs = 30
        self.keep_rate = 0.8
        self.LR = 0.001
        self.kick_out_sore = -500
        self.recursive_iterations = 10
        self.recursive_score_check_runs = 200
        self.recursive_initial_games = 10000
        self.recursive_target_training_data_length = 100
        self.recursive_epochs = 10
        self.target_score_increase = 1
        self.mutation_rate = 0.10
        self.score_requirement = 0
        # self.game_size = [72, 48]
        self.game_size = [36, 24]
        self.snake_speed = 45
        self.games_to_play = 10
        self.observation_space_type = 2  # 1 for entire matrix, 2 for array of perception
        self.now = datetime.now()  # current date and time
        self.model_dir = 'Models'
        self.log_dir = 'log'
        self.model_name = 'Snake_Model_' + self.now.strftime("%Y_%m_%d___%H_%M_%S")
        self.run_dir = self.now.strftime("%Y_%m_%d___%H_%M_%S")
        self.run_random_game = True

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
        if not self.generate_training_data and not self.load_training_data and self.train_model:
            raise Exception('This shit wont run. No training data for model')
