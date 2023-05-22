from datetime import datetime


class Parameters:

    def __init__(self):
        # True
        # False
        self.load_model = False
        self.load_specific_model = False
        self.load_training_data = True
        self.load_specific_training_data = True
        self.generate_training_data = False
        self.train_model = True
        self.recursive_training = False
        self.plot_graphs = True
        self.profile_run = False
        self.run_to_load = 'Snake_Model-18330'
        self.training_data_to_load = 'req2_1milGames.npy'
        self.reward_system = 0
        self.cores_for_games = 16
        self.cores_for_training = 12
        self.gpu_memory_fraction = 0.5
        # self.game_size = [72, 48]
        self.game_size = [36, 24]
        self.snake_speed = 45
        self.observation_space_type = 2  # 1 for entire matrix, 2 for array of perception
        self.score_check_runs = 200
        self.accepted_percentile = 98
        self.force_score_requirement = True
        self.forced_score_requirement = 2
        self.initial_games = 5000000
        self.target_score = 4
        self.target_sample_size = 10000
        self.epochs = 30
        self.keep_rate = 0.8
        self.LR = 0.001
        self.goal_steps = 400
        self.now = datetime.now()  # current date and time
        self.model_dir = 'Models'
        self.log_dir = 'log'
        self.model_name = 'Snake_Model_' + self.now.strftime("%Y_%m_%d___%H_%M_%S")
        self.run_dir = self.now.strftime("%Y_%m_%d___%H_%M_%S")
        self.kick_out_sore = -500
        self.games_to_play = 10
        self.recursive_iterations = 10
        self.recursive_score_check_runs = 200
        self.recursive_initial_games = 10000
        self.recursive_epochs = 20
        self.mutation_rate = 0.05

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
            self.plot_graphs = False
        if not self.generate_training_data and not self.load_training_data and self.train_model:
            raise Exception('This shit wont run. No training data for model')
        if not self.generate_training_data and not self.load_training_data and self.train_model:
            raise Exception('This shit wont run. No training data for model')
