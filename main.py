import os
import shutil
import time
import numpy as np
import random
from SnakeEnvironment import SnakeEnv
from DeepNeuralNetworkDriver import DnnDriver
from statistics import mean
from tqdm import tqdm
from datetime import datetime
import multiprocessing
from parallelbar import progress_map
from functools import partial
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# from example import SnakeNN
# from Player import some_random_games_first, generate_population, create_dummy_model, train_model_ex, evaluate

random.seed(time.time())
profile_run = False

cores_for_games = 12
cores_for_training = 8
gpu_memory_fraction = 0.25
# game_size = [72, 48]
game_size = [36, 24]
snake_speed = 45
observation_space_type = 2  # 1 for entire matrix, 2 for array of perception
score_check_runs = 5000
accepted_percentile = 95
initial_games = 100000
epochs = 5
keep_rate = 0.85
LR = 0.001
model_dir = 'Models'
model_name = 'Snake_Model'
kick_out_sore = -500
games_to_play = 10
recursive_iterations = 10
recursive_score_check_runs = 250
recursive_initial_games = 5000
recursive_epochs = 20


def main():
    clean_model_dir()
    # SnakeNN().train()
    # True
    # False
    reward_system = 0  # 0: old, 1: new
    load_model = False
    load_specific_model = False
    load_training_data = False
    load_specific_training_data = False
    generate_training_data = True
    train_model = True
    recursive_training = False
    plot_graphs = False

    now = datetime.now()  # current date and time
    run_dir = now.strftime("%Y_%m_%d___%H_%M_%S")
    run_to_load = ['2023_05_19___01_00_11', 'Snake_Model-2177208']
    training_data_to_load = ['2022_11_19___00_45_36']

    [load_model, generate_training_data, load_training_data, train_model, run_to_load, training_data_to_load,
     recursive_training, run_dir, plot_graphs] = parse_inputs(load_model, load_specific_model, generate_training_data,
                                                              load_training_data, load_specific_training_data,
                                                              train_model, run_to_load, training_data_to_load,
                                                              recursive_training, run_dir, plot_graphs)

    start_time = time.time()
    run_tflearn_trainer(run_dir, load_model, generate_training_data, load_training_data, train_model, run_to_load,
                        training_data_to_load, plot_graphs, recursive_training, reward_system)
    duration = time.time() - start_time
    print('Total Elapsed Time: ' + str(int(duration / 60)) + ' minutes')


def parse_inputs(load_model, load_specific_model, generate_training_data, load_training_data,
                 load_specific_training_data, train_model, run_to_load, training_data_to_load, recursive_training,
                 run_dir, plot_graphs):
    if not load_specific_training_data:
        training_data_to_load = None
    if not load_specific_model:
        run_to_load = None
    if recursive_training:
        run_dir = run_dir + '\\Iteration_1'
        plot_graphs = False
    if not generate_training_data and not load_training_data and train_model:
        raise Exception('This shit wont run. No training data for model')
    if not generate_training_data and not load_training_data and train_model:
        raise Exception('This shit wont run. No training data for model')
    return load_model, generate_training_data, load_training_data, train_model, run_to_load, \
           training_data_to_load, recursive_training, run_dir, plot_graphs


def spawn_trainer(run_dir, reward_system):
    training_driver = DnnDriver(model_name, model_dir, run_dir, LR, epochs, keep_rate, cores_for_training,
                                gpu_memory_fraction, profile_run)
    training_driver.check_version()
    env = SnakeEnv(game_size, observation_space_type)
    training_driver.train_type = reward_system
    if reward_system == 1:
        env.observation_space_length = env.observation_space_length + 1  # add 1 for reward
        training_driver.output_size = 1
    training_driver.observation_space_length = env.observation_space_length
    return training_driver


def run_tflearn_trainer(run_dir, load_model, generate_training_data, load_training_data, train_model, run_to_load,
                        training_data_to_load, plot_graphs, recursive_training, reward_system):
    training_driver = spawn_trainer(run_dir, reward_system)
    if load_model:
        training_driver.load_model(run_to_load)
    else:
        training_driver.neural_network_model()
    if load_training_data:
        training_driver.load_training_data(training_data_to_load)
    if generate_training_data or train_model:
        training_driver.change_run_dir(run_dir)
        training_driver.create_save_dir(model_dir, run_dir)
    if generate_training_data:
        training_driver = make_training_data(run_dir, training_driver, load_model, reward_system)
    if train_model:
        training_driver.train_model()
    if not train_model and not load_model:
        [final_scores, choices] = run_x_games(True)
    else:
        [final_scores, choices] = run_x_games(True, training_driver.model)
        save_outputs(training_driver, final_scores, choices)
    if plot_graphs:
        training_driver.plot_graphs()

    if recursive_training:
        load_model = True
        iteration = 1
        training_driver.epochs = recursive_epochs
        while iteration < recursive_iterations:
            iteration += 1
            tmp = run_dir.split('\\')
            tmp = '\\'.join(tmp[:-1])
            run_dir = tmp + '\\Iteration_' + str(iteration)
            training_driver.change_run_dir(run_dir)
            training_driver.create_save_dir(model_dir, run_dir)
            training_driver = make_training_data(run_dir, training_driver, load_model, reward_system, True)
            training_driver.train_model()
            [final_scores, choices] = run_x_games(True, training_driver)
            save_outputs(training_driver, final_scores, choices)


def make_training_data(run_dir, training_driver, load_model, reward_system, in_recursion=False):
    if in_recursion:
        games = [recursive_score_check_runs, recursive_initial_games]
    else:
        games = [score_check_runs, initial_games]
    test_run_scores, score_requirement = get_score_requirement(training_driver, games[0], load_model)
    print('Setting ' + str(score_requirement) + ' as the score requirement')
    time.sleep(5)
    training_driver.create_save_dir(model_dir, run_dir)
    training_driver.training_data, accepted_scores = \
        initial_population(training_driver, games[1], load_model, reward_system, score_requirement)
    time.sleep(5)
    if len(training_driver.training_data) == 0:
        return
    save_inputs(run_dir, accepted_scores, test_run_scores, score_requirement)
    return training_driver


def get_score_requirement(training_driver, num_of_runs, load_model):
    _, accepted_scores = initial_population(training_driver, num_of_runs, load_model)
    idx = int(accepted_percentile / 100 * len(accepted_scores))
    req = sorted(accepted_scores)[idx - 1]
    return accepted_scores, req


def initial_population(training_driver, num_of_runs, load_model, reward_system=0, score_requirement=-9e9):
    length = training_driver.observation_space_length
    if load_model:
        model = training_driver.model
    else:
        model = None
    if cores_for_games > 1 and not load_model:  # solve in parallel (cant pickle model objects so cant run in parallel)
        manager = multiprocessing.Manager()
        accepted_scores_dict = manager.dict()
        scores_dict = manager.dict()
        training_data_dict = manager.dict()
        func = partial(run_a_game, length, score_requirement, scores_dict, accepted_scores_dict, training_data_dict,
                       reward_system, model, False, None)
        progress_map(func, range(num_of_runs), process_timeout=3, n_cpu=cores_for_games)
        [scores, accepted_scores, training_data] = sort_dict_into_list(scores_dict, accepted_scores_dict,
                                                                       training_data_dict)
    else:  # solve with a single core
        accepted_scores = []
        scores = []
        training_data = []
        for _ in tqdm(range(num_of_runs)):  # iterate through however many games we want:
            [training_data, scores, accepted_scores, _] = \
                run_a_game(length, score_requirement, scores, accepted_scores, training_data, reward_system, model,
                           False, None)
    if score_requirement > -9999999:
        save_training_data(training_data, training_driver)
    if len(accepted_scores) > 0:
        print_score_stats(accepted_scores, 1)
    else:
        raise Exception('\n\n\n**************Out of ' + str(num_of_runs) + ' runs, no one got a score higher than ' +
                        str(score_requirement) + '.\n' + '**************Average score:', mean(scores) + '\n\n\n')
    return training_data, accepted_scores


def run_a_game(length=None, score_requirement=9e-9, scores=None, accepted_scores=None, training_data=None,
               reward_system=0, model=None, render=False, choices=None, run_id=None):
    env = SnakeEnv(game_size, observation_space_type)
    if choices is None:
        choices = []
    if scores is None:
        scores = []
    if model is not None:
        length = model.inputs[0].shape.dims[1].value
        if length > env.observation_space_length:
            reward_system = 1
    game_memory = []
    score = 0
    reward = 0
    prev_observation = env.state
    while True:  # for each frame in goal steps
        if render:
            env.render()
            time.sleep(1 / snake_speed)
            print(score)
        if model is not None:
            if reward_system == 1:
                actions = []
                for _ in range(0, 3):
                    actions.append(model.predict(np.append(prev_observation, reward).reshape(-1, length, 1))[0])
                action = np.argmax(actions)
            else:
                action = np.argmax(model.predict(prev_observation.reshape(-1, length, 1))[0])
        else:
            action = random.randint(0, env.action_space_size - 1)
        choices.append(action)
        [observation, reward, done] = env.step(action)
        if reward_system == 0:
            game_memory.append([prev_observation, action])
        elif reward_system == 1:
            game_memory.append([np.insert(prev_observation, 0, action, axis=0), reward])
        prev_observation = observation
        score += reward
        if done or score < kick_out_sore:
            break
    if run_id is None:
        [scores, accepted_scores, training_data] = append_data(
            scores, accepted_scores, training_data, score, score_requirement, game_memory, reward_system)
    else:
        [scores, accepted_scores, training_data] = add_to_dict(
            scores, accepted_scores, training_data, score, score_requirement, game_memory, reward_system, run_id)
    env.close()
    return training_data, scores, accepted_scores, choices


def append_data(scores, accepted_scores, training_data, score, score_requirement, game_memory, reward_system):
    scores.append(score)  # save overall scores
    if accepted_scores is not None:
        if score >= score_requirement:
            accepted_scores.append(score)
            if reward_system == 0:
                training_data.extend(parse_game_memory(game_memory))
            elif reward_system == 1:
                training_data.extend(game_memory)
    return scores, accepted_scores, training_data


def add_to_dict(scores, accepted_scores, training_data, score, score_requirement, game_memory, reward_system, run_id):
    scores[run_id] = score  # save overall scores
    if accepted_scores is not None:
        if score >= score_requirement:
            accepted_scores[run_id] = score
            if reward_system == 0:
                training_data[run_id] = parse_game_memory(game_memory)
            elif reward_system == 1:
                training_data[run_id] = game_memory
    return scores, accepted_scores, training_data


def run_x_games(render, model=None):
    scores = []
    choices = []
    for _ in range(games_to_play):
        [_, score, _, choice] = run_a_game(model=model, render=render)
        scores.extend(score)
        choices.extend(choice)
    print_score_stats(scores, 2)
    count_and_print_choices(choices)
    return scores, choices


def parse_game_memory(game_memory):
    new_mem = []
    output = []
    for data in game_memory:
        # convert to one-hot (this is the output layer for our neural network)
        if data[1] == 0:
            output = [1, 0, 0]
        elif data[1] == 1:
            output = [0, 1, 0]
        elif data[1] == 2:
            output = [0, 0, 1]
        new_mem.append([data[0], output])
    return new_mem  # saving our training data


def save_training_data(training_data, training_driver):
    training_data_save = np.array(training_data)  # just in case you wanted to reference later
    save_dir = os.path.join(os.getcwd(), model_dir, training_driver.run_dir, 'training_data.npy')
    np.save(save_dir, training_data_save)


def print_score_stats(accepted_scores, score_type):
    score_string = []
    if score_type == 1:
        score_string = 'accepted score'
    elif score_type == 2:
        score_string = 'score'
    print('Number of ' + score_string + 's: ' + str(len(accepted_scores)))
    print('Max ' + score_string + ': ' + str(max(accepted_scores)))
    print('Average ' + score_string + ': ' + str(round(mean(accepted_scores), 0)))
    print('Min ' + score_string + ': ' + str(min(accepted_scores)))


def sort_dict_into_list(scores_dict, accepted_scores_dict, training_data_dict):
    accepted_scores = []
    scores = []
    training_data = []
    for key in scores_dict:
        scores.append(scores_dict[key])
    start_time2 = time.time()
    print('Sorting dictionaries into arrays')
    for key, v in tqdm(accepted_scores_dict.items()):
        accepted_scores.append(accepted_scores_dict[key])
        training_data.extend(training_data_dict[key])
    elapsed_time = time.time() - start_time2
    return scores, accepted_scores, training_data


def count_and_print_choices(choices):
    # 0 = Straight, 1: Turn Left, 2: Turn Right
    print('Straight:{}%  Turn Left:{}%  Turn Right:{}%'.format(
        round(choices.count(0) / len(choices) * 100, 2),
        round(choices.count(1) / len(choices) * 100, 2),
        round(choices.count(2) / len(choices) * 100, 2)))


def save_inputs(run_dir, accepted_scores, test_run_scores, score_requirement):
    env = SnakeEnv(game_size, observation_space_type)
    filename = model_dir + '\\' + run_dir + '\\' + 'Summary' + '.txt'
    file = open(filename, "w")
    file.write('Model Inputs\n\n')
    file.write('Percentile                : ' + str(accepted_percentile) + '\n')
    file.write('Epochs                    : ' + str(epochs) + '\n')
    file.write('Keep Rate                 : ' + str(keep_rate) + '\n')
    file.write('Learning Rate             : ' + str(LR) + '\n\n')
    file.write('Number of test scores     : ' + str(len(test_run_scores)) + '\n')
    file.write('Max of test scores        : ' + str(max(test_run_scores)) + '\n')
    file.write('Average of test scores    : ' + str(round(mean(test_run_scores), 0)) + '\n')
    file.write('Min of tests score        : ' + str(min(test_run_scores)) + '\n\n')
    file.write('Score Requirement         : ' + str(score_requirement) + '\n\n')
    file.write('Initial_games             : ' + str(initial_games) + '\n')
    file.write('Number of accepted scores : ' + str(len(accepted_scores)) + '\n')
    file.write('Max accepted score        : ' + str(max(accepted_scores)) + '\n')
    file.write('Average accepted score    : ' + str(round(mean(accepted_scores), 0)) + '\n')
    file.write('Min accepted score        : ' + str(min(accepted_scores)) + '\n\n')
    file.write('Alive Weight              : ' + str(env.alive_weight) + '\n')
    file.write('Score Weight              : ' + str(env.score_weight) + '\n')
    file.write('Dead Weight               : ' + str(env.dead_weight) + '\n')
    file.write('Loop Weight               : ' + str(env.loop_weight) + '\n')
    file.write('Towards Weight            : ' + str(env.towards_weight) + '\n')
    file.write('Away Weight               : ' + str(env.away_weight) + '\n')
    file.write('\n\nModel Outputs\n\n')
    file.close()
    env.close()


def save_outputs(training_driver, final_scores, choices):
    filename = training_driver.model_dir + '\\' + training_driver.run_dir + '\\' + 'Summary' + '.txt'
    file = open(filename, "a")
    file.write('Model Run\n')
    file.write('Number of final scores    : ' + str(len(final_scores)) + '\n')
    file.write('Max Score                 : ' + str(max(final_scores)) + '\n')
    file.write('Average Score             : ' + str(round(mean(final_scores), 0)) + '\n')
    file.write('Min Score                 : ' + str(min(final_scores)) + '\n')
    file.write('Choices [Straight]        : ' + str(round(choices.count(0) / len(choices) * 100, 2)) + '%\n')
    file.write('Choices [Left]            : ' + str(round(choices.count(1) / len(choices) * 100, 2)) + '%\n')
    file.write('Choices [Right]           : ' + str(round(choices.count(2) / len(choices) * 100, 2)) + '%\n\n')
    file.close()


def clean_model_dir():
    model_folders = os.listdir(model_dir)
    for model_folder in model_folders:
        files = []
        for (dir_path, dir_names, file_names) in os.walk(model_dir + '\\' + model_folder):
            for file in file_names:
                files.append(file)
        if len(files) == 0:
            shutil.rmtree(os.path.join(model_dir, model_folder))


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()
