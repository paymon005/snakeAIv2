import os
import time
import numpy as np
import random
from SnakeEnvironment import SnakeEnv
from DeepNeuralNetworkDriver import DnnDriver
from statistics import mean
from collections import Counter
from tqdm import tqdm
from datetime import datetime
import multiprocessing
from parallelbar import progress_map
from functools import partial

start_time = time.time()
random.seed(time.time())
now = datetime.now()  # current date and time
run_dir = now.strftime("%Y_%m_%d___%H_%M_%S")
profile_run = False

number_of_cores = 12

game_size = [72, 48]
# game_size = [36, 24]
snake_speed = 45
observation_space_type = 2  # 1 for entire matrix, 2 for array of perception
score_check_runs = 50000
accepted_percentile = 95
initial_games = 1000000
epochs = 100
keep_rate = 0.8
LR = 0.01
model_dir = 'Models'
model_name = 'Snake_Model'
kickout_sore = -500


def main():
    # True
    # False
    load_model = True
    load_specific_model = True
    generate_training_data = False
    load_training_data = True
    load_specific_training_data = False
    train_model = False
    plot_graphs = True
    run_to_load = ['2023_05_19___01_00_11', 'Snake_Model-2177208']
    training_data_to_load = ['2022_11_19___00_45_36']
    [run_to_load, training_data_to_load] = parse_inputs(load_model, load_specific_model, generate_training_data,
                                                        load_training_data, load_specific_training_data,
                                                        train_model, run_to_load, training_data_to_load)
    run_tflearn_trainer(load_model, generate_training_data, load_training_data, train_model, run_to_load,
                        training_data_to_load, plot_graphs)
    duration = time.time() - start_time
    print('Total Elapsed Time: ' + str(int(duration / 60)) + ' minutes')


def parse_inputs(load_model, load_specific_model, generate_training_data, load_training_data,
                 load_specific_training_data, train_model, run_to_load, training_data_to_load):
    if not load_specific_training_data:
        training_data_to_load = None
    if not load_specific_model:
        run_to_load = None
    if not generate_training_data and not load_training_data and train_model:
        raise Exception('This shit wont run. No training data for model')
    if not generate_training_data and not load_training_data and train_model:
        raise Exception('This shit wont run. No training data for model')
    return run_to_load, training_data_to_load


def spawn_trainer():
    training_driver = DnnDriver(model_name, model_dir, run_dir, LR, epochs, keep_rate, profile_run)
    training_driver.check_version()
    env = SnakeEnv(game_size, observation_space_type)
    training_driver.observation_space_size = env.observation_space_size
    training_driver.observation_space_length = env.observation_space_length
    return training_driver


def run_tflearn_trainer(load_model, generate_training_data, load_training_data, train_model, run_to_load,
                        training_data_to_load, plot_graphs):
    training_driver = spawn_trainer()
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
        training_driver = make_training_data(training_driver, load_model)
    if train_model:
        training_driver.train_model()
    if not train_model and not load_model:
        run_model(10)
    else:
        final_scores = run_model(10, training_driver)
        save_outputs(training_driver, final_scores)
    if plot_graphs:
        training_driver.plot_graphs()


def make_training_data(training_driver, load_model):
    test_run_scores, score_requirement = get_score_requirement(training_driver, score_check_runs, load_model)
    print('Setting ' + str(score_requirement) + ' as the score requirement')
    time.sleep(5)
    training_driver.create_save_dir(model_dir, run_dir)
    training_driver.training_data, accepted_scores = \
        initial_population(training_driver, initial_games, load_model, score_requirement)
    time.sleep(5)
    if len(training_driver.training_data) == 0:
        return
    save_inputs(accepted_scores, test_run_scores, score_requirement)
    return training_driver


def get_score_requirement(training_driver, num_of_runs, load_model):
    _, accepted_scores = initial_population(training_driver, num_of_runs, load_model)
    idx = int(accepted_percentile / 100 * len(accepted_scores))
    req = sorted(accepted_scores)[idx-1]
    return accepted_scores, req


def initial_population(training_driver, num_of_runs, load_model, score_requirement=-9e9):
    length = training_driver.observation_space_length
    if load_model:
        model = training_driver.model
    else:
        model = None
    if number_of_cores > 1 and not load_model:  # solve in parallel (cant pickle model objects so cant run in parallel)
        manager = multiprocessing.Manager()
        accepted_scores_dict = manager.dict()
        scores_dict = manager.dict()
        training_data_dict = manager.dict()
        func = partial(run_a_game, length, score_requirement, scores_dict, accepted_scores_dict, training_data_dict, model)
        progress_map(func, range(num_of_runs), process_timeout=1.5, n_cpu=number_of_cores)
        [scores, accepted_scores, training_data] = sort_dict_into_list(scores_dict, accepted_scores_dict, training_data_dict)
    else:  # solve with a single core
        accepted_scores = []
        scores = []
        training_data = []
        for _ in tqdm(range(num_of_runs)):  # iterate through however many games we want:
            [training_data, scores, accepted_scores] = \
                run_a_game(length, score_requirement, scores, accepted_scores, training_data, model)
    if score_requirement > -9999999:
        save_training_data(training_data, training_driver)
    if len(accepted_scores) > 0:
        print_score_stats(accepted_scores, 1)
    else:
        raise Exception('\n\n\n**************Out of ' + str(num_of_runs) + ' runs, no one got a score higher than ' +
                        str(score_requirement) + '.\n' + '**************Average score:', mean(scores) + '\n\n\n')
    return training_data, accepted_scores


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
    print('Number of ', score_string, 's: ', len(accepted_scores))
    print('Max ', score_string, ': ', max(accepted_scores))
    print('Average ', score_string, ': ', round(mean(accepted_scores), 0))
    print('Min ', score_string, ': ', min(accepted_scores))
    print(Counter(accepted_scores))


def sort_dict_into_list(scores_dict, accepted_scores_dict, training_data_dict):
    accepted_scores = []
    scores = []
    training_data = []
    for key in scores_dict:
        scores.append(scores_dict[key])
    start_time2 = time.time()
    cnt = int(0)
    dict_len = len(accepted_scores_dict)
    for key in accepted_scores_dict:
        print(str(cnt / dict_len * 100) + '% complete')
        accepted_scores.append(accepted_scores_dict[key])
        training_data.extend(training_data_dict[key])
        cnt = cnt + 1
    elapsed_time = time.time() - start_time2
    print(str(elapsed_time) + ' seconds')
    return scores, accepted_scores, training_data


def run_model(num_of_runs, training_driver=None):
    env = SnakeEnv(game_size, observation_space_type)
    length = env.observation_space_length
    scores = []
    choices = []
    for each_game in range(num_of_runs):
        score = 0
        game_memory = []
        prev_obs = env.state
        env.reset()
        while True:
            env.render()
            time.sleep(1 / snake_speed)
            if training_driver is not None:
                action = np.argmax(training_driver.model.predict(prev_obs.reshape(-1, length, 1))[0])
            else:
                action = random.randint(0, env.action_space_size - 1)
            choices.append(action)
            [new_observation, reward, done, []] = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            print('Score: ' + str(score))
            if done or score < kickout_sore:
                break
        scores.append(score)
    print_score_stats(scores, 2)
    count_and_print_choices(choices)
    env.close()
    return scores


def count_and_print_choices(choices):
    # 0 = Straight, 1: Turn Left, 2: Turn Right
    print('Straight:{}%  Turn Left:{}%  Turn Right:{}%'.format(
        round(choices.count(0) / len(choices) * 100, 2),
        round(choices.count(1) / len(choices) * 100, 2),
        round(choices.count(2) / len(choices) * 100, 2)))


def run_a_game(length, score_requirement, scores, accepted_scores, training_data, model=None, run_id=None):
    env = SnakeEnv(game_size, observation_space_type)
    game_memory = []
    score = 0
    prev_observation = env.state
    while True:  # for each frame in goal steps
        if model is not None:
            action = np.argmax(model.predict(prev_observation.reshape(-1, length, 1))[0])
        else:
            action = random.randint(0, env.action_space_size - 1)
        [observation, reward, done, []] = env.step(action)
        game_memory.append([prev_observation, action])
        prev_observation = observation
        score += reward
        if done or score < kickout_sore:
            break
    if run_id is None:
        scores.append(score)  # save overall scores
        if score >= score_requirement:
            accepted_scores.append(score)
            training_data.extend(parse_game_memory(game_memory))
    else:
        scores[run_id] = score  # save overall scores
        if score >= score_requirement:
            accepted_scores[run_id] = score
            training_data[run_id] = parse_game_memory(game_memory)
    env.close()
    return training_data, scores, accepted_scores


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


def save_inputs(accepted_scores, test_run_scores, score_requirement):
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


def save_outputs(training_driver, final_scores):
    filename = training_driver.model_dir + '\\' + training_driver.run_dir + '\\' + 'Summary' + '.txt'
    file = open(filename, "a")
    file.write('Number of final scores    : ' + str(len(final_scores)) + '\n')
    file.write('Max Score                 : ' + str(max(final_scores)) + '\n')
    file.write('Average Score             : ' + str(round(mean(final_scores), 0)) + '\n')
    file.write('Min Score                 : ' + str(min(final_scores)) + '\n')
    file.close()


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()
