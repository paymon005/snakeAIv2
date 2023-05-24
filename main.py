import os
import shutil
import time
import numpy as np
import random
from SnakeEnvironment import SnakeEnv
from DeepNeuralNetworkDriver import DnnDriver
from statistics import mean
from tqdm import tqdm
import multiprocessing
from parallelbar import progress_map
from functools import partial
import MyTools
from Parameters import Parameters
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def main():
    param = Parameters()
    param.parse_inputs()
    clean_dir(param.model_dir)
    clean_dir(param.log_dir)
    start_time = time.time()
    run_tflearn_trainer(param)
    duration = time.time() - start_time
    print('Total Elapsed Time: ' + str(int(duration / 60)) + ' minutes')


def run_tflearn_trainer(param):
    training_driver = spawn_trainer(param)
    accepted_scores, test_run_scores = None, None
    if param.load_model:
        training_driver.load_model(param.run_to_load)
    else:
        training_driver.neural_network_model()
    param.first_layer_type = training_driver.layer_types[0]
    if param.load_training_data:
        training_driver.load_training_data(param.training_data_to_load)
    if param.generate_training_data or param.train_model:
        training_driver.change_run_dir(param.run_dir)
        MyTools.create_save_dir(param.model_dir, param.run_dir)
    if param.generate_training_data:
        [training_driver, accepted_scores, test_run_scores] = make_training_data(training_driver, param)
    if param.train_model:
        save_inputs(param, training_driver, accepted_scores, test_run_scores)
        training_driver.train_model()
    if param.run_random_game:
        run_x_games(True, param)
    elif param.load_model or param.train_model:
        [final_scores, choices] = run_x_games(True, param, training_driver.model)
        save_outputs(training_driver, final_scores, choices)
    if param.recursive_training:
        param.load_model = True
        iteration = 1
        training_driver.epochs = param.recursive_epochs
        while iteration < param.recursive_iterations:
            iteration += 1
            param.score_requirement = param.score_requirement + param.target_score_increase
            param.forced_training_data_length = param.recursive_target_training_data_length
            param.run_dir = '\\'.join(param.run_dir.split('\\')[:-1]) + '\\Iteration_' + str(iteration)
            training_driver.change_run_dir(param.run_dir)
            MyTools.create_save_dir(param.model_dir, param.run_dir)
            [training_driver, accepted_scores, test_run_scores] = make_training_data(training_driver, param, True)
            save_inputs(param, training_driver, accepted_scores, test_run_scores)
            training_driver.train_model()
            [final_scores, choices] = run_x_games(True, param, training_driver.model)
            save_outputs(training_driver, final_scores, choices)
    if param.plot_graphs:
        training_driver.print_tensorboard_plots()


def spawn_trainer(param):
    training_driver = DnnDriver(param.model_name, param.model_dir, param.log_dir, param.run_dir, param.LR, param.epochs,
                                param.cores_for_training, param.gpu_memory_fraction, param.profile_run)
    env = SnakeEnv(param.game_size, param.first_layer_type)
    if param.include_reward_in_obs:
        training_driver.observation_space_length = env.observation_space_length + 1
    else:
        training_driver.observation_space_length = env.observation_space_length
    training_driver.game_size = param.game_size
    return training_driver


def make_training_data(training_driver, param, in_recursion=False):
    if in_recursion:
        games = [param.recursive_score_check_runs, param.recursive_initial_games]
    else:
        games = [param.score_check_runs, param.initial_games]
    if param.force_score_requirement:
        test_run_scores = [param.forced_score_requirement, param.forced_score_requirement]
    else:
        test_run_scores, param.score_requirement = get_score_requirement(training_driver, games[0], param, in_recursion)
    print('Setting ' + str(param.score_requirement) + ' as the score requirement')
    time.sleep(5)
    MyTools.create_save_dir(param.model_dir, param.run_dir)
    training_driver.training_data, accepted_scores = initial_population(training_driver, games[1], param, False, True,
                                                                        in_recursion)
    time.sleep(5)
    if len(training_driver.training_data) == 0:
        return
    return training_driver, accepted_scores, test_run_scores


def get_score_requirement(training_driver, num_of_runs, param, in_recursion):
    _, accepted_scores = initial_population(training_driver, num_of_runs, param, True, False, in_recursion)
    idx = int(param.accepted_percentile / 100 * len(accepted_scores))
    req = sorted(accepted_scores)[idx - 1]
    return accepted_scores, req


def initial_population(training_driver, num_of_runs, param, score_checking, save_data, in_recursion):
    length = training_driver.observation_space_length
    if (param.load_model and param.use_model_when_making_training_data) or in_recursion:
        model = training_driver.model
    else:
        model = None
    if param.cores_for_games > 1 and not param.load_model:  # solve in parallel (cant pickle model objects so cant run in parallel)
        [scores, accepted_scores, training_data] = submit_parallel_core_jobs(param, length, num_of_runs, model,
                                                                             score_checking)
    else:  # solve with a single core
        [scores, accepted_scores, training_data] = submit_single_core_jobs(param, length, num_of_runs, model,
                                                                           score_checking)
    print('Total Games           : ' + str(len(scores)))
    print('Total Accepted Scores : ' + str(len(accepted_scores)))
    print('Training Data Length  : ' + str(len(training_data)))
    if save_data:
        save_training_data(param, training_data, training_driver)
    if len(accepted_scores) > 0:
        if score_checking:
            print_score_stats(accepted_scores, 2)
        else:
            print_score_stats(accepted_scores, 1)
    else:
        raise Exception('\n\n\n**************Out of ' + str(num_of_runs) + ' runs, no one got a score higher than ' +
                        str(param.sscore_requirement) + '.\n' + '**************Average score:', mean(scores) + '\n\n\n')
    return training_data, accepted_scores


def submit_parallel_core_jobs(param, length, num_of_runs, model, score_checking):
    manager = multiprocessing.Manager()
    scores, accepted_scores, training_data = manager.list(), manager.list(), manager.list()
    force_training_data_length = param.force_training_data_length
    if score_checking:
        force_training_data_length = False
    if force_training_data_length:
        print('Getting ' + str(param.forced_training_data_length) + ' games with a score over ' + str(
            param.score_requirement))
        time.sleep(0.5)
        func = partial(run_target_games, param, scores, accepted_scores, training_data, length, model, False, None)
        with multiprocessing.Pool(param.cores_for_games) as p:
            p.map(func, range(param.cores_for_games))
    else:
        if score_checking:
            print('Getting ' + str(param.score_check_runs) + ' games to get a score requirement')
        else:
            print('Getting ' + str(param.initial_games) + ' games and trimming to a score of ' + str(
                param.score_requirement))
        time.sleep(0.5)
        func = partial(run_a_game, param, length, scores, accepted_scores, training_data, model, False, None)
        progress_map(func, range(num_of_runs), process_timeout=3, n_cpu=param.cores_for_games)

    # [scores, accepted_scores, training_data] = sort_dict_into_list(parallel_data_dict)
    return scores, accepted_scores, training_data


def submit_single_core_jobs(param, length, num_of_runs, model, score_checking):
    accepted_scores, scores, training_data = [], [], []
    force_training_data_length = param.force_training_data_length
    if score_checking:
        force_training_data_length = False
    if force_training_data_length:
        print('Getting ' + str(param.forced_training_data_length) + ' games with a score over ' + str(
            param.score_requirement))
        time.sleep(0.5)
        cnt = 1
        while len(accepted_scores) < param.forced_training_data_length:
            [training_data, scores, accepted_scores, _] = \
                run_a_game(param, length, scores, accepted_scores, training_data, model, False, None)
            print(str(len(accepted_scores)) + '/' + str(param.forced_training_data_length) + ' [' + str(cnt) + ']')
            cnt += 1
    else:
        if score_checking:
            print('Getting ' + str(param.score_check_runs) + ' games to get a score requirement')
        else:
            print('Getting ' + str(param.initial_games) + ' games and trimming to a score of ' + str(
                param.score_requirement))
        time.sleep(0.5)
        for _ in tqdm(range(num_of_runs)):  # iterate through however many games we want:
            [training_data, scores, accepted_scores, _] = \
                run_a_game(param, length, scores, accepted_scores, training_data, model, False, None)
    return scores, accepted_scores, training_data


def run_a_game(param, length=None, scores=None, accepted_scores=None, training_data=None, model=None, render=False,
               choices=None, run_id=None):
    env = SnakeEnv(param.game_size, param.first_layer_type)
    if choices is None:
        choices = []
    if scores is None:
        scores = []
    if model is not None:
        length = model.inputs[0].shape.dims[1].value
        if length == env.observation_space_length + 1:
            param.include_reward_in_obs = True
    game_memory, this_games_choices = [], []
    score, step, reward = 0, 0, 0
    prev_observation = env.state
    mutate = False
    step_to_stop = param.goal_steps
    if param.play_full_game:
        step_to_stop = 9e99
    while step < step_to_stop:  # for each frame in goal steps
        if render:
            env.render(score)
            time.sleep(1 / param.snake_speed)
        else:
            mutate = random.randint(1, 101) <= param.mutation_rate * 100
        if param.include_reward_in_obs:
            prev_observation = np.append(prev_observation, reward)
        action = get_action(model, mutate, prev_observation, length, env, param)
        if param.print_direction_during_game and render:
            print(env.get_action_meaning(action))
        this_games_choices.append(action)
        [observation, reward, done] = env.step(action, this_games_choices)
        game_memory.append([prev_observation, action])
        prev_observation = observation
        score += reward
        if done or score < param.kick_out_sore:
            break
        step += 1
    choices.extend(this_games_choices)
    [scores, accepted_scores, training_data] = append_data(param, scores, accepted_scores, training_data, score,
                                                           game_memory)
    env.close()
    if render:
        print('Final Score: ' + str(score))
        count_and_print_choices(this_games_choices)
    return training_data, scores, accepted_scores, choices


def run_target_games(param, scores, accepted_scores, training_data, length=None, model=None, render=False, choices=None,
                     run_id=None):
    while len(accepted_scores) < param.forced_training_data_length:
        [training_data, scores, accepted_scores, _] = run_a_game(param, length, scores, accepted_scores, training_data,
                                                                 model, render, choices, run_id)
        print(str(len(accepted_scores)) + '/' + str(param.forced_training_data_length) + ' [' + str(len(scores)) + ']')


def get_action(model, mutate, prev_observation, length, env, param):
    if model is not None and not mutate:
        if param.first_layer_type == 'conv_2d':
            action = np.argmax(model.predict(prev_observation.reshape((-1, param.game_size[1], param.game_size[0], 1)))[0])
        else:
            action = np.argmax(model.predict(prev_observation.reshape((-1, length, 1)))[0])
    else:
        action = random.randint(0, env.action_space_size - 1)
    return action


def append_data(param, scores, accepted_scores, training_data, score, game_memory):
    scores.append(score)  # save overall scores
    if accepted_scores is not None:
        if score >= param.score_requirement:
            accepted_scores.append(score)
            training_data.extend(parse_game_memory(game_memory))
    return scores, accepted_scores, training_data


def run_x_games(render, param, model=None):
    scores, choices = [], []
    for _ in range(param.games_to_play):
        [_, score, _, choice] = run_a_game(param, model=model, render=render)
        scores.extend(score)
        choices.extend(choice)
    print_score_stats(scores, 2)
    count_and_print_choices(choices)
    return scores, choices


def parse_game_memory(game_memory):
    new_mem, output = [], []
    for data in game_memory:
        # convert to one-hot (this is the output layer for the neural network)
        if data[1] == 0:
            output = [1, 0, 0]
        elif data[1] == 1:
            output = [0, 1, 0]
        elif data[1] == 2:
            output = [0, 0, 1]
        new_mem.append([data[0], output])
    return new_mem


def save_training_data(param, training_data, training_driver):
    filename = training_driver.run_dir.replace('\\', '_') + '_training_data.npy'
    print('Saving training data to ' + filename + ' [' + str(len(training_data)) + ' samples]')
    training_data_save = np.array(training_data)  # just in case you wanted to reference later
    save_dir = os.path.join(os.getcwd(), param.model_dir, training_driver.run_dir, filename)
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


def count_and_print_choices(choices):
    # 0 = Straight, 1: Turn Left, 2: Turn Right
    print('Straight:{}%  Turn Left:{}%  Turn Right:{}%'.format(
        round(choices.count(0) / len(choices) * 100, 2),
        round(choices.count(1) / len(choices) * 100, 2),
        round(choices.count(2) / len(choices) * 100, 2)))


def save_inputs(param, training_driver, accepted_scores=None, test_run_scores=None):
    env = SnakeEnv(param.game_size, param.first_layer_type)
    filename = param.model_dir + '\\' + param.run_dir + '\\' + 'Summary' + '.txt'
    file = open(filename, "w")
    file.write('Model Inputs\n\n')
    file.write('Percentile                : ' + str(param.accepted_percentile) + '\n')
    file.write('Epochs                    : ' + str(param.epochs) + '\n')
    file.write('Learning Rate             : ' + str(param.LR) + '\n')
    file.write('Goal Steps                : ' + str(param.goal_steps) + '\n\n')
    if test_run_scores is not None:
        file.write('Number of test scores     : ' + str(len(test_run_scores)) + '\n')
        file.write('Max of test scores        : ' + str(max(test_run_scores)) + '\n')
        file.write('Average of test scores    : ' + str(round(mean(test_run_scores), 0)) + '\n')
        file.write('Min of tests score        : ' + str(min(test_run_scores)) + '\n\n')
        file.write('Score Requirement         : ' + str(param.score_requirement) + '\n\n')
        file.write('Initial_games             : ' + str(param.initial_games) + '\n')
    if accepted_scores is not None:
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
    file.write('Game Size                 : [' + str(param.game_size[0]) + ',' + str(param.game_size[1]) + ']\n\n')
    file.write('Network Data\n')
    file.write('Observation Space Length  : ' + str(training_driver.observation_space_length) + '\n')
    file.write('Include Reward in Obs     : ' + str(param.include_reward_in_obs) + '\n')
    file.write('Input Layer Nodes         : ' + str(training_driver.observation_space_length) + '\n')
    for i in range(0, len(training_driver.layer_types)):
        file.write('Layer ' + str(i+1) + ' Type              : ' + str(training_driver.layer_types[i]) + '\n')
        file.write('Layer ' + str(i+1) + ' Elements          : ' + str(training_driver.layer_num_of_elements[i]) + '\n')
        file.write('Layer ' + str(i+1) + ' Filter Size       : ' + str(training_driver.layer_filter_sizes[i]) + '\n')
        file.write('Layer ' + str(i+1) + ' Strides           : ' + str(training_driver.layer_strides[i]) + '\n')
        file.write('Layer ' + str(i+1) + ' Activation        : ' + str(training_driver.activations[i]) + '\n')
        file.write('Layer ' + str(i+1) + ' Dropout           : ' + str(training_driver.dropouts[i]) + '\n')
        file.write('Layer ' + str(i+1) + ' Dropout Keep Prob : ' + str(training_driver.keep_probability[i]) + '\n')
    file.write('Output Layer Nodes        : ' + str(training_driver.action_space_size) + '\n')
    file.write('Output Layer Activation   : ' + str(training_driver.output_layer_activation) + '\n')
    file.write('Regression Optimizer      : ' + str(training_driver.regression_optimizer) + '\n')
    file.write('Loss Function             : ' + str(training_driver.loss_function) + '\n')
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


def clean_dir(dir_to_search):
    folders = os.listdir(dir_to_search)
    for folder in folders:
        if os.path.isfile(dir_to_search + '\\' + folder):
            continue
        files = []
        for (dir_path, dir_names, file_names) in os.walk(dir_to_search + '\\' + folder):
            for file in file_names:
                files.append(file)
        if len(files) == 0:
            shutil.rmtree(os.path.join(dir_to_search, folder))


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()
