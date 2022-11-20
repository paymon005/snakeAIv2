from SnakeEnvironment import SnakeEnv
import time
import numpy as np
import random
from statistics import mean
from collections import Counter
from tqdm import tqdm
from datetime import datetime
from DeepNeuralNetworkDriver import DnnDriver
random.seed(time.time())
now = datetime.now()  # current date and time
run_dir = now.strftime("%Y_%m_%d___%H_%M_%S")

# 0:  Run one random case
# 1:  Train a new model
# 2:  Load the last model
# 3:  Load a specific model
run_mode = 2
profile_run = False

run_to_load = '2022_11_19___00_45_36'
model_to_resume = 'Snake_Model-6'

# game_size = [72, 48]
game_size = [40, 27]
snake_speed = 45
env = SnakeEnv(game_size)
score_check_runs = 10000
accepted_percentile = 5
initial_games = 100000
epochs = 1000
keep_rate = 0.8
LR = 5e-5
model_dir = 'Models'
model_name = 'Snake_Model'
kickout_sore = -500


def main():
    if run_mode == 0:
        run_random_agent()
        env.close()
    else:
        run_tflearn_trainer()


def run_tflearn_trainer():
    training_driver = DnnDriver(env.action_space_size, model_name, model_dir, run_dir, LR, epochs, keep_rate,
                                profile_run)
    if run_mode == 2 or run_mode == 3:  # load model
        training_driver.observation_space_size = game_size
        if run_mode == 3:
            training_driver.load_model(run_to_load, model_to_resume)
        else:
            training_driver.load_model()
        run_model(training_driver.model, 10)
        env.close()
    elif run_mode == 1:  # make model
        test_run_scores, score_requirement = get_score_requirement(score_check_runs)
        print('Setting ' + str(score_requirement) + ' as the score requirement')
        time.sleep(5)
        training_driver.training_data, accepted_scores = initial_population(initial_games, score_requirement)
        time.sleep(5)
        if len(training_driver.training_data) == 0:
            return
        training_driver.create_save_dir(model_dir, run_dir)
        save_inputs(accepted_scores, test_run_scores, score_requirement)
        training_driver.train_model()
        final_scores = run_model(training_driver.model, 10)
        env.close()
        save_outputs(final_scores)
        training_driver.plot_graphs()


def get_score_requirement(num_of_runs):
    _, accepted_scores = initial_population(num_of_runs)
    idx = int(accepted_percentile * len(accepted_scores) / 100)
    req = sorted(accepted_scores)[len(accepted_scores)-idx-1]
    return accepted_scores, req


def run_random_agent():
    env.reset()
    while True:
        action = env.action_space.sample()
        [obs, reward, done, info] = env.step(action)
        env.render()
        time.sleep(1/snake_speed)
        if done:
            break
    return obs, reward, done, info


def initial_population(num_of_runs=initial_games, score_requirement=-9e9):
    training_data = []  # [OBS, MOVES]
    scores = []  # all scores:
    accepted_scores = []  # just the scores that met our threshold:
    output = []
    for _ in tqdm(range(num_of_runs)):  # iterate through however many games we want:
        score = 0
        game_memory = []  # moves specifically from this environment:
        prev_observation = []  # previous observation that we saw
        while True:  # for each frame in goal steps
            # action = env.action_space.sample()
            action = random.randint(0, env.action_space_size-1)
            [observation, reward, done, []] = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])  # saving our training data
        env.reset()  # reset env to play again
        scores.append(score)  # save overall scores
    # training_data_save = np.array(training_data)  # just in case you wanted to reference later
    # np.save('saved.npy', training_data_save)
    # some stats here, to further illustrate the neural network magic!
    if len(accepted_scores) > 0:
        print('Number of accepted scores: ', len(accepted_scores))
        print('Max accepted score: ', max(accepted_scores))
        print('Average accepted score: ', round(mean(accepted_scores), 0))
        print('Min accepted score: ', min(accepted_scores))
        print(Counter(accepted_scores))
    else:
        raise Exception('Out of ' + str(initial_games) + ' runs, no one got a score higher than ' +
                        str(score_requirement) + '.\n' + 'Average score:', mean(scores))
    return training_data, accepted_scores


def run_model(model, num_of_runs):
    length = env.game_size[0] * env.game_size[1]
    scores = []
    choices = []
    for each_game in range(num_of_runs):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        while True:
            env.render()
            time.sleep(1/snake_speed)
            if len(prev_obs) == 0:
                action = random.randrange(0, env.action_space_size)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, length, 1))[0])
            choices.append(action)
            [new_observation, reward, done, []] = env.step(action)
            # print('Reward: ' + str(reward))
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            print('Score: ' + str(score))
            if done or score < kickout_sore:
                break
        scores.append(score)
    print('Number of final scores: ', len(scores))
    print('Max Score: ', max(scores))
    print('Average Score: ', round(sum(scores) / len(scores), 0))
    print('Min Score: ', min(scores))
    # 0 = Straight, 1: Turn Left, 2: Turn Right
    print('Straight:{}%  Turn Left:{}%  Turn Right:{}%'.format(
        round(choices.count(0) / len(choices) * 100, 2),
        round(choices.count(1) / len(choices) * 100, 2),
        round(choices.count(2) / len(choices) * 100, 2)))
    env.close()
    return scores


def save_inputs(accepted_scores, test_run_scores, score_requirement):
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


def save_outputs(final_scores):
    filename = model_dir + '\\' + run_dir + '\\' + 'Summary' + '.txt'
    file = open(filename, "a")
    file.write('Number of final scores    : ' + str(len(final_scores)) + '\n')
    file.write('Max Score                 : ' + str(max(final_scores)) + '\n')
    file.write('Average Score             : ' + str(round(mean(final_scores), 0)) + '\n')
    file.write('Min Score                 : ' + str(min(final_scores)) + '\n')
    file.close()


if __name__ == '__main__':
    main()
