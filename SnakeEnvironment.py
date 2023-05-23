import gym
import pygame
from gym import spaces
from HeadlessSnake import HeadlessSnake
import MyTools
import ctypes
from ctypes import wintypes, POINTER, WINFUNCTYPE, windll
from ctypes.wintypes import BOOL, HWND, RECT


class SnakeEnv(gym.Env):

    def __init__(self, game_size, first_layer_type):
        self.alive_weight = 0
        self.score_weight = 1
        self.dead_weight = -1
        self.loop_weight = 0
        self.towards_weight = 0
        self.away_weight = 0
        self.last_score = 0
        self.choices_to_check_for_looping = 10
        self.window = None
        self.fps = None
        self.x = 0
        self.y = 0
        self.game_size = game_size
        self.first_layer_type = first_layer_type
        self.reward_range = (0, 200)
        self.action_space_size = 3
        self.controller = HeadlessSnake(game_size)
        self.prototype = WINFUNCTYPE(BOOL, HWND, POINTER(RECT))
        self.paramflags = (1, "hwnd"), (2, "lprect")
        self.GetWindowRect = self.prototype(("GetWindowRect", windll.user32), self.paramflags)
        self.user32 = ctypes.WinDLL("user32")
        self.user32.SetWindowPos.restype = wintypes.HWND
        self.user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, wintypes.INT, wintypes.INT, wintypes.INT,
                                             wintypes.INT, wintypes.UINT]
        if self.first_layer_type == 'conv_2d':
            self.state = self.controller.update_matrix()
        else:
            self.state = self.controller.get_array()
        self.observation_space_length = self.state.size
        self.action_space = spaces.Discrete(self.action_space_size)
        # self.observation_space = spaces.Box(low=np.zeros(self.game_size),
        #                                    high=np.ones(self.game_size),
        #                                    dtype=np.int)   # 0 = up, 1 = down, 2 = left, 3 = right

    def step(self, action, choices=None):
        done = False
        assert self.action_space.contains(action), "Invalid Action"
        self.controller.change_direction(self.get_action_meaning(action))
        self.controller.change_position()
        self.controller.check_yumyum_and_move_body()  # check if we ate
        if self.controller.check_game_over():
            done = True
        else:
            if self.first_layer_type == 'conv_2d':
                self.state = self.controller.update_matrix()
            else:
                self.state = self.controller.get_array()
        rewards = self.calculate_reward(choices)
        return self.state, rewards, done

    def reset(self, *, seed=None, return_info=False, options=None):
        del self.controller
        self.controller = HeadlessSnake(self.game_size)
        if self.first_layer_type == 'conv_2d':
            self.state = self.controller.update_matrix().flatten()
        else:
            self.state = self.controller.get_array()

    def render(self, score=''):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.controller.window_x, self.controller.window_y))
            self.x = pygame.display.Info().current_w
            self.y = pygame.display.Info().current_h
            # MyTools.on_top(pygame.display.get_wm_info()['window'])
        hwnd = pygame.display.get_wm_info()['window']
        rect = self.GetWindowRect(hwnd)
        self.user32.SetWindowPos(hwnd, -1, rect.left, rect.top, 0, 0, 0x0001)
        pygame.display.set_caption('Snake [Score: ' + str(score) + ']')
        self.draw_game()
        # MyTools.on_top(pygame.display.get_wm_info()['window'])
        pygame.event.pump()

    def close(self):
        pygame.quit()
        self.window = None

    @staticmethod
    def get_action_meaning(action):
        if action == 0:
            return "STRAIGHT"
        elif action == 1:
            return "TURN_LEFT"
        elif action == 2:
            return "TURN_RIGHT"

    def calculate_reward(self, choices):
        reward = 0
        score_diff = self.controller.score - self.last_score
        fruit_distance_old = MyTools.calc_distance(self.controller.fruit_position, self.controller.last_position[0])
        fruit_distance_new = MyTools.calc_distance(self.controller.fruit_position, self.controller.snake.position)
        fruit_distance_diff = fruit_distance_old - fruit_distance_new

        reward += score_diff * self.score_weight

        if choices is not None:
            looping = False
            choices = list(filter(MyTools.is_not_zero, choices))
            if len(choices) > self.choices_to_check_for_looping:
                looping = True
                choice1 = choices[-self.choices_to_check_for_looping]
                for choice in choices[-self.choices_to_check_for_looping + 1:]:
                    if choice != choice1:
                        looping = False
                        break
            if looping:
                reward += self.loop_weight
        else:
            self_distance = []
            for pos in self.controller.last_position:
                self_distance.append(MyTools.calc_distance(pos, self.controller.snake.position))
            if len(self.controller.last_position) == 10 and max(self_distance) < 2:
                reward += self.loop_weight

        if fruit_distance_diff > 0:
            reward += self.towards_weight
        else:
            reward += self.away_weight

        if self.controller.snake.alive:
            reward += self.alive_weight
        if not self.controller.snake.alive:
            reward += self.dead_weight

        self.last_score = self.controller.score
        self.controller.last_position.insert(0, self.controller.snake.position)
        if len(self.controller.last_position) > 10:
            self.controller.last_position.pop()
        return reward

    def draw_game(self):
        spacing = self.controller.spacing
        half_space = spacing / 2
        double_space = spacing * 2
        window_x = self.game_size[0] * spacing
        window_y = self.game_size[1] * spacing

        self.window.fill(pygame.Color(0, 0, 0))
        # snake
        for pos in self.controller.snake.body:
            pygame.draw.rect(self.window, pygame.Color(0, 255, 0),
                             pygame.Rect(pos[0] * 10, pos[1] * 10, 10, 10))
        # top bar
        pygame.draw.rect(self.window, pygame.Color(255, 255, 255), pygame.Rect(0, 0, window_x - half_space, spacing))
        pygame.draw.rect(self.window, pygame.Color(0, 0, 0), pygame.Rect(0, 0, window_x, half_space))
        # bottom bar
        pygame.draw.rect(self.window, pygame.Color(255, 255, 255),
                         pygame.Rect(0, window_y - spacing, window_x, spacing))
        pygame.draw.rect(self.window, pygame.Color(0, 0, 0),
                         pygame.Rect(0, window_y - half_space, window_x, half_space))
        # left bar
        pygame.draw.rect(self.window, pygame.Color(255, 255, 255), pygame.Rect(
            0, spacing, spacing, window_y - double_space))
        pygame.draw.rect(self.window, pygame.Color(0, 0, 0), pygame.Rect(
            0, 0, half_space, window_y))
        # right bar
        pygame.draw.rect(self.window, pygame.Color(255, 255, 255), pygame.Rect(
            window_x - spacing, spacing, spacing, window_y - double_space))
        pygame.draw.rect(self.window, pygame.Color(0, 0, 0), pygame.Rect(
            window_x - half_space, spacing, half_space, window_y))
        # fruit
        pygame.draw.rect(self.window, pygame.Color(255, 0, 0), pygame.Rect(
            self.controller.fruit_position[0] * spacing,
            self.controller.fruit_position[1] * spacing, spacing, spacing))
        pygame.display.update()
