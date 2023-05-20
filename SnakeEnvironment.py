import math
import gym
import pygame
from gym import spaces
from HeadlessSnake import HeadlessSnake
from ctypes import windll, Structure, c_long, byref


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, game_size, observation_type):
        self._alive_weight = 0
        self._score_weight = 100
        self._dead_weight = -100
        self._loop_weight = -10
        self._towards_weight = 1
        self._away_weight = 0
        self._last_score = 0
        self._window = None
        self._fps = None
        self._x = 0
        self._y = 0
        self._game_size = game_size
        self._observation_type = observation_type
        self._reward_range = (0, 200)
        self._action_space_size = 3
        self._controller = HeadlessSnake(game_size)
        if self._observation_type == 1:
            self._state = self._controller.update_matrix()
        elif self._observation_type == 2:
            self._state = self._controller.get_array()
        self._observation_space_length = self._state.size
        self._action_space = spaces.Discrete(self._action_space_size)
        # self.observation_space = spaces.Box(low=np.zeros(self.game_size),
        #                                    high=np.ones(self.game_size),
        #                                    dtype=np.int)   # 0 = up, 1 = down, 2 = left, 3 = right

    def step(self, action):
        done = False
        assert self._action_space.contains(action), "Invalid Action"
        self._controller.change_direction(self.get_action_meaning(action))
        self._controller.change_position()
        self._controller.check_yumyum_and_move_body()  # check if we ate
        if self._controller.check_game_over():
            done = True
        else:
            if self._observation_type == 1:
                self._state = self._controller.update_matrix()
            elif self._observation_type == 2:
                self._state = self._controller.get_array()
        rewards = self.calculate_reward()
        return self._state, rewards, done

    def reset(self, *, seed=None, return_info=False, options=None):
        del self._controller
        self._controller = HeadlessSnake(self.game_size)
        if self._observation_type == 1:
            self._state = self._controller.update_matrix().flatten()
        elif self._observation_type == 2:
            self._state = self._controller.get_array()

    def render(self, mode='human', close=False):
        if self._window is None:
            pygame.init()
            pygame.display.set_caption('Snake: ' + str(self._controller.score))
            self._window = pygame.display.set_mode((self._controller.window_x, self._controller.window_y))
            self._x = pygame.display.Info().current_w
            self._y = pygame.display.Info().current_h
        self.draw_game()
        on_top(pygame.display.get_wm_info()['window'])
        pygame.event.pump()

    def close(self):
        pygame.quit()
        self._window = None

    @staticmethod
    def get_action_meaning(action):
        if action == 0:
            return "STRAIGHT"
        elif action == 1:
            return "TURN_LEFT"
        elif action == 2:
            return "TURN_RIGHT"

    def calculate_reward(self):
        reward = 0
        score_diff = self._controller.score - self._last_score
        fruit_distance_old = self.calc_distance(self._controller.fruit_position, self._controller.last_position[0])
        fruit_distance_new = self.calc_distance(self._controller.fruit_position, self._controller.snake.position)
        fruit_distance_diff = fruit_distance_old - fruit_distance_new
        self_distance = []
        for pos in self._controller.last_position:
            self_distance.append(self.calc_distance(pos, self._controller.snake.position))

        reward += score_diff * self._score_weight

        if len(self._controller.last_position) == 10 and max(self_distance) < 3:
            reward += self._loop_weight

        if fruit_distance_diff > 0:
            reward += self._towards_weight
        else:
            reward += self._away_weight

        if self._controller.snake.alive:
            reward += self._alive_weight
        if not self._controller.snake.alive:
            reward += self._dead_weight

        self._last_score = self._controller.score
        self._controller.last_position.insert(0, self._controller.snake.position)
        if len(self._controller.last_position) > 10:
            self._controller.last_position.pop()
        return reward

    def draw_game(self):
        spacing = self._controller.spacing
        half_space = spacing / 2
        double_space = spacing * 2
        window_x = self._game_size[0] * spacing
        window_y = self._game_size[1] * spacing

        self._window.fill(pygame.Color(0, 0, 0))
        # snake
        for pos in self._controller.snake.body:
            pygame.draw.rect(self._window, pygame.Color(0, 255, 0),
                             pygame.Rect(pos[0] * 10, pos[1] * 10, 10, 10))
        # top bar
        pygame.draw.rect(self._window, pygame.Color(255, 255, 255), pygame.Rect(0, 0, window_x - half_space, spacing))
        pygame.draw.rect(self._window, pygame.Color(0, 0, 0), pygame.Rect(0, 0, window_x, half_space))
        # bottom bar
        pygame.draw.rect(self._window, pygame.Color(255, 255, 255),
                         pygame.Rect(0, window_y - spacing, window_x, spacing))
        pygame.draw.rect(self._window, pygame.Color(0, 0, 0),
                         pygame.Rect(0, window_y - half_space, window_x, half_space))
        # left bar
        pygame.draw.rect(self._window, pygame.Color(255, 255, 255), pygame.Rect(
            0, spacing, spacing, window_y - double_space))
        pygame.draw.rect(self._window, pygame.Color(0, 0, 0), pygame.Rect(
            0, 0, half_space, window_y))
        # right bar
        pygame.draw.rect(self._window, pygame.Color(255, 255, 255), pygame.Rect(
            window_x - spacing, spacing, spacing, window_y - double_space))
        pygame.draw.rect(self._window, pygame.Color(0, 0, 0), pygame.Rect(
            window_x - half_space, spacing, half_space, window_y))
        # fruit
        pygame.draw.rect(self._window, pygame.Color(255, 0, 0), pygame.Rect(
            self._controller.fruit_position[0] * spacing,
            self._controller.fruit_position[1] * spacing, spacing, spacing))
        pygame.display.update()

    @staticmethod
    def calc_distance(pt1, pt2):
        x2 = (pt1[0] - pt2[0]) ** 2
        y2 = (pt1[1] - pt2[1]) ** 2
        return math.sqrt(x2 + y2)

    @property
    def action_space_size(self):
        return self._action_space_size

    @action_space_size.setter
    def action_space_size(self, value):
        self._action_space_size = value

    @action_space_size.deleter
    def action_space_size(self):
        del self._action_space_size

    @property
    def game_size(self):
        return self._game_size

    @game_size.setter
    def game_size(self, value):
        self._game_size = value

    @game_size.deleter
    def game_size(self):
        del self._game_size

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @state.deleter
    def state(self):
        del self._state

    @property
    def observation_space_size(self):
        return self._observation_space_size

    @observation_space_size.setter
    def observation_space_size(self, value):
        self._observation_space_size = value

    @observation_space_size.deleter
    def observation_space_size(self):
        del self._observation_space_size

    @property
    def observation_space_length(self):
        return self._observation_space_length

    @observation_space_length.setter
    def observation_space_length(self, value):
        self._observation_space_length = value

    @observation_space_length.deleter
    def observation_space_length(self):
        del self._observation_space_length

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @action_space.deleter
    def action_space(self):
        del self._action_space

    @property
    def last_score(self):
        return self._last_score

    @last_score.setter
    def last_score(self, value):
        self._last_score = value

    @last_score.deleter
    def last_score(self):
        del self._last_score

    @property
    def alive_weight(self):
        return self._alive_weight

    @alive_weight.setter
    def alive_weight(self, value):
        self._alive_weight = value

    @alive_weight.deleter
    def alive_weight(self):
        del self._alive_weight

    @property
    def score_weight(self):
        return self._score_weight

    @score_weight.setter
    def score_weight(self, value):
        self._score_weight = value

    @score_weight.deleter
    def score_weight(self):
        del self._score_weight

    @property
    def dead_weight(self):
        return self._dead_weight

    @dead_weight.setter
    def dead_weight(self, value):
        self._dead_weight = value

    @dead_weight.deleter
    def dead_weight(self):
        del self._dead_weight

    @property
    def loop_weight(self):
        return self._loop_weight

    @loop_weight.setter
    def loop_weight(self, value):
        self._loop_weight = value

    @loop_weight.deleter
    def loop_weight(self):
        del self._loop_weight

    @property
    def towards_weight(self):
        return self._towards_weight

    @towards_weight.setter
    def towards_weight(self, value):
        self._towards_weight = value

    @towards_weight.deleter
    def towards_weight(self):
        del self._towards_weight

    @property
    def away_weight(self):
        return self._away_weight

    @away_weight.setter
    def away_weight(self, value):
        self._away_weight = value

    @away_weight.deleter
    def away_weight(self):
        del self._away_weight


class RECT(Structure):
    _fields_ = [
        ('left', c_long),
        ('top', c_long),
        ('right', c_long),
        ('bottom', c_long),
    ]

    def width(self): return self.right - self.left

    def height(self): return self.bottom - self.top


def on_top(window):
    SetWindowPos = windll.user32.SetWindowPos
    GetWindowRect = windll.user32.GetWindowRect
    rc = RECT()
    GetWindowRect(window, byref(rc))
    SetWindowPos(window, -1, rc.left, rc.top, 0, 0, 0x0001)
