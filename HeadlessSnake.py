import time
import pygame
import numpy as np
import random
from Snake import Snake
import MyTools
random.seed(time.time())


class HeadlessSnake:

    def __init__(self, game_size):
        self._snake = Snake()
        self._last_position = []
        self._last_position.append(self._snake.position)
        self._spacing = 10
        self._speed = 1
        self._window_x = game_size[0] * self._spacing  # 720
        self._window_y = game_size[1] * self._spacing  # 480
        self._game_size = [game_size[0], game_size[1]]
        self._fps = pygame.time.Clock()
        self._fruit_position = [random.randrange(1, (self._game_size[0] // 10)) * 10,
                                random.randrange(1, (self._game_size[1] // 10)) * 10]
        self._score = 0
        self._score_per_fruit = 1
        self._matrix = np.zeros((self._window_x, self._window_y))
        self._matrix[:, 0] = 1
        self._matrix[:, self._window_y - 1] = 1
        self._matrix[0, :] = 1
        self._matrix[self._window_x - 1, :] = 1
        self._observation_array = None

    def spawn_fruit(self):
        while self._fruit_position in self._snake.body or self._fruit_position == self._snake.position:
            self._fruit_position = [random.randrange(1, (self._game_size[0] // 10)) * 10 - 1,
                                    random.randrange(1, (self._game_size[1] // 10)) * 10 - 1]

    def check_yumyum_and_move_body(self):
        snake_body = self._snake.body
        snake_body.insert(0, list(self._snake.position))
        if self._snake.position[0] == self._fruit_position[0] and self._snake.position[1] == \
                self._fruit_position[1]:
            self._score += self._score_per_fruit
            self.spawn_fruit()
        else:
            snake_body.pop()
        self._snake.body = snake_body

    def check_game_over(self):
        if self._snake.position[0] <= 0 or self._snake.position[0] >= self._game_size[0] - 1:
            return True
        if self._snake.position[1] <= 0 or self._snake.position[1] >= self._game_size[1] - 1:
            return True
        # Touching the snake body
        for block in self._snake.body[1:]:
            if self._snake.position[0] == block[0] and self._snake.position[1] == block[1]:
                self._snake.alive = False
                return True

    def change_position(self):
        current_position = self._snake.position
        if self._snake.direction == 'UP':
            self._snake.position = [current_position[0], current_position[1] - self._speed]
        elif self._snake.direction == 'DOWN':
            self._snake.position = [current_position[0], current_position[1] + self._speed]
        elif self._snake.direction == 'LEFT':
            self._snake.position = [current_position[0] - self._speed, current_position[1]]
        elif self._snake.direction == 'RIGHT':
            self._snake.position = [current_position[0] + self._speed, current_position[1]]

    def change_direction(self, direction):
        if self._snake.direction == 'UP':
            if direction == 'TURN_LEFT':
                self._snake.direction = 'LEFT'
            elif direction == 'TURN_RIGHT':
                self._snake.direction = 'RIGHT'
        elif self._snake.direction == 'DOWN':
            if direction == 'TURN_LEFT':
                self._snake.direction = 'RIGHT'
            elif direction == 'TURN_RIGHT':
                self._snake.direction = 'LEFT'
        elif self._snake.direction == 'LEFT':
            if direction == 'TURN_LEFT':
                self._snake.direction = 'DOWN'
            elif direction == 'TURN_RIGHT':
                self._snake.direction = 'UP'
        elif self._snake.direction == 'RIGHT':
            if direction == 'TURN_LEFT':
                self._snake.direction = 'UP'
            elif direction == 'TURN_RIGHT':
                self._snake.direction = 'DOWN'

    def update_matrix(self):
        # 0 = empty, -1 = obstacle/body, 2 = player, 1 = food
        self._matrix = np.zeros((self._game_size[1], self._game_size[0]))
        self._matrix[:, 0] = -1
        self._matrix[:, self._game_size[0] - 1] = -1
        self._matrix[0, :] = -1
        self._matrix[self._game_size[1] - 1, :] = -1
        for part in self._snake.body:
            self._matrix[part[1], part[0]] = -1
        self._matrix[self.fruit_position[1], self.fruit_position[0]] = 1
        self._matrix[self._snake.position[1], self._snake.position[0]] = 2
        return self._matrix

    def get_array(self):
        self._matrix = self.update_matrix()
        self._observation_array = np.array([])
        whats_left = 0
        whats_right = 0
        whats_in_front = 0
        snake_y = self._snake.position[1]
        snake_x = self._snake.position[0]
        fruit_y = self._fruit_position[1]
        fruit_x = self._fruit_position[0]
        if self._snake.direction == 'UP':
            whats_left = self._matrix[snake_y, snake_x-1]
            whats_right = self._matrix[snake_y, snake_x+1]
            whats_in_front = self._matrix[snake_y-1, snake_x]
        elif self._snake.direction == 'DOWN':
            whats_left = self._matrix[snake_y, snake_x+1]
            whats_right = self._matrix[snake_y, snake_x-1]
            whats_in_front = self._matrix[snake_y+1, snake_x]
        elif self._snake.direction == 'LEFT':
            whats_left = self._matrix[snake_y+1, snake_x]
            whats_right = self._matrix[snake_y-1, snake_x]
            whats_in_front = self._matrix[snake_y, snake_x-1]
        elif self._snake.direction == 'RIGHT':
            whats_left = self._matrix[snake_y-1, snake_x]
            whats_right = self._matrix[snake_y+1, snake_x]
            whats_in_front = self._matrix[snake_y, snake_x+1]

        normal_angle = MyTools.get_normalized_angle(snake_y, snake_x, fruit_y, fruit_x)
        y_dist = (snake_y - fruit_y) / (self._game_size[1]-2)
        x_dist = (snake_x - fruit_x) / (self._game_size[0]-2)

        self._observation_array = np.append(self._observation_array, whats_left)
        self._observation_array = np.append(self._observation_array, whats_right)
        self._observation_array = np.append(self._observation_array, whats_in_front)
        self._observation_array = np.append(self._observation_array, normal_angle)
        self._observation_array = np.append(self._observation_array, x_dist)
        self._observation_array = np.append(self._observation_array, y_dist)
        return self._observation_array

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @score.deleter
    def score(self):
        del self._score

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, value):
        self._spacing = value

    @spacing.deleter
    def spacing(self):
        del self._spacing

    @property
    def fruit_position(self):
        return self._fruit_position

    @fruit_position.setter
    def fruit_position(self, value):
        self._fruit_position = value

    @fruit_position.deleter
    def fruit_position(self):
        del self._fruit_position

    @property
    def window_x(self):
        return self._window_x

    @window_x.setter
    def window_x(self, value):
        self._window_x = value

    @window_x.deleter
    def window_x(self):
        del self._window_x

    @property
    def window_y(self):
        return self._window_y

    @window_y.setter
    def window_y(self, value):
        self._window_y = value

    @window_y.deleter
    def window_y(self):
        del self._window_y

    @property
    def snake(self):
        return self._snake

    @snake.setter
    def snake(self, value):
        self._snake = value

    @snake.deleter
    def snake(self):
        del self._snake

    @property
    def last_position(self):
        return self._last_position

    @last_position.setter
    def last_position(self, value):
        self._last_position = value

    @last_position.deleter
    def last_position(self):
        del self._last_position
