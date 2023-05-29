import time
import pygame
import numpy as np
import random
from Snake import Snake
import MyTools

random.seed(time.time())


class HeadlessSnake:
    def __init__(self, game_size):
        self.snake = Snake()
        self.last_position = []
        self.last_position.append(self.snake.position)
        self.spacing = 10
        self.speed = 1
        self.window_x = game_size[0] * self.spacing  # 720
        self.window_y = game_size[1] * self.spacing  # 480
        self.game_size = [game_size[0], game_size[1]]
        self.fps = pygame.time.Clock()
        self.fruit_position = [
            random.randrange(1, (self.game_size[0] // 10)) * 10,
            random.randrange(1, (self.game_size[1] // 10)) * 10,
        ]
        self.score = 0
        self.score_per_fruit = 1
        self.matrix = np.zeros((self.window_x, self.window_y))
        self.matrix[:, 0] = 1
        self.matrix[:, self.window_y - 1] = 1
        self.matrix[0, :] = 1
        self.matrix[self.window_x - 1, :] = 1
        self.observation_array = None

    def spawn_fruit(self):
        while self.fruit_position in self.snake.body:
            self.fruit_position = [
                random.randint(1, self.game_size[0] - 2),
                random.randint(1, self.game_size[1] - 2),
            ]

    def check_yumyum_and_move_body(self):
        snake_body = self.snake.body
        snake_body.insert(0, list(self.snake.position))  # add a snake segment
        if (
            self.snake.position[0] == self.fruit_position[0]
            and self.snake.position[1] == self.fruit_position[1]
        ):  # if we got fruit
            self.score += self.score_per_fruit  # increase score
            self.spawn_fruit()  # spawn a new fruit
        else:  # if we didn't get a fruit, remove the last snake segment
            snake_body.pop()
        self.snake.body = snake_body

    def check_game_over(self):
        if (
            self.snake.position[0] <= 0
            or self.snake.position[0] >= self.game_size[0] - 1
        ):
            return True  # we collided with an edge in the y
        if (
            self.snake.position[1] <= 0
            or self.snake.position[1] >= self.game_size[1] - 1
        ):
            return True  # we collided with an edge in the x
        # Touching the snake body
        for block in self.snake.body[1:]:
            if (
                self.snake.position[0] == block[0]
                and self.snake.position[1] == block[1]
            ):
                self.snake.alive = False
                return True

    def change_position(self):  # move snake
        current_position = self.snake.position
        if self.snake.direction == "UP":
            self.snake.position = [
                current_position[0],
                current_position[1] - self.speed,
            ]
        elif self.snake.direction == "DOWN":
            self.snake.position = [
                current_position[0],
                current_position[1] + self.speed,
            ]
        elif self.snake.direction == "LEFT":
            self.snake.position = [
                current_position[0] - self.speed,
                current_position[1],
            ]
        elif self.snake.direction == "RIGHT":
            self.snake.position = [
                current_position[0] + self.speed,
                current_position[1],
            ]

    def change_direction(self, direction):
        if self.snake.direction == "UP":
            if direction == "TURN_LEFT":
                self.snake.direction = "LEFT"
            elif direction == "TURN_RIGHT":
                self.snake.direction = "RIGHT"
        elif self.snake.direction == "DOWN":
            if direction == "TURN_LEFT":
                self.snake.direction = "RIGHT"
            elif direction == "TURN_RIGHT":
                self.snake.direction = "LEFT"
        elif self.snake.direction == "LEFT":
            if direction == "TURN_LEFT":
                self.snake.direction = "DOWN"
            elif direction == "TURN_RIGHT":
                self.snake.direction = "UP"
        elif self.snake.direction == "RIGHT":
            if direction == "TURN_LEFT":
                self.snake.direction = "UP"
            elif direction == "TURN_RIGHT":
                self.snake.direction = "DOWN"

    def update_matrix(self):
        # 0 = empty, -1 = obstacle/body, 1 = food, 2 = player
        self.matrix = np.zeros((self.game_size[1], self.game_size[0]))
        self.matrix[:, 0] = -1
        self.matrix[:, self.game_size[0] - 1] = -1
        self.matrix[0, :] = -1
        self.matrix[self.game_size[1] - 1, :] = -1
        for part in self.snake.body:
            self.matrix[part[1], part[0]] = -1
        self.matrix[self.fruit_position[1], self.fruit_position[0]] = 1
        self.matrix[self.snake.position[1], self.snake.position[0]] = 2
        return self.matrix

    def get_array(self):  # define perception array to feed as observation
        self.matrix = self.update_matrix()
        self.observation_array = np.array([])
        whats_left, whats_right, whats_in_front = 0, 0, 0
        snake_y = self.snake.position[1]
        snake_x = self.snake.position[0]
        fruit_y = self.fruit_position[1]
        fruit_x = self.fruit_position[0]
        # get what is to the left, right, and in front of us
        if self.snake.direction == "UP":
            whats_left = self.matrix[snake_y, snake_x - 1]
            whats_right = self.matrix[snake_y, snake_x + 1]
            whats_in_front = self.matrix[snake_y - 1, snake_x]
        elif self.snake.direction == "DOWN":
            whats_left = self.matrix[snake_y, snake_x + 1]
            whats_right = self.matrix[snake_y, snake_x - 1]
            whats_in_front = self.matrix[snake_y + 1, snake_x]
        elif self.snake.direction == "LEFT":
            whats_left = self.matrix[snake_y + 1, snake_x]
            whats_right = self.matrix[snake_y - 1, snake_x]
            whats_in_front = self.matrix[snake_y, snake_x - 1]
        elif self.snake.direction == "RIGHT":
            whats_left = self.matrix[snake_y - 1, snake_x]
            whats_right = self.matrix[snake_y + 1, snake_x]
            whats_in_front = self.matrix[snake_y, snake_x + 1]

        normal_angle = MyTools.get_normalized_angle(
            snake_y, snake_x, fruit_y, fruit_x
        )  # calculate normalized angle to fruit
        # y_dist = (snake_y - fruit_y) / (self.game_size[1] - 2)
        # x_dist = (snake_x - fruit_x) / (self.game_size[0] - 2)

        # create observation array from perception values
        self.observation_array = np.append(self.observation_array, whats_left)
        self.observation_array = np.append(self.observation_array, whats_right)
        self.observation_array = np.append(self.observation_array, whats_in_front)
        self.observation_array = np.append(self.observation_array, normal_angle)
        # self.observation_array = np.append(self.observation_array, x_dist)
        # self.observation_array = np.append(self.observation_array, y_dist)
        return self.observation_array
