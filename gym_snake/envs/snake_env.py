import gym
from gym import spaces

import sys
from io import StringIO
from contextlib import closing

import numpy as np
from random import randint, choice
from collections import deque
from itertools import islice


def randlist(y, x):
    return [randint(0, y - 1), randint(0, x - 1)]

def addFood(snake, height, width):
    food = randlist(height, width)
    while food in snake:
        food = randlist(height, width)
    return food

def updateHead(snake, direction, height, width):
    row, col = snake[0]

    # 0: right, 1: up, 2: left, 3: down
    if direction == 0:
        snake.appendleft([row + 1, col])
    elif direction == 1:
        snake.appendleft([row, col - 1])
    elif direction == 2:
        snake.appendleft([row - 1, col])
    else:
        snake.appendleft([row, col + 1])
    
    row, col = snake[0]
    
    if row == -1:
        snake[0][0] = height - 1
    elif col == -1:
        snake[0][1] = width - 1
    elif row == height:
        snake[0][0] = 0
    elif col == width:
        snake[0][1] = 0
    
    return snake[0]

def addBorder(m, height, width):
    vertical = np.full((height, 1), '|')
    
    horizontal = np.full((1, width + 2), '-')
    horizontal[0, 0] = horizontal[0, width + 1] = '+'
    
    m = np.concatenate((vertical, m, vertical), axis=1)
    m = np.concatenate((horizontal, m, horizontal))
    
    return m


class SnakeEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, height=None, width=None):
        if height == None:
            height = randint(3, 30)
        if width == None:
            width = randint(3, 30)

        self.action_space = spaces.Discrete(3)      # 0: turn right, 1: maintain direction, 2: turn left
        self.observation_space = spaces.Tuple((
            spaces.Discrete(4),                     # direction of head
            spaces.MultiDiscrete([height, width]),  # position of head
            spaces.MultiDiscrete([height, width]),  # position of food
            spaces.MultiBinary(height * width)))    # position of body
        
        self.height = height
        self.width = width
        self.reset()
    
    def step(self, action):
        self.direction = (self.direction + action - 1) % 4

        reward = -1
        done = False
        
        head = updateHead(self.snake, self.direction, self.height, self.width)

        if head == self.food:
            self.length += 1
            reward += 10
            self.food = addFood(self.snake, self.height, self.width)
        else:
            tail_row, tail_col = self.snake.pop()
            self.map[tail_row, tail_col] = 0

        if head in islice(self.snake, 1, self.length):
            reward -= 10 * (self.length - 1)
            done = True
        
        self.map[head[0], head[1]] = 1
        
        return self._obs(), reward, done, {}
    
    def _obs(self):
        return (self.direction,
            np.array([self.snake[0][0], self.snake[0][1]]),
            np.array([self.food[0], self.food[1]]),
            self.map)
    
    def reset(self):
        self.snake = deque([randlist(self.height, self.width)])
        self.length = 1
        self.food = addFood(self.snake, self.height, self.width)
        self.direction = self.observation_space[0].sample()
        self.map = np.zeros([self.height, self.width], dtype=np.int8)
        self.map[self.snake[0][0], self.snake[0][1]] = 1
        return self._obs()
    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        m = np.where(self.map == 0, ' ', 'o')
        m[self.food[0], self.food[1]] = '*'
        
        m = addBorder(m, self.height, self.width)

        outfile.write("\n".join(''.join(line) for line in m)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()