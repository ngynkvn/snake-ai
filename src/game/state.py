"""
Map pygame key codes to snake directions
"""
from dataclasses import dataclass
import logging
import random
import numpy as np
import pygame



UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
keymap = {
    pygame.K_UP: UP,
    pygame.K_DOWN: DOWN,
    pygame.K_LEFT: LEFT,
    pygame.K_RIGHT: RIGHT,
}
backwards = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}
delta = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}
class Snake:
    def __init__(self, x, y):
        self.body = [(x, y)]
        self.direction = keymap[pygame.K_RIGHT]

    @property
    def head(self):
        return self.body[0]
    
    def grow(self):
        self.body = [self.body[0]] + self.body

    def tick(self, direction: int | None) -> str | None:
        if direction is not None and direction != backwards[self.direction]:
            self.direction = direction
        head = self.body[0]
        dx, dy = delta[self.direction]
        head = (head[0] + dx, head[1] + dy)
        self.body = [head] + self.body[:-1]
        if head in self.body[1:]:
            return "gameover"
    
    def copy(self):
        s = Snake(0, 0)
        s.body = self.body.copy()
        s.direction = self.direction
        return s

class SnakeGame:
    def __init__(self, width = 50, height = 50):
        self.width = width
        self.height = height
        self.reset()

    @dataclass
    class GameState:
        snake: Snake
        food: tuple[int, int]
        score: int
        gameover: bool
        width: int
        height: int

    @dataclass
    class Event:
        type: str
    
    def reset(self) -> GameState:
        self.snake = Snake(self.width // 2, self.height // 2)
        self.score = 0
        self.food_pos = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        self.gameover = False
        return self.state
    
    def out_of_bounds(self, pos):
        return pos[0] < 0 \
            or pos[0] >= self.width \
            or pos[1] < 0 \
            or pos[1] >= self.height

    def place_food(self):
        pos = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        if pos not in self.snake.body:
            self.food_pos = pos
    

    def tick(self, key: int | None) -> Event | None:
        if self.gameover:
            return self.Event("gameover")
        prev_head = self.snake.head
        event = self.snake.tick(key)
        next_head = self.snake.head
        if event == "gameover":
            self.gameover = True
            return self.Event("gameover")
        if self.out_of_bounds(self.snake.head):
            self.gameover = True
            logging.info(f"Game over: Snake hit the wall {prev_head} -> {next_head}")
            logging.info(f"{agent_state_labels(create_agent_state(self.state))}")
            return self.Event("gameover")
        if self.snake.body[0] == self.food_pos:
            self.score += 1
            self.place_food()
            self.snake.grow()
            return self.Event("ate")
    
    
    @property
    def state(self) -> GameState:
        return self.GameState(
            snake=self.snake.copy(),
            food=self.food_pos,
            score=self.score,
            gameover=self.gameover,
            width=self.width,
            height=self.height,
        )
    
def create_agent_state(state: SnakeGame.GameState):
    hx, hy = state.snake.head
    fx, fy = state.food

    width = state.width
    height = state.height
    right_kills = hx + 1 >= width or (hx + 1, hy) in state.snake.body
    left_kills = hx - 1 <= 0 or (hx - 1, hy) in state.snake.body
    up_kills = hy - 1 < 0 or (hx, hy - 1) in state.snake.body
    down_kills = hy + 1 >= height or (hx, hy + 1) in state.snake.body

    return np.array([
        hx / state.width, hy / state.height, # head normalized position
        fx / state.width, fy / state.height, # food normalized position
        int(fx > hx), int(fx < hx), # food direction
        int(fy < hy), int(fy > hy),
        # danger detection
        int(right_kills), 
        int(left_kills), 
        int(up_kills), 
        int(down_kills),
    ])

def agent_state_labels(state: np.ndarray):
    return {
        "head_x": f"{state[0]:.2f}",
        "head_y": f"{state[1]:.2f}",
        "food_x": f"{state[2]:.2f}",
        "food_y": f"{state[3]:.2f}",
        "food_right": f"{state[4]:.2f}",
        "food_left": f"{state[5]:.2f}",
        "food_up": f"{state[6]:.2f}",
        "food_down": f"{state[7]:.2f}",
        "right_kills": f"{state[8]:.2f}",
        "left_kills": f"{state[9]:.2f}",
        "up_kills": f"{state[10]:.2f}",
        "down_kills": f"{state[11]:.2f}",
    }