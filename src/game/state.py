"""
Map pygame key codes to snake directions
"""
from dataclasses import dataclass
import random
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
        return pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height

    def place_food(self):
        pos = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        if pos not in self.snake.body:
            self.food_pos = pos
    

    def tick(self, key: int | None) -> Event | None:
        if self.gameover:
            return self.Event("gameover")

        event = self.snake.tick(key)
        if event == "gameover":
            self.gameover = True
            return self.Event("gameover")
        if self.out_of_bounds(self.snake.head):
            self.gameover = True
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
    