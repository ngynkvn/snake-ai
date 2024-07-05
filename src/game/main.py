# Example file showing a basic pygame "game loop"
import logging
import pygame
import random

logging.basicConfig(level=logging.INFO)

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
    
    def turn(self, direction):
        self.direction = direction
    
    def grow(self):
        self.body = [self.body[0]] + self.body

    @property
    def head(self):
        return self.body[0]

    def tick(self) -> str | None:
        head = self.body[0]
        dx, dy = delta[self.direction]
        head = (head[0] + dx, head[1] + dy)
        self.body = [head] + self.body[:-1]
        if head in self.body[1:]:
            return "gameover"

class SnakeGame:
    def __init__(self, width = 50, height = 50):
        self.width = width
        self.height = height
        self.snake = Snake(width // 2, height // 2)
        self.food_pos = random.randint(0, width - 1), random.randint(0, height - 1)
        self.surface = pygame.Surface((width, height))
        self.gameover = False
    
    def out_of_bounds(self, pos):
        return pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height

    def tick(self, key):
        if self.gameover:
            return
        if key and keymap[key] != backwards[self.snake.direction]:
            self.snake.turn(keymap[key])
        
        event = self.snake.tick()
        if event == "gameover":
            self.gameover = True
        if self.out_of_bounds(self.snake.head):
            self.gameover = True
        if self.snake.body[0] == self.food_pos:
            self.food_pos = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.snake.grow()
    
    @property
    def score(self):
        return len(self.snake.body) - self.gameover*9999
    
    def data(self):
        return {
            "snake": self.snake.body,
            "score": self.score,
            "food": self.food_pos,
            "gameover": self.gameover,
        }
    
    def render(self):
        for x, y in self.snake.body:
            pygame.draw.rect(screen, (255, 255, 255), (x*10, y*10, 10, 10))
        pygame.draw.rect(screen, (255, 0, 0), (self.food_pos[0]*10, self.food_pos[1]*10, 10, 10))
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

if __name__ == "__main__":
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    clock = pygame.time.Clock()
    running = True

    game = SnakeGame()
    pygame.font.init()
    font = pygame.font.SysFont(pygame.font.get_default_font(), 50)

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        directionInput = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

            if event.type == pygame.KEYDOWN and event.key in keymap:
                directionInput = event.key


        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # RENDER YOUR GAME HERE
        game.tick(directionInput)
        game.render()


        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(12)  # limits FPS to 60

    pygame.quit()
