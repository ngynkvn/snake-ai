"""
Simple snake game
"""

from dataclasses import dataclass
import logging
import pygame

import torch

from game.state import DOWN, LEFT, RIGHT, UP, SnakeGame, keymap
from qlearn.model import QNetwork, create_agent_state

logging.basicConfig(level=logging.INFO)

namemap = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}

def render(game: SnakeGame):
    for x, y in game.snake.body:
        pygame.draw.rect(screen, (255, 255, 255), (x*10, y*10, 10, 10))
    pygame.draw.rect(screen, (255, 0, 0), (game.food_pos[0]*10, game.food_pos[1]*10, 10, 10))
    score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
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

    logging.info(f"Starting game with width={game.width} and height={game.height}")

    # Pytorch model
    model = QNetwork(len(create_agent_state(game.state)), 64, 4)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        directionInput = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

            # if event.type == pygame.KEYDOWN and event.key in keymap:
            #     directionInput = keymap[event.key]
        with torch.no_grad():
            state = create_agent_state(game.state)
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state)
            directionInput = q_values.argmax().item()


        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # RENDER YOUR GAME HERE
        game.tick(directionInput)
        render(game)

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(12)  # limits FPS to 60

    pygame.quit()
