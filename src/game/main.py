"""
Simple snake game
"""

import logging
from typing import Optional
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def render(game: SnakeGame):
    for x, y in game.snake.body:
        pygame.draw.rect(screen, (255, 255, 255), (x*10, y*10, 10, 10))
    pygame.draw.rect(screen, (255, 0, 0), (game.food_pos[0]*10, game.food_pos[1]*10, 10, 10))
    score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

class Player:
    def play(self, state, keyEvent):
        raise NotImplementedError
    def render_debug(self):
        raise NotImplementedError

class HumanPlayer(Player):
    def play(self, state, keyEvent) -> Optional[int]:
        if keyEvent == pygame.K_UP:
            return UP
        elif keyEvent == pygame.K_DOWN:
            return DOWN
        elif keyEvent == pygame.K_LEFT:
            return LEFT
        elif keyEvent == pygame.K_RIGHT:
            return RIGHT
        else:
            return None
    def render_debug(self):
        pass
class AIPlayer(Player):
    def __init__(self, input_size) -> None:
        super().__init__()
        # Pytorch model
        # TODO: make this path configurable
        state_dict = torch.load("model.pth")
        model = QNetwork(input_size, 128, 4)
        model.load_state_dict(state_dict)
        self.model = model.to(device=device).eval()
        self.agent_state = []
        self.nodes = []
    def play(self, state, keyEvent) -> Optional[int]:
        with torch.no_grad():
            agent_state = create_agent_state(game.state)
            state = torch.FloatTensor(agent_state).unsqueeze(0).to(device=device)
            q_values: torch.Tensor = self.model(state)
            directionInput = int(q_values.argmax().item())
            self.agent_state = agent_state
            self.nodes = (q_values / q_values.sum()).squeeze().cpu().numpy()
        return directionInput
    def render_debug(self):
        for i, n in enumerate(self.agent_state):
            pygame.draw.rect(screen, (0, max(min(255, 255*n), 0), 0), (i*12, 80, 10, 10))
        for i, n in enumerate(self.nodes):
            pygame.draw.rect(screen, (0, max(min(255, 255*n), 0), 0), (i*12, 100, 10, 10))

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

    # player = HumanPlayer()
    player = AIPlayer(len(create_agent_state(game.state)))


    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        keyEvent = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap:
                keyEvent = event.key
        
        playerInput = player.play(game.state, keyEvent)


        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # RENDER YOUR GAME HERE
        prev_state = game.state
        result = game.tick(playerInput)
        render(game)
        player.render_debug()

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(24)  # limits FPS to 60

    pygame.quit()
