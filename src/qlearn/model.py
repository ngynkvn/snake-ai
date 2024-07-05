import json
import logging
from collections import deque
import random
from turtle import down, up, width

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from game.state import SnakeGame

def create_agent_state(state: SnakeGame.GameState):
    hx, hy = state.snake.head
    fx, fy = state.food

    width = state.width
    height = state.height
    right_kills = hx + 1 >= width or (hx + 1, hy) in state.snake.body
    left_kills = hx - 1 < 0 or (hx - 1, hy) in state.snake.body
    up_kills = hy - 1 < 0 or (hx, hy - 1) in state.snake.body
    down_kills = hy + 1 >= height or (hx, hy + 1) in state.snake.body

    return np.array([
        hx / state.width, hy / state.height, # head normalized position
        fx / state.width, fy / state.height, # food normalized position
        int(fx > hx), int(fx < hx), # food direction
        int(fy > hy), int(fy < hy),
        # danger detection
        int(right_kills), 
        int(left_kills), 
        int(up_kills), 
        int(down_kills),
    ])


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QLearningAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_size = action_size
        self.q_network = QNetwork(state_size, hidden_size, action_size)
        self.target_network = QNetwork(state_size, hidden_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=10000)
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        target_q_values = target_q_values.unsqueeze(1)

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()        
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def calc_reward(result: SnakeGame.Event | None, prev_state: SnakeGame.GameState, state: SnakeGame.GameState):
    if result is None:
        hx, hy = prev_state.snake.head
        fx, fy = prev_state.food
        prev_dist = abs((fx - hx) + (fy - hy))

        hx, hy = state.snake.head
        fx, fy = state.food
        curr_dist = abs((fx - hx) + (fy - hy))
        # Reward for moving towards food
        return 1 if curr_dist < prev_dist else -1
    elif result.type == "gameover":
        return -10
    elif result.type == "ate":
        return 1
    else: 
        logging.warning(f"Unknown event type: {result.type}")
        return 0

def train(episodes, batch_size = 32, target_update = 10):
    env = SnakeGame(width=50, height=50)
    agent = QLearningAgent(state_size=len(create_agent_state(env.state)), action_size=4)
    scores = []

    prog = tqdm(range(episodes))
    for episode in prog:
        state = env.reset()
        total_reward = 0
        best_score = 0

        while not env.gameover:
            prev_game_state = env.state
            state = create_agent_state(prev_game_state)
            action = agent.act(state)
            result = env.tick(action)
            next_state = create_agent_state(env.state)
            reward = calc_reward(result, prev_game_state, env.state)
            done = env.gameover
            agent.remember(state, action, reward, next_state, done)

            state = env.state
            total_reward += reward

            agent.replay(batch_size)
            best_score = max(env.score, best_score)

        if episode % target_update == 0:
            agent.update_target_network()
            prog.set_description(f'{np.mean(scores):0.2f}: {best_score}')
        scores.append(total_reward)
    return agent, scores


if __name__ == "__main__":
    # Run the training
    agent, scores = train(episodes=1000)
    model_state = agent.target_network.state_dict()

    print(f"Average score over last 100 episodes: {np.mean(scores[-100:]):.2f}")
    torch.save(model_state, "model.pth")