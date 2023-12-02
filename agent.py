import torch
import random
import numpy as np
from game import Game
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.game = Game()
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(16, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return np.array([[cell.value if cell is not None else 0 for cell in row] for row in game.board.grid]).flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 150 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    highscore = 0
    score_sum = 0
    agent = Agent()
    game = Game()

    while True:
        game.update()
        state_old = agent.get_state(game)
        game.update()
        final_move = agent.get_action(state_old)
        game.update()

        reward, done, score = game.play_step(final_move)
        game.update()
        state_new = agent.get_state(game)
        game.update()

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        game.update()
        agent.remember(state_old, final_move, reward, state_new, done)
        game.update()

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            score_sum += score

            if score > highscore:
                highscore = score
                agent.model.save()

            print('Game:', agent.n_games, '\tScore:', score, '\tHighscore:', highscore, '\tAverage:', round(score_sum / agent.n_games))


if __name__ == '__main__':
    train()
