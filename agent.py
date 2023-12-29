import math
import random
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque
import logging
from game import Game
from model import DQN

import torch
import torch.nn as nn
import torch.optim as optim


def mypause(interval): # to prevent stealing window focus by matplotlib
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, source_path = None, dest_path = None, batch_size = 64, gamma = 0.99, eps_start = 0.95, eps_end = 0.05, eps_decay = 500, tau = 0.005, lr = 1e-3, memory_capacity = 1000, plotting = True, logs = False):
        """
        Agent class for training the DQN model. It can be also used for testing the trained model.
        If you want to test the trained model, you can ignore all the parameters and use the default values.

        Args:
            source_path (str, optional): Import model path. Defaults to None.
            dest_path (str, optional): Export model path. Defaults to None.
            batch_size (int, optional): number of transitions sampled from the replay buffer. Defaults to 64.
            gamma (float, optional): discount factor as mentioned in the previous section. Defaults to 0.99.
            eps_start (float, optional): starting value of epsilon. Defaults to 0.95.
            eps_end (float, optional): final value of epsilon. Defaults to 0.05.
            eps_decay (int, optional): controls the rate of exponential decay of epsilon, higher means a slower decay. Defaults to 500.
            tau (float, optional): update rate of the target network. Defaults to 0.005.
            lr (float, optional): learning rate of the ``AdamW`` optimizer. Defaults to 1e-3.
            memory_capacity (int, optional): capacity of the replay buffer. Defaults to 1000.
            plotting (bool, optional): whether to plot the training result. Defaults to True.
            logs (bool, optional): whether to save the training logs in ``training.log``. Defaults to False.

        Returns:
            Agent: an Agent object
        """

        self.source_path = source_path
        self.dest_path = dest_path
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.memory_capacity = memory_capacity
        self.plotting = plotting
        self.logs = logs
        self.env = Game()

        if(self.logs):
            logging.basicConfig(filename='training.log', level=logging.INFO, format='[%(asctime)s] - %(message)s')

        n_actions = 4  # up, down, left, right
        state = self.env.reset()
        n_observations = len(state)

        self.policy_net = DQN(n_observations, n_actions).to(device)
        if(self.source_path is not None):
            self.policy_net.load_state_dict(torch.load(self.source_path))

        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(memory_capacity)

        if(self.plotting):
            matplotlib.use("TkAgg")
            plt.ion()
        self.steps_done = 0

        self.episode_scores = []


    def plot_scores(self, show_result=False):
        plt.figure(1)
        scores_t = torch.tensor(self.episode_scores, dtype=torch.float)

        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(scores_t.numpy())

        # Take 100 episode averages and plot them too
        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        mypause(0.001)  # pause a bit so that plots are updated


    def select_action(self, state, train=True):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        if(not train):
            eps_threshold = self.eps_end
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            choose = random.randint(0, 3)
            return torch.tensor([[choose]], device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    @staticmethod
    def onehot_encode(state):
        output = []
        for i in range(4):
            for j in range(4):
                output.append([1 if state[i][j] == k else 0 for k in range(1, 18)])
        return torch.tensor(output, dtype=torch.float32).flatten()

    @staticmethod
    def onehot_decode(state):
        output = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(0, 17):
                    if (state[i*17*4 + j*17 + k] == 1):
                        output[i][j] = k+1
                        break
        return torch.tensor(output, dtype=torch.float32)


    def train(self, num_episodes = 100000):
        logging.info(f'Start training for {num_episodes} episodes')
        if(self.plotting):
            self.plot_scores()

        for i in range(num_episodes):
            state = self.env.reset()
            state = state.to(dtype=torch.float32).unsqueeze(0)
            prev = (0, 0)

            while (1):
                self.env.update()
                action = self.select_action(state)

                if(prev[0] == action and prev[1] == -15): # if the previous action is invalid, choose another action
                    choose = random.randint(0, 3)
                    action =  torch.tensor([[choose]], device=device, dtype=torch.long)
                observation, reward, terminated, truncated = self.env.step(action)
                prev = (action, reward)
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = observation.clone().detach().to(dtype=torch.float32).unsqueeze(0)
                self.memory.push(state, action, next_state, reward)

                # Augmentation
                state = state.squeeze(0)
                state = self.onehot_decode(state)
                observation = self.onehot_decode(observation)
                action = action.unsqueeze(0)

                # Augmentation by flipping horizontally
                state = torch.flip(state, [1])
                observation = torch.flip(observation, [1])
                if(action % 2 == 1):
                    action = (action + 2) % 4

                new_state = self.onehot_encode(state).clone().detach().to(dtype=torch.float32).unsqueeze(0)
                new_next_state = self.onehot_encode(observation).clone().detach().to(dtype=torch.float32).unsqueeze(0)
                new_action = torch.tensor([[action]], device=device, dtype=torch.long)
                self.memory.push(new_state, new_action, new_next_state, reward)
                if(action % 2 == 1):
                    action = (action + 2) % 4

                # Augmentation by flipping vertically
                state = torch.flip(state, [0, 1])
                observation = torch.flip(observation, [0, 1])
                if(action % 2 == 0):
                    action = (action + 2) % 4
                new_state = self.onehot_encode(state).clone().detach().to(dtype=torch.float32).unsqueeze(0)
                new_next_state = self.onehot_encode(observation).clone().detach().to(dtype=torch.float32).unsqueeze(0)
                new_action = torch.tensor([[action]], device=device, dtype=torch.long)
                self.memory.push(new_state, new_action, new_next_state, reward)
                if(action % 2 == 0):
                    action = (action + 2) % 4

                # back to original state
                state = torch.flip(state, [0])
                observation = torch.flip(observation, [0])


                # Augmentation by rotation
                for _ in range(3):
                    state = torch.rot90(state)
                    observation = torch.rot90(observation)
                    action = (action - 1) % 4
                    new_state = self.onehot_encode(state).clone().detach().to(dtype=torch.float32).unsqueeze(0)
                    new_next_state = self.onehot_encode(observation).clone().detach().to(dtype=torch.float32).unsqueeze(0)
                    new_action = torch.tensor([[action]], device=device, dtype=torch.long)
                    self.memory.push(new_state, new_action, new_next_state, reward)


                state = next_state
                self.optimize_model()
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                if done:
                    if(self.plotting):
                        self.plot_scores()
                    self.episode_scores.append(self.env.current_score())
                    break

            if (i != 0 and i % 50 == 0 and self.dest_path is not None):
                if(self.dest_path is not None):
                    torch.save(self.policy_net.state_dict(), self.dest_path)
                if(self.logs):
                    avg = sum(self.episode_scores[-50:]) / 50
                    logging.info(f'Episode {i + 1}/{num_episodes} - Average score: {avg}')

        print('Complete')
        if(self.plotting):
            self.plot_scores(show_result=True)
            plt.ioff()
            plt.show()

    def play(self, num_episodes=50):
        if(self.plotting):
            self.plot_scores()
        for _ in range(num_episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            while (1):
                self.env.update()
                action = self.select_action(state, False)
                _, _, terminated, truncated = self.env.step(action)
                done = terminated or truncated

                if done:
                    self.episode_scores.append(self.env.current_score())
                    if(self.plotting):
                        self.plot_scores()
                    break

        print('Complete')
        if(self.plotting):
            self.plot_scores(show_result=True)
            plt.ioff()
            plt.show()


    def random_play(self, num_episodes=50):
        if(self.plotting):
            self.plot_scores()
        for _ in range(num_episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            while (1):
                self.env.update()
                action = random.randint(0, 3)
                _, _, terminated, truncated = self.env.step(action)
                done = terminated or truncated

                if done:
                    self.episode_scores.append(self.env.current_score())
                    if(self.plotting):
                        self.plot_scores()
                    break

        print('Complete')
        if(self.plotting):
            self.plot_scores(show_result=True)
            plt.ioff()
            plt.show()