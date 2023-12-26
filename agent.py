import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

from game import Game
from model import DQN

import torch
import torch.nn as nn
import torch.optim as optim

env = Game()
plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 512  # number of transitions sampled from the replay buffer
GAMMA = 0.99  # discount factor as mentioned in the previous section
EPS_START = 0.95  # starting value of epsilon
EPS_END = 0.05  # final value of epsilon
# controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_DECAY = 500
TAU = 0.005  # update rate of the target network
LR = 1e-3  # learning rate of the ``AdamW`` optimizer

n_actions = 4  # up, down, left, right
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
matplotlib.use("TkAgg")
steps_done = 0


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def plot_scores(show_result=False):
    plt.figure(1)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
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


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        choose = random.randint(0, 3)
        return torch.tensor([[choose]], device=device, dtype=torch.long)


episode_scores = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def onehot_encode(state):
    output = []
    for i in range(4):
        for j in range(4):
            output.append([1 if state[i][j].value == k else 0 for k in range(1, 18)])
    return torch.tensor(output, dtype=torch.float32).flatten()


def onehot_decode(state):
    output = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(1, 18):
                if (state[i*17+j*4+k] == 1):
                    output[i][j] = k
                    break
    return torch.tensor(output, dtype=torch.float32)


def train(path):
    num_episodes = 50000
    plot_scores()

    for i in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        shaped = env.shaped_current_state()
        while (1):
            action = select_action(state)
            env.update()
            observation, reward, terminated, truncated = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action, next_state, reward)

            # Augmentation by rotation and flipping state
            state = state.reshape(4, 4, 17)
            next_state = next_state.reshape(4, 4, 17)
            for _ in range(3):
                state = torch.rot90(state, 1, [0, 1])
                next_state = torch.rot90(next_state, 1, [0, 1])
                memory.push(state.flatten(), action, next_state.flatten(), reward)
                env.update()

            env.update()

            state = next_state
            optimize_model()
            env.update()
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            if done:
                episode_scores.append(env.score.value)
                plot_scores()
                break

        if (i % 100 == 0):
            torch.save(policy_net.state_dict(), path)

    print('Complete')
    plot_scores(show_result=True)
    plt.ioff()
    plt.show()


path = 'model.pth'
train(path)
