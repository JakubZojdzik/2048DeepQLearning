# Deep Q-Learning playing 2048 game

## Introduction
This repository contains a Python implementation of a neural network trained using Deep Q-Learning to play the game 2048. It is very basic project created mostly by following [pytorch reinforcement learning tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

Network contains 4 layers:
- Input layer: 272 nodes (board 4x4, onehot encoding for tile value)
- 2 hidden layers: 256 nodes
- Output layer: 4 nodes (4 actions: up, right, down, left)
Activation function is leaky ReLU between each layer. Cost is calculated with `SmoothL1Loss` function with beta = 1.

Rewards

Agent implements experience replay with memory of 1000 (state, action, next_state, reward) tuples in queue. Each batch is randomly chosen from memory.


## Project Structure
- `agent.py`: Contains the class with Deep Q-Learning agent implementation. You can experiment with various hyperparameters such as BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, and LR to improve the performance of the model.
- `model.py`: Defines the neural network architecture using PyTorch. The model consists of an input layer, two hidden layers, and an output layer with leaky ReLU activation functions between each layer.
- `game.py`: Implements enviroment. It inherits from my [2048 game](https://github.com/JakubZojdzik/2048) repository.
- `trained.pth`: Includes a pre-trained model ready to use for playing the 2048 game.

## Getting Started
1. Install the required dependencies:
```sh
pip install requirements.txt
```
2. Run the main.py script to see the trained model in action:
```sh
python main.py
```

## Train the model
If you wish to train the model with different hyperparameters, pass your custom parameters to Agent constructor and run `train()` method. You can change network structure in `model.py` file.

## Results
The average score of the trained model is around 2000 points. However, the model is not able to win the game and create a 2048 tile. Feel free to iterate on the hyperparameters and network architecture to improve performance.
