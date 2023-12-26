import pygame
import sys
from board import Board
from score import Score
import math
import torch

pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("2048")


class Game:
    def __init__(self, moves_limit=10000, wrong_moves_limit=32):
        self.game_over = False
        self.board = Board(200, 120)
        self.score = Score(screen_width, screen_height)
        self.moves_limit = moves_limit
        self.wrong_moves_limit = wrong_moves_limit
        self.moves = 0
        self.wrong_moves = 0

    def update(self):
        if(self.board.is_game_over()):
            self.game_over = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((255, 255, 255))
        self.board.draw(screen, self.game_over)
        self.score.draw(screen)
        pygame.display.flip()

    def step(self, action): # 0, 1, 2, 3 -> left, right, up, down
        prev = self.score.value
        res = 0
        reward = 0
        if(action == 0):
            res = self.board.move_tiles('l')
        elif(action == 1):
            res = self.board.move_tiles('r')
        elif(action == 2):
            res = self.board.move_tiles('u')
        elif(action == 3):
            res = self.board.move_tiles('d')

        if(res == -1): # invalid move
            reward = -15
            self.wrong_moves += 1
        else:
            self.score.add_points(res)
            self.wrong_moves = 0
            if(self.score.value - prev == 0):
                reward = 0
            else:
                reward = math.log2(self.score.value - prev)

        self.moves += 1
        game_over = self.game_over
        if(game_over):
            reward = -10

        truncate = (self.moves > self.moves_limit) or (self.wrong_moves > self.wrong_moves_limit)

        return self.current_state(), reward, game_over, truncate

    def reset(self):
        self.game_over = False
        self.board.reset_board()
        self.score.reset()
        self.moves = 0
        self.wrong_moves = 0
        return self.current_state()

    # hotend encoded board
    def current_state(self):
        output = []
        for i in range(4):
            for j in range(4):
                if(self.board.grid[i][j] is not None):
                    output.append([1 if self.board.grid[i][j].value == 2**k else 0 for k in range(2, 19)])
                else:
                    output.append([0 for _ in range(17)])
        return torch.tensor(output, dtype=torch.float32).flatten()
