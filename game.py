import pygame
import sys
from board import Board
from score import Score
import math

pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("2048")

class Game:
    def __init__(self):
        self.game_over = False
        self.board = Board(200, 120)
        self.score = Score(screen_width, screen_height)

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if(self.board.is_game_over()):
            self.game_over = True
        screen.fill((255, 255, 255))
        self.board.draw(screen, self.game_over)
        self.score.draw(screen)
        pygame.display.flip()

    def play_step(self, action): # [0, 0, 0, 1] -> [left, right, up, down]
        prev = self.score.value
        res = 0
        if(action[0] == 1):
            res = self.board.move_tiles('l')
            if(res == -1):
                reward = -2
            else:
                self.score.add_points(res)
        elif(action[1] == 1):
            res = self.board.move_tiles('r')
            if(res == -1):
                reward = -2
            else:
                self.score.add_points(res)
        elif(action[2] == 1):
            res = self.board.move_tiles('u')
            if(res == -1):
                reward = -2
            else:
                self.score.add_points(res)
        elif(action[3] == 1):
            res = self.board.move_tiles('d')
            if(res == -1):
                reward = -2
            else:
                self.score.add_points(res)

        if(res != -1):
            if(self.score.value == prev):
                reward = 0
            else:
                reward = math.log2(self.score.value - prev)
        game_over = self.game_over
        if(game_over):
            reward = -1
        score = self.score.value
        return reward, game_over, score

    def reset(self):
        self.game_over = False
        self.board.reset_board()
        self.score.reset()

    def main_loop(self): # needs to be run in separete thred
        while True:
            screen.fill((255, 255, 255))
            self.update()
            pygame.display.flip()