import pygame
import sys
from board import Board
from score import Score

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
        self.board.draw(screen, self.game_over)
        self.score.draw(screen)

    def play_step(self, action): # [0, 0, 0, 1] -> [left, right, up, down]
        prev = self.score.value
        if(action[0] == 1):
            self.score.add_points(self.board.move_tiles('l'))
        elif(action[1] == 1):
            self.score.add_points(self.board.move_tiles('r'))
        elif(action[2] == 1):
            self.score.add_points(self.board.move_tiles('u'))
        elif(action[3] == 1):
            self.score.add_points(self.board.move_tiles('d'))

        reward = self.score.value - prev
        game_over = self.game_over
        if(game_over):
            reward = -50
        score = self.score.value
        return reward, game_over, score

    def reset(self):
        self.game_over = False
        self.board.reset_board()
        self.score.reset_score()

    def main_loop(self): # needs to be run in separete thred
        while True:
            screen.fill((255, 255, 255))
            self.update()
            pygame.display.flip()