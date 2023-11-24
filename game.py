import pygame
import sys
from board import Board
from score import Score

class Game:
    def __init__(self, screen_width, screen_height):
        self.game_over = False
        self.board = Board(200, 120)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.score = Score(self.screen_width, self.screen_height)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.score.add_points(self.board.move_tiles('l'))
                elif event.key == pygame.K_RIGHT:
                    self.score.add_points(self.board.move_tiles('r'))
                elif event.key == pygame.K_UP:
                    self.score.add_points(self.board.move_tiles('u'))
                elif event.key == pygame.K_DOWN:
                    self.score.add_points(self.board.move_tiles('d'))

    def update(self):
        # Update the game state based on input and other logic
        pass

    def draw(self, screen):
        self.board.draw(screen)
        self.score.draw(screen)