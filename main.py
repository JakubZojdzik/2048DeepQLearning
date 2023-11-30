import pygame
import os
from game import Game

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2048")

game = Game(width, height)

while True:
    game.update()

    screen.fill((255, 255, 255))
    game.draw(screen)

    pygame.display.flip()

pygame.quit()