import pygame
import os
from game import Game

pygame.init()

# Get the screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2048")


# Create an instance of the game
game = Game()

# Main game loop
while not game.game_over:
    game.handle_events()
    game.update()

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the game elements
    game.draw(screen)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()