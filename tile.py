import pygame

texture_path = 'assets/tiles/'
tile_size = 85
padding = 12

class Tile:
    def __init__(self, value):
        self.value = value
        self.texture = pygame.image.load(texture_path + str(value) + '.png').convert()
        self.joinable = True

    def update(self, value):
        self.value = value
        self.texture = pygame.image.load(texture_path + str(value) + '.png').convert()

    def draw(self, screen, col, row, offset_x, offset_y):
        x = padding * (col + 1) + col * tile_size + offset_x
        y = padding * (row + 1) + row * tile_size + offset_y
        screen.blit(self.texture, (x, y))