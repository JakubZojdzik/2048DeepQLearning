import pygame

texture_path = 'assets/tiles/'
tile_size = 85
padding = 12
move_speed = 20
alpha_speed = 20

class Tile:
    def __init__(self, value, col, row, offset_x, offset_y, alpha=255):
        self.value = value
        self.texture = pygame.image.load(texture_path + str(value) + '.png').convert()
        self.joinable = True
        self.pos_x = padding * (col + 1) + col * tile_size + offset_x
        self.pos_y = padding * (row + 1) + row * tile_size + offset_y
        self.target_x = self.pos_x
        self.target_y = self.pos_y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.alpha = alpha
        self.texture.set_alpha(self.alpha)

    def update(self, value):
        self.value = value
        self.texture = pygame.image.load(texture_path + str(value) + '.png').convert()

    def animate(self):
        if(self.pos_x != self.target_x):
            self.pos_x += min(move_speed * ((self.target_x - self.pos_x) / abs(self.target_x - self.pos_x)), self.target_x - self.pos_x, key=abs)
        if(self.pos_y != self.target_y):
            self.pos_y += min(move_speed * ((self.target_y - self.pos_y) / abs(self.target_y - self.pos_y)), self.target_y - self.pos_y, key=abs)
        if(self.alpha < 255):
            self.alpha += min(alpha_speed, 255 - self.alpha)
            self.texture.set_alpha(self.alpha)

    def move(self, col, row):
        self.target_x = padding * (col + 1) + col * tile_size + self.offset_x
        self.target_y = padding * (row + 1) + row * tile_size + self.offset_y

    def draw(self, screen):
        screen.blit(self.texture, (self.pos_x, self.pos_y))