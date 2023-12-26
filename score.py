import pygame

pygame.font.init()
my_font = pygame.font.Font('assets/rajdhani.ttf', 60)


class Score:
    def __init__(self, screen_width, screen_height):
        self.value = 0
        self.screen_width = screen_width
        self.screen_height = screen_height

    def add_points(self, value):
        self.value += value

    def reset(self):
        self.value = 0

    def draw(self, screen):
        text_surface = my_font.render('Score: ' + str(self.value), True, (119, 110, 101))
        screen.blit(text_surface, ((self.screen_width - text_surface.get_width()) // 2, 0))