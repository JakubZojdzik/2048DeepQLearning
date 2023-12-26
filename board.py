from tile import Tile
# import pygame
import random
class Board:
    def __init__(self, pos_x, pos_y):
        self.grid = [[None for _ in range(4)] for _ in range(4)]
        self.trash = []
        self.pos_x = pos_x
        self.pos_y = pos_y
        # self.texture = pygame.image.load('assets/board.png').convert()
        # self.game_over_texture = pygame.image.load('assets/restart.png').convert_alpha()

        self.add_tile()
        self.add_tile()

    def add_tile(self):
        empty_positions = [(i, j) for i in range(4) for j in range(4) if self.grid[i][j] is None]
        if empty_positions:
            i, j = random.choice(empty_positions)
            value = random.choices([2, 4], weights=[0.9, 0.1], k=1)[0]
            self.grid[i][j] = Tile(value, i, j, self.pos_x, self.pos_y, 0)

    def move_tiles(self, direction):
        points = 0
        for i in range(4):
            for j in range(4):
                if(self.grid[i][j] is not None):
                    self.grid[i][j].joinable = True
        action = False
        if(direction == 'u'):
            for _ in range(3):
                for x in range(4):
                    for y in range(1, 4):
                        if(self.grid[x][y] is None):
                            continue
                        if(self.grid[x][y-1] is None):
                            self.grid[x][y].move(x, y-1)
                            self.grid[x][y-1] = self.grid[x][y]
                            self.grid[x][y] = None
                            action = True
                        elif(self.grid[x][y-1].value == self.grid[x][y].value and self.grid[x][y-1].joinable and self.grid[x][y].joinable):
                            self.grid[x][y-1].update(self.grid[x][y-1].value * 2)
                            points += self.grid[x][y-1].value * 2
                            self.grid[x][y-1].joinable = False
                            self.grid[x][y].move(x, y-1)
                            self.trash.append(self.grid[x][y])
                            self.grid[x][y] = None
                            action = True
        elif(direction == 'd'):
            for _ in range(3):
                for x in range(4):
                    for y in range(2, -1, -1):
                        if(self.grid[x][y] is None):
                            continue
                        if(self.grid[x][y+1] is None):
                            self.grid[x][y].move(x, y+1)
                            self.grid[x][y+1] = self.grid[x][y]
                            self.grid[x][y] = None
                            action = True
                        elif(self.grid[x][y+1].value == self.grid[x][y].value and self.grid[x][y+1].joinable and self.grid[x][y].joinable):
                            self.grid[x][y+1].update(self.grid[x][y+1].value * 2)
                            points += self.grid[x][y+1].value * 2
                            self.grid[x][y+1].joinable = False
                            self.grid[x][y].move(x, y+1)
                            self.trash.append(self.grid[x][y])
                            self.grid[x][y] = None
                            action = True
        elif(direction == 'l'):
            for _ in range(3):
                for x in range(1, 4):
                    for y in range(4):
                        if(self.grid[x][y] is None):
                            continue
                        if(self.grid[x-1][y] is None):
                            self.grid[x][y].move(x-1, y)
                            self.grid[x-1][y] = self.grid[x][y]
                            self.grid[x][y] = None
                            action = True
                        elif(self.grid[x-1][y].value == self.grid[x][y].value and self.grid[x-1][y].joinable and self.grid[x][y].joinable):
                            self.grid[x-1][y].update(self.grid[x-1][y].value * 2)
                            points += self.grid[x-1][y].value * 2
                            self.grid[x-1][y].joinable = False
                            self.grid[x][y].move(x-1, y)
                            self.trash.append(self.grid[x][y])
                            self.grid[x][y] = None
                            action = True
        elif(direction == 'r'):
            for _ in range(3):
                for x in range(2, -1, -1):
                    for y in range(4):
                        if(self.grid[x][y] is None):
                            continue
                        if(self.grid[x+1][y] is None):
                            self.grid[x][y].move(x+1, y)
                            self.grid[x+1][y] = self.grid[x][y]
                            self.grid[x][y] = None
                            action = True
                        elif(self.grid[x+1][y].value == self.grid[x][y].value and self.grid[x+1][y].joinable and self.grid[x][y].joinable):
                            self.grid[x+1][y].update(self.grid[x+1][y].value * 2)
                            points += self.grid[x+1][y].value * 2
                            self.grid[x+1][y].joinable = False
                            self.trash.append(self.grid[x][y])
                            self.grid[x][y].move(x+1, y)
                            self.grid[x][y] = None
                            action = True
        if not action:
            return -1
        else:
            self.add_tile()

        return points

    def is_game_over(self):
        for x in range(4):
            for y in range(4):
                if self.grid[x][y] is None:
                    return False
                if x > 0 and self.grid[x-1][y] and self.grid[x][y].value == self.grid[x-1][y].value:
                    return False
                if x < 3 and self.grid[x+1][y] and self.grid[x][y].value == self.grid[x+1][y].value:
                    return False
                if y > 0 and self.grid[x][y-1] and self.grid[x][y].value == self.grid[x][y-1].value:
                    return False
                if y < 3 and self.grid[x][y+1] and self.grid[x][y].value == self.grid[x][y+1].value:
                    return False
        return True

    def reset_board(self):
        self.grid = [[None for _ in range(4)] for _ in range(4)]
        self.add_tile()
        self.add_tile()

    # def draw(self, screen, game_over):
    #     screen.blit(self.texture, (self.pos_x, self.pos_y))

    #     for t_tile in self.trash:
    #         t_tile.draw(screen)
    #         t_tile.animate()
    #         if(t_tile.pos_x == t_tile.target_x and t_tile.pos_y == t_tile.target_y):
    #             self.trash.remove(t_tile)

    #     for x in range(4):
    #         for y in range(4):
    #             if self.grid[x][y] is not None:
    #                 self.grid[x][y].draw(screen)
    #                 self.grid[x][y].animate()

    #     if(game_over):
    #         screen.blit(self.game_over_texture, (self.pos_x, self.pos_y))
