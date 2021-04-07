import pygame

pygame.init()
WIDTH, HEIGHT = 400, 400
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
g = grid(28, 28, width, height)
main()

class pixel(object): #Define the pixel as an object
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255,255,255)
        self.neighbors = []

def draw(self, surface): #Create the drawing surface and split it into pixels
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

class grid(object):
    pixels = []

def main():
    clock = pygame.time.clock()
    run = True
    while run == True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            elif event.type == MOUSEWHEEL:



