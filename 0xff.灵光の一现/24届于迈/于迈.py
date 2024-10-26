import pygame
import sys

import random
import time

pygame.init()
screen = pygame.display.set_mode([800, 600])
bg_color = (0, 0, 0)


class Ball:
    def __init__(self):

        self.ball_color = (100, 100, 100)
        self.ball_x = 0
        self.ball_y = 0
        self.rect = (self.ball_x, self.ball_y, 50, 50)
        self.speed = 1

    def draw_ball(self):
        pygame.draw.circle(screen, self.ball_color, (self.ball_x, self.ball_y), 50)

    def ball_y_change(self):
        pygame.time.wait(1)
        self.ball_y += self.speed
        self.rect = (self.ball_x, self.ball_y, 50, 50)
        if self.ball_y >= 600:
            self.ball_x = random.randint(50, 750)
            self.ball_y = 0
        if self.ball_y < 0:
            self.ball_x = random.randint(50, 750)
            self.ball_y = 0
            ball.speed = - ball.speed


class Rect:
    def __init__(self):
        self.width = 200
        self.height = 25
        self.x = 300
        self.y = 550
        self.rect_color = (255, 255, 255)
        self.rect = (self.x, self.y, self.width, self.height)
        self.rect_1 = pygame.Rect(self.rect)

    def draw_rect(self):
        screen.fill(bg_color)
        pygame.draw.rect(screen, self.rect_color, self.rect)

    def check_mouse_events(self):
        self.x = pygame.mouse.get_pos()[0]
        self.x -= self.width / 2
        self.rect = (self.x, self.y, self.width, self.height)
        self.rect_1 = pygame.Rect(self.rect)


class Board:
    def __init__(self):
        self.text_color = (0, 0, 0)


rect = Rect()
ball = Ball()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    if rect.rect_1.colliderect(ball.rect):
        print("1")
        ball.speed = -ball.speed

    # print(rect.rect)
    ball.ball_y_change()
    rect.draw_rect()
    rect.check_mouse_events()
    ball.draw_ball()
    pygame.display.flip()
