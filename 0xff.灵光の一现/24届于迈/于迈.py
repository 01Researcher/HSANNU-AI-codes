import pygame
import sys

import random

pygame.init()
screen = pygame.display.set_mode([800, 600])
bg_color = (0, 0, 0)


class Ball:
    def __init__(self):
        self.stat = 0  # 0xia1shang
        self.ball_color = (100, 100, 100)
        self.ball_x = random.randint(0, 800)
        self.ball_y = 0
        self.rect = (self.ball_x, self.ball_y, 50, 50)
        self.y_speed = random.randint(1, 2)
        self.x_speed = (random.randint(-10, 10)) / 10

    def draw_ball(self):
        pygame.draw.circle(screen, self.ball_color, (self.ball_x, self.ball_y), 50)

    def ball_change(self):
        ball.rect = (ball.ball_x, self.ball_y, 50, 50)

        if self.ball_y >= 650:
            self.ball_x = random.randint(50, 750)
            self.ball_y = 0
            self.x_speed = (random.randint(-10, 10)) / 10
            self.y_speed = random.randint(1, 2)
        if self.ball_y < -50:
            self.ball_x = random.randint(50, 750)
            self.ball_y = 0
            self.x_speed = (random.randint(-10, 10)) / 10
            self.y_speed = random.randint(1, 2)
            self.stat = 0
        if self.ball_x <= 0 or self.ball_x >= 800:
            self.x_speed = -self.x_speed


class Rect:
    def __init__(self):
        self.width = 200
        self.height = 5
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
        self.real_score = 0

        self.color = (255, 255, 255)
        self.font = pygame.font.SysFont("华文行楷", 48)

    def draw_board(self):
        self.text = self.font.render(f"分数：{self.real_score}", True, self.color, None)
        screen.blit(self.text, (250, 10))


board = Board()
rect = Rect()
ball = Ball()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if rect.rect_1.colliderect(ball.rect):
        ball.stat = 1
        board.real_score += 1

    pygame.time.wait(1)

    if ball.stat == 0:
        ball.ball_y += ball.y_speed
        ball.ball_x += ball.x_speed
    elif ball.stat == 1:
        ball.ball_y -= ball.y_speed
        ball.ball_x += ball.x_speed

    ball.ball_change()
    rect.draw_rect()
    rect.check_mouse_events()
    ball.draw_ball()
    board.draw_board()
    pygame.display.flip()
