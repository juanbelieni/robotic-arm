import numpy as np
import pygame as pg
from robotic_arm import RoboticArm
from typing import List

screen = pg.display.set_mode((1200, 800))
clock = pg.time.Clock()

n = 5
arm = RoboticArm(np.random.uniform(-np.pi, np.pi, n), np.ones(n) * 600 / n)


def calc_grad(arm: RoboticArm, T: np.ndarray) -> np.ndarray:
    n = arm.size
    grad = np.zeros(n)

    for k in range(n):
        Px, Py = arm.end_point
        Tx, Ty = T

        for i in range(k, n):
            l = arm.lengths[i]
            sum = np.sum(arm.angles[: i + 1])
            grad[k] += -(Px - Tx) * l * np.sin(sum) + (Py - Ty) * l * np.cos(sum)

        grad[k] *= 2

    return grad


def calc_alpha(arm: RoboticArm, T: np.ndarray, d: np.ndarray) -> float:
    alpha = 1
    P = arm.end_point

    while True:
        arm_ = arm.copy()
        arm_.move(alpha * d)
        P_ = arm_.end_point

        if np.linalg.norm(P_ - T) < np.linalg.norm(P - T):
            print(np.linalg.norm(P_ - T), np.linalg.norm(P - T))
            break

        alpha *= 0.5

    return alpha


while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()

    dt = clock.get_time() / 1000
    screen.fill((0, 0, 0))

    mx, my = pg.mouse.get_pos()
    cx, cy = screen.get_width() / 2, screen.get_height() / 2
    target = np.array([mx - cx, cy - my])

    grad = calc_grad(arm, target)
    grad /= np.linalg.norm(grad)
    d = -grad 

    alpha = calc_alpha(arm, target, d)
    d *= alpha * dt

    arm.move(d)
    arm.draw(screen)

    pg.display.flip()
    clock.tick(60)
