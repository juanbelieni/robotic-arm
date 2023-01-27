import numpy as np
import pygame as pg
from dataclasses import dataclass
from typing import List, Self


@dataclass
class RoboticArm:
    angles: np.ndarray
    lengths: np.ndarray

    def __post_init__(self):
        assert len(self.angles) == len(self.lengths)

    @property
    def size(self):
        return len(self.angles)
    
    @property
    def points(self) -> np.ndarray:
        points_ = np.zeros((self.size, 2))
        x, y = 0, 0

        for i in range(self.size):
            l = self.lengths[i]
            sum = np.sum(self.angles[: i + 1])
            x += l * np.cos(sum)
            y += l * np.sin(sum)

            points_[i] = x, y
        
        return points_

    @property
    def end_point(self) -> np.ndarray:
        return self.points[-1]

    def move(self, angle_velocities: np.ndarray):
        assert len(angle_velocities) == self.size

        for i in range(self.size):
            self.angles[i] += angle_velocities[i]
    
    def draw(self, screen: pg.Surface):
        cx, cy = screen.get_width() / 2, screen.get_height() / 2
        px, py = cx, cy

        for point in self.points:
            x, y = int(cx + point[0]), int(cy - point[1])
            pg.draw.line(screen, (255, 255, 255), (px, py), (x, y), 2)
            px, py = x, y
    
    def copy(self) -> Self:
        return RoboticArm(self.angles.copy(), self.lengths.copy())