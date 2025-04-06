from typing import List
import numpy as np
import cv2 as cv

from utils.bbox import BoundingBox


def draw_board_rect(image: np.ndarray, rect: np.ndarray) -> None:
    """
    Рисует рамку вокруг платы
    """
    colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
    
    for ((x, y), color) in zip(rect, colors):
        cv.circle(image, (int(x), int(y)), 25, color, 25)

    for i in range(4):
        pt1 = (int(rect[i][0]), int(rect[i][1]))
        pt2 = (int(rect[(i+1)%4][0]), int(rect[(i+1)%4][1]))
        cv.line(image, pt1, pt2, (0, 0, 255), 3, lineType=cv.LINE_AA)


def draw_component_rect(image: np.ndarray, components: List[BoundingBox]) -> None:
    """
    Рисует рамку вокруг компонента
    """

    for component in components:
        xmin, ymin, xmax, ymax = map(int, component.get_min_max())
        cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        