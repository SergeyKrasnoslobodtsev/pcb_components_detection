from dataclasses import dataclass, field

import numpy as np

@dataclass
class BoundingBox:
    points: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=np.float32))

    def __post_init__(self):
        # Приводим точки к нужному типу и проверяем форму массива
        self.points = np.array(self.points, dtype=np.float32)
        if self.points.shape != (4, 2):
            raise Exception("BoundingBox должен содержать 4 точки с формой (4, 2)")

    @classmethod
    def from_scaled_points(cls, raw_points: np.ndarray, scale: float = 1.0) -> "BoundingBox":
        """
        Создаёт BoundingBox, преобразуя координаты из масштабированного изображения
        в координаты оригинального.

        Args:
            raw_points (np.ndarray): Массив точек (4, 2) из масштабированного изображения.
            scale (float): Коэффициент масштабирования, применённый к изображению.

        Returns:
            BoundingBox: Объект с координатами, приведёнными к оригинальному масштабу.
        """
        if scale == 0:
            raise Exception("Scale не может быть равен 0")
        original_points = np.array(raw_points, dtype=np.float32) / scale
        return cls(points=original_points)
    
    @classmethod
    def from_yolo_box(cls, xyxy, scale: float = 1.0) -> "BoundingBox":
        """
        Создает BoundingBox из координат YOLO (x1, y1, x2, y2), преобразуя их к
        оригинальному масштабу с помощью scale.
        
        Args:
            xyxy: Координаты [x1, y1, x2, y2] (могут быть в формате list, tuple или np.ndarray).
            scale: Коэффициент масштабирования, который переводит координаты из
                   масштабированного изображения в систему координат оригинала.
                   
        Returns:
            BoundingBox: Объект с преобразованными координатами.
        """
        xyxy = np.array(xyxy, dtype=np.float32)
        if scale == 0:
            raise Exception("Scale не может быть равен 0")
        
        # Преобразуем координаты (умножаем, если scale_factor вычислен как отношение размеров оригинала к масштабированному)

        x1, y1, x2, y2 = xyxy
        points = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32) / scale
        return cls(points=points)

    def as_tuple(self) -> tuple:
        """Возвращает точки в виде кортежа кортежей."""
        return tuple(map(tuple, self.points))

    def get_min_max(self) -> tuple:
        """
        Возвращает ограничивающий прямоугольник в формате (xmin, ymin, xmax, ymax).
        """
        x_min, y_min = np.min(self.points, axis=0)
        x_max, y_max = np.max(self.points, axis=0)
        return x_min, y_min, x_max, y_max
