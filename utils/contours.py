
import cv2 as cv
import numpy as np

from scipy.spatial import distance as dist

def find_contours(image: np.ndarray) -> list:
   
    try:
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  

        if not contours:
            return None
        
        max_contour = max(contours, key=cv.contourArea)
            
        rect = cv.minAreaRect(max_contour)
        box = cv.boxPoints(rect)

        ordered_points = order_points(box)
        
        return ordered_points
    
    except Exception as e:
        raise Exception(f"Ошибка поиска контуров: {str(e)}")
    

def order_points(pts):
    """ Упорядочивает точки в против часовой стрелке, начиная с верхнего левого угла."""
    # Сортируем точки по X-координате
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # запоминаем самую левую и самую правую точки
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # Сортируем точки по Y-координате
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # Вычисляем расстояние между точками
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # Возвращаем упорядоченный список точек
    return np.array([tl, tr, br, bl], dtype="float32")


def transform_perspective(image: np.ndarray, rect)->np.ndarray:
    """ Преобразует изображение в перспективе."""
    (tl, tr, br, bl) = rect

    # Вычисляем ширину нового изображения
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Вычисляем высоту нового изображения
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Задаём точки назначения для перспективного преобразования
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Вычисляем матрицу перспективного преобразования и применяем её
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped