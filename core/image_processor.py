import numpy as np
import cv2 as cv


class ImageProcessor:
    def __init__(self):
        self.processors = []
    
    def add_processor(self, processor_func, **kwargs):
        self.processors.append((processor_func, kwargs))
        return self
    
    def process(self, image):
        result = image
        for processor_func, kwargs in self.processors:
            result = processor_func(result, **kwargs)
        return result

def remove_shadows(image: np.ndarray, canny_low: int, canny_high: int, kernel:float) -> np.ndarray:
    """Удаляет тени с изображения
    
    Args:
        image (np.ndarray): входное изображение
        canny_low (int): нижний порог для Canny
        canny_high (int): верхний порог для Canny
        kernel (float): размер ядра для морфологических операций
    
    Returns:
        np.ndarray: изображение без теней
    """
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    edged = cv.Canny(hsv_image[:,:, 1], canny_low, canny_high)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel)
    edged = cv.dilate(edged, kernel, iterations=1)
    result = cv.erode(edged, kernel, iterations=1)

    return result


def resize_image(image: np.ndarray, scale=None, width=None, height=None, inter=cv.INTER_AREA) -> np.ndarray:
    """изменяет размер изображения с сохранением пропорций

    Args:
        image (np.ndarray): входное изображение
        scale (_type_, optional): масштаб для изменения размера. Defaults to None.
        width (_type_, optional): ширина для изменения размера. Defaults to None.
        height (_type_, optional): высота для изменения размера. Defaults to None.
        inter (_type_, optional): метод интерполяции. Defaults to cv.INTER_AREA.

    Returns:
        np.ndarray: измененное изображение
    """
    if image is None or image.size == 0:
        return None
    # Если задан scale, меняем размер на основе масштаба
    if scale is not None:
        return cv.resize(image, None, fx=scale, fy=scale, interpolation=inter)

    # Получаем исходные размеры изображения
    (h, w) = image.shape[:2]
    
    # Если ни ширина, ни высота не заданы, возвращаем оригинальное изображение
    if width is None and height is None:
        return image

    # Рассчитываем новые размеры, сохраняя пропорции
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # Изменяем размер изображения
    return cv.resize(image, dim, interpolation=inter)