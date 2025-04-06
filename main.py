
import cv2 as cv

from core.base_detector import BoardDetector, ComponentDetector
from vizual.drawer import draw_board_rect, draw_component_rect
from services.app_config import AppConfig
import os
from pathlib import Path

from services.logger import LoggerFactory
base_dir = Path(os.getcwd())

def main():
    # Инициализация
    config_path = base_dir / "configs/app_settings.yaml"
    logging_config_path = base_dir / "configs/logging.yaml"
    image_path = base_dir / "data/IMG_3141.jpg"
    config = AppConfig.load_config(config_path=config_path)
    LoggerFactory.load_config(logging_config_path)

    board_preprocessor = BoardDetector(config=config)
    detector = ComponentDetector(config=config)

    image = cv.imread(image_path)

    # Предобработка изображения
    board_image, bbox = board_preprocessor.detect(image)

    # Рисование рамки вокруг платы
    draw_board_rect(image, bbox.points)

    components = detector.detect(board_image)

    draw_component_rect(board_image, components)
    
    title = 'Original image'
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, image)
    cv.resizeWindow(title, 640, 480)

    title = 'Board'
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, board_image)
    cv.resizeWindow(title, 640, 480)

    cv.waitKey(0)
    cv.destroyAllWindows()
   

if __name__ == "__main__":
    main()