
from abc import ABC, abstractmethod

from ultralytics import YOLO
from core.image_processor import ImageProcessor, remove_shadows, resize_image
from utils.bbox import BoundingBox
from utils.contours import find_contours, transform_perspective
from services.app_config import AppConfig
from services.logger import LoggerFactory


class BaseDetector(ABC):
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger("app." + self.__class__.__name__)
        self.preprocessor = ImageProcessor()
    
    @abstractmethod
    def detect(self, image):
        """Метод для обнаружения объектов на изображении"""
        pass


class BoardDetector(BaseDetector):
    def __init__(self, config: AppConfig):
        super().__init__(config=config)
        self.preprocessor.add_processor(
            remove_shadows,
            canny_low=self.config.board_detection.canny_low,
            canny_high=self.config.board_detection.canny_high,
            kernel=self.config.board_detection.gaussian_kernel,
        ).add_processor(
            resize_image,
            scale=self.config.board_detection.scale)
        self.logger.info("Инициализация детектора платы завершена")
        self.logger.debug(f"Конфигурация детектора платы: {self.config.board_detection}")
    
    def detect(self, image):
        """Метод для обнаружения платы на изображении"""
        self.logger.info("Начало обнаружения платы")
        
        # Предобработка изображения
        preprocessed_image = self.preprocessor.process(image)

        self.logger.debug(f"Размер предобработанного изображения: {preprocessed_image.shape}")
        points = find_contours(preprocessed_image)

        if points is None:
            self.logger.warning("Плата не обнаружена")
            return None
        
        self.logger.info("Плата обнаружена")
        bbox = BoundingBox.from_scaled_points(points, scale=self.config.board_detection.scale)

        board_image = transform_perspective(image, bbox.points)
        self.logger.debug(f"Размер изображения платы: {board_image.shape}")
        self.logger.info("Обнаружение платы завершено")
        return board_image, bbox

class ComponentDetector(BaseDetector):
    def __init__(self, config: AppConfig):
        super().__init__(config=config)
        self.preprocessor.add_processor(resize_image, scale=self.config.component_detection.scale)
        self.logger.info("Инициализация детектора компонентов завершена")
        self.logger.debug(f"Конфигурация детектора компонентов: {self.config.component_detection}")
    
    def detect(self, image):
        """Метод для обнаружения компонентов на изображении"""
        self.logger.info("Начало обнаружения компонентов")
        
        try:
            model = YOLO(self.config.component_detection.model_path)

        except Exception as e:
            raise Exception(f"YOLO модель не загружена {str(e)}")

        # Предобработка изображения
        preprocessed_image = self.preprocessor.process(image)

        results = model.predict(preprocessed_image, 
                                conf=self.config.component_detection.confidence,
                                imgsz=self.config.component_detection.image_size)
        components = []
        for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):

                    xyxy = box.xyxy[0]
                    confidence = box.conf.item()

                    bbox = BoundingBox.from_yolo_box(xyxy, scale=self.config.component_detection.scale)
                    self.logger.debug(f'Компонент: {i} conf: {confidence:.2f}')
                    components.append(bbox)


        self.logger.info("Обнаружение компонентов завершено")
        return components