from typing import Tuple

import yaml
from pydantic import BaseModel


class BoardDetectionConfig(BaseModel):
    gaussian_kernel: Tuple[int, int]
    canny_low: int
    canny_high: int
    scale: float
    epsilon_factor: float

class ComponentDetectionConfig(BaseModel):
    model_path: str
    confidence: float
    image_size: int
    scale: float


class AppConfig(BaseModel):
    _instance = None
    board_detection: BoardDetectionConfig
    component_detection: ComponentDetectionConfig

    @classmethod
    def load_config(cls, config_path: str) -> 'AppConfig':
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)
    
    @classmethod
    def get_instance(cls): 
        if cls._instance is None:
            raise ValueError("AppConfig не инициализирован")
        return cls._instance