import logging
import logging.config
import os
from typing import Dict, Optional
import yaml

class LoggerFactory:
    _loggers: Dict[str, logging.Logger] = {}
    _config_loaded = False
    
    @classmethod
    def load_config(cls, config_path: str = "configs/logging.yaml") -> None:
        """Загрузка конфигурации логгера"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logging.config.dictConfig(config)
                cls._config_loaded = True
        else:
            cls._setup_default_config()

    @classmethod
    def _setup_default_config(cls) -> None:
        """Базовая конфигурация логгера"""
        config = {
            'version': 1,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                    'level': 'INFO'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': 'logs/app.log',
                    'formatter': 'standard',
                    'level': 'DEBUG'
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console']
            },
            'loggers': {
                'app': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False
                }
            }
        }
        logging.config.dictConfig(config)
        cls._config_loaded = True

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Получение инстанса логгера"""
        if not cls._config_loaded:
            cls.load_config()

        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]

class TestService():
    def __init__(self):
        self.logger = LoggerFactory.get_logger("test." + self.__class__.__name__)

    def test_logging(self):
        self.logger.debug("Тестовое сообщение DEBUG")
        self.logger.info("Тестовое сообщение INFO")
        self.logger.warning("Тестовое сообщение WARNING")
        self.logger.error("Тестовое сообщение ERROR")



from pathlib import Path

def main():
    # Инициализация
    config_path = Path("configs/logging.yaml")

    LoggerFactory.load_config(config_path=config_path)
    logger = LoggerFactory.get_logger()
    # Тестовые сообщения
    logger.info("Запуск тестовой программы")
    TestService().test_logging()
    logger.info("Тестирование завершено")

if __name__ == "__main__":
    main()