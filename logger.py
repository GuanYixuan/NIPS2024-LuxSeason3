import sys
import logging

class StepFormatter(logging.Formatter):
    """用于显示当前回合的Formatter"""
    def format(self, record):
        record.step = Logger()._instance.step if Logger()._instance else 0  # type: ignore
        return super().format(record)

class Logger:
    _instance = None

    logger: logging.Logger

    step: int = 0
    """当前的回合数"""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化logger
            cls._instance.logger = logging.getLogger('app')
            cls._instance.logger.setLevel(logging.INFO)

            handler = logging.StreamHandler(sys.stderr)
            # 使用自定义的格式化器，添加 step 字段
            formatter = StepFormatter('%(asctime)s - %(levelname)s - Step %(step)d - %(message)s',
                                      datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            cls._instance.logger.addHandler(handler)

        return cls._instance

    def set_step(self, step: int) -> None:
        self.step = step

    def info(self, message: str) -> None:
        self.logger.info(message)
    def error(self, message: str) -> None:
        self.logger.error(message)
    def debug(self, message: str) -> None:
        self.logger.debug(message)
    def warning(self, message: str) -> None:
        self.logger.warning(message)
