import sys
import logging

PRODUCTION = (sys.platform != 'win32')

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

            cls._instance.refresh_handler()

        return cls._instance

    def set_step(self, step: int) -> None:
        self.step = step
        self.refresh_handler()
        if PRODUCTION:
            self.logger.info(f'Step {step}')

    def refresh_handler(self) -> None:
        if len(self.logger.handlers):
            self.logger.removeHandler(self.logger.handlers[0])
        handler = logging.StreamHandler(sys.stdout if PRODUCTION else sys.stderr)

        if PRODUCTION:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = StepFormatter('%(asctime)s - %(levelname)s - Step %(step)d - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        self.logger.info(message)
    def error(self, message: str) -> None:
        self.logger.error(message)
    def debug(self, message: str) -> None:
        self.logger.debug(message)
    def warning(self, message: str) -> None:
        self.logger.warning(message)
