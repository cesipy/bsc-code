import datetime
import inspect
import os
import socket

FILENAME_PREFIX = "logs/log_"



class Logger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # only initialize once
        if Logger._initialized:
            return

        os.makedirs("logs", exist_ok=True)
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        if socket.gethostname() == "c703i-gpu5":
            self.filename = FILENAME_PREFIX + today+ "gpu5" + ".txt"
        elif socket.gethostname() == "c703i-gpu10":
            self.filename = FILENAME_PREFIX + today+ "gpu10" + ".txt"
        elif socket.gethostname() == "c703i-gpu11":
            self.filename = FILENAME_PREFIX + today+ "gpu11" + ".txt"
        elif socket.gethostname() == "c703i-gpu1":
            self.filename = FILENAME_PREFIX + today+ "gpu1" + ".txt"
        else:
            self.filename = FILENAME_PREFIX + today + ".txt"
        Logger._initialized = True

    def info(self, message: str) -> None:
        self._log("info", message)

    def warn(self, message: str) -> None:
        self._log("warn", message)

    def error(self, message: str) -> None:
        self._log("error", message)

    def _log(self, tag: str, message: str) -> None:
        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tag_padded: str = tag.upper().ljust(5)
        caller_info: str = self._get_caller_info()
        log_entry: str = f"{timestamp} - {tag_padded} - {caller_info} - {message}\n"
        with open(self.filename, "a") as f:
            f.write(log_entry)

    def _get_caller_info(self):
        stack = inspect.stack()
        caller = stack[3]
        return f"{os.path.basename(caller.filename)}:{caller.function}:{caller.lineno}"


if __name__ == '__main__':
    logger = Logger()
    logger.info("This is an info message.")
    logger.warn("This is a warning message.")
    logger.error("This is an error message.")