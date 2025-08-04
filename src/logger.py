import datetime
import inspect
import os

FILENAME_PREFIX = "logs/log_"


class Logger:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        self.filename = FILENAME_PREFIX + today + ".txt"


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
        '''
        include caller of log instance in order to see the function calling the wrapperof `_log`.
        '''
        stack = inspect.stack()
        # The caller will be 3 levels up in the stack
        caller = stack[3]
        return f"{os.path.basename(caller.filename)}:{caller.function}:{caller.lineno}"


if __name__ == '__main__':
    logger = Logger()
    logger.info("This is an info message.")
    logger.warn("This is a warning message.")
    logger.error("This is an error message.")
    logger.performance("This is a performance log message.")
