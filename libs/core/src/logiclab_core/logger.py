import io
import sys
import json
import queue
import atexit
import logging
from typing import Any, Literal
from logging.handlers import QueueHandler, QueueListener

import structlog
from rich.syntax import Syntax
from rich.console import Console
from rich.traceback import Traceback

LogLevel = int | str


class NonBlockingQueueHandler(QueueHandler):
    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass


class SmartFormatter(logging.Formatter):
    """
    A formatter that adds the name of the logger (library) to the message,
    ONLY if it is not our main application logger.
    """

    def format(self, record: logging.LogRecord) -> str:
        if record.name.startswith("app"):
            return record.getMessage()

        return f"[{record.name}] {record.getMessage()}"


class RichRenderer:
    def __init__(self) -> None:
        self.console = Console(file=io.StringIO(), force_terminal=True, width=120)

    def __call__(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> str:
        timestamp = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", "info").upper()
        event = event_dict.pop("event", "")
        exc_info = event_dict.pop("exception", None)

        level_colors = {
            "DEBUG": "dim blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red bold",
            "CRITICAL": "red bold reverse",
        }
        level_color = level_colors.get(level, "white")

        buf = io.StringIO()
        self.console.file = buf

        self.console.print(
            f"[dim cyan]{timestamp}[/] [{level_color}]{level:8}[/] [bold white]{event}[/]"
        )

        for key, value in event_dict.items():
            if isinstance(value, (dict, list)):
                try:
                    json_str = json.dumps(value, indent=2, ensure_ascii=False)
                    syntax = Syntax(
                        json_str,
                        "json",
                        theme="monokai",
                        line_numbers=False,
                        word_wrap=True,
                        background_color="default",
                    )
                    self.console.print(f"  [cyan]{key}[/]:")
                    self.console.print(syntax)
                except (TypeError, ValueError):
                    self.console.print(f"  [cyan]{key}[/] = [yellow]{value}[/]")
            else:
                self.console.print(f"  [cyan]{key}[/] = [yellow]{value}[/]")

        if exc_info:
            self.console.print(Traceback.from_exception(*sys.exc_info()))  # type: ignore

        return buf.getvalue()


class AppLogger:
    def __init__(
        self,
        mode: Literal["dev", "debug", "prod"] = "prod",
        log_level: LogLevel | None = None,
        log_to_console: bool = False,
        log_file: str | None = None,
        loki_url: str | None = None,
        app_name: str = "my_app",
    ) -> None:
        self.mode = mode

        if self.mode == "dev" and (log_file or loki_url):
            raise ValueError(
                "The 'dev' mode uses RichRenderer (ANSI colors) and cannot be used with log_file or loki_url."
            )

        self.log_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.handlers: list[logging.Handler] = []

        if log_level is None:
            log_level = logging.INFO if self.mode == "prod" else logging.DEBUG

        self._setup_handlers(log_level, log_to_console, log_file, loki_url, app_name)

        self.listener = QueueListener(self.log_queue, *self.handlers, respect_handler_level=True)
        self.listener.start()
        atexit.register(self.listener.stop)

        self._configure_structlog()
        self._configure_stdlib(log_level)

    def _setup_handlers(
        self,
        log_level: LogLevel,
        console: bool,
        file_path: str | None,
        loki_url: str | None,
        app_name: str,
    ) -> None:
        formatter = SmartFormatter()

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.handlers.append(console_handler)

        if file_path:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.handlers.append(file_handler)

        if loki_url:
            try:
                from logging_loki import LokiHandler

                loki_handler = LokiHandler(
                    url=loki_url,
                    tags={"application": app_name, "env": self.mode},
                    version="1",
                )
                loki_handler.setLevel(log_level)
                loki_handler.setFormatter(formatter)
                self.handlers.append(loki_handler)
            except ImportError:
                pass

    def _configure_structlog(self) -> None:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ]

        if self.mode == "prod":
            processors.extend([structlog.processors.JSONRenderer(ensure_ascii=False)])
        elif self.mode == "debug":
            processors.extend(
                [structlog.processors.JSONRenderer(indent=2, sort_keys=False, ensure_ascii=False)]
            )
        else:  # dev
            processors.extend([RichRenderer()])

        structlog.configure(
            processors=processors,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def _configure_stdlib(self, log_level: LogLevel) -> None:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(log_level)
        root_logger.addHandler(NonBlockingQueueHandler(self.log_queue))

        noisy_libraries = ["urllib3", "httpx", "httpcore", "hpack", "openai"]
        for lib_name in noisy_libraries:
            logging.getLogger(lib_name).setLevel(logging.WARNING)

    @staticmethod
    def get_logger(**initial_context: Any) -> structlog.typing.FilteringBoundLogger:
        return structlog.get_logger("app").bind(**initial_context)
