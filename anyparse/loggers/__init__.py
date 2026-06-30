import os
from .logurus import LoguruHandler


default_log_level = os.getenv("ANYPARSE_LOG_LEVEL", "DEBUG")

anyparse_logger = LoguruHandler(
    level = default_log_level
)
logger = anyparse_logger.logger