"""Structured logging for GAIA"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredLogger:
    """Logger that outputs structured JSON logs"""

    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler for structured logs
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)

    def log_event(self, event_type: str, **kwargs):
        """Log a structured event"""
        entry = {"timestamp": datetime.utcnow().isoformat(), "event": event_type, **kwargs}
        self.logger.info(json.dumps(entry))

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log_event("info", message=message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log_event("error", message=message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log_event("warning", message=message, **kwargs)


# Global logger instance
_default_logger = StructuredLogger("gaia")


def get_logger(name: str = "gaia") -> StructuredLogger:
    """Get a logger instance"""
    return StructuredLogger(name)
