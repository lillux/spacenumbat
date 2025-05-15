#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:49:53 2025

@author: lillux


Logging utilities

This module provides functions to configure and retrieve loggers for the package.
It supports flexible logging configurations, including console and file handlers,
with options for log rotation and customizable formats.

Usage
-----
To configure logging:
    from spacenumbat._log import configure
    configure(level="DEBUG", log_dir="~/spacenumbat_logs")

To obtain a logger in other modules:
    from spacenumbat._log import get_logger
    log = get_logger(__name__)
    log.info("This is an info message.")
"""

from __future__ import annotations
import logging
import logging.handlers
import sys
from pathlib import Path
from logging.config import dictConfig
from typing import Optional, Union


_DEFAULT_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def _get_unique_log_filename(base_name: str, log_dir: Union[str, Path]) -> Path:
    """
    Generate a unique log filename by appending an incrementing number if needed.

    Parameters
    ----------
    base_name : str
        The base name for the log file (e.g., 'run.log').
    log_dir : Union[str, Path]
        The directory where the log file will be stored.

    Returns
    -------
    Path
        A Path object with a unique filename that does not already exist.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_path = log_dir / base_name
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    i = 1
    while True:
        candidate = log_dir / f"{stem}{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def _build_default_config(
    log_dir: Optional[Union[str, Path]] = None,
    level: str = "INFO"
) -> dict:
    """
    Constructs a default logging configuration dictionary.

    Parameters
    ----------
    log_dir : Optional[Union[str, Path]], optional
        Directory path for log files. If None, no file handler is added.
    level : str, optional
        The logging level to set. Default is 'INFO'.

    Returns
    -------
    dict
        A dictionary suitable for passing to logging.config.dictConfig().
    """
    handlers: dict[str, dict] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stderr,
            "formatter": "standard",
        }
    }
    if log_dir:
        log_file = _get_unique_log_filename("run.log", log_dir)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "formatter": "standard",
        }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": _DEFAULT_FMT}},
        "handlers": handlers,
        "root": {"handlers": list(handlers), "level": level},
    }


def configure(
    level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Configures logging for the package.

    This function sets up logging handlers and formatters, and applies the
    configuration using logging.config.dictConfig().

    Parameters
    ----------
    level : str, optional
        The logging level to set. Default is 'INFO'.
    log_dir : Optional[Union[str, Path]], optional
        Directory path for log files. If None, no file handler is added.
    """
    dictConfig(_build_default_config(log_dir, level))
    

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a logger for the specified name.

    Parameters
    ----------
    name : str, optional
        The name of the logger. If None, the root logger is returned.

    Returns
    -------
    logging.Logger
        A logger instance for the specified name.
    """
    return logging.getLogger(name)