# -*- coding: utf-8 -*-

import logging


def get_logger(_: str | None = None) -> logging.Logger:
    return logging.getLogger("cogkit.api")
