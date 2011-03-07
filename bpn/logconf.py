#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2011 Christopher D. Lasher
#
# This software is released under the MIT License. Please see
# LICENSE.txt for details.


"""Logging configuration for BPN programs."""

import logging


def _set_up_root_stream_logger(logger):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)


def _set_up_root_file_logger(logger, logfile):
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)


def set_up_root_logger(logfile=None):
    logger = logging.getLogger('bpn')
    logger.setLevel(logging.INFO)
    _set_up_root_stream_logger(logger)
    if logfile is not None:
        _set_up_root_file_logger(logger, logfile)

