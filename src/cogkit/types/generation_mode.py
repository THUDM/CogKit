# -*- coding: utf-8 -*-


import enum


class GenerationMode(enum.Enum):
    TextToVideo = "t2v"
    ImageToVideo = "i2v"
    TextToImage = "t2i"
