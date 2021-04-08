from enum import Enum


class Activation(str, Enum):
    GeLU = "gelu"
    ReLU = "relu"
