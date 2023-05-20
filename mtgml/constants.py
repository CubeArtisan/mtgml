ACTIVATION_CHOICES = ("elu", "selu", "relu", "tanh", "sigmoid", "linear", "swish")

OPTIMIZER_CHOICES = (
    "adam",
    "adamax",
    "lazyadam",
    "rectadam",
    "novograd",
    "lamb",
    "adadelta",
    "nadam",
    "rmsprop",
    "adafactor",
)

LARGE_INT = 1e04

MAX_CARDS_IN_PACK = 16
MAX_BASICS = 16
MAX_PICKED = 48
MAX_SEEN_PACKS = 48
MAX_CUBE_SIZE = 1080
MAX_DECK_SIZE = 64
MAX_SIDEBOARD_SIZE = 32
MAX_COPIES = 16

RISKINESS_CONSTANT = 8

EPSILON = 1e-05

_debug = True


def is_debug() -> bool:
    return _debug


def set_debug(enable):
    global _debug
    _debug = enable
