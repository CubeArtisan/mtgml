ACTIVATION_CHOICES = ("elu", "selu", "relu", "tanh", "sigmoid", "linear", "swish")

OPTIMIZER_CHOICES = (
    "adam",
    "adamax",
    "adadelta",
    "nadam",
    "rmsprop",
    "adafactor",
)

LARGE_INT = 1e05

MAX_CARDS_IN_PACK = 16
MAX_BASICS = 16
MAX_PICKED = 48
MAX_SEEN_PACKS = 48
MAX_CUBE_SIZE = 1080
MAX_DECK_SIZE = 256
MAX_COPIES = 18
DEFAULT_PACKS_PER_PLAYER = 3
DEFAULT_PICKS_PER_PACK = 15

RISKINESS_CONSTANT = 8

EPSILON = 1e-08

_debug = False


def is_debug() -> bool:
    return _debug


def set_debug(enable):
    global _debug
    _debug = enable


_ensure_shape = False


def should_ensure_shape() -> bool:
    return _ensure_shape


def set_ensure_shape(enable):
    global _ensure_shape
    _ensure_shape = enable


_log_histograms = False


def should_log_histograms() -> bool:
    return _log_histograms


def set_log_histograms(enable):
    global _log_histograms
    _log_histograms = enable
