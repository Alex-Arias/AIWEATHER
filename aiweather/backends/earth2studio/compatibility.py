import earth2studio

from packaging.version import Version

VERSION = Version(earth2studio.__version__)


def is_v08():
    return VERSION.major == 0 and VERSION.minor == 8


def is_v17():
    return VERSION >= Version("0.17")