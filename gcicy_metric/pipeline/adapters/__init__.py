"""Built-in gCICY geometry adapters."""

from .p1p1p5_22 import P1P1P5Type22Adapter
from .p3p1p1p1_21 import P3P1P1P1Type21Adapter
from .p4p1_hirzebruch_11 import (
    P4P1HirzebruchM4Type11Adapter,
    P4P1HirzebruchType11Adapter,
)
from .p4p1p1_hirzebruch_21 import P4P1P1HirzebruchType21Adapter
from .p5p1_k3_21 import P5P1K3Type21Adapter

__all__ = [
    "P1P1P5Type22Adapter",
    "P3P1P1P1Type21Adapter",
    "P4P1HirzebruchM4Type11Adapter",
    "P4P1HirzebruchType11Adapter",
    "P4P1P1HirzebruchType21Adapter",
    "P5P1K3Type21Adapter",
]
