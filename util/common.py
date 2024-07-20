from typing import Dict


def value(d: Dict[str, float]) -> float:
    return list(d.values())[0]


def key(d: Dict[str, float]) -> float:
    return list(d.keys())[0]
