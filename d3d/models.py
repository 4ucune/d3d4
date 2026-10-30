from dataclasses import dataclass, field
from typing import List


@dataclass
class NEFISElement:
    name: str
    type: str
    single_bytes: int
    quantity: str = ''
    unit: str = ''
    description: str = ''
    count: int = 0
    dimensions: List[int] = field(default_factory=list)


@dataclass
class NEFISCell:
    name: str
    elements: List[str] = field(default_factory=list)


@dataclass
class NEFISGroupDef:
    name: str
    defined: int
    cell: str = ''
    dim_count: int = 0
    dimensions: List[int] = field(default_factory=list)
    order: str = ''
    max_index: int = 0
