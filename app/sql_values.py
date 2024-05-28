from dataclasses import dataclass, field
from typing import List


@dataclass
class SQLValues:
    """Dataclass for SQL query parameters."""
    limit: int = 10000000
    judge_names: List[str] = field(default_factory=list)
    county_names: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "limit": self.limit,
            "judge_names": self.judge_names,
            "county_names": self.county_names,
        }
