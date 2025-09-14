from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class TraceEvent:
    name: str
    info: Dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class TraceRecorder:
    def __init__(self) -> None:
        self.events: List[TraceEvent] = []

    def add(self, name: str, **info: Any) -> None:
        self.events.append(TraceEvent(name=name, info=info))

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [
            {"name": e.name, "at": e.at, **({k: v for k, v in e.info.items()})}
            for e in self.events
        ]


