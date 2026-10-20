"""人格库与调度。

对外承诺「数十万级人格」，实现方式是按需生成 + 去重，
而不是预先落盘几十万条记录——后者会占用大量存储且没有价值。
调度器为每个新会话分配一个人格，并保证同一号码多次会话人格不一致。
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Iterator

from ..persona.model import Person, PersonaFactory


class PersonaLibrary:
    """线程安全的人格库。"""

    def __init__(self, config) -> None:
        self.factory = PersonaFactory(config)
        self._lock = threading.Lock()
        self._used: set[str] = set()
        self._seed = 0
        self.size = self.factory.library_size

    def _next_seed(self) -> int:
        with self._lock:
            self._seed += 1
            return self._seed

    def acquire(self, *, fresh: bool = True) -> Person:
        """取一个人格。fresh=True 时尽量避开已用过的指纹。"""
        for _ in range(24):
            seed = self._next_seed()
            person = self.factory.create(seed=seed)
            fp = person.fingerprint()
            if not fresh:
                return person
            with self._lock:
                if fp not in self._used:
                    self._used.add(fp)
                    return person
        # 指纹耗尽（超出库容量），接受重复
        return self.factory.create(seed=self._next_seed())

    def stream(self, count: int) -> Iterator[Person]:
        for _ in range(count):
            yield self.acquire()

    def get_by_id(self, persona_id: str) -> Person | None:
        """按 id 重建人格。仅用于会话回放，不保证与原实例同一对象。"""
        for seed in range(self.size * 2):
            person = self.factory.create(seed=seed)
            if person.persona_id == persona_id:
                return person
        return None

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"library_size": self.size, "issued": len(self._used)}


class MissionRegistry:
    """任务登记处。客户可针对某号码或某渠道下发任务。"""

    def __init__(self) -> None:
        self._missions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def register(self, scope: str, mission) -> None:
        with self._lock:
            self._missions[scope] = mission

    def get(self, scope: str):
        with self._lock:
            return self._missions.get(scope)

    def all(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._missions)