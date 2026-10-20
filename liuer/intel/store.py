"""情报库：SQLite 持久化 + 字段级加密 + 客户维度隔离。

为什么加密：情报内容是收款账号、钱包地址这类可直接致害的信息，
落盘必须是密文，客户侧也要能按维度隔离。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from cryptography.fernet import Fernet

from ..intel.rules import Finding

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS intel (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    value_enc TEXT NOT NULL,
    value_hash TEXT NOT NULL,
    confidence REAL NOT NULL,
    context TEXT DEFAULT '',
    source TEXT DEFAULT 'rule',
    tactic TEXT DEFAULT '',
    brand TEXT DEFAULT '',
    axis TEXT DEFAULT '',
    client_id TEXT DEFAULT 'default',
    session_id TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_intel_dedup
    ON intel(kind, value_hash, client_id);
CREATE INDEX IF NOT EXISTS idx_intel_kind ON intel(kind);
CREATE INDEX IF NOT EXISTS idx_intel_session ON intel(session_id);
CREATE INDEX IF NOT EXISTS idx_intel_client ON intel(client_id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    peer TEXT DEFAULT '',
    persona_id TEXT DEFAULT '',
    stage TEXT DEFAULT '',
    mission TEXT DEFAULT '',
    turn_count INTEGER DEFAULT 0,
    started_at REAL NOT NULL,
    ended_at REAL,
    status TEXT DEFAULT 'active',
    transcript TEXT DEFAULT '[]',
    summary TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

CREATE TABLE IF NOT EXISTS metrics (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL DEFAULT 0
);
"""


@dataclass
class IntelRecord:
    id: str
    kind: str
    value: str
    confidence: float
    context: str
    source: str
    tactic: str
    brand: str
    axis: str
    client_id: str
    session_id: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntelStore:
    def __init__(self, db_path: Path, fernet_key: bytes) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(fernet_key)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _hash(value: str) -> str:
        import hashlib
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _enc(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def _dec(self, value_enc: str) -> str:
        return self._fernet.decrypt(value_enc.encode("ascii")).decode("utf-8")

    def upsert(self, finding: Finding, *, client_id: str = "default",
               session_id: str = "", tactic: str = "", brand: str = "",
               axis: str = "") -> tuple[IntelRecord, bool]:
        """写入情报。已存在则提升置信度。返回 (记录, 是否新建)。"""
        vh = self._hash(finding.value)
        now = time.time()
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "SELECT id, confidence FROM intel WHERE kind=? AND value_hash=? AND client_id=?",
                (finding.kind, vh, client_id))
            row = cur.fetchone()
            if row:
                new_conf = max(float(row["confidence"]), finding.confidence)
                conn.execute("UPDATE intel SET confidence=?, session_id=? WHERE id=?",
                             (new_conf, session_id or row["id"], row["id"]))
                conn.commit()
                record = IntelRecord(
                    id=row["id"], kind=finding.kind, value=finding.value,
                    confidence=new_conf, context=finding.context, source=finding.source,
                    tactic=tactic, brand=brand, axis=axis, client_id=client_id,
                    session_id=session_id, created_at=now)
                return record, False

            rec_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO intel (id, kind, value_enc, value_hash, confidence, context,"
                " source, tactic, brand, axis, client_id, session_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec_id, finding.kind, self._enc(finding.value), vh, finding.confidence,
                 finding.context[:500], finding.source, tactic, brand, axis,
                 client_id, session_id, now))
            conn.commit()
        return IntelRecord(
            id=rec_id, kind=finding.kind, value=finding.value,
            confidence=finding.confidence, context=finding.context,
            source=finding.source, tactic=tactic, brand=brand, axis=axis,
            client_id=client_id, session_id=session_id, created_at=now), True

    def query(self, *, client_id: str | None = None, kind: str | None = None,
              min_confidence: float = 0.0, limit: int = 200) -> list[IntelRecord]:
        sql = "SELECT * FROM intel WHERE confidence >= ?"
        args: list[Any] = [min_confidence]
        if client_id:
            sql += " AND client_id = ?"
            args.append(client_id)
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [
            IntelRecord(
                id=r["id"], kind=r["kind"], value=self._dec(r["value_enc"]),
                confidence=r["confidence"], context=r["context"], source=r["source"],
                tactic=r["tactic"], brand=r["brand"], axis=r["axis"],
                client_id=r["client_id"], session_id=r["session_id"],
                created_at=r["created_at"])
            for r in rows
        ]

    def stats(self, client_id: str | None = None) -> dict[str, Any]:
        with self._conn() as conn:
            base = "SELECT COUNT(*) c FROM intel"
            args: list[Any] = []
            if client_id:
                base += " WHERE client_id=?"
                args.append(client_id)
            total = conn.execute(base, args).fetchone()["c"]

            sql = "SELECT kind, COUNT(*) c FROM intel"
            if client_id:
                sql += " WHERE client_id=?"
            by_kind = {r["kind"]: r["c"] for r in conn.execute(sql, args).fetchall()}

            sql2 = "SELECT tactic, COUNT(*) c FROM intel WHERE tactic != ''"
            if client_id:
                sql2 += " AND client_id=?"
            by_tactic = {r["tactic"]: r["c"] for r in conn.execute(sql2, args).fetchall()}

            sql3 = "SELECT brand, COUNT(*) c FROM intel WHERE brand != ''"
            if client_id:
                sql3 += " AND client_id=?"
            by_brand = {r["brand"]: r["c"] for r in conn.execute(sql3, args).fetchall()}

            sql4 = "SELECT axis, COUNT(*) c FROM intel WHERE axis != ''"
            if client_id:
                sql4 += " AND client_id=?"
            by_axis = {r["axis"]: r["c"] for r in conn.execute(sql4, args).fetchall()}

        return {"total": total, "by_kind": by_kind, "by_tactic": by_tactic,
                "by_brand": by_brand, "by_axis": by_axis}

    # ---------- 会话 ----------

    def create_session(self, session_id: str, channel: str, peer: str,
                       persona_id: str, mission: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, channel, peer, persona_id,"
                " mission, turn_count, started_at, status) VALUES (?,?,?,?,?,0,?,'active')",
                (session_id, channel, peer, persona_id, mission, time.time()))
            conn.commit()

    def update_session(self, session_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "transcript" in fields and not isinstance(fields["transcript"], str):
            fields["transcript"] = json.dumps(fields["transcript"], ensure_ascii=False)
        allowed = {"stage", "turn_count", "status", "transcript", "summary", "ended_at", "peer"}
        cols = [k for k in fields if k in allowed]
        if not cols:
            return
        sql = f"UPDATE sessions SET {', '.join(f'{c}=?' for c in cols)} WHERE session_id=?"
        with self._lock, self._conn() as conn:
            conn.execute(sql, [fields[c] for c in cols] + [session_id])
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        try:
            d["transcript"] = json.loads(d.get("transcript") or "[]")
        except json.JSONDecodeError:
            d["transcript"] = []
        return d

    def list_sessions(self, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT session_id, channel, peer, persona_id, stage, turn_count," \
              " started_at, ended_at, status FROM sessions"
        args: list[Any] = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]

    # ---------- 指标 ----------

    def incr(self, key: str, delta: float = 1.0) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("INSERT INTO metrics (key, value) VALUES (?, ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = value + ?",
                         (key, delta, delta))
            conn.commit()

    def metrics(self) -> dict[str, float]:
        with self._conn() as conn:
            return {r["key"]: r["value"]
                    for r in conn.execute("SELECT key, value FROM metrics").fetchall()}