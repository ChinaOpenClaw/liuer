"""配置加载：所有模块统一从这里取配置，禁止硬编码路径与模型名。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """按点号路径读取，例如 get('llm.chat_model')。"""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path_of(self, dotted: str, default: str) -> Path:
        """读取配置中的路径值并解析为绝对路径。"""
        raw_value = self.get(dotted, default) or default
        p = Path(raw_value)
        return p if p.is_absolute() else (self.root / p)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    return Config(raw=raw, root=cfg_path.resolve().parent)


@dataclass
class Keystore:
    """对称加密密钥管理。密钥落盘但权限受限，仅用于情报字段加密。"""

    key: bytes
    path: Path

    @classmethod
    def load_or_create(cls, path: Path, configured_key: str = "") -> "Keystore":
        from cryptography.fernet import Fernet

        path.parent.mkdir(parents=True, exist_ok=True)
        if configured_key:
            key = configured_key.encode("utf-8")
        else:
            key = None
            if path.exists():
                try:
                    key = path.read_bytes().strip()
                except OSError as exc:
                    raise RuntimeError(
                        f"无法读取密钥文件 {path}: {exc}。"
                        "若被安全软件拦截，请在 config.yaml 的 "
                        "security.encryption_key 中直接配置密钥。") from exc
            if not key:
                key = Fernet.generate_key()
                try:
                    path.write_bytes(key)
                except OSError as exc:
                    raise RuntimeError(
                        f"无法写入密钥文件 {path}: {exc}。"
                        "请在 config.yaml 的 security.encryption_key 中配置密钥，"
                        "或换一个可写路径。") from exc
        try:
            Fernet(key)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"加密密钥格式非法: {exc}") from exc
        if os.name != "nt":
            os.chmod(path, 0o600)
        return cls(key=key, path=path)


@dataclass
class RuntimePaths:
    """运行期目录，延迟到真正需要时创建。"""

    root: Path
    raw: dict = field(default_factory=dict)
    data: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data = self.root / "data"
        self.data.mkdir(parents=True, exist_ok=True)

    def _configured(self, dotted_key: str) -> str:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return ""
            node = node[part]
        return node if isinstance(node, str) else ""

    def ensure(self, dotted_key: str, default: str) -> Path:
        """解析配置中的路径并确保目录存在。

        绝对路径原样使用；相对路径相对 data 目录。
        """
        value = self._configured(dotted_key) or default
        p = Path(value)
        if p.is_absolute():
            target = p
        elif value.startswith("data/"):
            target = self.data / p.name
        else:
            target = self.data / p

        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
        return target