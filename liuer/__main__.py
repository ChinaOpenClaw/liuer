"""启动入口：python -m liuer"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="liuer",
        description="LiuEr — 本地部署的主动反诈骗智能体系统")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--profile", default=None,
                        help="人格档案路径，例如 profiles/mother-1966.yaml")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from .api import create_server
    from .config import load_config

    cfg = load_config(args.config)
    host = args.host or str(cfg.get("deployment.host", "127.0.0.1"))
    port = args.port or int(cfg.get("deployment.port", 8787))

    if args.reload:
        uvicorn.run("liuer.api:create_server", factory=True, host=host,
                    port=port, reload=True, log_level=args.log_level)
        return 0

    server = create_server(args.config, profile_path=args.profile)
    uvicorn.run(server, host=host, port=port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())