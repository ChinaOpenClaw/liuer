"""GitHub 直推：git push 被代理拦截时的回退通道。

用 Git Data API 组装一次完整提交：建 blob → 建 tree → 建 commit → 更新 ref。
只推送指定 commit 里改动的文件，其余沿用 base_tree。
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

REPO = "ChinaOpenClaw/liuer"


def gh(args: list[str], *, input_data: str | None = None) -> dict | list | None:
    cmd = ["gh", "api"] + args
    if input_data is not None:
        cmd += ["--input", "-"]
        res = subprocess.run(cmd, input=input_data, capture_output=True, text=True)
    else:
        res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[gh 失败] {' '.join(args[:4])}\n{res.stderr[:400]}", file=sys.stderr)
        return None
    out = res.stdout.strip()
    return json.loads(out) if out else None


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    # 1) 本次 commit 改了哪些文件
    files = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
        capture_output=True, text=True, cwd=root).stdout.split()
    files = [f for f in files if f.strip()]
    if not files:
        print("没有待推送文件")
        return 1
    print(f"待推送 {len(files)} 个文件")

    # 2) 父 commit 与 base tree
    ref = gh([f"repos/{REPO}/git/ref/heads/main"])
    parent_sha = ref["object"]["sha"]
    commit = gh([f"repos/{REPO}/git/commits/{parent_sha}"])
    base_tree = commit["tree"]["sha"]
    print(f"父 commit {parent_sha[:8]}  base tree {base_tree[:8]}")

    # 3) 逐个建 blob。跳过目录与不存在路径——git 的 rename 检测
    # 可能把目录项混进 --name-only，直接当文件读会抛 PermissionError。
    tree_entries = []
    for rel in files:
        p = root / rel
        if p.is_dir():
            print(f"  跳过（目录）{rel}")
            continue
        if not p.exists():
            print(f"  删除 {rel}")
            tree_entries.append({"path": rel, "mode": "100644",
                                 "type": "blob", "sha": None})
            continue
        content = p.read_bytes()
        b64 = base64.b64encode(content).decode()
        # 大文件用 stdin 传 JSON，避免命令行长度限制
        blob = gh(["-X", "POST", f"repos/{REPO}/git/blobs"],
                  input_data=json.dumps({"content": b64, "encoding": "base64"}))
        if not blob:
            print(f"  blob 失败 {rel}")
            return 1
        tree_entries.append({"path": rel, "mode": "100644",
                             "type": "blob", "sha": blob["sha"]})
        print(f"  blob {rel} ({len(content)} B)")

    # 4) 建 tree
    tree = gh(["-X", "POST", f"repos/{REPO}/git/trees"],
              input_data=json.dumps({"base_tree": base_tree,
                                     "tree": tree_entries}))
    if not tree:
        return 1
    print(f"tree {tree['sha'][:8]}")

    # 5) 建 commit
    subject = subprocess.run(["git", "log", "-1", "--pretty=%s"],
                             capture_output=True, text=True, cwd=root).stdout.strip()
    body = subprocess.run(["git", "log", "-1", "--pretty=%b"],
                          capture_output=True, text=True, cwd=root).stdout.strip()
    message = subject + ("\n\n" + body if body else "")
    new_commit = gh(["-X", "POST", f"repos/{REPO}/git/commits"],
                    input_data=json.dumps({"message": message,
                                           "tree": tree["sha"],
                                           "parents": [parent_sha]}))
    if not new_commit:
        return 1
    print(f"commit {new_commit['sha'][:8]}")

    # 6) 更新 ref
    gh(["-X", "PATCH", f"repos/{REPO}/git/refs/heads/main"],
       input_data=json.dumps({"sha": new_commit["sha"], "force": False}))
    print("已更新 main")
    return 0


if __name__ == "__main__":
    sys.exit(main())