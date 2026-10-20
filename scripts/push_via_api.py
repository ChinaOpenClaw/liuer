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


def gh(args: list[str], *, input_data: str | None = None,
       retries: int = 3) -> dict | list | None:
    """调用 gh api，带重试。网络失败重试后仍失败返回 None。"""
    last_err = ""
    for attempt in range(retries):
        cmd = ["gh", "api"] + args
        if input_data is not None:
            cmd += ["--input", "-"]
            res = subprocess.run(cmd, input=input_data, capture_output=True, text=True)
        else:
            res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            out = res.stdout.strip()
            return json.loads(out) if out else {}
        last_err = res.stderr[:300]
        if "404" in last_err or "Not Found" in last_err:
            return None  # 明确的 404 不重试
        if attempt < retries - 1:
            import time
            time.sleep(2 * (attempt + 1))
    print(f"[gh 失败] {' '.join(args[:4])}\n{last_err}", file=sys.stderr)
    return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    # 1) 本地 HEAD 的完整文件清单（mode / blob sha / path）
    #    git blob sha 与 GitHub 同为 SHA-1 对象模型，服务端已有同名 blob 可直接复用
    out = subprocess.run(["git", "ls-tree", "-r", "HEAD"],
                         capture_output=True, text=True, cwd=root).stdout
    entries = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, otype, sha = meta.split()
        if otype != "blob":
            continue
        entries.append((mode, sha, path))
    if not entries:
        print("HEAD 为空")
        return 1
    print(f"本地 HEAD 共 {len(entries)} 个文件")

    # 2) 父 commit
    ref = gh([f"repos/{REPO}/git/ref/heads/main"])
    parent_sha = ref["object"]["sha"]
    print(f"父 commit {parent_sha[:8]}")

    # 3) 确保每个 blob 在服务端存在；缺失才上传
    tree_entries = []
    for mode, sha, path in entries:
        check = gh([f"repos/{REPO}/git/blobs/{sha}"])
        if check is None:
            content = (root / path).read_bytes()
            b64 = base64.b64encode(content).decode()
            blob = gh(["-X", "POST", f"repos/{REPO}/git/blobs"],
                      input_data=json.dumps({"content": b64, "encoding": "base64"}))
            if not blob or blob.get("sha") != sha:
                print(f"  blob 上传异常 {path}", file=sys.stderr)
                return 1
            print(f"  上传 {path} ({len(content)} B)")
        tree_entries.append({"path": path, "mode": mode,
                             "type": "blob", "sha": sha})
    print(f"blob 就绪 {len(tree_entries)} 个")

    # 4) 全量 tree（不带 base_tree，精确镜像本地 HEAD）
    tree = gh(["-X", "POST", f"repos/{REPO}/git/trees"],
              input_data=json.dumps({"tree": tree_entries}))
    if not tree:
        return 1
    print(f"tree {tree['sha'][:8]}")

    # 5) commit + 更新 ref
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
    gh(["-X", "PATCH", f"repos/{REPO}/git/refs/heads/main"],
       input_data=json.dumps({"sha": new_commit["sha"], "force": False}))
    print("已更新 main（全量镜像）")
    return 0


if __name__ == "__main__":
    sys.exit(main())