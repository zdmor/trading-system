#!/usr/bin/env python3
"""将Claude端对话同步到飞书共享上下文"""
import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
CTX_FILE = os.path.join(BASE, ".feishu_context.json")
MAX_HISTORY = 10

def load_ctx():
    try:
        with open(CTX_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_ctx(msgs):
    with open(CTX_FILE, "w", encoding="utf-8") as f:
        json.dump(msgs[-MAX_HISTORY * 2:], f, ensure_ascii=False, indent=2)

def append(user_text, reply_text):
    ctx = load_ctx()
    ctx.append({"role": "user", "content": user_text})
    ctx.append({"role": "assistant", "content": reply_text})
    save_ctx(ctx)
    print(f"ok: synced to .feishu_context.json ({len(ctx)} msgs)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python claude_to_feishu.py <user_text> <reply_text>")
        sys.exit(1)
    append(sys.argv[1], sys.argv[2])
