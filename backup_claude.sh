#!/bin/bash
# Claude 全量备份脚本 — 每日 12:00 定时执行
# 用法: bash backup_claude.sh
#
# 目录结构:
#   00_Backup/
#   ├── current/          ← 最新备份（固定文件名，每日覆盖）
#   │   ├── claude_config.tar.gz
#   │   ├── claude_workspace.tar.gz
#   │   └── vault.tar.gz
#   ├── archive/          ← 旧版本（含日期后缀）
#   │   ├── claude_config_20260521.tar.gz
#   │   ├── claude_workspace_20260521.tar.gz
#   │   └── vault_20260521.tar.gz
#   ├── requirements_frozen.txt
#   ├── restore.sh
#   └── 恢复指南.md
#
# 版本管理: current(最新) + archive(10天前)
# 目标目录: D:\我的坚果云\00_Backup\（坚果云同步，换机时可恢复）

BACKUP_DIR="/d/我的坚果云/00_Backup"
CURRENT_DIR="$BACKUP_DIR/current"
ARCHIVE_DIR="$BACKUP_DIR/archive"
DATE_TAG=$(date +%Y%m%d)
ERRORS=0

mkdir -p "$CURRENT_DIR" "$ARCHIVE_DIR"

echo "=== Claude 全量备份 $DATE_TAG ==="

# ── 0. 冻结 Python 环境 ──
if command -v python &>/dev/null; then
  python -m pip freeze > /d/ClaudeWorkspace/trading_system/requirements_frozen.txt 2>&1
elif [ -x "C:/Users/sut-b/AppData/Local/Programs/Python/Python312/python.exe" ]; then
  "C:/Users/sut-b/AppData/Local/Programs/Python/Python312/python.exe" -m pip freeze > /d/ClaudeWorkspace/trading_system/requirements_frozen.txt 2>&1
fi
echo "[0/5] Python 环境已冻结 ($(wc -l < /d/ClaudeWorkspace/trading_system/requirements_frozen.txt) packages)"

# ── 0.5. system_state 快照 ──
echo "[0.5/5] system_state snapshot..."
python -c "import sys; sys.path.insert(0, '/d/ClaudeWorkspace/trading_system'); from snapshot_state import snapshot_all; snapshot_all()"

# ── 1. 归档当前版本（每10天一次） ──
echo "[1/5] 归档旧版本..."
DO_ARCHIVE=false
# 检查最近一次归档是否已超过10天
latest_archive=""
for name in claude_config claude_workspace vault; do
  f=$(ls -t "$ARCHIVE_DIR/${name}"_*.tar.gz 2>/dev/null | head -1)
  [ -n "$f" ] && latest_archive="$f"
done
if [ -n "$latest_archive" ]; then
  arc_date=$(basename "$latest_archive" | sed 's/.*_\([0-9]\{8\}\).tar.gz/\1/')
  if [ -n "$arc_date" ]; then
    arc_sec=$(date -d "$arc_date" +%s 2>/dev/null)
    now_sec=$(date +%s)
    days_diff=$(( (now_sec - arc_sec) / 86400 ))
    [ "$days_diff" -ge 10 ] && DO_ARCHIVE=true
  fi
else
  DO_ARCHIVE=true  # 从未归档过，首次执行
fi

if [ "$DO_ARCHIVE" = true ]; then
  for name in claude_config claude_workspace vault; do
    src="$CURRENT_DIR/${name}.tar.gz"
    if [ -f "$src" ]; then
      arc_date=$(stat -c %Y "$src" 2>/dev/null | xargs -I{} date -d @{} +%Y%m%d 2>/dev/null)
      [ -z "$arc_date" ] && arc_date="$DATE_TAG"
      dst="$ARCHIVE_DIR/${name}_${arc_date}.tar.gz"
      [ ! -f "$dst" ] && mv "$src" "$dst" && echo "  归档: ${name}_${arc_date}.tar.gz"
    fi
  done
  # 清理 archive：只保留每组最新的1份
  for name in claude_config claude_workspace vault; do
    ls -t "$ARCHIVE_DIR/${name}"_*.tar.gz 2>/dev/null | tail -n +2 | xargs rm -f 2>/dev/null
  done
  echo "  10天周期归档完成（archive 只保留最新1份）"
else
  echo "  距上次归档不足10天，跳过归档"
fi

# ── 2. Claude 完整配置（含飞书凭证） → current/ ──
echo "[2/5] 备份 Claude 完整配置..."
tar -czf "$CURRENT_DIR/claude_config.tar.gz" \
  /c/Users/sut-b/.claude/CLAUDE.md \
  /c/Users/sut-b/.claude/settings.json \
  /c/Users/sut-b/.claude/.last-cleanup \
  /c/Users/sut-b/.claude/hooks/ \
  /c/Users/sut-b/.claude/skills/ \
  /c/Users/sut-b/.claude/sessions/ \
  /c/Users/sut-b/.claude/tasks/ \
  /c/Users/sut-b/.claude/plans/ \
  "/c/Users/sut-b/.claude/projects/D--ClaudeWorkspace/memory/" \
  /c/Users/sut-b/.claude.json \
  /c/Users/sut-b/.agents/ \
  /c/Users/sut-b/.feishu-user-plugin/ \
  /c/Users/sut-b/.lark-channel/ \
  2>/dev/null
TAR_EXIT=$?
if [ $TAR_EXIT -ne 0 ]; then echo "  ERROR: claude_config 备份失败 (tar exit=$TAR_EXIT)"; ERRORS=$((ERRORS+1)); else echo "  OK: claude_config.tar.gz"; fi

# ── 3. 项目工作区（含 file-history + cron任务） → current/ ──
echo "[3/5] 备份项目工作区..."
tar -czf "$CURRENT_DIR/claude_workspace.tar.gz" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  /d/ClaudeWorkspace/CLAUDE.md \
  /d/ClaudeWorkspace/trading_system/ \
  /d/ClaudeWorkspace/.claude/scheduled_tasks.json \
  /c/Users/sut-b/.claude/file-history/ \
  2>/dev/null
TAR_EXIT=$?
if [ $TAR_EXIT -ne 0 ]; then echo "  ERROR: workspace 备份失败 (tar exit=$TAR_EXIT)"; ERRORS=$((ERRORS+1)); else echo "  OK: claude_workspace.tar.gz"; fi

# ── 4. Vault 第二大脑 → current/ ──
echo "[4/5] 备份 Vault..."
tar -czf "$CURRENT_DIR/vault.tar.gz" \
  --exclude='node_modules' \
  "/d/DiskMigration/MySecondBrain/" \
  2>/dev/null
TAR_EXIT=$?
if [ $TAR_EXIT -ne 0 ]; then echo "  ERROR: vault 备份失败 (tar exit=$TAR_EXIT)"; ERRORS=$((ERRORS+1)); else echo "  OK: vault.tar.gz"; fi

# ── 5. 写入备份清单 ──
echo "[5/5] 写入备份清单..."
cat > "$BACKUP_DIR/backup_manifest_${DATE_TAG}.txt" << EOF
备份日期: $DATE_TAG
目录结构:
  current/  → 最新备份（claude_config.tar.gz / workspace / vault）
  archive/  → 过期版本（含日期后缀）
内容清单:
  1. current/claude_config.tar.gz — CLAUDE.md + settings + hooks + skills + sessions + tasks + plans + memory + .claude.json(mcp) + .agents(skills)
  2. current/claude_workspace.tar.gz — 项目CLAUDE.md + trading_system + file-history(会话JSON)
  3. current/vault.tar.gz — 第二大脑全部笔记
  4. requirements_frozen.txt — Python 全量依赖锁定
  恢复工具: restore.sh（自动适配路径，推荐使用）
  恢复指南: 恢复指南.md（含完整恢复步骤）
状态: $([ $ERRORS -eq 0 ] && echo "全部成功" || echo "有 $ERRORS 个错误")
EOF

echo ""
if [ $ERRORS -eq 0 ]; then
  echo "=== 备份完成: $DATE_TAG ==="
else
  echo "=== 备份完成: $DATE_TAG (有 $ERRORS 个错误) ==="
fi
echo "位置: $BACKUP_DIR"
ls -lh "$CURRENT_DIR/"
exit $ERRORS
