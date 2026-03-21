#!/usr/bin/env bash
# 掃描專案目錄體積與大型單檔（排除 .git），供本機稽核用。
# 用法：bash scripts/scan_large_files.sh
#      bash scripts/scan_large_files.sh | tee logs/large_files_scan.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== du Top 50（排除 .git）==="
# head 會關閉 pipe，避免 pipefail 將 SIGPIPE 視為失敗
du -ah "$ROOT" 2>/dev/null | grep -v '/\.git/' | sort -rh | head -50 || true
echo ""
echo "=== 單檔 >50MB ==="
find "$ROOT" -type f -size +50M ! -path '*/.git/*' 2>/dev/null | sort
