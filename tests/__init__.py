"""
測試套件

提供單元測試和整合測試的基礎設施。
"""

import sys
from pathlib import Path

# 將 src 目錄添加到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

__all__ = []
