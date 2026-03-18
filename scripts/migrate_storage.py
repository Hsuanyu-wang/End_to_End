#!/usr/bin/env python3
"""
Storage 遷移腳本

將舊的分散式 storage 遷移到新的集中化結構
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

# 定義舊路徑和新路徑的映射
OLD_ROOT = "/home/End_to_End_RAG"
NEW_ROOT = "/home/End_to_End_RAG/storage"


def find_old_storages() -> List[Tuple[str, str, str]]:
    """
    找出所有舊的 storage 目錄
    
    Returns:
        List of (old_path, storage_type, suggested_new_path)
    """
    storages = []
    
    # 1. 找出 lightrag_storage
    old_lightrag = os.path.join(OLD_ROOT, "lightrag_storage")
    if os.path.exists(old_lightrag):
        for item in os.listdir(old_lightrag):
            old_path = os.path.join(old_lightrag, item)
            if os.path.isdir(old_path):
                # 例如: DI_lightrag_default -> storage/lightrag/DI_lightrag_default
                new_path = os.path.join(NEW_ROOT, "lightrag", item)
                storages.append((old_path, "lightrag", new_path))
    
    # 2. 找出 graph_index_*
    for item in os.listdir(OLD_ROOT):
        if item.startswith("graph_index"):
            old_path = os.path.join(OLD_ROOT, item)
            if os.path.isdir(old_path):
                # 例如: graph_index_DI_propertyindex -> storage/graph_index/DI_propertyindex
                name = item.replace("graph_index_", "") if item != "graph_index" else "default"
                new_path = os.path.join(NEW_ROOT, "graph_index", name)
                storages.append((old_path, "graph_index", new_path))
    
    # 3. 找出 *.pkl (CSR graph cache)
    for item in os.listdir(OLD_ROOT):
        if item.endswith(".pkl") and not item.startswith("."):
            old_path = os.path.join(OLD_ROOT, item)
            if os.path.isfile(old_path):
                # 例如: some_cache.pkl -> storage/csr_graph/some_cache.pkl
                new_path = os.path.join(NEW_ROOT, "csr_graph", item)
                storages.append((old_path, "csr_graph", new_path))
    
    return storages


def migrate_storage(old_path: str, new_path: str, dry_run: bool = True) -> bool:
    """
    遷移單個 storage
    
    Args:
        old_path: 舊路徑
        new_path: 新路徑
        dry_run: 是否為試運行（不實際移動檔案）
    
    Returns:
        是否成功
    """
    if not os.path.exists(old_path):
        print(f"⚠️  舊路徑不存在: {old_path}")
        return False
    
    if os.path.exists(new_path):
        print(f"⚠️  新路徑已存在: {new_path}")
        return False
    
    if dry_run:
        print(f"[DRY RUN] 將遷移: {old_path} -> {new_path}")
        return True
    else:
        try:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            if os.path.isdir(old_path):
                shutil.move(old_path, new_path)
            else:
                shutil.move(old_path, new_path)
            
            print(f"✅ 已遷移: {old_path} -> {new_path}")
            return True
        except Exception as e:
            print(f"❌ 遷移失敗: {old_path} -> {new_path}")
            print(f"   錯誤: {e}")
            return False


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="遷移舊 storage 到新結構")
    parser.add_argument("--dry-run", action="store_true", help="試運行（不實際移動檔案）")
    parser.add_argument("--force", action="store_true", help="強制執行（不詢問確認）")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Storage 遷移腳本")
    print("=" * 60)
    print()
    
    # 建立新的 storage 目錄結構
    os.makedirs(NEW_ROOT, exist_ok=True)
    os.makedirs(os.path.join(NEW_ROOT, "vector_index"), exist_ok=True)
    os.makedirs(os.path.join(NEW_ROOT, "graph_index"), exist_ok=True)
    os.makedirs(os.path.join(NEW_ROOT, "lightrag"), exist_ok=True)
    os.makedirs(os.path.join(NEW_ROOT, "csr_graph"), exist_ok=True)
    os.makedirs(os.path.join(NEW_ROOT, "cache"), exist_ok=True)
    
    # 找出所有舊 storage
    storages = find_old_storages()
    
    if not storages:
        print("✨ 沒有找到需要遷移的舊 storage")
        return
    
    print(f"找到 {len(storages)} 個需要遷移的 storage:")
    print()
    
    # 按類型分組顯示
    by_type = {}
    for old_path, storage_type, new_path in storages:
        if storage_type not in by_type:
            by_type[storage_type] = []
        by_type[storage_type].append((old_path, new_path))
    
    for storage_type, items in by_type.items():
        print(f"📁 {storage_type.upper()} ({len(items)} 個):")
        for old_path, new_path in items:
            print(f"   {old_path}")
            print(f"   -> {new_path}")
            print()
    
    # 確認
    if not args.force and not args.dry_run:
        response = input("確定要執行遷移嗎？(yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("已取消遷移")
            return
    
    # 執行遷移
    print()
    print("=" * 60)
    print("開始遷移...")
    print("=" * 60)
    print()
    
    success_count = 0
    fail_count = 0
    
    for old_path, storage_type, new_path in storages:
        if migrate_storage(old_path, new_path, dry_run=args.dry_run):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("=" * 60)
    print("遷移完成")
    print("=" * 60)
    print(f"成功: {success_count} 個")
    print(f"失敗: {fail_count} 個")
    
    if args.dry_run:
        print()
        print("這是試運行模式，未實際移動任何檔案")
        print("若要實際執行，請移除 --dry-run 參數")


if __name__ == "__main__":
    main()
