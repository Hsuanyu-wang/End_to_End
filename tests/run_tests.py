#!/usr/bin/env python3
"""
測試執行腳本

執行所有單元測試和整合測試
"""

import sys
import os
import unittest

# 加入專案路徑
sys.path.insert(0, '/home/End_to_End_RAG')

def run_all_tests():
    """執行所有測試"""
    print("=" * 70)
    print("開始執行測試套件")
    print("=" * 70)
    
    # 切換到專案根目錄
    os.chdir('/home/End_to_End_RAG')
    
    # 建立測試載入器
    loader = unittest.TestLoader()
    
    # 載入所有測試
    test_suite = unittest.TestSuite()
    
    # 單元測試
    print("\n[1/2] 載入單元測試...")
    unit_tests_dir = 'tests/unit'
    if os.path.exists(unit_tests_dir):
        unit_tests = loader.discover(unit_tests_dir, pattern='test_*.py', top_level_dir='.')
        test_suite.addTests(unit_tests)
        print(f"✅ 已載入單元測試")
    else:
        print(f"⚠️  單元測試目錄不存在: {unit_tests_dir}")
    
    # 整合測試
    print("\n[2/2] 載入整合測試...")
    integration_tests_dir = 'tests/integration'
    if os.path.exists(integration_tests_dir):
        integration_tests = loader.discover(integration_tests_dir, pattern='test_*.py', top_level_dir='.')
        test_suite.addTests(integration_tests)
        print(f"✅ 已載入整合測試")
    else:
        print(f"⚠️  整合測試目錄不存在: {integration_tests_dir}")
    
    # 執行測試
    print("\n" + "=" * 70)
    print("執行測試")
    print("=" * 70 + "\n")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 輸出結果摘要
    print("\n" + "=" * 70)
    print("測試結果摘要")
    print("=" * 70)
    print(f"執行測試數: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"錯誤: {len(result.errors)}")
    print(f"跳過: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有測試通過！")
        return 0
    else:
        print("\n❌ 部分測試失敗")
        return 1


def run_unit_tests_only():
    """只執行單元測試"""
    print("=" * 70)
    print("執行單元測試")
    print("=" * 70 + "\n")
    
    # 切換到專案根目錄
    os.chdir('/home/End_to_End_RAG')
    
    loader = unittest.TestLoader()
    unit_tests_dir = 'tests/unit'
    
    if not os.path.exists(unit_tests_dir):
        print(f"❌ 單元測試目錄不存在: {unit_tests_dir}")
        return 1
    
    test_suite = loader.discover(unit_tests_dir, pattern='test_*.py', top_level_dir='.')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return 0 if result.wasSuccessful() else 1


def run_integration_tests_only():
    """只執行整合測試"""
    print("=" * 70)
    print("執行整合測試")
    print("=" * 70 + "\n")
    
    # 切換到專案根目錄
    os.chdir('/home/End_to_End_RAG')
    
    loader = unittest.TestLoader()
    integration_tests_dir = 'tests/integration'
    
    if not os.path.exists(integration_tests_dir):
        print(f"❌ 整合測試目錄不存在: {integration_tests_dir}")
        return 1
    
    test_suite = loader.discover(integration_tests_dir, pattern='test_*.py', top_level_dir='.')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="執行測試套件")
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["all", "unit", "integration"],
        help="測試類型 (all/unit/integration)"
    )
    
    args = parser.parse_args()
    
    if args.type == "unit":
        exit_code = run_unit_tests_only()
    elif args.type == "integration":
        exit_code = run_integration_tests_only()
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)
