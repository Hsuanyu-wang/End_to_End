#!/usr/bin/env python3
"""
全面性 RAG 測試/實驗執行腳本

提供互動式介面，讓使用者選擇要執行的測試或實驗
支援快速測試模式、Schema Cache、Pipeline 元件和 Ablation Study
"""

import sys
import os
import subprocess
from typing import List, Dict, Tuple
from datetime import datetime

# 定義所有可用的測試配置
TEST_CONFIGS = {
    "vector": {
        "hybrid": {
            "name": "Vector Hybrid RAG",
            "args": ["--vector_method", "hybrid"]
        },
        "vector": {
            "name": "Vector Only RAG",
            "args": ["--vector_method", "vector"]
        },
        "bm25": {
            "name": "BM25 RAG",
            "args": ["--vector_method", "bm25"]
        }
    },
    "advanced_vector": {
        "parent_child": {
            "name": "Parent-Child RAG",
            "args": ["--adv_vector_method", "parent_child"]
        },
        "self_query": {
            "name": "Self-Query RAG",
            "args": ["--adv_vector_method", "self_query"]
        }
    },
    "graph": {
        "propertyindex": {
            "name": "Property Graph RAG",
            "args": ["--graph_rag_method", "propertyindex"]
        },
        "dynamic_schema": {
            "name": "Dynamic Schema Graph RAG",
            "args": ["--graph_rag_method", "dynamic_schema"]
        },
        "autoschema": {
            "name": "AutoSchemaKG",
            "args": ["--graph_rag_method", "autoschema"]
        }
    },
    "lightrag_schema": {
        "lightrag_default": {
            "name": "LightRAG (Default Schema)",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "hybrid", "--lightrag_schema_method", "lightrag_default"],
            "use_cache": False  # 預設 schema 不需要快取
        },
        "iterative_evolution": {
            "name": "LightRAG (Iterative Evolution Schema)",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "hybrid", "--lightrag_schema_method", "iterative_evolution"],
            "use_cache": True  # 支援 Schema Cache
        },
        "llm_dynamic": {
            "name": "LightRAG (LLM Dynamic Schema)",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "hybrid", "--lightrag_schema_method", "llm_dynamic"],
            "use_cache": True  # 支援 Schema Cache
        },
        "autoschema_cached": {
            "name": "LightRAG (AutoSchemaKG with Cache)",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "hybrid", "--lightrag_schema_method", "autoschema"],
            "use_cache": True  # 支援 Schema Cache
        },
        "llamaindex_dynamic": {
            "name": "LightRAG (LlamaIndex Dynamic Schema)",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "hybrid", "--lightrag_schema_method", "llamaindex_dynamic"],
            "use_cache": True  # 支援 Schema Cache
        }
    },
    "lightrag_mode": {
        "local": {
            "name": "LightRAG Local Mode",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "local", "--lightrag_schema_method", "lightrag_default"]
        },
        "global": {
            "name": "LightRAG Global Mode",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "global", "--lightrag_schema_method", "lightrag_default"]
        },
        "hybrid": {
            "name": "LightRAG Hybrid Mode",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "hybrid", "--lightrag_schema_method", "lightrag_default"]
        },
        "mix": {
            "name": "LightRAG Mix Mode",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "mix", "--lightrag_schema_method", "lightrag_default"]
        },
        "naive": {
            "name": "LightRAG Naive Mode",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "naive", "--lightrag_schema_method", "lightrag_default"]
        },
        "bypass": {
            "name": "LightRAG Bypass Mode",
            "args": ["--graph_rag_method", "lightrag", "--lightrag_mode", "bypass", "--lightrag_schema_method", "lightrag_default"]
        }
    },
    "modular_combo": {
        "autoschema_lightrag": {
            "name": "AutoSchemaKG + LightRAG Retriever",
            "args": ["--graph_preset", "autoschema_lightrag"]
        },
        "lightrag_csr": {
            "name": "LightRAG Builder + CSR Retriever",
            "args": ["--graph_preset", "lightrag_csr"]
        },
        "dynamic_csr": {
            "name": "DynamicSchema Builder + CSR Retriever",
            "args": ["--graph_preset", "dynamic_csr"]
        },
        "dynamic_lightrag": {
            "name": "DynamicSchema Builder + LightRAG Retriever",
            "args": ["--graph_preset", "dynamic_lightrag"]
        }
    },
    "advanced_retrieval": {
        "k_hop": {
            "name": "LightRAG + K-Hop Traversal",
            "args": ["--graph_type", "lightrag", "--lightrag_mode", "hybrid", "--graph_retrieval", "k_hop", "--graph_hop_k", "1"],
            "description": "可控 hop 的實體關聯遍歷"
        },
        "ppr": {
            "name": "LightRAG + PPR Traversal",
            "args": [
                "--graph_type", "lightrag",
                "--lightrag_mode", "hybrid",
                "--graph_retrieval", "ppr",
                "--ppr_alpha", "0.85",
                "--ppr_weight_mode", "semantic"
            ],
            "description": "PPR 走訪策略"
        },
        "pcst": {
            "name": "LightRAG + PCST Traversal",
            "args": [
                "--graph_type", "lightrag",
                "--lightrag_mode", "hybrid",
                "--graph_retrieval", "pcst",
                "--pcst_cost_mode", "inverse_weight"
            ],
            "description": "PCST 走訪策略"
        },
        "tog": {
            "name": "LightRAG + ToG Traversal",
            "args": [
                "--graph_type", "lightrag",
                "--lightrag_mode", "hybrid",
                "--graph_retrieval", "tog",
                "--tog_max_iterations", "2",
                "--tog_beam_width", "3"
            ],
            "description": "Think-on-Graph 迭代式檢索"
        }
    },
    "ablation_study": {
        "token_budget_hybrid": {
            "name": "Token Budget 實驗（基準：Vector Hybrid）",
            "args": [
                "--vector_method", "hybrid",
                "--graph_type", "lightrag",
                "--lightrag_mode", "hybrid",
                "--graph_retrieval", "native",
                "--enable_token_budget",
                "--token_budget_baseline", "vector_hybrid"
            ]
        },
        "token_budget_vector": {
            "name": "Token Budget 實驗（基準：Vector）",
            "args": [
                "--vector_method", "vector",
                "--graph_type", "lightrag",
                "--lightrag_mode", "hybrid",
                "--graph_retrieval", "native",
                "--enable_token_budget",
                "--token_budget_baseline", "vector_vector"
            ]
        },
        "token_budget_bm25": {
            "name": "Token Budget 實驗（基準：BM25）",
            "args": [
                "--vector_method", "bm25",
                "--graph_type", "lightrag",
                "--lightrag_mode", "hybrid",
                "--graph_retrieval", "native",
                "--enable_token_budget",
                "--token_budget_baseline", "vector_bm25"
            ]
        },
        "token_budget_tog": {
            "name": "Token Budget 實驗（基準：Vector Hybrid，+ ToG）",
            "args": [
                "--vector_method", "hybrid",
                "--graph_type", "lightrag",
                "--lightrag_mode", "hybrid",
                "--graph_retrieval", "tog",
                "--tog_max_iterations", "1",
                "--tog_beam_width", "2",
                "--enable_token_budget",
                "--token_budget_baseline", "vector_hybrid"
            ]
        }
    }
}


def clear_screen():
    """清除螢幕"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """印出標題"""
    print("=" * 60)
    print("RAG 測試/實驗執行腳本")
    print("=" * 60)
    print()


def display_main_menu() -> str:
    """顯示主選單並取得使用者選擇"""
    print("請選擇測試類型：")
    print("  1. Vector RAG 測試 (3 種方法)")
    print("  2. Advanced Vector RAG 測試 (2 種方法)")
    print("  3. Graph RAG 測試 (3 種方法)")
    print("  4. LightRAG Schema 方法測試 (4 種 schema) ✨ 支援 Cache")
    print("  5. LightRAG 檢索模式測試 (6 種模式)")
    print("  6. 模組化組合測試 (4 種組合)")
    print("  7. Graph Traversal 策略測試 (4 種方法)")
    print("  8. Token Budget 實驗 (4 種配置)")
    print("  9. 完整實驗 (所有方法)")
    print(" 10. 自訂選擇")
    print(" 11. Schema Cache 管理工具")
    print("  0. 退出")
    print()
    
    choice = input("請輸入選項 (0-11): ").strip()
    return choice


def display_category_menu(category: str, configs: Dict) -> List[str]:
    """顯示分類選單並取得使用者選擇"""
    category_names = {
        "vector": "Vector RAG",
        "advanced_vector": "Advanced Vector RAG",
        "graph": "Graph RAG",
        "lightrag_schema": "LightRAG Schema 方法 (支援 Cache)",
        "lightrag_mode": "LightRAG 檢索模式",
        "modular_combo": "模組化組合",
        "advanced_retrieval": "Graph Traversal 策略",
        "ablation_study": "Token Budget 實驗"
    }
    
    print(f"\n{category_names.get(category, category)} 測試選項：")
    keys = list(configs.keys())
    for i, key in enumerate(keys, 1):
        name = configs[key]['name']
        desc = configs[key].get('description', '')
        if desc:
            print(f"  {i}. {name} - {desc}")
        else:
            print(f"  {i}. {name}")
    print(f"  {len(keys) + 1}. 全部執行")
    print("  0. 返回主選單")
    print()
    
    choice = input(f"請選擇 (0-{len(keys) + 1})，可用逗號分隔多個選項: ").strip()
    
    if choice == "0":
        return []
    
    if choice == str(len(keys) + 1):
        return keys
    
    # 處理多個選擇
    selections = []
    for c in choice.split(","):
        c = c.strip()
        if c.isdigit():
            idx = int(c) - 1
            if 0 <= idx < len(keys):
                selections.append(keys[idx])
    
    return selections


def ask_test_mode() -> bool:
    """詢問測試模式"""
    print("\n選擇測試模式：")
    print("  1. 快速測試 (--qa_dataset_fast_test, 僅測試前 2 題)")
    print("  2. 完整測試 (使用完整資料集)")
    print()
    
    choice = input("請選擇 (1-2): ").strip()
    return choice == "1"


def ask_data_type() -> str:
    """詢問資料類型"""
    print("\n選擇資料類型：")
    print("  1. DI (Domain-specific Inquiries)")
    print("  2. GEN (Generate)")
    print("  3. LIGHTRAG_CS")
    print("  4. LIGHTRAG_AGRICULTURE")
    print("  5. LIGHTRAG_LEGAL")
    print("  6. LIGHTRAG_MIX")
    print()
    
    choice = input("請選擇 (1-6，預設 1): ").strip()
    mapping = {
        "1": "DI",
        "2": "GEN",
        "3": "LIGHTRAG_CS",
        "4": "LIGHTRAG_AGRICULTURE",
        "5": "LIGHTRAG_LEGAL",
        "6": "LIGHTRAG_MIX",
    }
    return mapping.get(choice, "DI")


def ask_use_cache() -> bool:
    """詢問是否使用 Schema Cache"""
    print("\n使用 Schema Cache？")
    print("  1. 是 (啟用快取，加速重複實驗)")
    print("  2. 否 (強制重建 Schema)")
    print()
    
    choice = input("請選擇 (1-2，預設 1): ").strip()
    return choice != "2"


def ask_force_rebuild() -> bool:
    """詢問是否強制重建"""
    print("\n強制重建 Schema？")
    print("  1. 否 (使用快取)")
    print("  2. 是 (忽略快取，重新生成)")
    print()
    
    choice = input("請選擇 (1-2，預設 1): ").strip()
    return choice == "2"


def build_command(test_config: Dict, fast_test: bool, data_type: str) -> List[str]:
    """建立執行命令"""
    base_cmd = ["python", "scripts/run_evaluation.py"]
    
    # 加入測試配置的參數
    base_cmd.extend(test_config["args"])
    
    # 加入資料類型
    base_cmd.extend(["--data_type", data_type])
    
    # 加入快速測試參數
    if fast_test:
        base_cmd.append("--qa_dataset_fast_test")

    return base_cmd


def run_test(test_name: str, command: List[str], test_num: int, total_tests: int) -> Tuple[bool, float]:
    """執行單一測試"""
    print("\n" + "=" * 60)
    print(f"測試 {test_num}/{total_tests}: {test_name}")
    print("=" * 60)
    print(f"執行命令: {' '.join(command)}")
    print()
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            command,
            cwd="/home/End_to_End_RAG",
            check=True,
            capture_output=False
        )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"\n✅ {test_name} 測試成功 (耗時: {elapsed:.1f} 秒)")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"\n❌ {test_name} 測試失敗 (耗時: {elapsed:.1f} 秒)")
        print(f"錯誤: {e}")
        return False, elapsed


def run_tests_from_selections(selections: List[Tuple[str, Dict]], fast_test: bool, data_type: str,
                             use_cache: bool = True, force_rebuild: bool = False):
    """執行選擇的測試"""
    if not selections:
        print("\n沒有選擇任何測試。")
        return
    
    total_tests = len(selections)
    results = []
    
    # 檢查是否有測試支援 Schema Cache
    has_cache_support = any(config.get("use_cache", False) for _, config in selections)
    
    print(f"\n將執行 {total_tests} 個測試")
    print(f"測試模式: {'快速測試' if fast_test else '完整測試'}")
    print(f"資料類型: {data_type}")
    if has_cache_support:
        print(f"Schema Cache: {'啟用' if use_cache else '停用'}")
        if force_rebuild:
            print(f"強制重建: 是")
    input("\n按 Enter 開始執行...")

    # Schema Cache 相關參數在 run_evaluation.py 目前不支援；
    # 因此用「清理快取檔案」來達到「不用快取 / 強制重建」的效果。
    if has_cache_support and (not use_cache or force_rebuild):
        methods_to_clean = set()
        for _, cfg in selections:
            if not cfg.get("use_cache", False):
                continue
            args = cfg.get("args", [])
            for idx, token in enumerate(args):
                if token in ("--lightrag_schema_method", "--schema_method") and idx + 1 < len(args):
                    method = args[idx + 1]
                    if method and method != "lightrag_default":
                        methods_to_clean.add(method)

        if methods_to_clean:
            print("\n🗑️  開始清理 Schema Cache（用於重新生成 Schema）")
            for m in sorted(methods_to_clean):
                print(f"  - 清理方法：{m}")
                subprocess.run(
                    ["python", "scripts/manage_schema_cache.py", "--clean", "--method", m],
                    cwd="/home/End_to_End_RAG",
                    check=True,
                    capture_output=False,
                )
        else:
            print("\n未偵測到需清理的 schema 方法，將直接執行。")
    
    for i, (test_name, test_config) in enumerate(selections, 1):
        command = build_command(test_config, fast_test, data_type)
        success, elapsed = run_test(test_name, command, i, total_tests)
        results.append({
            "name": test_name,
            "success": success,
            "time": elapsed
        })
    
    # 顯示總結
    print("\n" + "=" * 60)
    print("測試總結")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r["success"])
    total_time = sum(r["time"] for r in results)
    
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} {r['name']}: {r['time']:.1f} 秒")
    
    print(f"\n成功: {success_count}/{total_tests}")
    print(f"總耗時: {total_time:.1f} 秒 ({total_time / 60:.1f} 分鐘)")


def handle_category_selection(category: str):
    """處理分類選擇"""
    configs = TEST_CONFIGS[category]
    selections = display_category_menu(category, configs)
    
    if not selections:
        return
    
    fast_test = ask_test_mode()
    data_type = ask_data_type()
    
    # 檢查是否有測試支援 Schema Cache
    has_cache_support = any(configs[key].get("use_cache", False) for key in selections)
    use_cache = True
    force_rebuild = False
    
    if has_cache_support:
        use_cache = ask_use_cache()
        if use_cache:
            force_rebuild = ask_force_rebuild()
    
    # 建立測試列表
    tests = [(configs[key]["name"], configs[key]) for key in selections]
    
    run_tests_from_selections(tests, fast_test, data_type, use_cache, force_rebuild)


def handle_all_tests():
    """處理執行所有測試"""
    print("\n⚠️  警告：執行所有測試可能需要數小時！")
    confirm = input("確定要執行所有測試？(yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("已取消。")
        return
    
    fast_test = ask_test_mode()
    data_type = ask_data_type()
    
    # 收集所有測試
    all_tests = []
    for category, configs in TEST_CONFIGS.items():
        for key, config in configs.items():
            all_tests.append((config["name"], config))
    
    run_tests_from_selections(all_tests, fast_test, data_type)


def handle_schema_cache_management():
    """處理 Schema Cache 管理"""
    print("\nSchema Cache 管理工具")
    print("=" * 60)
    print("  1. 列出所有快取")
    print("  2. 查看快取統計")
    print("  3. 清理快取（選擇方法）")
    print("  4. 清理所有快取")
    print("  5. 匯出快取報告")
    print("  0. 返回")
    print()
    
    choice = input("請選擇 (0-5): ").strip()
    
    if choice == "0":
        return
    elif choice == "1":
        subprocess.run(["python", "scripts/manage_schema_cache.py", "--list"], 
                      cwd="/home/End_to_End_RAG")
    elif choice == "2":
        subprocess.run(["python", "scripts/manage_schema_cache.py", "--info"], 
                      cwd="/home/End_to_End_RAG")
    elif choice == "3":
        method = input("輸入要清理的方法名稱 (iterative_evolution/llm_dynamic/autoschema): ").strip()
        if method:
            subprocess.run(["python", "scripts/manage_schema_cache.py", "--clean", "--method", method], 
                          cwd="/home/End_to_End_RAG")
    elif choice == "4":
        confirm = input("確定要清理所有快取？(yes/no): ").strip().lower()
        if confirm == "yes":
            subprocess.run(["python", "scripts/manage_schema_cache.py", "--clean"], 
                          cwd="/home/End_to_End_RAG")
    elif choice == "5":
        output = input("輸出檔案名稱 (預設: schema_report.json): ").strip()
        if not output:
            output = "schema_report.json"
        subprocess.run(["python", "scripts/manage_schema_cache.py", "--export", "--output", output], 
                      cwd="/home/End_to_End_RAG")
    
    input("\n按 Enter 繼續...")


def handle_custom_selection():
    """處理自訂選擇"""
    print("\n自訂測試選擇")
    print("=" * 60)
    
    selected_tests = []
    
    # 逐類別選擇
    for category, configs in TEST_CONFIGS.items():
        category_names = {
            "vector": "Vector RAG",
            "advanced_vector": "Advanced Vector RAG",
            "graph": "Graph RAG",
            "lightrag_schema": "LightRAG Schema 方法",
            "lightrag_mode": "LightRAG 檢索模式",
            "modular_combo": "模組化組合",
            "advanced_retrieval": "Graph Traversal 策略",
            "ablation_study": "Token Budget 實驗"
        }
        
        print(f"\n{category_names.get(category, category)}:")
        for key, config in configs.items():
            choice = input(f"  包含 {config['name']}? (y/n): ").strip().lower()
            if choice == 'y':
                selected_tests.append((config["name"], config))
    
    if not selected_tests:
        print("\n沒有選擇任何測試。")
        return
    
    fast_test = ask_test_mode()
    data_type = ask_data_type()
    
    # 檢查是否有測試支援 Schema Cache
    has_cache_support = any(config.get("use_cache", False) for _, config in selected_tests)
    use_cache = True
    force_rebuild = False
    
    if has_cache_support:
        use_cache = ask_use_cache()
        if use_cache:
            force_rebuild = ask_force_rebuild()
    
    run_tests_from_selections(selected_tests, fast_test, data_type, use_cache, force_rebuild)


def main():
    """主程式"""
    while True:
        clear_screen()
        print_header()
        
        choice = display_main_menu()
        
        if choice == "0":
            print("\n再見！")
            break
        elif choice == "1":
            handle_category_selection("vector")
        elif choice == "2":
            handle_category_selection("advanced_vector")
        elif choice == "3":
            handle_category_selection("graph")
        elif choice == "4":
            handle_category_selection("lightrag_schema")
        elif choice == "5":
            handle_category_selection("lightrag_mode")
        elif choice == "6":
            handle_category_selection("modular_combo")
        elif choice == "7":
            handle_category_selection("advanced_retrieval")
        elif choice == "8":
            handle_category_selection("ablation_study")
        elif choice == "9":
            handle_all_tests()
        elif choice == "10":
            handle_custom_selection()
        elif choice == "11":
            handle_schema_cache_management()
        else:
            print("\n無效的選項，請重新選擇。")
            input("按 Enter 繼續...")
            continue
        
        input("\n按 Enter 返回主選單...")


if __name__ == "__main__":
    main()
