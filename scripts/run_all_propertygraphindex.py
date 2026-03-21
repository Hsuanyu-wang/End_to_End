import itertools
import subprocess

# 僅重跑與 "text2cypher" 有關的 retriever 組合

# 定義所有的基礎選項
extractors_base = ["implicit", "schema", "simple", "dynamic"]
retrievers_base = ["vector", "synonym", "text2cypher"]

# 定義一個函數來產生所有的非空組合 (Power set)
def get_all_combinations(items):
    combinations = []
    # 從長度 1 到 全部數量 進行排列組合
    for r in range(1, len(items) + 1):
        for subset in itertools.combinations(items, r):
            # 將組合用逗號連接，例如 ('implicit', 'schema') 變成 'implicit,schema'
            combinations.append(",".join(subset))
    return combinations

# 取得所有的參數組合
all_extractors = get_all_combinations(extractors_base)
all_retrievers = get_all_combinations(retrievers_base)

# 篩選只含有 "text2cypher" 的 retriever 組合
text2cypher_retrievers = [r for r in all_retrievers if "text2cypher" in r.split(",")]

# 計算總共要跑多少次
total_runs = len(all_extractors) * len(text2cypher_retrievers)
print(f"總共將執行 {total_runs} 種組合 (Extractors: {len(all_extractors)} 種, Retrievers: {len(text2cypher_retrievers)} 種, 僅限含 text2cypher)\n")
print("-" * 50)

# 開始雙層迴圈迭代執行
run_count = 1
for ext in all_extractors:
    for ret in text2cypher_retrievers:
        print(f"[{run_count}/{total_runs}] 正在重跑組合:")
        print(f"Extractors: {ext}")
        print(f"Retrievers: {ret}")
        
        # 組裝指令
        cmd = [
            "python", "scripts/run_evaluation.py",
            "--unified_graph_type", "property_graph",
            "--pg_extractors", ext,
            "--pg_retrievers", ret,
            "--pg_combination_mode", "ensemble",
            "--data_type", "DI"
        ]
        
        # 執行指令 (這會將輸出直接印在你的 terminal)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"執行失敗，退出代碼: {e.returncode}。跳至下一個組合...")
            
        print("-" * 50)
        run_count += 1

print("所有含 text2cypher 的實驗組合執行完畢！")