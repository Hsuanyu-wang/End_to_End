"""
自動 QA 生成腳本

此腳本用於從 CSR_full.jsonl 自動生成 Local 和 Global 類型的問答對。
- Local QA: 針對單一維護紀錄的細節問題
- Global QA: 跨紀錄的趨勢分析與模式識別問題

使用方法:
    python scripts/generate_qa.py --local_count 50 --global_count 50
"""

import os
import sys
import json
import argparse
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# 確保可以 import src 內的模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag import QueryParam
from src.config.settings import get_settings
from src.data.processors import data_processing
from src.rag.graph.lightrag import get_lightrag_engine, build_lightrag_index
from src.storage import get_storage_path

import re

def extract_json_from_response(text: str) -> Optional[Dict]:
    """從 LLM 的輸出字串中提取並解析 JSON"""
    try:
        # 1. 嘗試直接解析（最理想狀況）
        return json.loads(text)
    except json.JSONDecodeError:
        # 2. 嘗試使用 Regex 尋找 {} 區塊
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 3. 如果還是失敗，嘗試清理常見問題（如 Markdown 標籤或尾隨逗號）
                clean_str = json_str.replace('```json', '').replace('```', '').strip()
                try:
                    return json.loads(clean_str)
                except:
                    return None
    return None


def check_lightrag_cache(storage_path: str) -> bool:
    """
    檢查 LightRAG cache 是否存在且完整
    
    Args:
        storage_path: LightRAG storage 路徑
    
    Returns:
        True 如果 cache 存在且完整，否則 False
    """
    if not os.path.exists(storage_path):
        return False
    
    # 檢查關鍵檔案是否存在
    required_files = [
        "kv_store_full_docs.json",
        "graph_chunk_entity_relation.graphml"
    ]
    
    for filename in required_files:
        file_path = os.path.join(storage_path, filename)
        if not os.path.exists(file_path):
            return False
    
    return True


def setup_or_load_lightrag(settings, force_rebuild: bool = False):
    """
    根據 cache 狀態決定載入或建構 LightRAG
    
    Args:
        settings: 設定物件
        force_rebuild: 是否強制重建索引
    
    Returns:
        LightRAG 實例
    """
    storage_path = get_storage_path(
        storage_type="lightrag",
        data_type="GEN",
        method="qa_generation"
    )
    cache_exists = check_lightrag_cache(storage_path)
    if cache_exists and not force_rebuild:
        print("✅ 使用現有 LightRAG 索引")
        print(f"📂 索引路徑: {storage_path}")
        rag = get_lightrag_engine(
            settings,
            data_type="GEN",
            sup="qa_generation",
            fast_test=False,
            mode=""
        )
    else:
        if force_rebuild:
            print("🔨 強制重建 LightRAG 索引")
        else:
            print("🔨 建構新的 LightRAG 索引（首次執行或 cache 不完整）")
        print(f"📂 索引路徑: {storage_path}")
        build_lightrag_index(
            settings,
            mode="natural_text",
            data_type="GEN",
            sup="qa_generation",
            fast_build=False,
            lightrag_mode=""
        )
        rag = get_lightrag_engine(
            settings,
            data_type="GEN",
            sup="qa_generation",
            fast_test=False,
            mode=""
        )
    return rag


def load_csr_records(settings) -> List[Dict[str, Any]]:
    """
    載入 CSR 原始紀錄
    
    Args:
        settings: 設定物件
    
    Returns:
        CSR 紀錄列表
    """
    raw_file_path = settings.data_config.raw_file_path_GEN
    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"找不到資料檔案: {raw_file_path}")
    
    records = []
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
    print(f"📊 載入 {len(records)} 筆 CSR 紀錄")
    return records


def generate_local_qa_improved(records, count, settings):
    qa_pairs = []
    total = min(count, len(records))
    if total == 0:
        return []
    sampled_records = random.sample(records, total)
    for idx, record in enumerate(sampled_records):
        prompt = f"""
你是一個資安與IT維護專家。請根據以下的維護紀錄，生成一個具體的技術問題以及詳細解答。

【維護紀錄】
單號: {record.get('NO')}
客戶: {record.get('Customer')}
描述: {record.get('Description')}
處置: {record.get('Action')}

請以 JSON 格式輸出：
{{
    "question": "針對該處置細節的具體問題",
    "answer": "基於上述紀錄的詳細解答"
}}
"""
        response = settings.builder_llm.complete(prompt)
        try:
            qa_json = json.loads(response.text)
        except Exception:
            # 若不是合法 JSON，嘗試進行自動修復
            try:
                qa_json = json.loads(response.text.replace("'", '"'))
                
            except Exception:
                print(f"⚠️  Local QA第{idx + 1}筆 LLM 輸出解析失敗，略過。")
                continue
        qa_pair = {
            "id": f"qa_local_{idx+1:08d}",
            "question": qa_json.get("question", ""),
            "answer": qa_json.get("answer", ""),
            "qa_type": "local",
            "source_doc_ids": [str(record.get("NO"))],
            "confidence": 0.95
        }
        qa_pairs.append(qa_pair)
        if (idx + 1) % 10 == 0:
            print(f"  已生成 {len(qa_pairs)}/{total} 組 Local QA")
    print(f"✅ Local QA 生成完成，共 {len(qa_pairs)} 組")
    return qa_pairs


def generate_global_qa_grouped(records, count, settings):
    from collections import defaultdict

    # 只產生 up to count 個 global QA，優先按客戶分組
    customer_groups = defaultdict(list)
    for r in records:
        customer = r.get('Customer')
        if customer:
            customer_groups[customer].append(r)
    qa_pairs = []
    group_customers = [c for c, recs in customer_groups.items() if len(recs) >= 3]
    random.shuffle(group_customers)
    idx = 0
    for customer in group_customers:
        if idx >= count:
            break
        group_records = customer_groups[customer]
        # 若群組資料量過大，亂數擷取10~20筆摘要
        group_size = len(group_records)
        if group_size > 20:
            context_records = random.sample(group_records, min(20, group_size))
        else:
            context_records = group_records
        context = ""
        src_ids = []
        for r in context_records:
            context += f"單號:{r.get('NO')} | 描述:{r.get('Description')} | 處置摘要:{str(r.get('Action', ''))[:100]}\n"
            src_ids.append(str(r.get('NO')))
        prompt = f"""
以下是客戶 {customer} 在一段期間內的多筆維護紀錄：
{context}

請進行跨紀錄的綜合分析，生成一個「趨勢型」或「共通問題型」的問答對。
例如：該客戶最常遭遇哪類設備異常？或該客戶的系統升級趨勢為何？

請以 JSON 格式輸出：
{{
    "question": "跨紀錄的綜合分析問題",
    "answer": "基於上述多筆資料的總結分析解答",
    "source_ids": ["單號A", "單號B", "單號C"] // 必須是你實際用來總結出這個答案的單號
}}
"""
        response = settings.builder_llm.complete(prompt)
        try:
            qa_json = json.loads(response.text)
        except Exception:
            try:
                qa_json = json.loads(response.text.replace("'", '"'))
            except Exception:
                print(f"⚠️  Global QA for {customer} 解析失敗，略過。")
                continue
        # 從 LLM 回傳內容 parse 出 source_ids，fallback 使用 context 中選擇的 src_ids
        ids_from_llm = qa_json.get('source_ids')
        if isinstance(ids_from_llm, list) and len(ids_from_llm) > 0:
            used_source_ids = ids_from_llm
        else:
            used_source_ids = src_ids
        qa_pair = {
            "id": f"qa_global_{idx+1:08d}",
            "question": qa_json.get("question", ""),
            "answer": qa_json.get("answer", ""),
            "qa_type": "global",
            "source_doc_ids": used_source_ids,
            "analysis": f"群組: {customer}，以其多筆紀錄 cross-record 統整自動生成。",
            "confidence": 0.97
        }
        qa_pairs.append(qa_pair)
        idx += 1
        if idx % 5 == 0:
            print(f"  已生成 {idx}/{count} 組 Global QA")
    # 若還不足 count，可補充隨機分組 or fallback
    print(f"✅ Global QA 生成完成，共 {len(qa_pairs)} 組")
    return qa_pairs


def save_qa_jsonl(qa_pairs: List[Dict[str, Any]], output_path: str) -> None:
    """
    儲存 QA 對為 JSONL 格式
    
    Args:
        qa_pairs: QA 對列表
        output_path: 輸出檔案路徑
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    print(f"💾 已儲存至: {output_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="自動生成 Local 和 Global QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 生成各 50 組 Local 和 Global QA
  python scripts/generate_qa.py --local_count 50 --global_count 50
  
  # 強制重建 LightRAG 索引
  python scripts/generate_qa.py --local_count 30 --global_count 30 --force_rebuild
  
  # 指定輸出目錄
  python scripts/generate_qa.py --local_count 100 --global_count 100 --output_dir ./output
        """
    )
    parser.add_argument(
        '--local_count',
        type=int,
        default=50,
        help='要生成的 Local QA 數量（預設: 50）'
    )
    parser.add_argument(
        '--global_count',
        type=int,
        default=50,
        help='要生成的 Global QA 數量（預設: 50）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/End_to_End_RAG/Data/generated_QA',
        help='輸出目錄路徑（預設: /home/End_to_End_RAG/Data/generated_QA）'
    )
    parser.add_argument(
        '--force_rebuild',
        action='store_true',
        help='強制重建 LightRAG 索引'
    )
    args = parser.parse_args()

    # 輸出執行資訊
    print("="*80)
    print("🚀 自動 QA 生成腳本")
    print("="*80)
    print(f"📊 目標數量: Local QA = {args.local_count}, Global QA = {args.global_count}")
    print(f"📂 輸出目錄: {args.output_dir}")
    print(f"🔧 強制重建: {'是' if args.force_rebuild else '否'}")
    print("="*80)

    try:
        # 1. 載入設定
        print("\n[步驟 1/5] 載入專案設定...")
        settings = get_settings()
        print("✅ 設定載入完成")
        
        # 2. 設定或載入 LightRAG
        print("\n[步驟 2/5] 設定 LightRAG 引擎...")
        rag = setup_or_load_lightrag(settings, force_rebuild=args.force_rebuild)
        print("✅ LightRAG 引擎就緒")
        
        # 3. 載入 CSR 紀錄
        print("\n[步驟 3/5] 載入 CSR 紀錄...")
        records = load_csr_records(settings)
        print("✅ CSR 紀錄載入完成")
        
        # 4. 生成 QA
        print("\n[步驟 4/5] 生成 QA 對...")
        local_qa = []
        global_qa = []
        if args.local_count > 0:
            local_qa = generate_local_qa_improved(records, args.local_count, settings)
        if args.global_count > 0:
            global_qa = generate_global_qa_grouped(records, args.global_count, settings)
        print(f"\n✅ QA 生成完成，共 {len(local_qa) + len(global_qa)} 組")
        
        # 5. 儲存結果
        print("\n[步驟 5/5] 儲存結果...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if local_qa:
            local_output = output_dir / f"qa_local_{len(local_qa)}.jsonl"
            save_qa_jsonl(local_qa, str(local_output))
        if global_qa:
            global_output = output_dir / f"qa_global_{len(global_qa)}.jsonl"
            save_qa_jsonl(global_qa, str(global_output))
        # 輸出摘要
        print("\n" + "="*80)
        print("✅ 所有任務完成！")
        print("="*80)
        print(f"📊 生成摘要:")
        print(f"  - Local QA:  {len(local_qa)} 組")
        print(f"  - Global QA: {len(global_qa)} 組")
        print(f"  - 總計:      {len(local_qa) + len(global_qa)} 組")
        print("="*80)
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 執行過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
