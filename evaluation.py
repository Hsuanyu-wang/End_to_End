import os
import json
import time
from llama_index.core.evaluation.retrieval.metrics import Recall
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator
from llama_index.core import Settings
import nest_asyncio
import evaluate
from advanced_vector_package import get_self_query_engine, get_parent_child_query_engine
# from ms_graphrag_package import MSGraphRAGWrapper

nest_asyncio.apply()

# ==========================================
# 1. 資料正規化 (Data Normalization)
# ==========================================
def load_and_normalize_qa_CSR_DI(csv_path: str) -> List[Dict[str, Any]]:
    """將不同格式的 QA 資料集統整為標準格式"""
    normalized_data = []
    
    # 處理 QA_43.csv (欄位：Q, GT, REF)
    df_csv = pd.read_csv(csv_path)
    for _, row in df_csv.iterrows():
        ref_id = str(row["REF"]).strip() if pd.notna(row["REF"]) else ""
        normalized_data.append({
            "source": "QA_43_clean.csv",
            "query": row["Q"],
            "ground_truth_answer": row["GT"],
            "ground_truth_doc_ids": [ref_id] if ref_id else []
        })
    return normalized_data

def load_and_normalize_qa_CSR_full(jsonl_path: str) -> List[Dict[str, Any]]:
    # 處理 qa_global_group_100.jsonl (欄位：question, answer, source_doc_ids)
    normalized_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            normalized_data.append({
                "source": "qa_global_group_100.jsonl",
                "query": data.get("question", ""),
                "ground_truth_answer": data.get("answer", ""),
                "ground_truth_doc_ids": data.get("source_doc_ids", [])
            })

    return normalized_data

# ==========================================
# 2. 檢索方法封裝 (Retriever Wrappers)
# ==========================================
class RAGPipelineWrapper:
    """
    統一的 RAG 封裝介面，抹平 Vector 與 Graph 的底層差異。
    """
    def __init__(self, name: str, query_engine):
        self.name = name
        self.query_engine = query_engine

    async def aquery_and_log(self, user_query: str) -> Dict[str, Any]: 
        # 紀錄開始時間
        start_time = time.time()
        
        generated_answer = ""
        retrieved_contexts = []
        retrieved_ids = []
        source_nodes = []
    
        try:
            response = await self.query_engine.aquery(user_query)
            generated_answer = str(response)
            
            retrieved_contexts = []
            retrieved_ids = []
            source_nodes = response.source_nodes
            
            for node in source_nodes:
                retrieved_contexts.append(node.get_content())
                doc_id = node.metadata.get("NO", None) 
                retrieved_ids.append(str(doc_id))
                
        except Exception as e:
            print(f"❌ {self.name} 查詢發生錯誤: {e}")
            generated_answer = f"Error: {e}"
        
        # 紀錄結束時間並計算耗時
        end_time = time.time()
        execution_time_sec = round(end_time - start_time, 4)
        
        # 提取生成的答案
        generated_answer = str(response)
        
        # 提取檢索到的 Context 與 Document IDs
        retrieved_contexts = []
        retrieved_ids = []
        
        for node in response.source_nodes:
            retrieved_contexts.append(node.get_content())
            doc_id = node.metadata.get("NO", None) 
            retrieved_ids.append(str(doc_id))
            
        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": retrieved_ids,
            "source_nodes": response.source_nodes,
            "execution_time_sec": execution_time_sec
        }

    def query_and_log(self, user_query: str) -> Dict[str, Any]:
        # 紀錄開始時間
        start_time = time.time()
        
        # 執行檢索與生成
        response = self.query_engine.query(user_query)
        
        # 紀錄結束時間並計算耗時
        end_time = time.time()
        execution_time_sec = round(end_time - start_time, 4)
        
        # 提取生成的答案
        generated_answer = str(response)
        
        # 提取檢索到的 Context 與 Document IDs
        retrieved_contexts = []
        retrieved_ids = []
        
        for node in response.source_nodes:
            retrieved_contexts.append(node.get_content())
            doc_id = node.metadata.get("id", node.node_id) 
            retrieved_ids.append(str(doc_id))
            
        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": retrieved_ids,
            "source_nodes": response.source_nodes,
            "execution_time_sec": execution_time_sec # 新增：檢索與生成耗時
        }

class LightRAGWrapper:
    def __init__(self, name: str, rag_instance, mode="hybrid"):
        self.name = name
        self.rag = rag_instance
        self.mode = mode
        self._initialized = False

    def query(self, question: str) -> str:
        # 呼叫 LightRAG 的 query 方法
        from lightrag.lightrag import QueryParam
        return self.rag.query(question, param=QueryParam(mode=self.mode))

    async def aquery_and_log(self, user_query: str) -> Dict[str, Any]:
        from lightrag.lightrag import QueryParam
        import inspect
        start_time = time.time()
        Settings = get_settings(model_type=args.model_type)
        try:
            # [新增] 解決 LightRAG 新版在 aquery 時發生 AttributeError: __aenter__ 的鎖定問題
            if not self._initialized and hasattr(self.rag, "initialize_storages"):
                await self.rag.initialize_storages()
                self._initialized = True
            # [METHOD: ROUTER]
            # async def route_lightrag_mode(query: str, llm) -> str:
            #     prompt = (
            #         f"問題：'{query}'\n"
            #         "若問題針對「特定實體、人物、具體細節」請回答 'local'。\n"
            #         "若問題針對「整體趨勢、跨文件總結、宏觀主題」請回答 'global'。\n"
            #         "只回答 'local' 或 'global'。"
            #     )
            #     response = await llm.acomplete(prompt)
            #     return "global" if "global" in response.text.lower() else "local"

            # # 在 Pipeline 呼叫時：
            # dynamic_mode = await route_lightrag_mode(user_query, Settings.llm)
            # raw_context = await self.rag.aquery_data(user_query, param=QueryParam(mode=dynamic_mode, only_need_context=True))
                
            # # [METHOD: QUERY REWRITE]1. 查詢改寫：強化實體抽取
            # rewrite_prompt = (
            #     f"請從以下問題中提取核心實體與關鍵字，並保留原意擴充同義詞。直接輸出改寫後的問題即可：\n"
            #     f"原問題：{user_query}"
            # )
            # rewritten_query_response = await Settings.llm.acomplete(rewrite_prompt)
            # enhanced_query = rewritten_query_response.text.strip()
            
            # # 2. 將改寫後的問題餵給 LightRAG (保留 user_query 給最終生成，enhanced_query 給檢索)
            # raw_context = await self.rag.aquery_data(
            #     enhanced_query, 
            #     param=QueryParam(mode=self.mode, only_need_context=True)
            # )

            # 要拿retrieval的context，所以only_need_context=True
            # 優先使用 LightRAG 原生的非同步方法 aquery，避免與外層 asyncio.run 衝突
            if hasattr(self.rag, "aquery"):
                # response = await self.rag.aquery(user_query, param=QueryParam(mode=self.mode))
                raw_context = await self.rag.aquery_data(user_query, param=QueryParam(mode=self.mode, only_need_context=True))
                # [METHOD: RERANKING]
                # raw_context = await self.rag.aquery_data(user_query, param=QueryParam(mode=self.mode, only_need_context=True, enable_rerank=True, top_k=60))
            elif inspect.iscoroutinefunction(self.rag.query):
                # response = await self.rag.query(user_query, param=QueryParam(mode=self.mode))
                raw_context = self.rag.query_data(user_query, param=QueryParam(mode=self.mode, only_need_context=True))
            else:
                # 備用方案：若版本較舊僅支援同步 (此情況套用 nest_asyncio)
                # response = self.rag.query(user_query, param=QueryParam(mode=self.mode))
                raw_context = self.rag.query_data(user_query, param=QueryParam(mode=self.mode, only_need_context=True))
                
            print(f"RAW_CONTEXT:\n\n{raw_context}\n\n")
            retrieved_contexts = [str(raw_context)] if raw_context else []
            
            context_prompt = await self.rag.aquery_data(user_query, param=QueryParam(mode=self.mode, only_need_prompt=True))
            print(f"CONTEXT_PROMPT:\n\n{context_prompt}\n\n")
            
            from lightrag.prompt import PROMPTS
            context_str = "\n".join(retrieved_contexts) if retrieved_contexts else ""

            prompt = (
                PROMPTS["rag_response"].format(
                    context_data=context_str,               # LightRAG 原生模板預期的上下文
                    response_type="Multiple Paragraphs",    # 補上預設的生成格式
                    user_prompt=user_query,                 # 👈 解決這次錯誤的核心：補上 user_prompt
                    query=user_query,                       # 保留 query 以防舊版相容性需要
                    retrieved_contexts=context_str          # 保留防呆
                )
            )
            # prompt = (
            #     PROMPTS["rag_response"].format(
            #         query=user_query,
            #         retrieved_contexts=retrieved_contexts
            #     )
            # )
            
            Settings = get_settings(model_type=args.model_type)
            llm_response = await Settings.llm.acomplete(prompt)
            response = llm_response.text
        except Exception as e:
            print(f"❌ LightRAG 查詢發生錯誤: {e}")
            response = f"發生錯誤: {e}"
        
        # 紀錄結束時間並計算耗時
        end_time = time.time()
        execution_time_sec = round(end_time - start_time, 4)
        
        # 封裝為標準輸出格式，確保後續評估系統不會出現 IndexError/TypeError
        return {
            "generated_answer": str(response) if response else "找不到答案",
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": [],            
            "source_nodes": [],              
            "execution_time_sec": execution_time_sec
        }

# class CombineRAGPipelineWrapper:
#     """
#     自定義的混合檢索 Wrapper：
#     同時從 Vector Engine 與 Graph Engine 撈取資料，合併後再丟給 LLM 生成。
#     """
#     def __init__(self, name: str, vector_retriever, graph_retriever, llm):
#         self.name = name
#         self.vector_retriever = vector_retriever
#         self.graph_retriever = graph_retriever
#         self.llm = llm # 傳入 Settings.llm

#     async def aquery_and_log(self, user_query: str) -> Dict[str, Any]:
#         start_time = time.time()
        
#         # 1. 平行或依序進行兩種檢索 (這裡以依序為例，可用 asyncio.gather 加速)
#         vector_nodes = await self.vector_retriever.aretrieve(user_query)
#         graph_nodes = await self.graph_retriever.aretrieve(user_query)
        
#         # 2. 合併與去重 Nodes (如有重複的 ID)
#         all_nodes = vector_nodes + graph_nodes
#         # 實務上可以在這裡加入 Reranker (如 bge-reranker) 進行重新排序
        
#         # 3. 萃取 Context 準備生成
#         retrieved_contexts = [node.get_content() for node in all_nodes]
#         retrieved_ids = [str(node.metadata.get("NO", node.node_id)) for node in all_nodes]
        
#         combined_text = "\n\n---\n\n".join(retrieved_contexts)
        
#         # 4. 組裝 Prompt 並呼叫 LLM
#         prompt = (
#             "請根據以下提供的參考資料（包含文件片段與知識圖譜關聯），回答使用者的問題。\n"
#             "如果資料中沒有答案，請回答「不知道」。\n\n"
#             f"【參考資料】\n{combined_text}\n\n"
#             f"【問題】: {user_query}\n\n"
#             "【解答】:"
#         )
#         response = await self.llm.acomplete(prompt)
        
#         end_time = time.time()
        
#         # 5. 回傳完整格式 (這次 contexts 有資料了，Faithfulness 就會正常運作！)
#         return {
#             "generated_answer": response.text,
#             "retrieved_contexts": retrieved_contexts,
#             "retrieved_ids": retrieved_ids,
#             "source_nodes": all_nodes,
#             "execution_time_sec": round(end_time - start_time, 4)
#         }

# ==========================================
# 3. 指標計算與評估引擎 (Evaluation Engine)
# ==========================================

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")
meteor_metric = evaluate.load("meteor")

def calculate_hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str]) -> int:
    if not ground_truth_ids:
        return 0
    return 1 if any(gt_id in retrieved_ids for gt_id in ground_truth_ids) else 0

def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    if not ground_truth_ids:
        return 0.0
    for i, r_id in enumerate(retrieved_ids):
        if r_id in ground_truth_ids:
            return 1.0 / (i + 1)
    return 0.0

def calculate_f1_score(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    if len(ground_truth_ids) == 0 or len(retrieved_ids) == 0:
        return 0.0, 0.0, 0.0
    recall = float(len(set(retrieved_ids) & set(ground_truth_ids)) / len(ground_truth_ids))
    precision = float(len(set(retrieved_ids) & set(ground_truth_ids)) / len(retrieved_ids))
    f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return recall, precision, f1_score

# generation metrics
def calculate_rouge_score(generated_answer: str, ground_truth_answer: str) -> tuple:
    if not generated_answer or not ground_truth_answer:
        return 0.0, 0.0, 0.0, 0.0
    # 預設支援英文，若為中文建議使用 tokenizer="zh" (需搭配結巴分詞) 或單純基於字元
    score = rouge_metric.compute(predictions=[generated_answer], references=[ground_truth_answer])
    return score.get("rouge1", 0.0), score.get("rouge2", 0.0), score.get("rougeL", 0.0), score.get("rougeLsum", 0.0)

def calculate_bleu_score(generated_answer: str, ground_truth_answer: str) -> float:
    if not generated_answer or not ground_truth_answer:
        return 0.0
    score = bleu_metric.compute(predictions=[generated_answer], references=[ground_truth_answer])
    return score.get("bleu", 0.0)

def calculate_bertscore_score(generated_answer: str, ground_truth_answer: str, lang="zh") -> tuple:
    if not generated_answer or not ground_truth_answer:
        return 0.0, 0.0, 0.0
    score = bertscore_metric.compute(predictions=[generated_answer], references=[ground_truth_answer], lang=lang)
    return score["precision"][0], score["recall"][0], score["f1"][0]

def calculate_meteor_score(generated_answer: str, ground_truth_answer: str) -> float:
    if not generated_answer or not ground_truth_answer:
        return 0.0
    score = meteor_metric.compute(predictions=[generated_answer], references=[ground_truth_answer])
    return score.get("meteor", 0.0)

def token_level_f1_score(generated_answer: str, ground_truth_answer: str) -> tuple:
    """使用字元集合計算交集來算出精準度與召回率 (相容中英混合)"""
    gen_tokens = set(generated_answer)
    gt_tokens = set(ground_truth_answer)
    
    if not gen_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0
        
    common_tokens = gen_tokens.intersection(gt_tokens)
    token_level_recall = len(common_tokens) / len(gt_tokens)
    token_level_precision = len(common_tokens) / len(gen_tokens)
    
    if (token_level_precision + token_level_recall) > 0:
        token_level_f1 = 2 * (token_level_precision * token_level_recall) / (token_level_precision + token_level_recall)
    else:
        token_level_f1 = 0.0
        
    return token_level_recall, token_level_precision, token_level_f1

import re
import jieba
from collections import Counter
from typing import Tuple
def jieba_f1_score(generated_answer: str, ground_truth_answer: str) -> Tuple[float, float, float]:
    """
    使用 Word-level 計算精準度、召回率與 F1 Score
    """
    def jieba_normalize_and_tokenize(text: str) -> list[str]:
        """
        使用 jieba 進行 Word-level 中英混合分詞：
        1. 轉小寫
        2. 透過 jieba.lcut 切分詞彙
        3. 過濾掉純標點符號與空白，只保留包含中文字、英文字母或數字的 Token
        """
        if not text:
            return []
        
        text = str(text).lower()
        
        # 使用 jieba 精確模式進行分詞
        raw_tokens = jieba.lcut(text)
        
        # 過濾：只保留含有實質內容（中/英/數）的詞彙，拋棄純標點符號如 "，", "。", " " 等
        valid_tokens = [
            token for token in raw_tokens 
            if re.search(r'[a-z0-9\u4e00-\u9fa5]', token)
        ]
        
        return valid_tokens

    gen_tokens = jieba_normalize_and_tokenize(generated_answer)
    gt_tokens = jieba_normalize_and_tokenize(ground_truth_answer)
    
    if not gen_tokens and not gt_tokens:
        return 1.0, 1.0, 1.0
        
    if not gen_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0
        
    gen_counter = Counter(gen_tokens)
    gt_counter = Counter(gt_tokens)
    
    # 計算交集 (Multiset Intersection)
    common_tokens_count = sum((gen_counter & gt_counter).values())
    
    if common_tokens_count == 0:
        return 0.0, 0.0, 0.0
        
    word_level_recall = common_tokens_count / len(gt_tokens)
    word_level_precision = common_tokens_count / len(gen_tokens)
    
    word_level_f1 = 2 * (word_level_precision * word_level_recall) / (word_level_precision + word_level_recall)
        
    return word_level_recall, word_level_precision, word_level_f1

async def compute_and_format_metrics(
    idx: int, source: str, query: str, gt_answer: str, gen_answer: str, 
    gt_ids: List[str], retrieved_ids: List[str], retrieved_contexts: List[str], 
    execution_time_sec: float, correctness_evaluator, faithfulness_evaluator
) -> Dict[str, Any]:
    """封裝所有指標計算，並回傳單筆結果字典"""
    
    # --- 1. 計算檢索指標 ---
    hit_rate = calculate_hit_rate(retrieved_ids, gt_ids)
    mrr = calculate_mrr(retrieved_ids, gt_ids)
    recall, precision, f1_score = calculate_f1_score(retrieved_ids, gt_ids)

    # --- 2. 計算生成指標 (傳統 NLP) ---
    jieba_r, jieba_p, jieba_f1 = jieba_f1_score(gen_answer, gt_answer)
    rouge1, rouge2, rougeL, rougeLsum = calculate_rouge_score(gen_answer, gt_answer)
    bleu = calculate_bleu_score(gen_answer, gt_answer)
    meteor = calculate_meteor_score(gen_answer, gt_answer)
    bert_p, bert_r, bert_f1 = calculate_bertscore_score(gen_answer, gt_answer, lang="zh")
    tok_r, tok_p, tok_f1 = token_level_f1_score(gen_answer, gt_answer)

    # --- 3. 計算 LLM-as-a-judge 評估 ---
    try:
        correctness_result = await correctness_evaluator.aevaluate(
            query=query, response=gen_answer, reference=gt_answer
        )
        correctness_score = correctness_result.score
        
        faithfulness_result = await faithfulness_evaluator.aevaluate(
            query=query, response=gen_answer, contexts=retrieved_contexts
        )
        faithfulness_score = 1.0 if faithfulness_result.passing else 0.0
    except Exception as e:
        print(f"  ⚠️ 評測 LLM API 呼叫失敗: {e}")
        correctness_score, faithfulness_score = None, None

    # --- 4. 統整回傳 Dictionary ---
    return {
        "idx": idx,
        "dataset_source": source,
        "query": query,
        "ground_truth_answer": gt_answer,
        "generated_answer": gen_answer,
        "ground_truth_ids": ", ".join(gt_ids),
        "retrieved_ids": ", ".join(retrieved_ids),
        # 以下為數值型指標 (未來新增指標請加在這裡)
        "execution_time_sec": execution_time_sec,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "retrieval_recall": recall,
        "retrieval_precision": precision,
        "retrieval_f1_score": f1_score,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "rougeLsum": rougeLsum,
        "bleu": bleu,
        "meteor": meteor,
        "bertscore_f1": bert_f1,
        "token_f1": tok_f1,
        "jieba_f1": jieba_f1,
        "correctness_score": correctness_score,
        "faithfulness_score": faithfulness_score,
    }

async def run_evaluation(qa_datasets: List[Dict], pipelines: List[RAGPipelineWrapper], postfix: str = ""):
    """執行評估流程並輸出各階層報告 - 詳細報表包含個別 row 及平均 row"""
    
    # 建立外層總資料夾 (以時間戳記命名)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_eval_dir = f"/home/End_to_End_RAG/results/evaluation_results_{timestamp}{postfix}"
    os.makedirs(base_eval_dir, exist_ok=True)
    
    print(f"📁 建立實驗數據主資料夾: {base_eval_dir}")

    correctness_evaluator = CorrectnessEvaluator(llm=Settings.eval_llm)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.eval_llm)

    summary_records = []

    for pipeline in pipelines:
        print(f"\n🚀 開始評測 Pipeline: {pipeline.name}")
        
        pipeline_dir = os.path.join(base_eval_dir, pipeline.name)
        os.makedirs(pipeline_dir, exist_ok=True)
        
        pipeline_detailed_results = []

        for idx, qa in enumerate(qa_datasets):
            print(f"  正在評估第 {idx+1}/{len(qa_datasets)} 題...")
            query = qa["query"]
            gt_answer = qa["ground_truth_answer"]
            gt_ids = qa["ground_truth_doc_ids"]
            gt_ids = gt_ids[0].splitlines()

            # 1. 執行檢索與生成
            try:
                result = await pipeline.aquery_and_log(query)

            # 2. 呼叫獨立函數計算所有指標
                metrics_row = await compute_and_format_metrics(
                    idx=idx + 1,
                    source=qa["source"],
                    query=query,
                    gt_answer=gt_answer,
                    gen_answer=result["generated_answer"],
                    gt_ids=gt_ids,
                    retrieved_ids=result["retrieved_ids"],
                    retrieved_contexts=result["retrieved_contexts"],
                    execution_time_sec=result["execution_time_sec"],
                    correctness_evaluator=correctness_evaluator,
                    faithfulness_evaluator=faithfulness_evaluator
                )
                pipeline_detailed_results.append(metrics_row)

            except Exception as e:
                print(f"⚠️ 第 {idx+1} 題評測崩潰，跳過。錯誤: {e}")
                continue

        # --- 儲存單一 Pipeline 詳細報表 ---
        df_pipeline = pd.DataFrame(pipeline_detailed_results)
        # 加入平均 row
        if not df_pipeline.empty:
            # 自動抓取所有數值欄位 (略過非評估指標的數值，例如 idx)
            numeric_cols = df_pipeline.select_dtypes(include='number').columns.tolist()
            if 'idx' in numeric_cols:
                numeric_cols.remove('idx')

            # 自動計算所有指標的平均
            avg_row = {col: df_pipeline[col].mean() for col in numeric_cols}
            avg_row["idx"] = "平均"
            
            # 填補文字類型的欄位為空字串
            for col in df_pipeline.columns:
                if col not in avg_row:
                    avg_row[col] = ""

            df_pipeline = pd.concat([df_pipeline, pd.DataFrame([avg_row])], ignore_index=True)

            # --- 計算當前 Pipeline 的平均表現供總表使用 (動態產生 summary) ---
            summary_record = {"pipeline_name": pipeline.name}
            for col in numeric_cols:
                summary_record[f"avg_{col}"] = avg_row[col]
            summary_records.append(summary_record)

        detailed_csv_path = os.path.join(pipeline_dir, "detailed_results.csv")
        df_pipeline.to_csv(detailed_csv_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ 已儲存明細報表至: {detailed_csv_path}")

    # --- 儲存外層總指標比較 CSV ---
    df_summary = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(base_eval_dir, "global_summary_report.csv")
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n📊 已儲存外層綜合評估報告至: {summary_csv_path}")
    
    print("\n" + "="*50)
    print("實驗結果總覽 (Summary)")
    print("="*50)
    print(df_summary.to_string(index=False))

# ==========================================
# 4. 主程式
# ==========================================
if __name__ == "__main__":
    import asyncio
    import argparse
    from model_settings import get_settings
    # from data_processing import data_processing
    from vector_package import get_vector_query_engine
    from graph_unstructured_text_package import get_graph_query_engine
    from graph_dynamic_schema_package import get_dynamic_schema_graph_query_engine
    
    from lightrag_package import get_lightrag_engine, build_lightrag_index

    
    # documents = data_processing(mode="natural_text", datatype="DI")

    parser = argparse.ArgumentParser()
    # method
    parser.add_argument("--vector_method", type=str, default="none", choices=["none", "hybrid", "vector", "bm25", "all"])
    parser.add_argument("--graph_rag_method", type=str, default="none", choices=["none", "propertyindex", "lightrag",  "dynamic_schema", "all"]) #"graphrag", "csr_khop", "csr_bridge", 
    parser.add_argument("--lightrag_mode", type=str, default="none", choices=["none", "local", "global", "hybrid", "mix", "naive", "bypass", "all"])
    """
    "Query Mode Differences:
        - local: Focuses on entities and their related chunks based on low-level keywords
        - global: Focuses on relationships and their connected entities based on high-level keywords
        - hybrid: Combines local and global results using round-robin merging
        - mix: Includes knowledge graph data plus vector-retrieved document chunks
        - naive: Only vector-retrieved chunks, entities and relationships arrays are empty
        - bypass: All data arrays are empty, used for direct LLM queries"
    """
    parser.add_argument("--adv_vector_method", type=str, default="none", choices=["none", "parent_child", "self_query", "all"])
    parser.add_argument("--ms_graphrag_mode", type=str, default="none", choices=["none", "local", "global", "all"])
    # parser.add_argument("--combine_rag_method", type=str, default="none", choices=["none", "graph_vector", "all"])

    # method parameter
    parser.add_argument("--top_k", type=int, default=2)

    # CSR graph anchor/extraction options
    parser.add_argument("--use_vector_docs", action="store_true", help="CSR graph：將 vector 檢索到的文件加入 anchor 來源")
    parser.add_argument(
        "--doc_entity_mode",
        type=str,
        default="metadata_only",
        choices=["metadata_only", "llm"],
        help="CSR graph：從 vector 文件抽取 entities 的方式",
    )
    parser.add_argument("--use_schema_hint", action="store_true", help="CSR graph：啟用 schema hint（用於 LLM entity 抽取前的 schema 更新/提示）")
    # parser.add_argument("--use_dynamic_schema_extractor", action="store_true", help="啟用 dynamic schema extractor")
    
    # data parameter
    parser.add_argument("--data_type", type=str, default="DI", choices=["DI", "GEN"])
    parser.add_argument("--data_mode", type=str, default="natural_text", choices=["natural_text", "markdown", "key_value_text", "unstructured_text"])
    parser.add_argument("--model_type", type=str, default="small", choices=["small", "big"])
    
    # evaluation parameter
    parser.add_argument("--qa_dataset_fast_test", action="store_true", help="只抽取 2 題 QA 進行快速測試")
    parser.add_argument("--vector_build_fast_test", action="store_true", help="只抽取 2 題 QA 進行快速測試")
    parser.add_argument("--graph_build_fast_test", action="store_true", help="只抽取 2 題 QA 進行快速測試")
    
    parser.add_argument("--postfix", type=str, default="", help="for postfix")
    parser.add_argument("-sup", "--sup", type=str, default="", help="for cache method")
    
    parser.add_argument("--schema_method", type=str, default="static_baseline", help="Schema 方法", choices=["lightrag_default", "iterative_evolution", "llm_dynamic"])
    
    args = parser.parse_args()

    Settings = get_settings(model_type=args.model_type)

    if args.qa_dataset_fast_test:
        args.vector_build_fast_test = True
        args.graph_build_fast_test = True
        print("⚡ 已啟用 --qa_dataset_fast_test，自動連動開啟 vector_build_fast_test 與 graph_build_fast_test！")

    pipelines_to_test = []

    PIPELINE_FACTORY = {
        "vector_vector": lambda: RAGPipelineWrapper("Vector", get_vector_query_engine(..., vector_method="vector")),
        "vector_hybrid": lambda: RAGPipelineWrapper("Hybrid", get_vector_query_engine(..., vector_method="hybrid")),
        "vector_bm25": lambda: RAGPipelineWrapper("BM25", get_vector_query_engine(..., vector_method="bm25")),
        "vector_vector": lambda: RAGPipelineWrapper("Vector", get_vector_query_engine(..., vector_method="vector")),
        "graph_propertyindex": lambda: RAGPipelineWrapper("Graph_PropertyIndex", get_graph_query_engine(..., graph_method="propertyindex")),
        "graph_dynamic_schema": lambda: RAGPipelineWrapper("Graph_DynamicSchema", get_dynamic_schema_graph_query_engine(..., graph_method="dynamic_schema")),
        "graph_csr_khop": lambda: RAGPipelineWrapper("CSR_KHop_Graph", get_graph_query_engine(..., graph_method="csr_khop")),
        "graph_csr_bridge": lambda: RAGPipelineWrapper("CSR_Bridge_Graph", get_graph_query_engine(..., graph_method="csr_bridge")),
        "graph_lightrag": lambda: RAGPipelineWrapper("LightRAG", get_lightrag_engine(..., mode="local")),
    }

    # -------------------------
    # VECTOR RAG Pipeline Setup
    # -------------------------
    if args.vector_method != "none":
        if args.vector_method in ["hybrid", "all"]:
            engine = get_vector_query_engine(Settings, vector_method="hybrid", top_k=args.top_k, data_mode=args.data_mode, data_type=args.data_type, fast_build=args.vector_build_fast_test)
            if engine is not None:
                pipelines_to_test.append(RAGPipelineWrapper(name="Vector_hybrid_RAG", query_engine=engine))
            
        if args.vector_method in ["vector", "all"]:
            engine = get_vector_query_engine(Settings, vector_method="vector", top_k=args.top_k, data_mode=args.data_mode, data_type=args.data_type, fast_build=args.vector_build_fast_test)
            if engine is not None:
                pipelines_to_test.append(RAGPipelineWrapper(name="Vector_vector_RAG", query_engine=engine))
        if args.vector_method in ["bm25", "all"]:
            engine = get_vector_query_engine(Settings, vector_method="bm25", top_k=args.top_k, data_mode=args.data_mode, data_type=args.data_type, fast_build=args.vector_build_fast_test)
            if engine is not None:
                pipelines_to_test.append(RAGPipelineWrapper(name="Vector_bm25_RAG", query_engine=engine))
    else:
        print("Skip Vector RAG")

    # -------------------------
    # Advanced Vector RAG Pipeline Setup
    # -------------------------
    if args.adv_vector_method != "none":
        if args.adv_vector_method in ["self_query", "all"]:
            engine = get_self_query_engine(Settings, data_mode=args.data_mode, data_type=args.data_type, top_k=args.top_k, fast_build=args.vector_build_fast_test)
            pipelines_to_test.append(RAGPipelineWrapper(name="Self_Query_RAG", query_engine=engine))
            
        if args.adv_vector_method in ["parent_child", "all"]:
            engine = get_parent_child_query_engine(Settings, data_mode=args.data_mode, data_type=args.data_type, top_k=args.top_k, fast_build=args.vector_build_fast_test)
            pipelines_to_test.append(RAGPipelineWrapper(name="Parent_Child_RAG", query_engine=engine))
    else:
        print("Skip Advanced Vector RAG")

    # -------------------------
    # Graph RAG Pipeline Setup
    # -------------------------
    if args.graph_rag_method != "none":
        if args.graph_rag_method in ["propertyindex", "all"]:
            engine = get_graph_query_engine(Settings, data_mode=args.data_mode, data_type=args.data_type, fast_build=args.graph_build_fast_test, graph_method="propertyindex")
            if engine is not None:
                pipelines_to_test.append(RAGPipelineWrapper(name="Graph_PropertyIndex_RAG", query_engine=engine))

        if args.graph_rag_method in ["dynamic_schema", "all"]:
            engine = get_dynamic_schema_graph_query_engine(Settings, data_mode=args.data_mode, data_type=args.data_type, fast_build=args.graph_build_fast_test, graph_method="dynamic_schema")
            if engine is not None:
                pipelines_to_test.append(RAGPipelineWrapper(name="Graph_DynamicSchema_RAG", query_engine=engine))

        if args.graph_rag_method in ["csr_khop", "all"]:
            engine = get_graph_query_engine(Settings, data_mode=args.data_mode, data_type=args.data_type, fast_build=args.graph_build_fast_test, graph_method="csr_khop", top_k=args.top_k, use_vector_docs=args.use_vector_docs, doc_entity_mode=args.doc_entity_mode, use_schema_hint=args.use_schema_hint)
            if engine is not None:
                pipelines_to_test.append(RAGPipelineWrapper(name="CSR_KHop_Graph", query_engine=engine))

        if args.graph_rag_method in ["csr_bridge", "all"]:
            engine = get_graph_query_engine(
                Settings,
                data_mode=args.data_mode,
                data_type=args.data_type,
                fast_build=args.graph_build_fast_test,
                graph_method="csr_bridge",
                top_k=args.top_k,
                use_vector_docs=args.use_vector_docs,
                doc_entity_mode=args.doc_entity_mode,
                use_schema_hint=args.use_schema_hint,
            )
            if engine is not None:
                pipelines_to_test.append(RAGPipelineWrapper(name="CSR_Bridge_Graph", query_engine=engine))
        

        if args.graph_rag_method in ["lightrag", "all"]:
            if args.sup:
                sup = "_" + args.sup
            else:
                sup = ""
            if not os.path.exists(os.path.join(Settings.lightrag_storage_path_DIR, args.data_type) + sup):
                if not os.path.exists(Settings.lightrag_storage_path_DIR):
                    os.mkdir(Settings.lightrag_storage_path_DIR)
                os.mkdir(os.path.join(Settings.lightrag_storage_path_DIR, args.data_type) + sup)
                # build_lightrag_index(Settings, mode=args.data_mode, data_type=args.data_type, sup=args.sup, fast_build=args.graph_build_fast_test)
                
                # 1. 取得動態 Schema
                from data_processing import data_processing
                from schema_factory import get_schema_by_method
                custom_entity_types = get_schema_by_method(
                    method=args.schema_method, 
                    text_corpus=data_processing(mode=args.data_mode, data_type=args.data_type), # 若為 llm_dynamic 則傳入樣本
                    llm=Settings.llm
                )
                print(f"🌟 LightRAG 將使用以下實體類別建圖: {custom_entity_types}")
                
                # 2. 將其寫入 Settings (確保 lightrag_package.py 讀得到)
                Settings.lightrag_entity_types = custom_entity_types
                
                # 3. 建立帶有動態 Schema 的 LightRAG 索引
                build_lightrag_index(
                    Settings, 
                    mode=args.data_mode, 
                    data_type=args.data_type, 
                    sup=args.sup, # 💡 關鍵：將 schema 名稱加入資料夾命名以隔離 Cache
                    fast_build=args.graph_build_fast_test
                )
            
            lightrag_instance = get_lightrag_engine(Settings, data_type=args.data_type, sup=args.sup + f"_{args.schema_method}")
            if args.lightrag_mode != "none":
                if args.lightrag_mode == "all":
                    # 若為 all，測試所有模式
                    modes_to_test = ["local", "global", "mix", "bypass", "hybrid", "naive"]
                    for mode in modes_to_test:
                        wrapper = LightRAGWrapper(name=f"LightRAG_{mode.capitalize()}", rag_instance=lightrag_instance, mode=mode)
                        pipelines_to_test.append(wrapper)
                else:
                    # 若非 all，只測試指定的 lightrag_mode
                    wrapper = LightRAGWrapper(name=f"LightRAG_{args.lightrag_mode.capitalize()}", rag_instance=lightrag_instance, mode=args.lightrag_mode)
                    pipelines_to_test.append(wrapper)
            else:
                print("Skip LightRAG")
    else:
        print("Skip Graph RAG")

    # if args.combine_rag_method != "none":
    #     vector_engine = get_vector_query_engine(Settings, vector_method="hybrid", top_k=args.top_k, data_mode="natural_text", data_type=args.data_type, fast_build=args.vector_build_fast_test)
    #     graph_engine = get_graph_query_engine(Settings, data_mode="natural_text", data_type=args.data_type, fast_build=args.graph_build_fast_test, graph_method="propertyindex")

    #     combine_pipeline = CombineRAGPipelineWrapper(
    #         name="Graph_Vector_Combine_RAG",
    #         vector_retriever=vector_engine.retriever,
    #         graph_retriever=graph_engine.retriever,
    #         llm=Settings.llm
    #     )
    #     pipelines_to_test.append(combine_pipeline)


    # -------------------------
    # MS End-to-End GraphRAG Pipeline Setup
    # -------------------------
    # if args.ms_graphrag_mode != "none":
    #     # 假設你的 MS GraphRAG 專案目錄放在此路徑
    #     ms_root_dir = f"./storage_ms_graphrag_{args.data_type}" 
        
    #     if args.ms_graphrag_mode in ["local", "all"]:
    #         wrapper = MSGraphRAGWrapper(name="MS_GraphRAG_Local", root_dir=ms_root_dir, mode="local")
    #         pipelines_to_test.append(wrapper)
            
    #     if args.ms_graphrag_mode in ["global", "all"]:
    #         wrapper = MSGraphRAGWrapper(name="MS_GraphRAG_Global", root_dir=ms_root_dir, mode="global")
    #         pipelines_to_test.append(wrapper)

    postfix = ""
    if args.data_type == "DI":
        postfix += "_DI"
    elif args.data_type == "GEN":
        postfix += "_GEN"
    if args.vector_method != "none":
        postfix += f"_{args.vector_method}"
    if args.graph_rag_method != "none":
        postfix += f"_{args.graph_rag_method}"
    if args.lightrag_mode != "none":
        postfix += f"_{args.lightrag_mode}"
    if args.use_schema_hint:
        postfix += "_schema_hint"
    if args.postfix:
        postfix += f"_{args.postfix}"
    if args.sup:
        postfix += f"_{args.sup}"
    if args.qa_dataset_fast_test:
        postfix += "_fast_test"
    


    if not pipelines_to_test:
        print("⚠️ 未選擇任何 RAG pipeline，請檢查啟動參數。")
    else:
        if args.data_type == "DI":
            datasets = load_and_normalize_qa_CSR_DI(csv_path=Settings.qa_file_path_DI)
        elif args.data_type == "GEN":
            datasets = load_and_normalize_qa_CSR_full(jsonl_path=Settings.qa_file_path_GEN)
            
        # for qa_data in datasets:
        # if qa_data: # 確保有讀取到資料
        if args.qa_dataset_fast_test:
            print("⚡ 啟用快速測試模式，僅抽取前 2 題進行評估...")
            datasets = datasets[:2]
        else:
            print("啟用完整測試模式，進行完整 QA 評估...")
        asyncio.run(run_evaluation(datasets, pipelines_to_test, postfix=postfix))
        
    # qa_data = load_and_normalize_qa_CSR_DI(csv_path=Settings.qa_file_path) # QA_DI
    # qa_data = load_and_normalize_qa_CSR_full(jsonl_path=Settings.qa_file_path) # QA_GEN