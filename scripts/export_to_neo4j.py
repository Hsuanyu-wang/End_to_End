#!/usr/bin/env python3
"""
GraphML → Neo4j 匯入工具

用法:
  python scripts/export_to_neo4j.py \
    --graphml storage/lightrag/DI_lightrag_default_hybrid/graph_chunk_entity_relation.graphml \
    --neo4j_uri bolt://localhost:7687 \
    --neo4j_user neo4j \
    --neo4j_password your_password \
    --clear
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="將 GraphML 圖譜匯入 Neo4j")
    parser.add_argument("--graphml", required=True, help="GraphML 檔案路徑")
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j_user", default="neo4j", help="Neo4j 使用者名稱")
    parser.add_argument("--neo4j_password", required=True, help="Neo4j 密碼")
    parser.add_argument("--neo4j_database", default="neo4j", help="Neo4j 資料庫名稱")
    parser.add_argument("--clear", action="store_true", help="匯入前先清空資料庫")
    args = parser.parse_args()

    graphml_path = args.graphml
    if not os.path.isabs(graphml_path):
        graphml_path = os.path.join(_PROJECT_ROOT, graphml_path)

    if not os.path.exists(graphml_path):
        print(f"❌ GraphML 檔案不存在: {graphml_path}")
        sys.exit(1)

    import networkx as nx
    print(f"📂 載入 GraphML: {graphml_path}")
    G = nx.read_graphml(graphml_path)
    print(f"   節點: {G.number_of_nodes()}, 邊: {G.number_of_edges()}")

    from src.graph_adapter.converters.neo4j_converter import networkx_to_neo4j

    neo4j_config = {
        "uri": args.neo4j_uri,
        "username": args.neo4j_user,
        "password": args.neo4j_password,
        "database": args.neo4j_database,
        "clear_before_import": args.clear,
    }

    print(f"🔗 連接 Neo4j: {args.neo4j_uri} (database={args.neo4j_database})")
    if args.clear:
        print("⚠️  將先清空資料庫再匯入")

    networkx_to_neo4j(G, neo4j_config)
    print("🎉 匯入完成！可在 Neo4j Browser 中查看圖譜。")


if __name__ == "__main__":
    main()
