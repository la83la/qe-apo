"""
在 MS MARCO v1.1 中搜尋特定 query。

用法:
    python search_query.py "your query here"
    python search_query.py "your query here" --exact    # 完全匹配
"""

import argparse
from datasets import load_dataset


def search(query_text: str, exact: bool = False):
    print("正在載入 MS MARCO v1.1...")
    ds = load_dataset("microsoft/ms_marco", "v1.1")

    results = []
    for split_name in ds:
        for i, row in enumerate(ds[split_name]):
            q = row["query"]
            if exact:
                match = q.strip().lower() == query_text.strip().lower()
            else:
                match = query_text.strip().lower() in q.lower()

            if match:
                results.append({
                    "split": split_name,
                    "index": i,
                    "query": q,
                    "query_id": row.get("query_id"),
                    "query_type": row.get("query_type"),
                })

    if results:
        print(f"\n找到 {len(results)} 筆匹配結果:\n")
        for r in results:
            print(f"  [{r['split']}][{r['index']}] id={r['query_id']}  type={r['query_type']}")
            print(f"    query: {r['query']}")
            print()
    else:
        print(f"\n找不到匹配的 query: \"{query_text}\"")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="要搜尋的 query 文字")
    parser.add_argument("--exact", action="store_true", help="完全匹配（預設為部分匹配）")
    args = parser.parse_args()

    search(args.query, args.exact)
