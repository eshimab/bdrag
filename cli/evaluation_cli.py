import argparse
import json
import os

import lib.hybrid_search as hybs


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    top_k = args.limit

    # load data
    with open("data/golden_dataset.json", "r") as jfile:
        gdata = json.load(jfile)["test_cases"]
        print("evaluation_cli > golden_dataset.json loaded")
    with open("data/movies.json") as jfile:
        documents = json.load(jfile)["movies"]
        print("evaluation_cli > movies.json loaded")
    # init hybrid search
    hss = hybs.HybridSearch(documents)
    #
    k_val = 60
    for gidx, gd in enumerate(gdata):
        query = gd["query"]
        rdocs = gd["relevant_docs"]
        rrfs = hss.rrf_search(query, k_val, top_k)
        # rr_map = {inner_dict["id"]: inner_dict for inner_dict in rrfs}
        rr_titles = [rr["title"] for rr in rrfs]
        # main stats
        gd["matched"] = set(rdocs) & set(rr_titles)
        gd["kpres"] = len(gd["matched"]) / len(rr_titles)
        gd["recall"] = len(gd["matched"]) / len(rdocs)
        # other stats
        gd["ret"] = sorted(set(rr_titles))
        gd["relevant"] = sorted(set(rdocs))
        gd["missed"] = set(rdocs) - gd["matched"]
        gd["not_rel"] = set(rr_titles) - gd["matched"]
        gd["pct_missed"] = len(gd["missed"]) / len(rdocs) * 100
        gd["pct_not_rel"] = len(gd["not_rel"]) / len(rr_titles) * 100
        gd["pct_kpres"] = gd["kpres"] * 100
        gd["matched"] = sorted(gd["matched"])
    # print
    for gd in gdata:
        query_list = ["cute british bear marmalade", "car racing", "dinosaur park"]
        # query_list = ["car racing"]
        if gd["query"] not in query_list:
            continue
        print(f"\n- Query: {gd["query"]}")
        print(f"  - Precision@{top_k}: {gd["kpres"]:.4f}")
        print(f"  - Recall@{top_k}: {gd["recall"]:.4f}")
        print(f"  - Relevant:  {", ".join(gd["relevant"])}")
        print(f"  - Retrieved: {", ".join(gd["ret"])}")
        print(f"  - Matched: ({gd["pct_kpres"]:.1f}%) {", ".join(gd["matched"])}")
        print(f"  - Missed: ({gd["pct_missed"]:.1f}%) {", ".join(gd["missed"])}")
        print(
            f"  - Not Relevant: ({gd["pct_not_rel"]:.1f}%) {", ".join(gd["not_rel"])}"
        )


if __name__ == "__main__":
    main()
