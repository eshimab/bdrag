import argparse
import json

import lib.hybrid_search as hybs


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_sp = subparsers.add_parser("normalize", help="normalize")
    normalize_sp.add_argument("scores", type=float, nargs="+", help="list of scores")

    weighted_sp = subparsers.add_parser("weighted-search", help="weighted search")
    weighted_sp.add_argument("query", type=str, help="query")
    weighted_sp.add_argument(
        "--alpha", type=float, nargs="?", default=0.5, help="alpha"
    )
    weighted_sp.add_argument("--limit", type=int, nargs="?", default=5, help="limit")

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            with open("data/movies.json") as jfile:
                documents = json.load(jfile)["movies"]
            hss = hybs.HybridSearch(documents)
            hs_list = hss.weighted_search(args.query, args.alpha, args.limit)
            hidx = 1
            for hs in hs_list:
                print(f"{hidx}. {hs["title"]}")
                print(f"Hybrid Score: {hs["hybrid_score"]:.3f}")
                print(
                    f"BM25: {hs["bm25_score"]:.4f}, Semantic: {hs["semantic_score"]:.4f}"
                )
                print(f"{hs["description"]}")
                # if hidx >= 5:
                # break
                hidx += 1

        case "normalize":
            num_list = args.scores
            # print(f"num_list = {num_list}")
            if min(num_list) == max(num_list):
                # print(f"min = {min(num_list)}, max = {max(num_list)}")
                norm_scores = [1] * len(num_list)
            else:
                norm_scores = list()
                score_dist = max(num_list) - min(num_list)
                for score in num_list:
                    norm_score = (score - min(num_list)) / score_dist
                    norm_scores.append(norm_score)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
