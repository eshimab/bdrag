import argparse
import json
import os
import time

import lib.hybrid_search as hybs
import lib.model_queries as mq
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

# load api key
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
# load model
client = genai.Client(api_key=api_key)


def desc_string(input_dict):
    return f"{input_dict["description"][: min(len(input_dict["description"]), 60)]}..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # normalize scores
    normalize_sp = subparsers.add_parser("normalize", help="normalize")
    normalize_sp.add_argument("scores", type=float, nargs="+", help="list of scores")
    # weighted search
    weighted_sp = subparsers.add_parser("weighted-search", help="weighted search")
    weighted_sp.add_argument("query", type=str, help="query")
    weighted_sp.add_argument(
        "--alpha", type=float, nargs="?", default=0.5, help="alpha"
    )
    weighted_sp.add_argument("--limit", type=int, nargs="?", default=5, help="limit")
    # rrf-search
    rrfs_sp = subparsers.add_parser("rrf-search", help="rrf search")
    rrfs_sp.add_argument("query", type=str, help="query")
    rrfs_sp.add_argument("-k", type=int, nargs="?", default=60, help="k")
    rrfs_sp.add_argument("--limit", type=int, nargs="?", default=5, help="limit")
    rrfs_sp.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite", "expand"], help="enhance"
    )
    rrfs_sp.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="rerank",
    )

    # init args
    args = parser.parse_args()

    match args.command:

        case "rrf-search":
            with open("data/movies.json") as jfile:
                documents = json.load(jfile)["movies"]
            hss = hybs.HybridSearch(documents)
            init_query = args.query
            if not args.enhance:
                query = init_query
            else:
                if args.enhance == "spell":
                    model_query = mq.model_spell(init_query)
                elif args.enhance == "rewrite":
                    model_query = mq.model_rewrite(init_query)
                elif args.enhance == "expand":
                    model_query = mq.model_expand(init_query)
                # send to model
                mresp = client.models.generate_content(
                    model="gemini-2.5-flash", contents=model_query
                )
                # get model data and report
                query = mresp.text
                print(f"Enhanced query ({args.enhance}): '{init_query}' -> '{query}'\n")
            # run query
            if args.rerank_method:
                print(f"rerank_method = {args.rerank_method}")
                total_limit = args.limit * 5
            else:
                total_limit = args.limit
            # get rrf scores
            rrfs = hss.rrf_search(query, args.k, total_limit)
            rr_map = {inner_dict["id"]: inner_dict for inner_dict in rrfs}
            for doc_id in rr_map:
                rr_map[doc_id]["init_query"] = init_query
                rr_map[doc_id]["query"] = query
                if args.enhance:
                    rr_map[doc_id]["enh_kind"] = args.enhance
                    rr_map[doc_id]["enh_query"] = model_query
                if args.rerank_method:
                    rr_map[doc_id]["rerank"] = args.rerank_method
            # model reranks
            if not args.rerank_method:
                rrfs_final = rrfs
            elif args.rerank_method and args.rerank_method == "individual":
                for ridx, rr in enumerate(rrfs):
                    doc_id = rr["id"]
                    doc = hss.semantic_search.document_map[doc_id]
                    model_query = mq.model_rerank_indv(query, doc)
                    rr["rerank_query"] = model_query
                    mresp = client.models.generate_content(
                        model="gemini-2.5-flash", contents=model_query
                    )
                    print(f"model_rank: {mresp.text} for {rr["title"]}")
                    rr["model_rank"] = int(mresp.text)
                    rrfs[ridx] = rr
                    time.sleep(3)
                rrfs_final = sorted(
                    rrfs, key=lambda inner_dict: inner_dict["model_rank"], reverse=True
                )
            elif args.rerank_method and args.rerank_method == "batch":
                # query model
                model_query = mq.model_rerank_batch(query, rrfs)
                mresp = client.models.generate_content(
                    model="gemini-2.5-flash", contents=model_query
                )
                model_text = mresp.text.strip()
                if model_text.startswith("```"):
                    model_text = model_text.strip("`").removeprefix("json").strip()
                model_ranks = json.loads(model_text)
                print(f"model_ranks = {model_ranks}")
                # re-rank rrfs list
                rrfs_final = list()
                ridx = 1
                for doc_id in model_ranks:
                    try:
                        doc_id = int(doc_id)
                    except (ValueError, KeyError):
                        print(f"Model doc_id = {doc_id} is not an int")
                        continue
                    if doc_id not in rr_map:
                        print(f"Model hallucinated doc_id = {doc_id}")
                        continue
                    rr = rr_map[doc_id]
                    print(f"model_rank: {ridx} for {rr["title"]}")
                    rr["model_rank"] = ridx
                    rr["rerank_query"] = model_query
                    rrfs_final.append(rr)
                    ridx += 1
            elif args.rerank_method and args.rerank_method == "cross_encoder":
                query_pairs = list()
                for rr in rrfs:
                    doc = hss.semantic_search.document_map[rr["id"]]
                    query_pairs.append(
                        [
                            query,
                            f"{doc.get('title', '')} - {doc.get('description', '')}",
                        ]
                    )
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(query_pairs)
                ridx = 1
                for rr, score in zip(rrfs, scores):
                    rr["cross_score"] = float(score)
                    print(f"{ridx}. {rr["title"]} - Score: {score}")
                    ridx += 1
                rrfs_final = sorted(
                    rrfs, key=lambda inner_dict: inner_dict["cross_score"], reverse=True
                )
            # print query meta
            print(f"\n========= Query Metadata ===============")
            print(f"   Original Query: {rrfs_final[0]["init_query"]}")
            if args.enhance:
                print(f"   Enhance Method: {rrfs_final[0]["enh_kind"]}")
                print(f"   Enhanced Query: {rrfs_final[0]["enh_query"]}")
            if args.rerank_method:
                print(f"   Rerank-method: {rrfs_final[0]["rerank"]}")
            # printing
            for ridx, rr in enumerate(rrfs_final, start=1):
                print(f"\n{ridx}. {rr["title"]}")
                if rr.get("model_rank"):
                    print(f"   Model Rank: {rr["model_rank"]}/{len(rrfs_final)}")
                    print(f"   Rerank Method: {args.rerank_method}")
                if rr.get("cross_score"):
                    print(f"   Cross Encoder Score: {rr["cross_score"]:.4f}")
                print(f"   RRF Score: {rr["rr_score"]:.4f} | k = {args.k}")
                print(f"   RRF Rank: {rr["rr_rank"]}")
                print(f"   BM25 Rank: {rr["bm_rank"]}, Semantic Rank: {rr["cs_rank"]}")
                print(
                    f"   BM25 Raw:  {rr["bm_raw"]:.4f}, Semantic Raw:  {rr["cs_raw"]:.4f}"
                )
                if rr.get("rerank_query"):
                    print(f"    Re-Rank Query: {rr["rerank_query"]}")
                if "The Land Before Time XI" in rr["title"]:
                    break
                # print(f"   {desc_string(rr)}")

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
                print(f"{desc_string(hs)}")
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
