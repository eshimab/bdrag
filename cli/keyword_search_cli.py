# hello
#!/usr/bin/env python3

import argparse
import json
import math
import string

import lib.keyword_search as ks
from search_utils import BM25_B, BM25_K1

# from nltk.stem import PorterStemmer


def main() -> None:

    # print(json.dumps(movie_dict, indent=4))
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    # subparse
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # subparse: search
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    # subparse: search > arg: query
    search_parser.add_argument("query", type=str, help="Search query")

    # subparse: build
    build_parser = subparsers.add_parser("build", help="build the movie index")

    # subparse: tf (term frequency)
    tf_parser = subparsers.add_parser("tf", help="get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term to search for")

    # subparse: idf
    idf_parser = subparsers.add_parser("idf", help="calculate inverse document freq")
    idf_parser.add_argument("term", type=str, help="term to calculate idf for")

    # subparse tfidf
    tfidf_parser = subparsers.add_parser("tfidf", help="calculate tf-idf score")
    tfidf_parser.add_argument("doc_id", type=int, help="document id")
    tfidf_parser.add_argument("term", type=str, help="term to search for")

    # subparse bmidf
    bmidf_parser = subparsers.add_parser("bm25idf", help="get bm25 idf")
    bmidf_parser.add_argument("term", type=str, help="term to get bmidf for")

    # subparse bm25tf
    bmtf_parser = subparsers.add_parser("bm25tf", help="get bm25 saturated tf")
    bmtf_parser.add_argument("doc_id", type=int, help="document id")
    bmtf_parser.add_argument("term", type=str, help="term to find")
    bmtf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="tuninig"
    )
    bmtf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="b tuninig"
    )

    # subparse bm25search
    bms_parser = subparsers.add_parser("bm25search", help="search with bm25")
    bms_parser.add_argument("query", type=str, help="query to search")
    bms_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="query to search"
    )

    args = parser.parse_args()

    inverted_index = ks.InvertedIndex(dict())
    try:
        inverted_index.load()
    except ValueError:
        print("inverted_index.load() failed")

    # begin args
    match args.command:
        case "search":
            search_tokens = set(inverted_index.tokenize(args.query))
            print(f"search_tokens = {search_tokens}")
            doc_id_set = set()
            for token in search_tokens:
                doc_ids = inverted_index.get_document(token)
                if doc_ids:
                    doc_id_set.update(doc_ids)
                if len(doc_id_set) >= 5:
                    break
            if len(doc_id_set) == 0:
                print(f"no results found for {args.query}")
                return None
            title_set = set()
            for doc_id in doc_id_set:
                # print(f"doc_id = {doc_id}")
                title_set.add(inverted_index.docmap[doc_id]["title"])
            print("printing doc_id and titles")
            mdx = 1
            for doc_id, title in zip(doc_id_set, title_set):
                print(f"{doc_id}: {title}")
                mdx += 1
                if mdx > 5:
                    break
            pass

        case "bm25search":
            bms = inverted_index.bm25_search(args.query, args.limit)
            didx = 0
            for bm_dict in bms:
                didx += 1
                title = bm_dict["title"]
                fstr = (
                    f"{didx}. ({bm_dict["id"]}) {title} - Score: {bm_dict["score"]:.4f}"
                )
                print(fstr)
                if didx == 5:
                    break
            pass

        case "bm25idf":
            bmidf = inverted_index.get_bm25_idf(args.term)
            print(f"bm25idf for '{args.term}': {bmidf:.2f}")
            pass

        case "bm25tf":
            bm25tf = inverted_index.get_bm25_tf(args.doc_id, args.term)
            print(f"bm25tf for '{args.term}' in doc '{args.doc_id:}': {bm25tf:.2f}")
            pass

        case "tfidf":
            doc_id = args.doc_id
            term = args.term
            tf_num = inverted_index.get_tf(doc_id, term)
            idf_num = inverted_index.get_idf(term)
            tf_idf = tf_num * idf_num
            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
            pass

        case "idf":
            total_doc_count = len(inverted_index.docmap)
            term_match_doc_count = len(inverted_index.get_document(args.term))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            pass

        case "tf":
            doc_id = args.doc_id
            term = args.term
            # term frequency search
            print(f"tf > doc_id = {doc_id} and term = {term}")
            tf_num = inverted_index.get_tf(doc_id, term)
            if tf_num:
                print(f"tf > {doc_id}: found {tf_num} occurences of {term}")
            else:
                print(f"tf > {doc_id}: {term} not found. 0 occurences")
            pass

        case "build":
            inverted_index.build()
            inverted_index.save()
            pass

        case _:
            print(f"index has {len(inverted_index.docmap)} entries")
            parser.print_help()
            pass


if __name__ == "__main__":
    main()
