#!/usr/bin/env python3

import argparse
import json
import logging
import math
import os
import re
import string

import lib.semantic_search as ss
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

# This is the most effective way to kill the 'Loading weights' bar
transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # verify
    verify_parser = subparsers.add_parser("verify", help="verify model")
    # embed_text
    embed_text_parser = subparsers.add_parser("embed_text", help="emded text")
    embed_text_parser.add_argument("text", type=str, help="text input")
    # verify_embed
    verify_emb_parser = subparsers.add_parser("verify_embeddings", help="verify embed")
    # embedquery
    embedquery_parser = subparsers.add_parser("embedquery", help="embed q")
    embedquery_parser.add_argument("query", type=str, help="query")
    # search
    search_parser = subparsers.add_parser("search", help="help")
    search_parser.add_argument("query", type=str, help="query")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="limit")
    # chunk
    chunk_parser = subparsers.add_parser("chunk", help="chunk")
    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, nargs="?", default=200, help="size"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="overlap"
    )
    # semantic chunk
    semchunk_parser = subparsers.add_parser("semantic_chunk", help="semantic chunks")
    semchunk_parser.add_argument("text", type=str, help="text")
    semchunk_parser.add_argument(
        "--max-chunk-size", type=int, nargs="?", default=4, help="max chunk"
    )
    semchunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="overlap"
    )
    # embed_chunks
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="embed chunks")
    # search_chunked
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="seach chunked"
    )
    search_chunked_parser.add_argument("query", type=str, help="query")
    search_chunked_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="limit"
    )

    args = parser.parse_args()
    match args.command:
        case "verify":
            ss.verify_model()

        case "search_chunked":
            with open("data/movies.json") as jfile:
                movies_dict = json.load(jfile)["movies"]
            css = ss.ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(movies_dict)
            # chunks = css.build_chunk_embeddings(movies_dict)
            print(f"num chunk_metadata = {len(css.chunk_metadata)}")
            print(f"num chunk_embedding = {len(css.chunk_embeddings)}")
            chunks = css.search_chunks(args.query, args.limit)
            cidx = 1
            for chunk in chunks:
                print(f"\n{cidx}. {chunk["title"]} (score: {chunk["score"]:.4f})")
                print(f"\n    {chunk["description"][:60]}")
                cidx += 1

        case "embed_chunks":
            with open("data/movies.json") as jfile:
                documents = json.load(jfile)["movies"]
            css = ss.ChunkedSemanticSearch()
            embeddings = css.build_chunk_embeddings(documents)
            # embeddings = css.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "semantic_chunk":
            text_input = args.text
            print(f"text_input ==={text_input}=== is there a space")
            if not text_input:
                print("semantic_chunk > args.text is empty")
                return
            text_input = text_input.strip()
            print(f"text_input ==={text_input}=== is there a space after strip")
            if not text_input:
                print("semantic_chunk > text_input is empty after strip")
                return
            text_all = re.split(r"(?<=[.!?])\s+", text_input)
            one_line = False
            # if one sentence AND no puncutation, process text_all as one sentence
            if len(text_all) == 1 and not any(
                char in string.punctuation for char in text_input[-1]
            ):
                one_line = True
                print(f"one_line = {one_line}")
            print(f"Semantically chunking {len(text_input)} chunks")
            minidx = 0
            cidx = 1
            while minidx < len(text_all):
                maxidx = min(minidx + args.max_chunk_size, len(text_all))
                if one_line:
                    maxidx = len(text_all)
                    print("one_line is true")
                print(f"{cidx}. {' '.join(text_all[minidx:maxidx])}")
                if maxidx == len(text_all):
                    break
                minidx = maxidx - args.overlap
                cidx += 1

        case "chunk":
            text_all = args.text.split()
            total_chunks = math.ceil(len(text_all) / args.chunk_size)
            print(f"Chunking {len(args.text)} characters")
            print(f"len(text_all) = {len(text_all)}")
            for cidx in range(0, total_chunks):
                minidx = cidx * args.chunk_size
                maxidx = min(minidx + args.chunk_size, len(text_all))
                if cidx > 0:
                    minidx = minidx - args.overlap
                print(f"minidx = {minidx} and maxidx = {maxidx}")
                print(f"{cidx + 1}. {' '.join(text_all[minidx:maxidx])}")

        case "search":
            sems = ss.verify_embeddings()
            docs_found = sems.search(args.query, args.limit)
            didx = 1
            for doc in docs_found:
                print(
                    f"{didx}. {doc['title']} (score: {doc['score']})\n   {doc['description'][:80]}..."
                )
                didx += 1
            pass

        case "embedquery":
            ss.embed_query_text(args.query)
        case "embed_text":
            ss.embed_text(args.text)
        case "verify_embeddings":
            ss.verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
