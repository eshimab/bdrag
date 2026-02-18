import json
import logging
import os
import re
import string

import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text_input):
    ss = SemanticSearch()
    emb = ss.generate_embedding(text_input)
    print(f"Text: {text_input}")
    print(f"First 3 dimensions: {emb[:3]}")
    print(f"Dimensions: {emb.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    with open("data/movies.json") as jfile:
        movies_dict = json.load(jfile)
    emb = ss.load_or_create_embeddings(movies_dict["movies"])
    print(f"Number of docs: {len(ss.documents)}")
    print(f"Embeddings shape: {emb.shape[0]} vectors in {emb.shape[1]} dimensions")
    return ss


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    ss = SemanticSearch()
    embq = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embq[:5]}")
    print(f"Shape: {embq.shape}")


class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def search(self, query, limit) -> list:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embedding` first"
            )
        embq = self.generate_embedding(query)
        eidx = 0
        score_list = list()
        for emb in tqdm.tqdm(self.embeddings):
            similarity_score = cosine_similarity(emb, embq)
            score_list.append((similarity_score, self.documents[eidx]))
            eidx += 1
        sorted_list = sorted(score_list, key=lambda x: x[0], reverse=True)
        final_list = list()
        for idx in range(0, limit):
            score = sorted_list[idx][0]
            doc = sorted_list[idx][1]
            dict_out = dict()
            dict_out["score"] = score
            dict_out["title"] = doc["title"]
            dict_out["description"] = doc["description"]
            final_list.append(dict_out)
        return final_list

    def generate_embedding(self, text_input):
        if not text_input or not text_input.strip():
            raise ValueError("generate_embedding > text_input is empty")
        embedding = self.model.encode([text_input])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        doc_strings = list()
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        else:
            print(f"cache/movie_embeddings.npy not found, rebulding")
            return self.build_embeddings(documents)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks = list()
        chunks_meta = list()
        for midx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
            doc_desc = doc["description"]
            if not len(doc_desc):
                print(f"skipped {doc["id"]} - {doc["title"]}")
                continue
            # strip whitespace before split for edge-case
            doc_desc = doc_desc.strip()
            # if empty after strip, continue
            if not doc_desc:
                print(f"skipped {doc["id"]} - {doc["title"]} after stripping")
                continue
            text_all = re.split(r"(?<=[.!?])\s+", doc_desc)
            one_line = False
            # if one sentence AND no puncutation, process text_all as one sentence
            if len(text_all) == 1 and not any(
                char in string.punctuation for char in doc_desc[-1]
            ):
                one_line = True
            # Init loop
            chunk_size = 4
            overlap = 1
            minidx = 0
            chunk_idx = 0
            num_chunks_init = len(chunks_meta)
            while minidx < len(text_all):
                # update idx
                maxidx = min(minidx + chunk_size, len(text_all))
                chunk_meta = dict()
                chunk_meta["id"] = doc["id"]
                chunk_meta["movie_idx"] = midx
                chunk_meta["chunk_idx"] = chunk_idx
                chunk_meta["total_chunks"] = chunk_idx
                # if one sentence and no punctuation, only loop once.
                if one_line:
                    maxidx = len(text_all)
                text_raw = text_all[minidx:maxidx]
                # save to meta lists
                chunk_text = " ".join(text_raw).strip()
                # strip before append to list
                chunk_text = chunk_text.strip()
                # skip if nothing after stripping
                if not chunk_text:
                    print(f"skipped chunk: {chunk_idx} due to nothing after strip")
                    continue
                chunks.append(chunk_text)
                chunks_meta.append(chunk_meta)
                # early exit
                if maxidx == len(text_all):
                    break
                # update idx
                minidx = maxidx - overlap
                chunk_idx += 1
            # update total chunks for all chunk_meta dicts for doc_id
            for midx in range(num_chunks_init, len(chunks_meta)):
                chunks_meta[midx]["total_chunks"] = chunk_idx
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_meta
        print(f"build_embeddings > num embeddings = {len(self.chunk_embeddings)}")
        print(f"build_embeddings > num metadata = {len(self.chunk_metadata)}")
        # save data
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as jfile:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks)},
                jfile,
                indent=2,
            )
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10):
        if not query:
            raise ValueError("generate_embedding > text_input is empty")
            return list()
        query.strip()
        if not query:
            raise ValueError("generate_embedding > text_input empty after strip")
            return list()
        embq = self.model.encode([query])[0]
        cidx = 0
        chunk_scores = list()
        movie_scores = dict()
        print(f"chunk_metadata = {len(self.chunk_metadata)}")
        print(f"chunk_embeddings = {len(self.chunk_embeddings)}")
        for chunk_emb in self.chunk_embeddings:
            # print(f"cidx = {cidx}")
            chunk_meta = self.chunk_metadata[cidx]
            cs_dict = dict()
            movie_idx = chunk_meta["movie_idx"]
            cossim = cosine_similarity(chunk_emb, embq)
            cs_dict["chunk_idx"] = chunk_meta["chunk_idx"]
            cs_dict["movie_idx"] = chunk_meta["movie_idx"]
            cs_dict["id"] = chunk_meta["id"]
            cs_dict["score"] = cossim
            chunk_scores.append(cs_dict)
            if movie_idx not in movie_scores or movie_scores[movie_idx] < cossim:
                movie_scores[movie_idx] = cossim
            cidx += 1
        scores_sorted = sorted(
            chunk_scores, key=lambda cs_dict: cs_dict["score"], reverse=True
        )
        final_list = list()
        doc_id_uni = set()
        for cs_dict in scores_sorted:
            fin_dict = dict()
            doc_id = cs_dict["id"]
            if doc_id in doc_id_uni:
                print(f"Duplicate entry for doc_id: {doc_id}, skipping")
                continue
            doc_id_uni.add(doc_id)
            doc = self.document_map[doc_id]
            fin_dict["id"] = doc_id
            fin_dict["title"] = doc["title"]
            fin_dict["description"] = doc["description"][:100]
            fin_dict["score"] = cs_dict["score"]
            fin_dict["metadata"] = {}
            final_list.append(fin_dict)
            if len(final_list) == limit:
                break
        return final_list

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("cache/chunk_embeddings.npy"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
        else:
            print(f"Could not load chunk_embeddings")
        if os.path.exists("cache/chunk_metadata.json"):
            with open("cache/chunk_metadata.json", "r") as jfile:
                self.chunk_metadata = json.load(jfile)["chunks"]
        else:
            print(f"Could not load chunk_metadata")
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            print(f"Could not load chunk_embeddings or metadata")
            return self.build_chunk_embeddings(documents)
        return self.chunk_embeddings
