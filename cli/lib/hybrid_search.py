import os

from dotenv import load_dotenv
from google import genai

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

# load api key
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def sort_list_of_dict_by_key(list_of_dicts, key_string, reverse_sort=True):
    return sorted(
        list_of_dicts,
        key=lambda inner_dict: inner_dict[key_string],
        reverse=True,
    )


def xfer_fields(new_dict, old_dict):
    fields = ["id", "title", "description"]
    for keyname in fields:
        new_dict[keyname] = old_dict[keyname]
    return new_dict


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex(documents)

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_search(self, query=str, k_val=60, limit=5):
        # init embeddings
        # these should come pre sorted
        bms = self._bm25_search(query, limit * 500)
        css = self.semantic_search.search_chunks(query, limit * 500)
        # init
        rrf_scores = {}  # Use this like the guide
        # Loop 1: BM25
        for rank, result in enumerate(bms, start=1):
            doc_id = result["id"]
            bm_score = 1 / (k_val + rank)
            rrf_scores[doc_id] = {
                "id": doc_id,
                "score": bm_score,
                "bm_rank": rank,
                "bm_score": bm_score,
                "cs_rank": None,
                "cs_score": 0,
            }
        # Loop 2: Semantic
        for rank, result in enumerate(css, start=1):
            doc_id = result["id"]
            cs_score = 1 / (k_val + rank)
            if doc_id in rrf_scores:
                rrf_scores[doc_id]["score"] += cs_score
            else:
                rrf_scores[doc_id] = {
                    "id": doc_id,
                    "score": cs_score,
                    "bm_rank": None,
                    "bm_score": 0,
                }
            rrf_scores[doc_id]["cs_rank"] = rank
            rrf_scores[doc_id]["cs_score"] = cs_score
        # get metadata and compute rrf score
        for doc_id in rrf_scores:
            rrf_scores[doc_id]["title"] = self.semantic_search.document_map[doc_id][
                "title"
            ]
            rrf_scores[doc_id]["description"] = self.semantic_search.document_map[
                doc_id
            ]["description"]
            rrf_scores[doc_id]["score"] = round(rrf_scores[doc_id]["score"], 3)
        # sort and limit
        rr_sorted = sorted(
            rrf_scores.values(), key=lambda rrf: rrf["score"], reverse=True
        )
        return rr_sorted[0:limit]

    def weighted_search(self, query, alpha, limit=5):
        bms = self._bm25_search(query, limit * 500)
        css = self.semantic_search.search_chunks(query, limit * 500)
        bm_map = {doc["id"]: doc for doc in bms}
        cs_map = {doc["id"]: doc for doc in css}
        bm_scores = [bm_dict["score"] for bm_dict in bms]
        cs_scores = [cs_dict["score"] for cs_dict in css]
        bm_ids = [bm_dict["id"] for bm_dict in bms]
        bm_id_set = set(bm_ids)
        cs_ids = [cs_dict["id"] for cs_dict in css]
        cs_id_set = set(cs_ids)
        id_set = set(bm_ids) | set(cs_ids)  # create union of sets
        print(f"weighted_search > bm_ids = {len(bm_ids)}, bm_id_set = {len(bm_id_set)}")
        print(f"weighted_search > cs_ids = {len(cs_ids)}, cs_id_set = {len(cs_id_set)}")
        print(f"weighted_search > id_set = {len(id_set)}")
        bm_min = min(bm_scores)
        bm_dist = max(bm_scores) - bm_min
        cs_min = min(cs_scores)
        cs_dist = max(cs_scores) - cs_min
        if cs_dist == 0 or bm_dist == 0:
            print(f"weighted_search > cs_dist = {cs_dist} and bm_dist = {bm_dist}")
            raise ValueError("divide by zero inc")
        hybrid_list = list()
        for doc_id in id_set:
            hybrid_dict = dict()
            # get bm_score
            if doc_id not in bm_ids:
                bm_score = 0.0
                bm_score_raw = 0.0
                bm_dict = dict()
            else:
                bm_dict = bm_map[doc_id]
                bm_score_raw = bm_dict["score"]
                bm_score = (bm_score_raw - bm_min) / bm_dist
            # get cs_score
            if doc_id not in cs_ids:
                cs_score = 0.0
                cs_score_raw = 0.0
                cs_dict = dict()
            else:
                cs_dict = cs_map[doc_id]
                cs_score_raw = cs_dict["score"]
                cs_score = (cs_score_raw - cs_min) / cs_dist
            # get hybrid_score
            hs_score = hybrid_score(bm_score, cs_score, alpha)
            # assign to output dict
            hybrid_dict["id"] = doc_id
            hybrid_dict["title"] = self.semantic_search.document_map[doc_id]["title"]
            hybrid_dict["description"] = self.semantic_search.document_map[doc_id][
                "description"
            ]
            hybrid_dict["semantic_score"] = cs_score
            hybrid_dict["bm25_score"] = bm_score
            hybrid_dict["hybrid_score"] = round(hs_score, 3)
            hybrid_dict["cs_raw"] = cs_score_raw
            hybrid_dict["bm_raw"] = bm_score_raw
            hybrid_list.append(hybrid_dict)
        hybrid_sorted = sorted(
            hybrid_list,
            key=lambda hybrid_dict: hybrid_dict["hybrid_score"],
            reverse=True,
        )
        print(f"num hybrid = {len(hybrid_sorted)}")
        return hybrid_sorted[0:limit]
