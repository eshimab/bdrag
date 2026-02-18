import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex(documents)
        # if not os.path.exists(self.idx.index_path):
        # self.idx.build()
        # self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bms = self._bm25_search(query, limit * 500)
        css = self.semantic_search.search_chunks(query, limit * 500)
        bm_scores = [bm_dict["score"] for bm_dict in bms]
        cs_scores = [cs_dict["score"] for cs_dict in css]
        bm_ids = [bm_dict["id"] for bm_dict in bms]
        cs_ids = [cs_dict["id"] for cs_dict in css]
        id_set = set(bm_ids) | set(cs_ids)  # create union of sets
        print(f"num css = {len(cs_scores)}")
        print(f"num bms = {len(bms)}")
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
            if not doc_id in bm_ids:
                bm_score = 0.0
                bm_score_raw = 0.0
                bm_dict = dict()
            else:
                bm_dict_list = [bmsc for bmsc in bms if bmsc["id"] == doc_id]
                if len(bm_dict_list) > 1:
                    print(f"weighted_search > bm_score_list len = {len(bm_dict_list)}")
                    raise ValueError(f"weighted_search > bm_score_list too long")
                bm_dict = bm_dict_list[0]
                bm_score_raw = bm_dict["score"]
                bm_score = (bm_score_raw - bm_min) / bm_dist
            # get cs_score
            if not doc_id in cs_ids:
                cs_score = 0.0
                cs_score_raw = 0.0
                cs_dict = dict()
            else:
                cs_dict_list = [cssc for cssc in css if cssc["id"] == doc_id]
                if len(cs_dict_list) > 1:
                    print(f"weighted_search > cs_dict_list len = {len(cs_dict_list)}")
                    raise ValueError(f"weighted_search > cs_score_list too long")
                cs_dict = cs_dict_list[0]
                cs_score_raw = cs_dict["score"]
                cs_score = (cs_score_raw - cs_min) / cs_dist
            # get hybrid_score
            hs_score = hybrid_score(bm_score, cs_score, alpha)
            # get doc_id metadata
            if not cs_dict:
                title = bm_dict["title"]
                description = bm_dict["description"]
            else:
                title = cs_dict["title"]
                description = cs_dict["description"]
            # assign to output dict
            hybrid_dict["id"] = doc_id
            hybrid_dict["title"] = title
            hybrid_dict["description"] = description
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

        # raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
