import collections
import json
import math
import os
import pickle
import string
from itertools import islice

import tqdm
from nltk.stem import PorterStemmer
from search_utils import BM25_B, BM25_K1


class InvertedIndex:
    def __init__(self, index: dict):
        self.index = index
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()
        # get stop words, create stop word set, read file once for perf
        with open("data/stopwords.txt") as sfile:
            self.stop_words = sfile.read().splitlines()

    def tokenize(self, text_input) -> list:
        text = text_input.lower()
        # remove punctuation
        translation_table = str.maketrans("", "", string.punctuation)
        text = text.translate(translation_table)
        # split string, remove empty, and create unique set
        tokens_init = [str_split for str_split in text.split() if str_split]
        tokens_nostop = [tok for tok in tokens_init if tok not in self.stop_words]
        # init stemmer, stem tokens_init, create unique set
        stemmer = PorterStemmer()
        tokens_all = [stemmer.stem(tok) for tok in tokens_nostop]
        # remove the self.stop_words set from the tokens_stem set
        # could also be written tokens_final = tokens_stem - stop_word_set
        return tokens_all

    def make_term(self, text_input) -> str:
        token_list = self.tokenize(text_input)
        if not token_list:
            raise ValueError("make_term > no terms.")
        if len(token_list) > 1:
            raise ValueError("make_term > error: requires one arg")
        else:
            term = token_list[0]
        return term

    def __add_document(self, doc_id, text_input) -> None:
        tokens_all = self.tokenize(text_input)
        # count token frequency
        self.term_frequencies[doc_id] = collections.Counter(tokens_all)
        self.doc_lengths[doc_id] = len(tokens_all)
        # make token set and then add to index
        for token in set(tokens_all):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def build(self) -> None:
        with open("data/movies.json") as jfile:
            movies_dict = json.load(jfile)
        # iterate thru movies
        for movie in tqdm.tqdm(movies_dict["movies"]):
            # get movie metadata
            doc_id = movie["id"]
            desc = movie["description"]
            title = movie["title"]
            movie_text = f"{title} {desc}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, movie_text)

    def get_document(self, term_input) -> set:
        print(f"get_document > finding term: {term_input} in documents")
        term = self.make_term(term_input)
        if term in self.index:
            doc_id_set = sorted(self.index[term])
            print(f"get_document > found {len(doc_id_set)} docs with term: {term}")
            return doc_id_set
        else:
            print(f"get_document > no results for {term_input}")
            return None

    def get_idf(self, term_input) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_document(term_input))
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_tf(self, doc_id, term_input) -> int:
        term = self.make_term(term_input)
        term_count = 0
        if self.term_frequencies.get(doc_id):
            count_dict = self.term_frequencies[doc_id]
            if count_dict:
                term_count = self.term_frequencies[doc_id][term]
            # print(f"get_tf > found {term_count} occurences of {term} in doc: {doc_id}")
        else:
            print(f"get_tf > No term: {term} in doc_id: {doc_id}")
        return term_count

    def get_bms(self, doc_id, term_input: str) -> float:
        tf = self.get_bm25_tf(doc_id, term_input)
        idf = self.get_bm25_idf(term_input)
        bm25_raw = tf * idf
        return bm25_raw

    def bm25_search(self, query, limit=5) -> dict:
        query_tokens = set(self.tokenize(query))
        scores = dict()
        for token in query_tokens:
            if token in self.index:
                doc_ids = self.index[token]
                idf = self.get_bm25_idf(token)
                for doc_id in doc_ids:
                    tf_component = self.get_bm25_tf(doc_id, token)
                    score = idf * tf_component
                    scores[doc_id] = scores.get(doc_id, 0.0) + score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_scores:
            if score > 0:
                doc = self.docmap[doc_id]
                results.append(
                    {
                        "id": doc_id,
                        "title": doc["title"],
                        "description": doc["description"],
                        "score": score,
                    }
                )
        return results[:limit]

    def get_bm25_idf(self, term_input: str) -> float:
        doc_count = len(self.docmap)
        term_made = self.make_term(term_input)
        if term_made not in self.index:
            return 0
        term_doc_count = len(self.index[term_made])
        # doc_freq = len(self.get_document(term_input))
        bmidf = math.log(
            (doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1
        )
        # print(f"get_bmidf > bmidf for {term_input} is {bmidf}")
        return bmidf

    def get_bm25_tf(self, doc_id, term_input, k1=BM25_K1, b=BM25_B) -> float:
        term_freq = self.get_tf(doc_id, term_input)
        avg_length = self.__get_avg_doc_length()
        length_norm = 1.0 - b + b * (self.doc_lengths[doc_id] / avg_length)
        tf_component = (term_freq * (k1 + 1)) / (term_freq + k1 * length_norm)
        return tf_component

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            avg_length = 0.0
        else:
            avg_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return avg_length

    def __load_path(self, file_name) -> None:
        file_path = f"cache/{file_name}.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as att_file:
                return pickle.load(att_file)
        else:
            raise FileNotFoundError(f"no file found for {file_path}")

    def load(self) -> None:
        # print("semantic_search.load > loading inverted index atts")
        # index
        try:
            self.index = self.__load_path("index")
        except FileNotFoundError as fnf:
            print(f"error loading filename: {fnf}")
        # docmap
        try:
            self.docmap = self.__load_path("docmap")
        except FileNotFoundError as fnf:
            print(f"error loading filename: {fnf}")
        # term_frequencies
        try:
            self.term_frequencies = self.__load_path("term_frequencies")
        except FileNotFoundError as fnf:
            print(f"error loading filename: {fnf}")
        # doc_lengths
        try:
            self.doc_lengths = self.__load_path("doc_lengths")
        except FileNotFoundError as fnf:
            print(f"error loading filename: {fnf}")

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as file_obj:
            pickle.dump(self.index, file_obj)
            print(f"successful save: {file_obj}")
        with open("cache/docmap.pkl", "wb") as file_obj:
            pickle.dump(self.docmap, file_obj)
            print(f"successful save: {file_obj}")
        with open("cache/term_frequencies.pkl", "wb") as file_obj:
            pickle.dump(self.term_frequencies, file_obj)
            print(f"successful save: {file_obj}")
        with open("cache/doc_lengths.pkl", "wb") as file_obj:
            pickle.dump(self.doc_lengths, file_obj)
            print(f"successful save: {file_obj}")


#
