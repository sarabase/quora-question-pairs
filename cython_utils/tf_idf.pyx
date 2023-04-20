from math import log
from collections import defaultdict
from typing import List, Tuple
from numpy import sqrt

def compute_tf_idf(docs: List[str]) -> List[Tuple[str, float]]:
    '''
    Args:
        docs (List[str]): Entire corpus of the dataset, idealy already preprocessed.
    
    Returns:
        List[Tuple[str, float]]: List of tuples indicating the word and its tf-idf value for each sentence.
    '''
    # Compute the term frequencies for each document
    term_freqs = [defaultdict(int) for _ in range(len(docs))]
    for i, doc in enumerate(docs):
        for word in doc.split():
            term_freqs[i][word] += 1

    # Compute the document frequencies for each term
    doc_freqs = defaultdict(int)
    for term_freq in term_freqs:
        for term in term_freq.keys():
            doc_freqs[term] += 1

    # Compute the tf-idf scores for each term in each document
    tf_idfs = []
    num_docs = len(docs)
    for i, term_freq in enumerate(term_freqs):
        tf_idf_scores = []
        for term, freq in term_freq.items():
            tf_idf = tf_idf_c(freq, doc_freqs[term], num_docs)
            tf_idf_scores.append((term, tf_idf))
        tf_idfs.append(tf_idf_scores)

    return tf_idfs

cdef double tf_idf_c(int term_freq, int doc_freq, int num_docs):
    cdef double tf = term_freq
    cdef double idf = log(num_docs / doc_freq)
    return tf * idf