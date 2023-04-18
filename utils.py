from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import nltk
from collections import Counter
import unidecode
import unicodedata

import pandas as pd
import scipy
import numpy as np
import re # To do ReGeX
import spacy # For NER
nlp = spacy.load('en_core_web_sm')
from fuzzywuzzy import fuzz # To compute similiraties.

import editdistance

import gensim.downloader as api
from scipy.spatial.distance import cosine

import fasttext.util


# ======================= Functions for feature engineering =======================
# Number of words for a given text
def words_count(text):
    '''
    Args:
      text (str): The input text to count the number of words

    Returns:
      int : The number of words in the given text
    '''
    return len(tokenize_text(text))


# Number of non ASCII words for a given text
def nonAscii_word_count(text):
    '''
    Args:
      text (str): The input text to count the number of non-ASCII words

    Returns:
      int : The number of non-ASCII words in the given text
    '''

    # Split sentence into words
    words = tokenize_text(text)

    # Initialize counter for non-ASCII words
    non_ascii_word_count = 0

    # Loop through words and check if each one contains non-ASCII characters
    for word in words:
        # Normalize the word to its canonical form (NFKD) to separate diacritics
        normalized_word = unicodedata.normalize('NFKD', word)
        # Check if any character in the normalized word has a non-ASCII category
        if any(not c.isascii() for c in normalized_word):
            non_ascii_word_count += 1

    return non_ascii_word_count


def common_words_count(text1, text2):
    '''
    Args:
      text1 (str): First sentence
      text2 (str): Second sentence

    Return:
      int: The number of common words that the two sentences have in common
    '''
    # Compute the tokens for each sentence
    tokens1 = set(tokenize_text(text1))
    tokens2 = set(tokenize_text(text2))

    # Return the number of common words
    return len(tokens1 & tokens2)


def one_hot_begin(corpus):
    '''
    Args:
      corpus (list): The whole corpus to create the one hot encoding

    Return:
      dataframe : A dataframe with the one hot encoding
    '''
    # Define the one-hot encoding labels
    labels = ['who', 'where', 'when', 'why', 'what', 'which', 'how']

    # Initialize an empty list to store the one-hot encodings
    one_hot_encodings = []

    # Iterate through each sentence in the dataset
    for question in corpus:
        # Initialize a list of zeros
        one_hot_encoding = [0] * len(labels)

        # Split the sentence into individual words
        words = tokenize_text(question)

        # Check if the first word of the sentence is in the labels list
        if words[0] in labels:
            one_hot_encoding[labels.index(words[0])] = 1

        # Add the one-hot encoding to the list of encodings
        one_hot_encodings.append(one_hot_encoding)

    # Convert the list of encodings to a pandas dataframe
    df_one_hot = pd.DataFrame(one_hot_encodings, columns=labels)

    return df_one_hot

def first_word_equal(row):
    """Computes whether the first word of the two questions are equal.

    Args:
        row: A pandas Series containing the 'question1' and 'question2' columns.

    Returns:
        A binary value indicating whether the first word of the two questions are equal.
    """
    q1_words = row['question1'].split()
    q2_words = row['question2'].split()
    return int(q1_words[0] == q2_words[0])

def last_word_equal(row):
    """Computes whether the last word of the two questions are equal.

    Args:
        row: A pandas Series containing the 'question1' and 'question2' columns.

    Returns:
        A binary value indicating whether the last word of the two questions are equal.
    """
    q1_words = row['question1'].split()
    q2_words = row['question2'].split()
    return int(q1_words[-1] == q2_words[-1])

def common_words_count(row):
    """Computes the number of common words between the two questions.

    Args:
        row: A pandas Series containing the 'question1' and 'question2' columns.

    Returns:
        An integer value indicating the number of common words between the two questions.
    """
    q1_words = row['question1'].split()
    q2_words = row['question2'].split()
    common_words = set(q1_words).intersection(set(q2_words))
    return len(common_words)

def common_words_ratio(row):
    """Computes the ratio of common words between the two questions to the total number of words in both questions.

    Args:
        row: A pandas Series containing the 'question1' and 'question2' columns.

    Returns:
        A float value indicating the ratio of common words between the two questions to the total number of words in both questions.
    """
    q1_words = row['question1'].split()
    q2_words = row['question2'].split()
    common_words = set(q1_words).intersection(set(q2_words))
    return len(common_words) / (len(q1_words) + len(q2_words))

def fuzz_ratio(row):
    """Computes the fuzzy string matching ratio between the two questions.

    Args:
        row: A pandas Series containing the 'question1' and 'question2' columns.

    Returns:
        An integer value indicating the fuzzy string matching ratio between the two questions.
    """
    return fuzz.ratio(row['question1'], row['question2'])

def longest_substring_ratio(row):
    """Computes the ratio of the length of the longest common substring between the two questions to the length of the shorter question.

    Args:
        row: A pandas Series containing the 'question1' and 'question2' columns.

    Returns:
        A float value indicating the ratio of the length of the longest common substring between the two questions to the length of the shorter question.
    """
    # Extract the values of 'question1' and 'question2' from the input row
    q1 = row['question1']
    q2 = row['question2']
    # If q1 is longer than q2, swap their values
    if len(q1) > len(q2):
        q1, q2 = q2, q1
    # Compute the length of q1 and create an empty list to store the substring scores
    len_q1 = len(q1)
    substr_scores = []
    # Iterate over all possible substrings of q1
    for i in range(len_q1):
        for j in range(i+1, len_q1+1):
            # Extract the substring from q1 and compute its ratio score with q2
            substr = q1[i:j]
            substr_scores.append(fuzz.ratio(substr, q2) / len(substr))
    # Return the maximum score in the list of substring scores
    return max(substr_scores)

def num_of_characters(q):
    return len(q)

def difference_word_count(q1, q2):
    len_q1 = len(q1.split())
    len_q2 = len(q2.split())
    return abs(len_q1 - len_q2)


def num_of_unique_words(q1, q2):
    q1q2 = q1 + " " + q2
    words = q1q2.split()
    num_unique_words = len(set(words))
    return num_unique_words

def num_of_words(q1, q2):
    q1q2 = q1 + " " + q2
    words = q1q2.split()
    num_words = len(words)
    return num_words

def total_unique_words_ratio(q1, q2):
    num_unique_words = num_of_unique_words(q1, q2)
    num_words = num_of_words(q1, q2)
    return num_unique_words/num_words


def oov_count(text, vocab):
    """
    Computes the number of out of vocabulary words in a text given a vocabulary.

    Parameters:
        text (str): The text to compute the OOV counts for.
        vocab (set): A set containing the vocabulary of known words.

    Returns:
        int: The number of out of vocabulary words in the text.
    """
    words = text.split()
    oov_words = [word for word in words if word.lower() not in vocab]
    return len(oov_words)


def rare_word_count(text, word_counts, threshold):
    """
    Computes the count of rare words in a text.

    Args:
        text (str): The input text to compute the rare word count for.
        word_counts (dict): A dictionary containing the counts of each word in the corpus.
        threshold (int): The threshold for a word to be considered "rare".

    Returns:
        int: The count of rare words in the input text.
    """
    # Split the text into words
    words = text.split()

    # Compute the count of rare words
    rare_word_count = sum([1 for word in words if word_counts.get(word, 0) < threshold])

    return rare_word_count


def named_entity_overlap(text1, text2):
    """
    Computes the named entity overlap between two texts.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The named entity overlap score between the two texts.
    """
    # Tokenize the texts into sentences
    sentences1 = nltk.sent_tokenize(text1)
    sentences2 = nltk.sent_tokenize(text2)

    # Identify the named entities in each text
    entities1 = set()
    for sentence in sentences1:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        named_entities = nltk.ne_chunk(tagged, binary=False)
        for entity in named_entities:
            if isinstance(entity, nltk.tree.Tree):
                entity_name = " ".join([token[0] for token in entity])
                entities1.add(entity_name)

    entities2 = set()
    for sentence in sentences2:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        named_entities = nltk.ne_chunk(tagged, binary=False)
        for entity in named_entities:
            if isinstance(entity, nltk.tree.Tree):
                entity_name = " ".join([token[0] for token in entity])
                entities2.add(entity_name)

    # Compute the named entity overlap between the two texts
    overlap = len(entities1.intersection(entities2)) / float(len(entities1.union(entities2)))

    return overlap

def compute_word2vec_embeddings(text):
    """
    Computes the word2vec embedding for a given text by taking the mean of embeddings of all the words in the text.

    Args:
        text (str): The input text for which the word embeddings need to be computed.

    Returns:
        numpy.ndarray or None: The computed embedding for the given text. If no embeddings are found, returns None.
    """

    model = api.load("word2vec-google-news-300")

    # Convert text to lowercase and split it into individual words
    words = text.lower().split()

    # Initialize empty list for embeddings
    embeddings = []

    # Iterate through each word in the text
    for word in words:
        # Check if the word is present in the word2vec model's vocabulary
        if word in model.index_to_key:
            # If the word is present, append its embedding to the list of embeddings
            embeddings.append(model[word])

    # If no embeddings were found, return None
    if len(embeddings) == 0:
        return None
    else:
        # Take the mean of all embeddings to get a single embedding for the entire text
        return np.mean(embeddings, axis=0)

def compute_cosine_similarity(embedding1, embedding2):
    """
    Computes the cosine similarity between two given word embeddings.

    Args:
        embedding1 (numpy.ndarray or None): The first word embedding.
        embedding2 (numpy.ndarray or None): The second word embedding.

    Returns:
        float or None: The cosine similarity between the two embeddings. If either of the embeddings is None, returns None.
    """
    # Check if either of the embeddings is None
    if embedding1 is None or embedding2 is None:
        return None
    else:
        # Compute the cosine similarity between the two embeddings
        return 1 - cosine(embedding1, embedding2)

def compute_fasttext_embeddings(text):
    """
    Computes the FastText embedding for a given text by taking the mean of embeddings of all the words in the text.

    Args:
        text (str): The input text for which the word embeddings need to be computed.

    Returns:
        numpy.ndarray or None: The computed embedding for the given text. If no embeddings are found, returns None.
    """
    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft_model = fasttext.load_model('cc.en.300.bin')
    # Convert text to lowercase and split it into individual words
    words = text.lower().split()

    # Initialize empty list for embeddings
    embeddings = []

    # Iterate through each word in the text
    for word in words:
        embeddings.append(ft_model.get_word_vector(word))
    # If no embeddings were found, return None
    if len(embeddings) == 0:
        return None
    else:
        # Take the mean of all embeddings to get a single embedding for the entire text
        return np.mean(embeddings, axis=0)

# ======================= Functions for text preprocessing =======================
# Tokenize a text
def tokenize_text(text):
    '''
    Args:
      text (str): The input text to be tokenize

    Returns:
      list: The tokenized text in a list

    '''
    return [token.lower() for token in nltk.word_tokenize(text)]


# Remove punctuation symbols
def remove_punctuation(text, question_mark=True):
    '''
    Args:
      text (str): The input text to remove punctuations
      question_mark (bool, default=True): If True, the question_mark is removed

    Returns:
      str: The final text without punctuation symbols
    '''
    if question_mark:
        return re.sub(r'[^\w\s]', '', text)
    else:
        return re.sub(r'[^\w\s?]', '', text)


# Remove english stopwords
def remove_stopwords(text):
    """
    Args:
      text (str): The input text to remove stop words

    Returns:
      str: The final text without stop words
    """
    # Get the stop words vocabulary
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # Tokenize the text
    words = tokenize_text(text)
    # Replace
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# Replace all consecutive whitespace characters in the text string with a single space.
def normalize_spaces(text):
    '''
    Args:
      text (str): The input text to normalize
    Returns:
      str: The final normalized text
    '''
    return re.sub(r'\s+', ' ', text)


# Replace all non-alphabetic characters in the text string with a single space.
def remove_nonAlphaWord(text, question_mark=True):
    '''
    Args:
      text (str): The input text to replace non alphabetic characters
      question_mark (bool, default=True): If True, the question_mark is removed
    Returns:
      str : The final replaced text
    '''
    if question_mark:
        return re.sub(r'[^a-zA-Z]', ' ', text)
    else:
        return re.sub(r'[^a-zA-Z?]', ' ', text)


def remove_accents(text):
    '''
    Args:
      text (str): The input text to remove accent
    Return:
      str : The final text without accents
    '''
    return unidecode.unidecode(text)


# If a token appears less than max_count, we change the word to a general one
def special_tokens(corpus, max_count=1):
    '''
    Args:
      corpus (list): The input corpus to treat special tokens
      max_count (int, default = 1): Number of times required for the word to appear.
      Otherwise, it is change it to special_token

    Returns:
      list: The final corpus as amended
    '''
    # Count the frequency of each word in the corpus
    word_counts = Counter(word for sentence in corpus for word in tokenize_text(sentence))
    modified_corpus = []
    # Replace single-word occurrences with "special_token"
    for question in corpus:
        modified_sentence = ' '.join(
            'special_token' if word_counts[word] == max_count else word for word in tokenize_text(question))
        modified_corpus.append(modified_sentence)

    return modified_corpus

def mask_entities(text):
    """
    Masks named entities of types PERSON, GPE, LOC, DATE, TIME, MONEY, and ORG with their respective entity labels.

    Args:
    text (str): The input text to be masked.

    Returns:
    str: The text with named entities masked.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'LOC', 'DATE', 'TIME', 'MONEY', 'ORG']:
            text = text.replace(ent.text, f'<{ent.label_}>')
    return text

def normalize_text(text, contractions_dict, abbreviations_dict):
    try:
        # Convert text to lowercase
        text = text.lower()

        # Expand contractions
        for contraction, expansion in contractions_dict.items():
            text = re.sub(r"\b" + contraction + r"\b", expansion, text)

        # Replace abbreviations
        for abbreviation, full_form in abbreviations_dict.items():
            text = re.sub(r"\b" + abbreviation + r"\b", full_form, text)

        return text
    except:
        # NANs
        print(text)
        return ''


class BKTree:
    def __init__(self, distfn, words):
        self.distfn = distfn

        it = iter(words)
        root = next(it)
        self.tree = (root, {})

        for i in it:
            self._add_word(self.tree, i)

    def _add_word(self, parent, word):
        pword, children = parent
        d = self.distfn(word, pword)
        if d in children:
            self._add_word(children[d], word)
        else:
            children[d] = (word, {})

    def _search_descendants(self, parent, max_distance, distance, query_word):
        node_word, children_dict = parent
        dist_to_node = distance(query_word, node_word)
        self.visited_nodes.append(node_word)
        results = []

        if dist_to_node <= max_distance:
            results.append((dist_to_node, node_word))

        I = range(max(0, dist_to_node - max_distance), dist_to_node + max_distance + 1)
        for dist in I:
            if dist in children_dict:
                child = children_dict[dist]
                if child[0] not in self.visited_nodes:
                    results.extend(self._search_descendants(child, max_distance, distance, query_word))
        return results

    def query(self, query_word, max_distance):
        self.visited_nodes = []
        results = self._search_descendants(self.tree, max_distance, self.distfn, query_word)
        sorted_results = sorted(results)
        return sorted_results

def spellchecker(q, V):
    bk_tree = BKTree(editdistance.eval, V)
    correction = []
    for word in q.split():
        if word in V:
            correction.append(word)
        else:
            candidates = bk_tree.query(word, 2)
            if len(candidates)>0:
                correction.append(candidates[0][1])
            else:
                correction.append(word)
    return ' '.join(correction)

# ====================== Functions for trian_models and reproduce_results ======================
def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    # assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted = cast_list_as_strings(list(df["question1"]))
    q2_casted = cast_list_as_strings(list(df["question2"]))

    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)
    X_q1q2 = scipy.sparse.hstack((X_q1, X_q2))

    return X_q1q2


def get_mistakes(clf, X, y):
    """
    returns the indices of the mistakes made by the classifier
    """
    predictions = clf.predict(X)
    incorrect_predictions = predictions != y
    incorrect_indices, = np.where(incorrect_predictions)

    if np.sum(incorrect_predictions) == 0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions


def evaluate_model(clf, X, y):
    """
    returns the accuracy, roc auc score, precision, recall and confusion matrix of the classifier
    """
    predictions = clf.predict(X)
    accuracy = clf.score(X, y)
    roc_auc = roc_auc_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    log_loss_score = log_loss(y, predictions)

    print(f"accuracy: {accuracy}")
    print(f"roc auc: {roc_auc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1 score: {f1}")
    print(f"log loss: {log_loss_score}")

    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        X,
        y,
        cmap=plt.cm.Blues,
    )
    disp.ax_.set_title('Confusion Matrix')
    plt.show()
