# https://excelsior-cjh.tistory.com/93
# https://lovit.github.io/nlp/2019/04/30/textrank/
# https://velog.io/@seolini43/%ED%8C%8C%EC%9D%B4%EC%8D%AC-TextRank%ED%85%8D%EC%8A%A4%ED%8A%B8%EB%9E%AD%ED%81%AC%EB%9E%80..-%EA%B0%84%EB%8B%A8%ED%95%98%EA%B2%8C-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EA%B8%B0
# https://github.com/spookyQubit/TextRank

from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import numpy as np

class RankText:
    def __init__(self):
        self.raw_sentences=''
        self.sentences=''
        self.tfidf_mat=''
        self.cv_mat=''
        self.vocab=''
        self.directed_graph_weights_sentences=''
        self.directed_graph_weights_words=''
        self.ranks_of_sentences=''
        self.ranks_of_words=''

    def get_sentences(self, doc):
        sentence_tokenizer = PunktSentenceTokenizer()
        return sentence_tokenizer.tokenize(doc)

    def remove_non_words(self, sentences):
        regex = re.compile('[^a-zA-Z~\-!@#$%^&*()+|<>?:{}\" "]')
        return [regex.sub('', s) for s in sentences]

    # Insert document (start method)
    def setSetences(self, document):
        self.raw_sentences = self.get_sentences(document) # From the document, extract the list of sentences
        self.sentences = self.remove_non_words(self.raw_sentences) # Remove all non-words from raw_sentences


    # A callable class which stems the word to its root according to the rules defined in ProterStemmer
    class PorterTokenizer(object):
        def __init__(self):
            self.porter = PorterStemmer()

        def __call__(self, *args, **kwargs):
            return [self.porter.stem(word) for word in args[0].split()]

    # We create a term-frequency-inverse-document-frequency vectorizer object
    # Input: List of sentences.
    # Processing: 1) Remove stop words defined in stop_words from the sentences and
    #             2) Stem the words to its roots according to PorterStemmer
    def setSentencesGraph(self):
        tfidf = TfidfVectorizer(preprocessor=None,
                                stop_words=stopwords.words('english'),
                                tokenizer=self.PorterTokenizer())
        self.tfidf_mat = tfidf.fit_transform(self.sentences).toarray()

    def setWordsGraph(self):
        cv = CountVectorizer(preprocessor=None,
                                stop_words=stopwords.words('english'),
                                tokenizer=self.PorterTokenizer())
        self.cv_mat = normalize(cv.fit_transform(self.sentences).toarray().astype(float), axis=0)
        self.vocab = cv.vocabulary_

    # directed_graph_weights_sentences: A square matrix with dimension (num_of_sentences x num_of_sentences).
    #                                 : This matrix is symmetric.
    #                                 : (i,j)th element of the matrix specifies the similarity between sentences i and j.
    def setWeights(self):
        self.directed_graph_weights_sentences = np.dot(self.tfidf_mat, self.tfidf_mat.T)
        self.directed_graph_weights_words = np.dot(self.cv_mat.T, self.cv_mat)

    def get_ranks(self, directed_graph_weights, d=0.85):
        A = directed_graph_weights
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0
            col_sum = np.sum(A[:, id])
            if col_sum != 0:
                A[:, id] /= col_sum
            A[:, id] *= -d
            A[id, id] = 1

        B = (1 - d) * np.ones((matrix_size, 1))

        ranks = np.linalg.solve(A, B)
        return {idx: r[0] for idx, r in enumerate(ranks)}

    def setRank(self):
        self.setSentencesGraph()
        self.setWordsGraph()
        self.setWeights()

        self.ranks_of_sentences = self.get_ranks(self.directed_graph_weights_sentences, 0.85)
        self.ranks_of_words = self.get_ranks(self.directed_graph_weights_words, 0.85)




    # 문장 출력
    def display_highlighted_sentences(self, sentences_to_highlight = 3):
        sorted_sentences_ranks_idx = sorted(self.ranks_of_sentences, key=lambda k: self.ranks_of_sentences[k], reverse=True)
        summary = []
        for idx in range(len(self.raw_sentences)):
            if idx in sorted_sentences_ranks_idx[:sentences_to_highlight]:
                summary.append(self.raw_sentences[idx])

        return summary


    # 단어 출력
    def display_highlighted_words(self, words_to_highlight = 10):
        sorted_words_ranks_idx = sorted(self.ranks_of_words, key=lambda k: self.ranks_of_words[k], reverse=True)
        term = []
        for s in self.raw_sentences:
            for w_ in s.split(' '):
                regex = re.compile('[^a-zA-Z\" "]')
                w = regex.sub('', w_)
                if len(self.PorterTokenizer().__call__(w))!=0:
                    stemmed_word = self.PorterTokenizer().__call__(w)[0].lower()
                else:
                    stemmed_word = " "
                if stemmed_word in self.vocab and self.vocab[stemmed_word] in sorted_words_ranks_idx[:words_to_highlight]:
                    # html += ' / '+w_ ????????????????????????
                    if stemmed_word in term:
                        pass
                    else:
                        term.append(stemmed_word)
        return term


rt = RankText()
rt.setSetences("insert the document~~")
rt.setRank()

sen = rt.display_highlighted_sentences()
print(sen)