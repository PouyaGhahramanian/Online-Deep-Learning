
import numpy as np

class TFIDF(object):

    def __init__(self, smoothing = 0.0):

        self.df = {}
        self.size = 0.0
        self.smoothing = smoothing

    def fit(self, docs):

        for doc in docs:
            self.size += 1
            for word in doc:
                if word in self.df:
                    self.df[word] = self.df[word] + 1
                else:
                    self.df[word] = 1

    def partial_fit(self, doc):

        self.size += 1
        for word in doc:
            if word in self.df:
                self.df[word] = self.df[word] + 1
            else:
                self.df[word] = 1

    def score(self, doc, word):

        s = self.smoothing
        N = float(len(doc))
        if word in doc:
            #tf = np.sum(np.asarray(doc[doc == word])) / N
            tf = 1.0 / N
            idf = np.log( ( self.size / self.df[word] ) )
            s = max(self.smoothing, tf * idf)

        return s

    def score(self, doc):

        N = float(len(doc))
        score_vec = []
        for word in doc:

            #tf = tf = np.sum(np.asarray(doc[doc == word])) / N
            tf = 1.0 / N
            idf = np.log( self.size / self.df[word] )
            s = max(self.smoothing, tf * idf)
            score_vec.append(s)

        return np.asarray(score_vec)

if __name__ == '__main__':

    document = ['here', 'is', 'a', 'very', 'nice', 'place', 'to', 'visit']
    document_2 = ['very', 'nice', 'place']
    tfidf = TFIDF(smoothing = 1e-5)
    tfidf.partial_fit(document)
    tfidf.partial_fit(document_2)
    ss = tfidf.score(document)
    print(ss)
    print(tfidf.df)
