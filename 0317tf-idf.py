import numpy as np

oringin_docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

def preprocess(docs):
    docs = [s.replace(',', "").split(' ') for s in docs]
    return docs

class Vocab():
    def __init__(self, docs):
        self.vocab = {'<unk>': 0}
        self.vocab_idx_to_word = ['<unk>']
        cnt = len(self.vocab)
        for sen in docs:
            for word in sen:
                if word not in self.vocab:
                    self.vocab[word] = cnt
                    self.vocab_idx_to_word.append(word)
                    cnt += 1


    def __getitem__(self, item):
        if item in self.vocab:
            return self.vocab[item]
        return 0

    def __len__(self):
        return len(self.vocab)

    def tolist(self):
        return self.vocab_idx_to_word



def build_vocab(docs):
    return Vocab(docs)

def token2idx(docs, vocab):
    return [[vocab[w] for w in s] for s in docs]

def cal_tf(docs, vocab_size):
    """
    why we need log(frequency+1.0) but not the frequency
    think about the unknown word that will cause the tf equals zero
    so the corresponding tf-idf equals zero
    we cannot verify which one is more important
    """
    vec = np.full([len(docs), vocab_size], 1.0)
    for i, sen in enumerate(docs):
        for w in sen:
            vec[i][w] += 1
        vec[i] /= len(sen)
    return np.log(vec)

def cal_idf(docs, vocab_size):
    # log(len(D) / (1+n) )
    vec = np.full(vocab_size, 1.0)
    for i in range(vocab_size):
        for sen in docs:
            if i in sen: vec[i] += 1
    return np.log(len(docs) / vec)

def cal_tfidf(vec_tf:np.ndarray, vec_idf:np.ndarray):
    return vec_tf * vec_idf

def cal_cosin_similarity(x, y):
    norm_x = np.sqrt(np.sum(x ** 2, axis=1))
    norm_y = np.sqrt(y @ y.T).squeeze(axis=1)
    temp = x @ y.T
    temp = temp.squeeze(axis=1) / norm_x / norm_y
    print(temp)
    return np.argsort(temp)


if __name__ == '__main__':
    docs = preprocess(oringin_docs)
    vocab = build_vocab(docs)
    docs = token2idx(docs, vocab)
    vec_tf = cal_tf(docs, len(vocab))
    vec_idf = cal_idf(docs, len(vocab))
    tfidf = cal_tfidf(vec_tf, vec_idf)
    print(vec_tf[:, 0])
    print(vec_idf[0])
    print(tfidf[:, 0])
    test = ['good day like bob have a cup of coffee bring']
    test = preprocess(test)
    test = token2idx(test, vocab)
    test_tfidf = cal_tfidf(cal_tf(test, len(vocab)), cal_idf(test, len(vocab)))
    most_similar_idx = cal_cosin_similarity(tfidf, test_tfidf)
    print(most_similar_idx)
    for i in most_similar_idx[-1:-4:-1]:
        print(oringin_docs[i])