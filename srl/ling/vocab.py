from collections import defaultdict

PAD = u'PAD'
UNK = u'UNKNOWN'
MARK = u'MARK'
NMARK = u'NMARK'


class Vocab(object):

    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def has_key(self, word):
        return self.w2i.has_key(word)

    def get_id(self, word):
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def set_init_word(self):
        self.add_word(PAD)

    def set_labels(self, corpus, vocab_cut_off=0):
        self.add_word('_')
        self.add_word('A0')
        self.add_word('A1')
        self.add_word('A2')
        self.add_word('A3')
        self.add_word('A4')
        label_freqs = self.get_label_freqs(corpus)
        for label, freq in sorted(label_freqs.items(), key=lambda (k, v): -v):
            if freq <= vocab_cut_off:
                break
            self.add_word(label)

    def add_vocab(self, word_freqs, vocab_cut_off=0):
        for w, freq in sorted(word_freqs.items(), key=lambda (k, v): -v):
            if freq <= vocab_cut_off:
                break
            self.add_word(w)

    def add_vocab_from_corpus(self, corpus, min_unit='word', vocab_cut_off=0):
        word_freqs = self.get_word_freqs(corpus)
        self.add_vocab(word_freqs, vocab_cut_off)

    def add_vocab_from_lists(self, corpus, vocab_cut_off=0):
        word_freqs = self.get_word_freqs_in_lists(corpus)
        self.add_vocab(word_freqs, vocab_cut_off)

    @staticmethod
    def get_word_freqs(corpus):
        word_freqs = defaultdict(int)
        for sent in corpus:
            for w in sent:
                word_freqs[w.form] += 1
        return word_freqs

    @staticmethod
    def get_label_freqs(corpus_set):
        word_freqs = defaultdict(int)
        for corpus in corpus_set:
            if corpus is None:
                continue
            for sent in corpus:
                for w in sent:
                    for label in w.labels:
                        word_freqs[label] += 1
        return word_freqs

    @staticmethod
    def get_word_freqs_in_lists(corpus):
        word_freqs = defaultdict(int)
        for n_best_list in corpus:
            for w in n_best_list.words:
                word_freqs[w.form] += 1
        return word_freqs

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab
