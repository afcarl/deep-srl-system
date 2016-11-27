import numpy as np

from abc import ABCMeta, abstractmethod
from ..utils.io_utils import say
from ..ling.vocab import UNK

NA = '_'
PRD = 'V'


class Sample(object):
    __metaclass__ = ABCMeta

    def __init__(self, sent, window, vocab_word, vocab_label):
        """
        sent: 1D: n_words; Word()
        word_ids: 1D: n_words; word id
        prd_indices: 1D: n_prds; prd index
        x: 1D: n_elems
        y: 1D: n_prds, 2D: n_words; label id
        """
        self.sent = sent
        self.n_words = len(sent)
        self.prd_indices = self._set_prd_indices(sent)
        self.n_prds = len(self.prd_indices)

        self.word_ids = self._set_word_ids(sent, vocab_word)
        self.label_ids = self._set_label_ids(sent, vocab_label)

        self.x = self._set_x(window)
        self.y = self._set_y()

    @staticmethod
    def _set_word_ids(sent, vocab_word):
        word_ids = []
        for w in sent:
            if w.form not in vocab_word.w2i:
                w_id = vocab_word.get_id(UNK)
            else:
                w_id = vocab_word.get_id(w.form)
            word_ids.append(w_id)
        return word_ids

    @abstractmethod
    def _set_label_ids(self, sent, vocab_label):
        raise NotImplementedError

    @staticmethod
    def _set_prd_indices(sent):
        return [word.index for word in sent if word.is_prd]

    @abstractmethod
    def _set_x(self, window):
        raise NotImplementedError

    @abstractmethod
    def _set_y(self):
        raise NotImplementedError

    @staticmethod
    def _numpize(sample):
        return np.asarray(sample, dtype='int32')


class BaseSample(Sample):

    def _set_label_ids(self, sent, vocab_label):
        labels = [[] for i in xrange(len(sent[0].labels))]
        for word in sent:
            for i, label in enumerate(word.labels):
                labels[i].append(vocab_label.get_id(label))
        return labels

    def _set_x(self, window):
        x_w = self._numpize(self._get_word_phi(window))
        x_p = self._numpize(self._get_posit_phi(window))
        return [x_w, x_p]

    def _set_y(self):
        return self._numpize(self.label_ids)

    def _get_word_phi(self, window):
        phi = []

        ###################
        # Argument window #
        ###################
        slide = window / 2
        sent_len = len(self.word_ids)
        pad = [0 for i in xrange(slide)]
        a_sent_w_ids = pad + self.word_ids + pad

        ####################
        # Predicate window #
        ####################
        p_window = 5
        p_slide = p_window / 2
        p_pad = [0 for i in xrange(p_slide)]
        p_sent_w_ids = p_pad + self.word_ids + p_pad

        for prd_index in self.prd_indices:
            prd_ctx = p_sent_w_ids[prd_index: prd_index + p_window]
            p_phi = []

            for arg_index in xrange(sent_len):
                arg_ctx = a_sent_w_ids[arg_index: arg_index + window] + prd_ctx
                p_phi.append(arg_ctx)
            phi.append(p_phi)

        assert len(phi) == len(self.prd_indices)
        return phi

    def _get_posit_phi(self, window):
        phi = []
        slide = window / 2

        for prd_index in self.prd_indices:
            p_phi = [self._get_mark(prd_index, arg_index, slide) for arg_index in xrange(self.n_words)]
            phi.append(p_phi)

        assert len(phi) == len(self.prd_indices)
        return phi

    @staticmethod
    def _get_mark(prd_index, arg_index, slide):
        if prd_index - slide <= arg_index <= prd_index + slide:
            return 0
        return 1
