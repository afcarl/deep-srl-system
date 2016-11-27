import math
import numpy as np

from ..utils.io_utils import say


class Eval(object):

    def __init__(self, vocab_label):
        self.vocab_label = vocab_label

        self.corrects = None
        self.results_sys = None
        self.results_gold = None

        self.precision = None
        self.recall = None
        self.f1 = None

        self.all_corrects = None
        self.all_results_sys = None
        self.all_results_gold = None

        self.all_precision = 0.
        self.all_recall = 0.
        self.all_f1 = 0.
        self.nll = 0.

        self._set_params()

    def _set_params(self):
        n_labels = self.vocab_label.size()-1
        self.corrects = np.zeros(n_labels, dtype='float32')
        self.results_sys = np.zeros(n_labels, dtype='float32')
        self.results_gold = np.zeros(n_labels, dtype='float32')

        self.precision = np.zeros(n_labels, dtype='float32')
        self.recall = np.zeros(n_labels, dtype='float32')
        self.f1 = np.zeros(n_labels, dtype='float32')

        self.all_corrects = np.zeros(n_labels, dtype='float32')
        self.all_results_sys = np.zeros(n_labels, dtype='float32')
        self.all_results_gold = np.zeros(n_labels, dtype='float32')

    def _summarize(self):
        self.precision = self.corrects / self.results_sys
        self.recall = self.corrects / self.results_gold
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        self.all_corrects = np.sum(self.corrects)
        self.all_results_sys = np.sum(self.results_sys)
        self.all_results_gold = np.sum(self.results_gold)

        self.all_precision = self.all_corrects / self.all_results_sys
        self.all_recall = self.all_corrects / self.all_results_gold
        self.all_f1 = 2 * self.all_precision * self.all_recall / (self.all_precision + self.all_recall)

    def show_results(self):
        self._summarize()
        say('\n\tNLL: %f' % self.nll)
        say('\n\n\tACCURACY')

        precision = self.corrects / self.results_sys
        recall = self.corrects / self.results_gold
        f1 = 2 * precision * recall / (precision + recall)

        """
        for case_index, (correct, result_sys, result_gold) in enumerate(zip(self.corrects,
                                                                            self.results_sys,
                                                                            self.results_gold)):
            case_name = self.vocab_label.get_word(case_index+1)
            say('\n\t%s:\n' % case_name)
            say('\tALL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
                f1[case_index], precision[case_index], int(correct), int(result_sys),
                recall[case_index], int(correct), int(result_gold)))
        """

        crr = int(self.all_corrects)
        r_sys = int(self.all_results_sys)
        r_gold = int(self.all_results_gold)

        say('\n\tTOTAL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
            self.all_f1, self.all_precision, crr, r_sys, self.all_recall, crr, r_gold)
            )

    def update_results(self, y_hat_batch, y_batch):
        assert len(y_hat_batch) == len(y_batch)
        for prd_i, (y_hat_prd, y_prd) in enumerate(zip(y_hat_batch, y_batch)):
            assert len(y_hat_prd) == len(y_prd)
            for word_index, (y_hat, y) in enumerate(zip(y_hat_prd, y_prd)):
                y_hat_case_index = y_hat - 1
                y_case_index = y - 1
                answer = self._judge_answer(y_hat, y)

                if -1 < y_hat_case_index:
                    self.results_sys[y_hat_case_index] += 1
                    self.corrects[y_hat_case_index] += answer
                if -1 < y_case_index:
                    self.results_gold[y_case_index] += 1

    @staticmethod
    def _judge_answer(y_hat, y):
        if y_hat == y:
            return 1
        return 0
