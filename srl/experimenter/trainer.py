from experimenter import Experimenter

from ..utils.io_utils import say


class Trainer(Experimenter):

    def __init__(self, argv, preprocessor, model_api, epoch_manager):
        super(Trainer, self).__init__(argv, preprocessor, model_api, epoch_manager, None)

    def _setup_word(self):
        say('\n\nSetting up vocabularies...\n')
        vocab_word, emb = self.preprocessor.load_init_emb()
        vocab_word, emb, untrainable_emb = self.preprocessor.create_trainable_emb(train_corpus=self.corpus_set[0],
                                                                                  vocab_word=vocab_word,
                                                                                  emb=emb)
        self.vocab_word = vocab_word
        self.trainable_emb = emb
        self.untrainable_emb = untrainable_emb

        if self.argv.save:
            self.save_word()

    def _setup_label(self):
        say('\n\nSetting up target labels...\n')
        self.vocab_label = self.preprocessor.create_vocab_label(self.corpus_set)
        if self.argv.save:
            self.save_label()

    def _setup_samples(self):
        say('\n\nSetting up samples...\n')
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)
        sample_set = self.preprocessor.create_sample_set(self.corpus_set)
        self.preprocessor.show_sample_stats(sample_set, self.vocab_label)

        self.train_samples = self.preprocessor.create_batch(sample_set[0])
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]
        say('\nMini-Batches: %d\n\n' % (self.train_samples.size()))

    def _setup_model_api(self):
        say('\n\nSetting up a model API...\n')
        self.model_api.compile(vocab_word=self.vocab_word, vocab_label=self.vocab_label, init_emb=self.trainable_emb)
        self.model_api.set_train_f()
        self.model_api.set_predict_f()

    def train(self):
        say('\n\nTRAINING START\n\n')
        self.epoch_manager.train(model_api=self.model_api,
                                 train_samples=self.train_samples,
                                 dev_samples=self.dev_samples,
                                 test_samples=self.test_samples,
                                 untrainable_emb=self.untrainable_emb)
