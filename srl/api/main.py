import theano
import numpy as np
import random

theano.config.floatX = 'float32'
np.random.seed(0)
random.seed(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='The Neural PAS System')

    ########
    # Mode #
    ########
    parser.add_argument('-mode', default='train', help='train/test')

    ##########
    # Inputs #
    ##########
    parser.add_argument('--train_data', default=None, help='path to training data')
    parser.add_argument('--dev_data', default=None, help='path to development data')
    parser.add_argument('--test_data', default=None, help='path to test data')
    parser.add_argument('--load_word', type=str, default=None, help='word')
    parser.add_argument('--load_label', type=str, default=None, help='label')
    parser.add_argument('--load_param', type=str, default=None, help='load trained parameters')
    parser.add_argument('--load_config', type=str, default=None, help='load configuration')

    ###########
    # Outputs #
    ###########
    parser.add_argument('--output_fn', type=str, default=None, help='Output file name')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')
    parser.add_argument('--output', type=str, default=None, help='n_best/pretrain')

    #########
    # Model #
    #########
    parser.add_argument('--model', type=str, default='base', help='base/rank/rerank/sep/jack')
    parser.add_argument('--save', type=int, default=1, help='save model')
    parser.add_argument('--result', type=bool, default=False, help='output results')

    ###############
    # Data Option #
    ###############
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--vocab_cut_off', type=int, default=0)
    parser.add_argument('--sec', type=int, default=None, help='The target section of the jack knife')

    ########################
    # Neural Architectures #
    ########################
    parser.add_argument('--unit', default='gru', help='unit')
    parser.add_argument('--fix', type=int, default=0, help='fix or not init embeddings')
    parser.add_argument('--layers',  type=int, default=1, help='number of layers')
    parser.add_argument('--output_layer',  type=int, default=0, help='softmax/memm/crf')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--dim_emb',    type=int, default=32, help='dimension of word embeddings')
    parser.add_argument('--dim_posit',  type=int, default=5, help='dimension of position embeddings')
    parser.add_argument('--dim_hidden', type=int, default=32, help='dimension of hidden layer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')

    #######################
    # Training Parameters #
    #######################
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch size')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')
    parser.add_argument('--n_best', type=int, default=10, help='How many best lists are created')

    argv = parser.parse_args()
    print
    print argv
    print

    ########
    # Mode #
    ########
    if argv.mode == 'train':
        import train
        train.main(argv)
    else:
        import test
        test.main(argv)
