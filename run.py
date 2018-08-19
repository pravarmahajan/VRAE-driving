import numpy as np
import time
import datetime
import os
from VRAE import VRAE
import pickle
import gzip
import argparse

import theano
import progressbar
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from classification import Classifier
from returnTrainAndTestData import returnTrainAndTestData

def copy_the_model(original_model, new_model):
    for (key, value) in original_model.params.items():
        new_model.params[key].set_value(value.get_value())

def numpy_ce_loss(model_out, expected_out):
    if model_out.shape[1] == 1:
        return -(np.sum(np.log(model_out) * expected_out + (np.log(1-model_out))*(1-expected_out))/model_out.shape[0])
    else:
        return -(np.sum(np.log(model_out) * expected_out)/model_out.shape[0])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=64,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=128,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs')
    parser.add_argument('--num_drivers', type=int, default=50,
                        help='Number of drivers')
    parser.add_argument('--num_trajs', type=int, default=200,
                        help='Number of trips per driver')
    parser.add_argument('--scale', type=int, default=10,
                        help='Scale factor')
    parser.add_argument('--lamda1', type=float, default='0.33',
                        help='weightage to driver loss')
    parser.add_argument('--lamda_l2', type=float, default='1.0',
                        help='l2 regularization')
    parser.add_argument('--lamda_l1', type=float, default='1.0',
                        help='l1 regularization')
    parser.add_argument('--suffix', type=str, default='_Hd',
                        help='suffix')
    args = parser.parse_args()

    return args

def load_data(args):
    train_data, train_labels, dev_data, dev_labels, test_data, test_labels, _, num_features = returnTrainAndTestData([args.num_drivers, args.num_trajs], args.suffix, args.scale)
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, num_features

if __name__ == "__main__":
    args = parse_args()
    MAX_EARLY_STOP_COUNT = 5
    save_path = os.path.join("saved_weights", datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H:%M:%S"))
    os.makedirs(save_path)
    with open(os.path.join(save_path, 'args.pkl'), 'w') as f:
        pickle.dump(args, f)

    x_train, t_train, x_valid, t_valid, x_test, t_test, num_features = load_data(args)
    
    model = VRAE(args.rnn_size, args.rnn_size, num_features, args.latent_size, args.num_drivers, batch_size=args.batch_size, lamda1=args.lamda1, lamda_l2=args.lamda_l2, lamda_l1=args.lamda_l1)
    saved_model = VRAE(args.rnn_size, args.rnn_size, num_features, args.latent_size, args.num_drivers, batch_size=args.batch_size, lamda1=args.lamda1, lamda_l2=args.lamda_l2, lamda_l1=args.lamda_l1)


    batch_order = np.arange(x_train.shape[0] // model.batch_size + 1)
    val_batch_order = np.arange(x_valid.shape[0] // model.batch_size + 1)
    epoch = 0
    LB_list = []

    model.create_gradientfunctions(x_train, t_train, x_valid, t_valid)
    saved_model.create_gradientfunctions(x_train, t_train, x_valid, t_valid)

    print("iterating")
    best_val_score = -float("inf")
    prev_val_score = -float("inf")
    early_stop_count = MAX_EARLY_STOP_COUNT

    while epoch < args.num_epochs and early_stop_count > 0:
        epoch += 1
        start = time.time()
        np.random.shuffle(batch_order)
        train_total_loss = 0.0
        train_driver_loss = 0.0
        val_total_loss = 0.0
        val_driver_loss = 0.0

        bar = progressbar.ProgressBar()

        for batch in bar(batch_order):
            batch_end = min(model.batch_size*(batch+1), x_train.shape[0])
            batch_start = model.batch_size*batch
            l1, l2 = model.updatefunction(epoch, batch_start, batch_end)
            train_total_loss += (l1+l2)*(batch_end-batch_start)
            train_driver_loss += l2*(batch_end-batch_start)

        train_total_loss /= x_train.shape[0]
        train_driver_loss /= x_train.shape[0]

        print("Epoch {0} finished. Total Training Loss: {1}, Driver Loss: {2}".format(epoch, train_total_loss, train_driver_loss))
        path = os.path.join(save_path, str(epoch))
        os.makedirs(path)
        model.save_parameters(path)

        bar = progressbar.ProgressBar()
        valid_LB = 0.0
        val_LB1 = 0.0
        val_LB2 = 0.0
        for batch in bar(val_batch_order):
            batch_end = min(model.batch_size*(batch+1), x_valid.shape[0])
            batch_start = model.batch_size*batch
            l1, l2 = model.likelihood(batch_start, batch_end)
            val_total_loss += (l1+l2)*(batch_end-batch_start)
            val_driver_loss += l2*(batch_end-batch_start)

        val_total_loss /= x_valid.shape[0]
        val_driver_loss /= x_valid.shape[0]

        print("Val loss: {}, Val driver loss: {}".format(val_total_loss, val_driver_loss))

        ###Classification
        h_train = []
        h_val = []

        for i in range(x_train.shape[0]//model.batch_size+1):
            h_train.append(model.encoder(x_train[i*model.batch_size:(i+1)*model.batch_size].transpose(1, 0, 2).astype(theano.config.floatX)))
        for i in range(x_valid.shape[0]//model.batch_size+1):
            h_val.append(model.encoder(x_valid[i*model.batch_size:(i+1)*model.batch_size].transpose(1, 0, 2).astype(theano.config.floatX)))

        h_train = np.concatenate(h_train)
        h_val = np.concatenate(h_val)

        clf = Classifier(hidden_units=(), learning_rate=0.001, dropout=0.0)
        oh_encoder = LabelBinarizer()
        Y_train = oh_encoder.fit_transform(t_train).astype('float32')
        Y_valid = oh_encoder.fit_transform(t_valid).astype('float32')
        Y_test = oh_encoder.transform(t_test).astype('float32')

        clf.fit(h_train, Y_train, batch_size=args.batch_size, num_epochs=200, verbose=False)
        train_score = clf.get_accuracy(h_train, Y_train)
        val_score = clf.get_accuracy(h_val, Y_valid)
        print("Accuracy on train: %0.4f" % train_score)
        print("Accuracy on val: %0.4f" % val_score)

        if val_score > best_val_score:
            print "Updating model"
            best_val_score = val_score
            copy_the_model(model, saved_model)

        if val_score < prev_val_score:
            early_stop_count -= 1
            print("Early stopping count reduced to " + str(early_stop_count))
        else:
            early_stop_count = MAX_EARLY_STOP_COUNT

        prev_val_score = val_score

    h_test = []
    for i in range(x_test.shape[0]//saved_model.batch_size+1):
        h_test.append(saved_model.encoder(x_test[i*saved_model.batch_size:(i+1)*saved_model.batch_size].transpose(1, 0, 2).astype(theano.config.floatX)))
    h_test = np.concatenate(h_test)
    print("Accuracy on test: %0.4f" % clf.get_accuracy(h_test, Y_test))
