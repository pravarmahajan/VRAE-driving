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
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from regression import Regressor

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
    parser.add_argument('--n_features', type=int, default=35,
                        help='Number of features')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs')
    parser.add_argument('--traj_data', type=str, default='data/smallSample_50_200',
                        help='path to trajectory data')
    parser.add_argument('--val_frac', type=float, default='0.2',
                        help='fraction to use for validation')
    parser.add_argument('--scale', type=float, default='1.0',
                        help='fraction to use for validation')
    parser.add_argument('--lamda', type=float, default='0.5',
                        help='weightage to driver loss')
    args = parser.parse_args()

    return args

def load_data(args):
    risk_path = "/users/PAS0536/osu9965/Telematics/Trajectories/Selected Drivers/DriverIdToRisk.csv"
    trip_segments = np.load('{}.npy'.format(args.traj_data))/args.scale
    with open(args.traj_data+"_keys.pkl", 'rb') as f:
        labels = np.array(pickle.load(f))[:, 0]

    with open(risk_path, 'r') as f:
        lines = f.readlines()
    risk_labels = dict()

    for line in lines[1:]:
        driver, risk = line.strip().split(',')
        risk = float(risk)
        risk_labels[driver] = risk
    risk_array = np.array([risk_labels[label] for label in labels])

    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    print("Number of samples: {}".format(trip_segments.shape[0]))
    rng_state = np.random.get_state()
    np.random.shuffle(trip_segments)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    np.random.set_state(rng_state)
    np.random.shuffle(risk_array)

    split_idx = int((1-args.val_frac) * trip_segments.shape[0])
    return trip_segments[:split_idx, 1:, :], labels[:split_idx], np.expand_dims(risk_array[:split_idx], -1), trip_segments[split_idx:, 1:, :], labels[split_idx:], np.expand_dims(risk_array[split_idx:], -1)

if __name__ == "__main__":
    args = parse_args()
    save_path = os.path.join("saved_weights", datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H:%M:%S"))
    os.makedirs(save_path)
    with open(os.path.join(save_path, 'args.pkl'), 'w') as f:
        pickle.dump(args, f)

    x_train, t_train, r_train, x_valid, t_valid, r_valid = load_data(args)
    num_drivers = np.max(t_train)+1
    model = VRAE(args.rnn_size, args.rnn_size, args.n_features, args.latent_size, num_drivers, batch_size=args.batch_size, lamda=args.lamda)


    batch_order = np.arange(x_train.shape[0] // model.batch_size + 1)
    val_batch_order = np.arange(x_valid.shape[0] // model.batch_size + 1)
    epoch = 0
    LB_list = []

    model.create_gradientfunctions(x_train, t_train, x_valid, t_valid)

    print("iterating")
    while epoch < args.num_epochs:
        epoch += 1
        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.
        batch_LB1 = 0.0
        batch_LB2 = 0.0
        bar = progressbar.ProgressBar()

        for batch in bar(batch_order):
            t1, t2 = model.updatefunction(epoch, model.batch_size*batch, min(model.batch_size*(batch+1), x_train.shape[0]))
            batch_LB1 += t1
            batch_LB2 += t2
            LB += t1 + t2

        LB /= len(batch_order)

        LB_list = np.append(LB_list, LB)
        print("Epoch {0} finished. LB: {1}, driver_lb: {2} time: {3}".format(epoch, LB, batch_LB2/len(batch_order), time.time() - start))
        path = os.path.join(save_path, str(epoch))
        os.makedirs(path)
        model.save_parameters(path)

        bar = progressbar.ProgressBar()
        valid_LB = 0.0
        val_LB1 = 0.0
        val_LB2 = 0.0
        for batch in bar(val_batch_order):
            t1, t2= model.likelihood(model.batch_size*batch, min(model.batch_size*(batch+1), x_valid.shape[0]))
            val_LB1 += t1
            val_LB2 += t2
            valid_LB += (t1+t2)

        val_LB1/=len(val_batch_order)
        val_LB2/=len(val_batch_order)
        print("LB loss = {}, driver_loss = {}".format(val_LB1, val_LB2))
        print("LB on validation set: {0}".format(valid_LB/len(val_batch_order)))

        ###Classification
        h_train = []
        h_val = []
        for i in range(x_train.shape[0]//model.batch_size+1):
            h_train.append(model.encoder(x_train[i*model.batch_size:(i+1)*model.batch_size].transpose(1, 0, 2).astype(theano.config.floatX)))
        for i in range(x_valid.shape[0]//model.batch_size+1):
            h_val.append(model.encoder(x_valid[i*model.batch_size:(i+1)*model.batch_size].transpose(1, 0, 2).astype(theano.config.floatX)))

        h_train = np.concatenate(h_train)
        h_val = np.concatenate(h_val)

        clf = MLPClassifier(hidden_layer_sizes=())
        clf.fit(h_train, t_train)
        print("Accuracy on train: %0.4f" % clf.score(h_train, t_train))
        print("Accuracy on val: %0.4f" % clf.score(h_val, t_valid))

        ###Risk Prediction
        risk_predictor = Regressor((64,), learning_rate = 0.001, dropout=0.5)
        risk_predictor.fit(h_train, r_train.astype(theano.config.floatX), batch_size=model.batch_size, num_epochs=50, verbose=False)
        mse = risk_predictor.mse(h_train, r_train.astype(theano.config.floatX))
        print "MSE for risk prediction on train set %.6f" %(mse)
        mse = risk_predictor.mse(h_val, r_valid.astype(theano.config.floatX))
        print "MSE for risk prediction on test set %.6f" %(mse)
