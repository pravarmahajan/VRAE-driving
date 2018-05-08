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

#np.random.seed(42)

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
    args = parser.parse_args()

    return args

def load_data(args):
    trip_segments = np.load('{}.npy'.format(args.traj_data))/args.scale
    with open(args.traj_data+"_keys.pkl", 'rb') as f:
        labels = np.array(pickle.load(f))[:, 0]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    print("Number of samples: {}".format(trip_segments.shape[0]))
    rng_state = np.random.get_state()
    np.random.shuffle(trip_segments)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    split_idx = int((1-args.val_frac) * trip_segments.shape[0])
    return trip_segments[:split_idx, 1:, :], labels[:split_idx], trip_segments[split_idx:, 1:, :], labels[split_idx:]

args = parse_args()

x_train, t_train, x_valid, t_valid = load_data(args)
num_drivers = np.max(t_train)+1
model = VRAE(args.rnn_size, args.rnn_size, args.n_features, args.latent_size, num_drivers, batch_size=args.batch_size)


batch_order = np.arange(int(x_train.shape[0] / model.batch_size))
val_batch_order = np.arange(int(x_valid.shape[0] / model.batch_size))
epoch = 0
LB_list = []
path = "./cache"

model.create_gradientfunctions(x_train, t_train, x_valid, t_valid)
if os.path.isfile(path + "params.pkl"):
    print("Restarting from earlier saved parameters!")
    model.load_parameters(path)
    LB_list = np.load(path + "LB_list.npy")
    epoch = len(LB_list)

if __name__ == "__main__":
    print("iterating")
    save_path = os.path.join("saved_weights", datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H:%M:%S"))
    while epoch < args.num_epochs:
        epoch += 1
        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.
        batch_LB1 = 0.0
        batch_LB2 = 0.0
        bar = progressbar.ProgressBar()

        for batch in bar(batch_order):
            t1, t2 = model.updatefunction(epoch, batch)
            batch_LB1 += t1
            batch_LB2 += t2
            LB += t1 + t2

        LB /= len(batch_order)

        LB_list = np.append(LB_list, LB)
        print("Epoch {0} finished. LB: {1}, time: {2}".format(epoch, LB, time.time() - start))
        path = os.path.join(save_path, str(epoch))
        os.makedirs(path)
        model.save_parameters(path)

        bar = progressbar.ProgressBar()
        valid_LB = 0.0
        val_LB1 = 0.0
        val_LB2 = 0.0
        for batch in bar(val_batch_order):
            t1, t2= model.likelihood(batch)
            val_LB1 += t1
            val_LB2 += t2
            valid_LB += (t1+t2)

        val_LB1/=len(val_batch_order)
        val_LB2/=len(val_batch_order)
        print("LB loss = {}, driver_loss = {}".format(val_LB1, val_LB2))
        print("LB on validation set: {0}".format(valid_LB/len(val_batch_order)))

