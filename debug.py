from VRAE import VRAE
import numpy as np
import os
from run import load_data

def load_params():
    dirs = os.listdir('./saved_weights')
    print(dirs)

    path = os.path.join('saved_weights', dirs[int(raw_input('enter your choice: '))])
    print("Loading params from: "+path)
    args = np.load(os.path.join(path, 'args.pkl'))
    return args, path

def init_model():
    model = VRAE(args.rnn_size, args.rnn_size, args.n_features, args.latent_size, num_drivers, batch_size=args.batch_size)
    model.create_gradientfunctions(x_train, t_train, x_valid, t_valid)
    return model

def load_epoch(model, e):
    model.load_parameters(os.path.join(path, str(e)))

def forward(model, batch_num=0):
    return model.encoder(x_train[batch_num*model.batch_size:(batch_num+1)*model.batch_size].transpose(1, 0, 2))

args, path = load_params()
x_train, t_train, x_valid, t_valid = load_data(args)
num_drivers = np.max(t_train)+1
model = init_model()
