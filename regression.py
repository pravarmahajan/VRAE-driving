import numpy as np
import theano
import theano.tensor as T
import progressbar
from theano import printing
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Regressor:
    
    def glorot_init_(self, fan_in, fan_out):
        sigma = np.sqrt(2.0/fan_in+fan_out)
        return np.random.rand(fan_in, fan_out).astype(theano.config.floatX)/sigma

    def __init__(self, hidden_units = (), hidden_activation = T.nnet.relu, learning_rate = 0.1, momentum = 0.9, dropout = 0.0):
        
        self.hidden_units = hidden_units
        self.activations = [hidden_activation]*len(self.hidden_units) + [T.nnet.sigmoid]
        self.lr = learning_rate
        self.lr_m = momentum
        self.saved_state = dict()
        self.num_layers = len(self.hidden_units) + 1
        self.keep_prob = np.array(1-dropout).astype(theano.config.floatX)
        self.srng = RandomStreams(np.random.RandomState().randint(999999))
        
    def fit(self, X, y, batch_size = 256, num_epochs = 50, val_split=0.2, verbose=True):
        val_idx = int(X.shape[0]*(1-val_split))
        num_samples = val_idx
        num_batches = num_samples//batch_size + 1 if not num_samples%batch_size==0 else num_samples//batch_size
        X_val = X[val_idx:]
        Y_val = y[val_idx:]

        self.in_units = X.shape[1]
        if y.ndim == 1:
            self.out_units = 1
        else:
            self.out_units = y.shape[1]

        self.biases = []
        self.weights = []

        for (i, h) in enumerate(self.hidden_units):
            self.biases.append(theano.shared(np.zeros(h).astype(theano.config.floatX)))
            if i>0:
                self.weights.append(theano.shared(self.glorot_init_(self.hidden_units[i-1], h)))


        self.biases.append(theano.shared(np.zeros(self.out_units).astype(theano.config.floatX)))

        if len(self.hidden_units) > 0:
            self.weights.append(theano.shared(self.glorot_init_(self.hidden_units[-1], self.out_units)))
            self.weights = [theano.shared(self.glorot_init_(self.in_units, self.hidden_units[0]))] + self.weights
        else:
            self.weights.append(theano.shared(self.glorot_init_(self.in_units, self.out_units)))

        x_train = T.matrix('x_train')
        y_train = T.matrix('y_train')

        x_val = T.matrix('x_val')
        y_val = T.matrix('y_val')

        keep_prob = T.scalar(dtype=theano.config.floatX)
        output = x_train

        for i in range(self.num_layers):
            output = self.activations[i](T.dot(output, self.weights[i])+self.biases[i])

            if i != self.num_layers - 1:
                mask = self.srng.binomial(p=keep_prob, size=output.shape).astype(theano.config.floatX)
            else:
                mask = T.ones_like(output)

            output = output * mask/keep_prob

        loss = T.sum(T.pow(output- y_train, 2))/y_train.shape[1]
        mean_loss = T.mean(T.pow(output- y_train, 2))

        batch = T.iscalar('batch')
        X_train = theano.shared(X[:val_idx])
        Y_train = theano.shared(y[:val_idx].astype(T.config.floatX))

        givens = {
            x_train: X_train[batch*batch_size: (batch+1)*batch_size],
            y_train: Y_train[batch*batch_size: (batch+1)*batch_size],
            keep_prob: self.keep_prob
        }

        params =  self.weights + self.biases
        gradients = T.grad(mean_loss, params)
        updates = OrderedDict()
        momenta = OrderedDict()

        for i, p in enumerate(params):
            momenta[p] = self.lr_m * momenta.get(p, 0.0) + self.lr * gradients[i]
            updates[p] = p - momenta[p]

        update_func = theano.function([batch], loss, givens = givens, updates=updates)

        self.mse = theano.function([x_val, y_val], mean_loss, givens = {x_train: x_val, y_train: y_val, keep_prob: np.array(1.0).astype(theano.config.floatX)})
        self.predict = theano.function([x_val], output, givens = {x_train: x_val, keep_prob: np.array(1.0).astype(theano.config.floatX)})

        current_best = float("inf")

        for epoch in range(num_epochs):
            train_loss = 0.0
            if verbose:
                bar = progressbar.ProgressBar()
            else:
                bar = lambda x: x

            for batch in bar(range(num_batches)):
                train_loss += update_func(batch)
            train_loss /= num_samples
            val_loss = self.mse(X_val, Y_val)

            if verbose:
                print("Epoch %d, Train loss: %.6f, Validation loss: %.6f" %(epoch+1, train_loss, val_loss))
            if val_loss < current_best:
                current_best = val_loss
                if verbose:
                    print "saving current state"
                self.save_state()


        self.restore_state()

    def save_state(self):
        for param in self.weights + self.biases:
            self.saved_state[param] = np.copy(param.get_value())

    def restore_state(self):
        if len(self.saved_state) < 1:
            return

        for param, value in self.saved_state.items():
            param.set_value(value)

if __name__ == "__main__":
    regression_obj = Regressor(hidden_units=(16,8,4), learning_rate=0.1, momentum=0.9)
    ndim = 20
    num_points = 2000
    nout = 5

    def generate_synthetic_output(X, w, b, min = None, max = None):

        n = X.shape[0]
        error = np.random.randn(n, w.shape[1])

        y = np.dot(X, w) + b + 0.1 * error

        if min == None:
            generate_synthetic_output.min = y.min()
            generate_synthetic_output.max = y.max()

        y = (y- generate_synthetic_output.min)/(generate_synthetic_output.max-generate_synthetic_output.min)

        return y

    X_train = np.random.rand(num_points, ndim)
    w = np.random.rand(ndim,nout)
    b = np.random.rand(1,nout)
    Y_train = generate_synthetic_output(X_train, w, b)

    regression_obj.fit(X_train, Y_train, batch_size = 16, num_epochs = 100)

    X_test = np.random.rand(int(num_points*0.2), ndim)
    Y_test = generate_synthetic_output(X_test, w, b, generate_synthetic_output.min, generate_synthetic_output.max)

    mse_loss = regression_obj.mse(X_test, Y_test)
    print("mse loss on test data = %.6f" % (mse_loss))
