import numpy as np
import theano
import theano.tensor as T
import progressbar
from theano import printing
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import math

class Classifier:
    
    def glorot_init_(self, fan_in, fan_out):
        sigma = np.sqrt(2.0/fan_in+fan_out)
        return np.random.rand(fan_in, fan_out).astype(theano.config.floatX)/sigma

    def __init__(self, hidden_units = (), hidden_activation = T.nnet.relu, learning_rate = 0.1, beta_1 = 0.9, beta_2 = 0.999, dropout = 0.0):
        
        self.hidden_units = hidden_units
        self.activations = [hidden_activation]*len(self.hidden_units) + [T.nnet.softmax]
        self.lr = learning_rate
        self.saved_state = dict()
        self.num_layers = len(self.hidden_units) + 1
        self.keep_prob = np.array(1-dropout).astype(theano.config.floatX)
        self.srng = RandomStreams(np.random.RandomState().randint(999999))
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-8
        self.eta = 0.001
        
    def fit(self, X, y, batch_size = 256, num_epochs = 50, val_split=0.2, verbose=True, lamda1 = 1e-3, lamda2 = 1e-3):
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
                mask = self.srng.binomial(p=keep_prob, size=output.shape).astype(theano.config.floatX)/keep_prob
            else:
                mask = T.ones_like(output)

            output = output * mask

        params =  self.weights + self.biases

        ce_loss = T.mean(T.nnet.categorical_crossentropy(output, y_train))
        l1_loss = T.sum([T.sum(abs(v)) for v in params])
        l2_loss = T.sum([T.sum(v**2) for v in params])
        total_loss = ce_loss + lamda1 * l1_loss + lamda2 * l2_loss
        mse_loss = T.mean(T.pow(output-y_train, 2))

        batch = T.iscalar('batch')
        X_train = theano.shared(X[:val_idx])
        Y_train = theano.shared(y[:val_idx].astype(T.config.floatX))

        givens = {
            x_train: X_train[batch*batch_size: (batch+1)*batch_size],
            y_train: Y_train[batch*batch_size: (batch+1)*batch_size],
            keep_prob: self.keep_prob
        }

        self.m = dict()
        self.v = dict()

        for param in params:
            self.m[param] = T.zeros_like(param)
            self.v[param] = T.zeros_like(param)

        gradients = T.grad(total_loss, params)
        updates = OrderedDict()

        epoch = T.iscalar('epoch')
        beta_1_hat = self.beta_1 ** (1+epoch)
        beta_2_hat = self.beta_2 ** (1+epoch)

        for (p, g) in zip(params, gradients):
            self.m[p] = (self.beta_1*self.m[p] + (1-self.beta_1)*g).astype(theano.config.floatX)
            self.v[p] = (self.beta_2*self.v[p] + (1-self.beta_2)*g*g).astype(theano.config.floatX)
            m_hat = (self.m[p]/(1-beta_1_hat)).astype(theano.config.floatX)
            v_hat = (self.v[p]/(1-beta_2_hat)).astype(theano.config.floatX)
            updates[p] = (p - self.eta*m_hat/(T.sqrt(v_hat) + self.epsilon)).astype(theano.config.floatX)

        update_func = theano.function([epoch, batch], [total_loss, output.mean()], givens = givens, updates=updates, on_unused_input='ignore')

        self.cross_entropy_loss = theano.function([x_val, y_val], ce_loss/x_val.shape[0], givens = {x_train: x_val, y_train: y_val, keep_prob: np.array(1.0).astype(theano.config.floatX)}, on_unused_input='ignore')
        self.mean_squared_error_loss = theano.function([x_val, y_val], mse_loss, givens = {x_train: x_val, y_train: y_val, keep_prob: np.array(1.0).astype(theano.config.floatX)}, on_unused_input='ignore')
        self.predict = theano.function([x_val], output, givens = {x_train: x_val, keep_prob: np.array(1.0).astype(theano.config.floatX)}, on_unused_input='ignore')

        current_best = float("inf")

        for epoch in range(num_epochs):
            train_loss = 0.0
            if verbose:
                bar = progressbar.ProgressBar()
            else:
                bar = lambda x: x

            for batch in bar(range(num_batches)):
                temp_train_loss, _ = update_func(epoch, batch)
                if math.isnan(temp_train_loss):
                    import ipdb; ipdb.set_trace()
                train_loss += temp_train_loss

            train_loss /= num_samples
            val_loss = self.cross_entropy_loss(X_val, Y_val)

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

    def get_accuracy(self, X, Y):
        output = self.predict(X)
        accuracy = accuracy_score(Y.argmax(axis=1), output.argmax(axis=1))
        return accuracy


if __name__ == "__main__":

    # Test on Synthetic Datset
    
    ## (1) Linear, multidimensional
    X, Y = make_classification(n_samples = 2000, n_features = 20, n_informative = 20,
                               n_classes = 5, n_redundant = 0)

    classification_obj = Classifier(hidden_units=(32,32), learning_rate=0.0001, dropout=0.0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)
    oh_encoder = LabelBinarizer()
    Y_train = oh_encoder.fit_transform(Y_train)
    Y_test = oh_encoder.transform(Y_test)

    classification_obj.fit(X_train, Y_train, batch_size = 16, num_epochs = 70, verbose=False)

    ce_loss = classification_obj.cross_entropy_loss(X_test, Y_test)
    print("cross entropy loss on test data = %.6f" % (ce_loss))

    accuracy_train = classification_obj.get_accuracy(X_train, Y_train)
    print("accuracy on train data = %.6f" % (accuracy_train))

    accuracy_test = classification_obj.get_accuracy(X_test, Y_test)
    print("accuracy on test data = %.6f" % (accuracy_test))

    # Comparison with sklearn benchmark
    from sklearn.neural_network import MLPClassifier
    classifier_sklearn = MLPClassifier(hidden_layer_sizes=(32, 32), verbose=0, max_iter=70, batch_size=16)
    classifier_sklearn.fit(X_train, Y_train)
    print("Sklearn MLP accuracy = %.6f" %(classifier_sklearn.score(X_test, Y_test)))

    ## (2) iris
    X, Y = load_iris(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)
    oh_encoder = LabelBinarizer()
    Y_train = oh_encoder.fit_transform(Y_train)
    Y_test = oh_encoder.transform(Y_test)

    classification_obj = Classifier(hidden_units=(32,16,8,4), learning_rate=0.0001, dropout=0.9)
    classification_obj.fit(X_train, Y_train, batch_size = 120, num_epochs = 200, verbose=False)

    ce_loss = classification_obj.cross_entropy_loss(X_test, Y_test)
    print("cross entropy loss on test data = %.6f" % (ce_loss))

    accuracy_train = classification_obj.get_accuracy(X_train, Y_train)
    print("accuracy on train data = %.6f" % (accuracy_train))

    accuracy_test = classification_obj.get_accuracy(X_test, Y_test)
    print("accuracy on test data = %.6f" % (accuracy_test))
