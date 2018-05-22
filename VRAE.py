import numpy as np
import theano
import theano.tensor as T
from theano import printing

import cPickle as pickle
from collections import OrderedDict

class VRAE:
    """This class implements the Variational Recurrent Auto Encoder"""
    def __init__(self, hidden_units_encoder, hidden_units_decoder, features, latent_variables, num_drivers, b1=0.9, b2=0.999, learning_rate=0.001, sigma_init=None, batch_size=256):
        self.batch_size = batch_size
        self.hidden_units_encoder = hidden_units_encoder
        self.hidden_units_decoder = hidden_units_decoder
        self.features = features
        self.latent_variables = latent_variables

        self.b1 = theano.shared(np.array(b1).astype(theano.config.floatX), name = "b1")
        self.b2 = theano.shared(np.array(b2).astype(theano.config.floatX), name = "b2")
        self.learning_rate = theano.shared(np.array(learning_rate).astype(theano.config.floatX), name="learning_rate")


        #Initialize all variables as shared variables so model can be run on GPU

        #encoder
        sigma_init = np.sqrt(2.0/(hidden_units_encoder+features))
        W_xhe = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_encoder)).astype(theano.config.floatX), name='W_xhe')

        sigma_init = np.sqrt(1.0/(hidden_units_encoder))
        W_hhe = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder,hidden_units_encoder)).astype(theano.config.floatX), name='W_hhe')
        
        b_he = theano.shared(np.zeros((hidden_units_encoder,1)).astype(theano.config.floatX), name='b_hhe', broadcastable=(False,True))

        sigma_init = np.sqrt(2.0/(latent_variables+hidden_units_encoder))
        W_hmu = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder, latent_variables)).astype(theano.config.floatX), name='W_hmu')
        b_hmu = theano.shared(np.zeros((latent_variables,1)).astype(theano.config.floatX), name='b_hmu', broadcastable=(False,True))

        W_hsigma = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder, latent_variables)).astype(theano.config.floatX), name='W_hsigma')
        b_hsigma = theano.shared(np.zeros((latent_variables,1)).astype(theano.config.floatX), name='b_hsigma', broadcastable=(False,True))

        #decoder
        W_zh = theano.shared(np.random.normal(0,sigma_init,(latent_variables, hidden_units_decoder)).astype(theano.config.floatX), name='W_zh')
        b_zh = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_zh', broadcastable=(False,True))

        sigma_init = np.sqrt(1.0/(hidden_units_encoder))
        W_hhd = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,hidden_units_decoder)).astype(theano.config.floatX), name='W_hhd')
        W_xhd = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_decoder)).astype(theano.config.floatX), name='W_hxd')
        
        b_hd = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_hxd', broadcastable=(False,True))
        
        sigma_init = np.sqrt(2.0/(features+hidden_units_encoder))
        W_hx = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder, features)).astype(theano.config.floatX), name='W_hx')
        b_hx = theano.shared(np.zeros((features,1)).astype(theano.config.floatX), name='b_hx', broadcastable=(False,True))

        sigma_init = np.sqrt(2.0/(hidden_units_encoder + num_drivers))
        W_driver = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder, num_drivers)).astype(theano.config.floatX), name='W_driver')
        b_driver = theano.shared(np.zeros((num_drivers,1)).astype(theano.config.floatX), name='b_driver', broadcastable=(False,True))

        self.params = OrderedDict([("W_xhe", W_xhe), ("W_hhe", W_hhe), ("b_he", b_he), ("W_hmu", W_hmu), ("b_hmu", b_hmu), \
            ("W_hsigma", W_hsigma), ("b_hsigma", b_hsigma), ("W_zh", W_zh), ("b_zh", b_zh), ("W_hhd", W_hhd), ("W_xhd", W_xhd), ("b_hd", b_hd),
            ("W_hx", W_hx), ("b_hx", b_hx), ("W_driver", W_driver), ("b_driver", b_driver)])

        #Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key,value in self.params.items():
            if 'b' in key:
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key, broadcastable=(False, True))
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key, broadcastable=(False, True))
            else:
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)


    def create_gradientfunctions(self, train_data, train_labels, val_data, val_labels):
        """This function takes as input the whole dataset and creates the entire model"""
        def encodingstep(x_t, h_t):
            return T.tanh(T.dot(x_t, self.params["W_xhe"]) + T.dot(h_t, self.params['W_hhe']) + self.params["b_he"].squeeze())

        x = T.tensor3("x")

        h0_enc = T.matrix("h0_enc")
        result, _ = theano.scan(encodingstep, 
                sequences = x, 
                outputs_info = h0_enc)

        h_encoder = result[-1]

        #log sigma encoder is squared
        mu_encoder = T.dot(h_encoder, self.params["W_hmu"]) + self.params["b_hmu"].squeeze()
        log_sigma_encoder = T.dot(h_encoder, self.params["W_hsigma"]) + self.params["b_hsigma"].squeeze()

        #Use a very wide prior to make it possible to learn something with Z
        logpz = 0.005 * T.sum(1 + log_sigma_encoder - mu_encoder**2 - T.exp(log_sigma_encoder), axis = 1)

        seed = 42
        
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            srng = T.shared_randomstreams.RandomStreams(seed=seed)

        #Reparametrize Z
        eps = srng.normal((self.batch_size, self.latent_variables), avg = 0.0, std = 1.0, dtype=theano.config.floatX)
        z = mu_encoder + T.exp(0.5 * log_sigma_encoder) * eps

        h0_dec = T.tanh(T.dot(z, self.params["W_zh"]) + self.params["b_zh"].squeeze())

        def decodingstep(x_t, h_t):
            h = T.tanh(h_t.dot(self.params["W_hhd"]) + x_t.dot(self.params["W_xhd"]) + self.params["b_hd"].squeeze())
            x = T.nnet.sigmoid(h.dot(self.params["W_hx"]) + self.params["b_hx"].squeeze())

            return x, h

        x0 = T.matrix("x0")
        [y, _], _ = theano.scan(decodingstep,
                n_steps = x.shape[0], 
                outputs_info = [x0, h0_dec])

        # Clip y to avoid NaNs, necessary when lowerbound goes to 0
        # 128 x 8 x 35
        y = T.clip(y, 1e-6, 1 - 1e-6)
        #logpxz = T.sum(-T.nnet.binary_crossentropy(y,x), axis = 0)
        logpxz = -T.sum(T.pow(y-x, 2), axis = 0)
        logpxz = T.mean(logpxz, axis = 1)

        #Average over batch dimension
        logpx = T.mean(logpxz + logpz) 

        #Driver output
        batch = T.iscalar('batch')

        driver_output = T.nnet.softmax(T.dot(h_encoder, self.params['W_driver']) + self.params['b_driver'].squeeze())

        train_labels = theano.shared(train_labels)
        driver_loss = -T.mean(T.nnet.categorical_crossentropy(driver_output, train_labels[batch*self.batch_size:(batch+1)*self.batch_size]))

        #Compute all the gradients
        gradients = T.grad(logpx+driver_loss, self.params.values(), disconnected_inputs='ignore')
        #gradients = T.grad(driver_loss, self.params.values(), disconnected_inputs='ignore')

        #Let Theano handle the updates on parameters for speed
        updates = OrderedDict()
        epoch = T.iscalar("epoch")
        gamma = (T.sqrt(1 - (1 - self.b2)**epoch)/(1 - (1 - self.b1)**epoch)).astype(theano.config.floatX)

        #Adam
        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):
            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v)+ 1e-8)
            updates[m] = new_m
            updates[v] = new_v

        #data = theano.shared(data[:,1:,:].swapaxes(1, 2))
        train_data = theano.shared(train_data.transpose(1,0,2)).astype(theano.config.floatX)

        givens = {
            h0_enc: np.zeros((self.batch_size, self.hidden_units_encoder)).astype(theano.config.floatX), 
            x0:     np.zeros((self.batch_size, self.features)).astype(theano.config.floatX),
            x:      train_data[:,batch*self.batch_size:(batch+1)*self.batch_size,:],
            
        }

        self.updatefunction = theano.function([epoch, batch], [logpx, driver_loss], updates=updates, givens=givens, allow_input_downcast=True)

        x_val = theano.shared(val_data.transpose(1, 0, 2)).astype(theano.config.floatX)
        givens[x] = x_val[:, batch*self.batch_size:(batch+1)*self.batch_size,:]
        self.likelihood = theano.function([batch], [logpxz.mean(), driver_loss], givens=givens)

        x_test = T.tensor3("x_test")
        test_givens = {
            x: x_test,
            h0_enc: np.zeros((self.batch_size, self.hidden_units_encoder)).astype(theano.config.floatX), 
        }

        self.encoder = theano.function([x_test], h_encoder, givens=test_givens)


        return True


    def save_parameters(self, path):
        """Saves all the parameters in a way they can be retrieved later"""
        pickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params.pkl", "wb"))
        pickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/m.pkl", "wb"))
        pickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/v.pkl", "wb"))

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way"""
        p_list = pickle.load(open(path + "/params.pkl", "rb"))
        m_list = pickle.load(open(path + "/m.pkl", "rb"))
        v_list = pickle.load(open(path + "/v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))

