import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import utility as my_util
from sklearn.utils import shuffle


class HiddenLayer(object):
    def __init__(self, M1, M2, layer_id):
        self.id = layer_id
        self.M1 = M1
        self.M2 = M2
        W, b = my_util.create_weight_and_bias(M1, M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward_one_step(self, X):
        return theano.tensor.nnet.relu(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers = []
        self.params = []

    def fit(self, X, Y, learning_rate=10e-8, mu=0.99, decay=0.999, reg=10e-12, epsilon=10e-10, epochs=50, batch_sz=50):
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        decay = np.float32(decay)
        reg = np.float32(reg)
        epsilon = np.float32(epsilon)

        X, Y = shuffle(X, Y)
        print('Shuffling done')
        X, Y = X[:-500], Y[:-500]
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid, Yvalid = X[-100:], Y[-100:]
        X, Y = X[:-100], Y[:-100]

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y))
        M1 = D
        id_counter = 0
        for M2 in self.hidden_layer_sizes:
            hiddenLayer = HiddenLayer(M1, M2, id_counter)
            self.hidden_layers.append(hiddenLayer)
            M1 = M2
            id_counter += 1
        W_last, b_last = my_util.create_weight_and_bias(M1, K)
        self.W_last = theano.shared(W_last, 'W_last')
        self.b_last = theano.shared(b_last, 'b_last')

        # maintain list of all parameters for momentum and adaptive learning rate calculation
        self.params.append(self.W_last)
        self.params.append(self.b_last)
        for param in self.hidden_layers:
            self.params.append(param.W)
            self.params.append(param.b)


        dParams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
        # cache_matrix = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        # setup inputs
        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        # cost function
        reg_cost = reg * T.sum([(p * p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + reg_cost

        # prediction function
        prediction = self.predict(thX)

        # defining theano operations

        updates = [
                      (param, param + mu * dP - learning_rate * T.grad(cost, param) ) for
                      dP, param in
                      zip(dParams, self.params)
                  ] + [
                      (dP, mu * dP - learning_rate * T.grad(cost, param) ) for
                      dP, param in zip(dParams, self.params)
                  ]
        predict_op = theano.function(inputs=[thX], outputs=[prediction])
        cost_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])
        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        n_batches = int(N / batch_sz)
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                Ybatch = Y[j * batch_sz:(j * batch_sz + batch_sz)]

                train_op(Xbatch, Ybatch)

                if j % 1 == 0:
                    c, p = cost_op(Xvalid, Yvalid)
                    for param in self.params:
                        print(param)
                    costs.append(c)
                    e = my_util.error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        plt.plot(costs)
        plt.show()

    def forward(self, X):
        Z = X
        for hidden_layer in self.hidden_layers:
            Z = hidden_layer.forward_one_step(Z)
        return T.nnet.softmax(Z.dot(self.W_last) + self.b_last)

    def predict(self, X):
        Y = self.forward(X)
        return T.argmax(Y, axis=1)

def main():
    X, Y = my_util.getData()
    model = ANN([20, 10])
    model.fit(X, Y)


if __name__ == '__main__':
    main()